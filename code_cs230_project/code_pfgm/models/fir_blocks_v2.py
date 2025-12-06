"""
SwinFIR Building Blocks v2 for PFGM-FIR with Cross-Attention.

This module extends the original fir_blocks.py with:
- FIRResBlockV2: Residual block with optional cross-attention
- Integration points for MRI context conditioning

The blocks maintain the same SFB (Spatial-Frequency Block) and Swin attention
from v1, but add cross-attention after self-attention for MRI conditioning.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_attention import SwinTransformerBlock
from .cross_attention import CrossAttentionBlock


def get_num_groups(channels: int, target: int = 32) -> int:
    """Get a valid number of groups for GroupNorm."""
    if channels >= target and channels % target == 0:
        return target
    for g in [32, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


class FourierUnit(nn.Module):
    """
    Core Fourier Transform unit.
    
    Transforms input to frequency domain, applies convolution,
    and transforms back to spatial domain.
    """
    
    def __init__(self, embed_dim: int, fft_norm: str = 'ortho'):
        super().__init__()
        self.conv_layer = nn.Conv2d(embed_dim * 2, embed_dim * 2, kernel_size=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.fft_norm = fft_norm
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        input_dtype = x.dtype
        input_shape = x.shape[-2:]
        
        x = x.float()
        
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=2)
        ffted = ffted.view(batch, -1, ffted.size(3), ffted.size(4))
        
        ffted = self.conv_layer(ffted.float()).float()
        ffted = self.relu(ffted)
        
        ffted = ffted.view(batch, -1, 2, ffted.size(2), ffted.size(3))
        ffted = torch.complex(ffted[:, :, 0, :, :], ffted[:, :, 1, :, :])
        
        output = torch.fft.irfftn(ffted, s=input_shape, dim=fft_dim, norm=self.fft_norm)
        return output.to(input_dtype)


class SpectralTransform(nn.Module):
    """Spectral Transform module from SwinFIR."""
    
    def __init__(self, embed_dim: int, last_conv: bool = False):
        super().__init__()
        self.last_conv = last_conv
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.fu = FourierUnit(embed_dim // 2)
        self.conv2 = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=1)
        
        if last_conv:
            self.final_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        else:
            self.final_conv = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        output = self.fu(x)
        output = self.conv2(x + output)
        
        if self.final_conv is not None:
            output = self.final_conv(output)
        
        return output


class ResB(nn.Module):
    """Simple Residual Block (spatial branch)."""
    
    def __init__(self, embed_dim: int, reduction: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // reduction, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(embed_dim // reduction, embed_dim, kernel_size=3, padding=1),
        )
        self.scale = 1.0 / math.sqrt(2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * (x + self.body(x))


class SFB(nn.Module):
    """
    Spatial-Frequency Block (SFB) from SwinFIR.
    
    Combines local spatial features with global frequency features.
    """
    
    def __init__(self, embed_dim: int, reduction: int = 1):
        super().__init__()
        
        self.S = ResB(embed_dim, reduction)
        self.F = SpectralTransform(embed_dim)
        self.norm = nn.GroupNorm(get_num_groups(embed_dim * 2), embed_dim * 2)
        self.fusion = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.S(x)
        f = self.F(x)
        out = torch.cat([s, f], dim=1)
        out = self.norm(out)
        out = self.fusion(out)
        return out


class Downsample(nn.Module):
    """Downsampling layer using strided convolution."""
    
    def __init__(self, channels: int, out_channels: Optional[int] = None):
        super().__init__()
        out_channels = out_channels or channels
        self.conv = nn.Conv2d(channels, out_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling layer using nearest-neighbor + convolution."""
    
    def __init__(self, channels: int, out_channels: Optional[int] = None):
        super().__init__()
        out_channels = out_channels or channels
        self.conv = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class FIRResBlockV2(nn.Module):
    """
    FIR Residual Block v2: Core building block with Cross-Attention.
    
    Extends the original FIRResBlock with optional cross-attention for
    MRI conditioning. Architecture:
    
    1. Conv + FiLM conditioning (radius/timestep)
    2. Optional Swin self-attention
    3. Optional Cross-attention (MRI context) <-- NEW
    4. SFB residual connection
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        radius_emb_dim: Dimension of radius embedding.
        use_swin: Whether to use Swin attention.
        use_sfb: Whether to use SFB for residual.
        use_cross_attn: Whether to use cross-attention for MRI conditioning.
        context_dim: Dimension of MRI context tokens (required if use_cross_attn).
        cross_attn_heads: Number of cross-attention heads.
        input_resolution: (H, W) for Swin attention.
        num_heads: Number of attention heads for Swin.
        window_size: Window size for Swin attention.
        mlp_ratio: MLP expansion ratio.
        drop_path: Stochastic depth rate.
        dropout: Dropout rate.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        radius_emb_dim: int,
        use_swin: bool = False,
        use_sfb: bool = True,
        use_cross_attn: bool = False,
        context_dim: Optional[int] = None,
        cross_attn_heads: int = 8,
        input_resolution: Tuple[int, int] = (64, 64),
        num_heads: int = 4,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_swin = use_swin
        self.use_sfb = use_sfb
        self.use_cross_attn = use_cross_attn
        
        # First conv block
        self.norm1 = nn.GroupNorm(get_num_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Radius conditioning via FiLM (scale and shift)
        self.radius_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(radius_emb_dim, out_channels * 2),
        )
        
        # Second conv block
        self.norm2 = nn.GroupNorm(get_num_groups(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Optional Swin self-attention
        if use_swin:
            self.swin_blocks = nn.ModuleList([
                SwinTransformerBlock(
                    dim=out_channels,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if i % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path,
                )
                for i in range(2)
            ])
        else:
            self.swin_blocks = None
        
        # Optional Cross-attention for MRI conditioning
        if use_cross_attn:
            assert context_dim is not None, "context_dim required when use_cross_attn=True"
            self.cross_attn = CrossAttentionBlock(
                dim=out_channels,
                context_dim=context_dim,
                heads=cross_attn_heads,
                dropout=dropout,
            )
        else:
            self.cross_attn = None
        
        # Residual connection with SFB or conv
        if use_sfb:
            self.residual = SFB(out_channels)
        else:
            self.residual = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Skip connection (if channels change)
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
        
        # Residual scaling factor for stability
        self.residual_scale = 1.0 / math.sqrt(2)
        
        # Initialize last conv to zero for better training dynamics
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        r_emb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        x_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            r_emb: Radius embedding of shape [B, radius_emb_dim]
            context: Optional MRI context tokens of shape [B, N_ctx, C_ctx]
            x_size: Optional (H, W) for Swin attention
        
        Returns:
            Output tensor of shape [B, out_channels, H, W]
        """
        B, C, H, W = x.shape
        
        # Skip connection
        skip = self.skip(x)
        
        # First conv
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Radius conditioning via FiLM
        r_cond = self.radius_proj(r_emb)
        scale, shift = r_cond.chunk(2, dim=-1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        h = h * (1 + scale) + shift
        
        # Second conv
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # Optional Swin self-attention
        if self.swin_blocks is not None:
            h_flat = h.flatten(2).transpose(1, 2)
            current_size = x_size if x_size is not None else (H, W)
            
            for swin_block in self.swin_blocks:
                h_flat = swin_block(h_flat, current_size)
            
            h = h_flat.transpose(1, 2).view(B, -1, H, W)
        
        # Optional Cross-attention with MRI context
        if self.cross_attn is not None and context is not None:
            h = self.cross_attn(h, context)
        
        # Residual with SFB (global features) - scaled for stability
        return self.residual_scale * (self.residual(skip) + h)


class DecoderBlockV2(nn.Module):
    """
    Decoder block with upsample, skip connection fusion, and cross-attention.
    
    Architecture:
        1. Upsample from previous stage
        2. Concatenate with encoder skip connection
        3. 1x1 conv to reduce channels
        4. FIRResBlockV2 (with optional cross-attention)
    
    Args:
        in_channels: Input channels (from previous decoder stage)
        skip_channels: Skip connection channels (from encoder)
        out_channels: Output channels
        context_dim: MRI context dimension for cross-attention
        use_cross_attn: Whether to use cross-attention
        num_res_blocks: Number of residual blocks
        **block_kwargs: Additional arguments for FIRResBlockV2
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        radius_emb_dim: int,
        context_dim: Optional[int] = None,
        use_cross_attn: bool = True,
        num_res_blocks: int = 2,
        use_swin: bool = False,
        use_sfb: bool = True,
        cross_attn_heads: int = 8,
        num_heads: int = 4,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Upsample
        self.upsample = Upsample(in_channels, out_channels)
        
        # Skip connection fusion (concat + 1x1 conv)
        self.skip_conv = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=1)
        
        # Residual blocks (first block gets cross-attention)
        self.blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            self.blocks.append(
                FIRResBlockV2(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    radius_emb_dim=radius_emb_dim,
                    use_swin=use_swin,
                    use_sfb=use_sfb,
                    use_cross_attn=use_cross_attn if i == 0 else False,  # Cross-attn only on first block
                    context_dim=context_dim,
                    cross_attn_heads=cross_attn_heads,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path,
                    dropout=dropout,
                )
            )
    
    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        r_emb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        x_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input from previous decoder stage [B, in_ch, H, W]
            skip: Skip connection from encoder [B, skip_ch, H*2, W*2]
            r_emb: Radius embedding [B, radius_emb_dim]
            context: MRI context tokens [B, N_ctx, C_ctx]
            x_size: Optional (H, W) for Swin attention
        
        Returns:
            Output [B, out_ch, H*2, W*2]
        """
        # Upsample
        h = self.upsample(x)
        
        # Concatenate with skip connection
        h = torch.cat([h, skip], dim=1)
        h = self.skip_conv(h)
        
        # Update size after upsample
        B, C, H, W = h.shape
        current_size = (H, W)
        
        # Apply blocks
        for block in self.blocks:
            h = block(h, r_emb, context=context, x_size=current_size)
        
        return h
