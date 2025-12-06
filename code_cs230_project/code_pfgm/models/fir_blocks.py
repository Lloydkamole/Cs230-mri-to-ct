"""
SwinFIR Building Blocks for PFGM-FIR.

This module contains the core building blocks from SwinFIR:
- SFB (Spatial-Frequency Block): Combines spatial convolution with FFT for global receptive field
- SpectralTransform: Fast Fourier Convolution component
- FourierUnit: Core FFT operation
- FIRResBlock: Residual block with SwinFIR components + radius conditioning

The key innovation from SwinFIR is the SFB block that gives image-wide receptive
field in early layers through Fast Fourier Convolution.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .swin_attention import SwinTransformerBlock


class FourierUnit(nn.Module):
    """
    Core Fourier Transform unit.
    
    Transforms input to frequency domain, applies convolution,
    and transforms back to spatial domain.
    
    This gives image-wide receptive field because convolution
    in frequency domain corresponds to global operations in spatial domain.
    """
    
    def __init__(self, embed_dim: int, fft_norm: str = 'ortho'):
        super().__init__()
        
        # Conv in frequency domain (operates on real+imag stacked)
        self.conv_layer = nn.Conv2d(embed_dim * 2, embed_dim * 2, kernel_size=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.fft_norm = fft_norm
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, C, H, W]
        
        Returns:
            Output tensor of shape [B, C, H, W]
        """
        batch = x.shape[0]
        input_dtype = x.dtype
        input_shape = x.shape[-2:]
        
        # FFT and torch.complex require float32 - doesn't support bfloat16/float16
        # Force everything to float32 for numerical stability
        x = x.float()
        
        # Transform to frequency domain
        # Output: complex tensor of shape [B, C, H, W//2+1]
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        
        # Stack real and imaginary parts: [B, C, 2, H, W//2+1]
        ffted = torch.stack((ffted.real, ffted.imag), dim=2)
        
        # Reshape for conv: [B, C*2, H, W//2+1]
        ffted = ffted.view(batch, -1, ffted.size(3), ffted.size(4))
        
        # Apply convolution in frequency domain (force float32)
        ffted = self.conv_layer(ffted.float()).float()
        ffted = self.relu(ffted)
        
        # Reshape back: [B, C, 2, H, W//2+1]
        ffted = ffted.view(batch, -1, 2, ffted.size(2), ffted.size(3))
        
        # Convert back to complex (requires float32)
        ffted = torch.complex(ffted[:, :, 0, :, :], ffted[:, :, 1, :, :])
        
        # Inverse FFT back to spatial domain
        output = torch.fft.irfftn(ffted, s=input_shape, dim=fft_dim, norm=self.fft_norm)
        
        # Convert back to original dtype for mixed precision compatibility
        return output.to(input_dtype)


class SpectralTransform(nn.Module):
    """
    Spectral Transform module from SwinFIR.
    
    Architecture:
    - Conv 1x1 to reduce channels
    - FourierUnit for global processing
    - Conv 1x1 to restore channels
    - Optional final 3x3 conv
    
    This provides global receptive field through FFT operations.
    """
    
    def __init__(self, embed_dim: int, last_conv: bool = False):
        super().__init__()
        self.last_conv = last_conv
        
        # Reduce channels before FFT (efficiency)
        self.conv1 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        
        # FFT processing
        self.fu = FourierUnit(embed_dim // 2)
        
        # Restore channels
        self.conv2 = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=1)
        
        # Optional final conv
        if last_conv:
            self.final_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        else:
            self.final_conv = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, C, H, W]
        
        Returns:
            Output tensor of shape [B, C, H, W]
        """
        x = self.conv1(x)
        output = self.fu(x)
        output = self.conv2(x + output)  # Residual around FourierUnit
        
        if self.final_conv is not None:
            output = self.final_conv(output)
        
        return output


class ResB(nn.Module):
    """
    Simple Residual Block (spatial branch).
    
    Two 3x3 convolutions with LeakyReLU, with scaled residual.
    """
    
    def __init__(self, embed_dim: int, reduction: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // reduction, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(embed_dim // reduction, embed_dim, kernel_size=3, padding=1),
        )
        self.scale = 1.0 / math.sqrt(2)  # Scale residual for stability
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * (x + self.body(x))


class SFB(nn.Module):
    """
    Spatial-Frequency Block (SFB) from SwinFIR.
    
    The key innovation: combines local spatial features with global frequency features.
    
    Architecture:
    - Spatial branch (S): ResB for local features via convolution
    - Frequency branch (F): SpectralTransform for global features via FFT
    - Fusion: Concatenate + GroupNorm + 1x1 conv
    
    This gives the network both local detail preservation and global context
    understanding simultaneously.
    """
    
    def __init__(self, embed_dim: int, reduction: int = 1):
        super().__init__()
        
        # Spatial branch: local features
        self.S = ResB(embed_dim, reduction)
        
        # Frequency branch: global features
        self.F = SpectralTransform(embed_dim)
        
        # Normalization before fusion (prevents explosion)
        def get_num_groups(channels, target=32):
            if channels >= target and channels % target == 0:
                return target
            for g in [32, 16, 8, 4, 2, 1]:
                if channels % g == 0:
                    return g
            return 1
        
        self.norm = nn.GroupNorm(get_num_groups(embed_dim * 2), embed_dim * 2)
        
        # Fusion: combine spatial and frequency features
        self.fusion = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, C, H, W]
        
        Returns:
            Output tensor of shape [B, C, H, W]
        """
        # Spatial features (local)
        s = self.S(x)
        
        # Frequency features (global)
        f = self.F(x)
        
        # Concatenate, normalize, and fuse
        out = torch.cat([s, f], dim=1)
        out = self.norm(out)
        out = self.fusion(out)
        
        return out


class ChannelAttention(nn.Module):
    """
    Channel Attention module (squeeze-and-excitation style).
    
    Used in the CAB (Channel Attention Block) from SwinFIR.
    """
    
    def __init__(self, num_feat: int, squeeze_factor: int = 16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, kernel_size=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attention(x)


class CAB(nn.Module):
    """
    Channel Attention Block from SwinFIR.
    
    Combines convolution with channel attention for adaptive feature recalibration.
    """
    
    def __init__(self, num_feat: int, compress_ratio: int = 3, squeeze_factor: int = 30):
        super().__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, kernel_size=3, padding=1),
            ChannelAttention(num_feat, squeeze_factor),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cab(x)


class FIRResBlock(nn.Module):
    """
    FIR Residual Block: Core building block for PFGM-FIR.
    
    Combines:
    1. Convolution with radius conditioning (FiLM)
    2. Optional Swin Transformer attention
    3. SFB residual connection for global receptive field
    
    This block processes features while being conditioned on the PFGM++ radius
    parameter, enabling the network to adapt its behavior based on noise level.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        radius_emb_dim: Dimension of radius embedding.
        use_swin: Whether to use Swin attention.
        use_sfb: Whether to use SFB for residual (vs simple conv).
        input_resolution: (H, W) for Swin attention.
        num_heads: Number of attention heads.
        window_size: Window size for Swin attention.
        mlp_ratio: MLP expansion ratio.
        drop_path: Stochastic depth rate.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        radius_emb_dim: int,
        use_swin: bool = False,
        use_sfb: bool = True,
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
        
        # Helper function for flexible GroupNorm
        def get_num_groups(channels, target=32):
            """Get a valid number of groups for GroupNorm."""
            if channels >= target and channels % target == 0:
                return target
            # Find largest divisor <= 32
            for g in [32, 16, 8, 4, 2, 1]:
                if channels % g == 0:
                    return g
            return 1
        
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
        
        # Optional Swin attention
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
                for i in range(2)  # Two Swin blocks (W-MSA + SW-MSA)
            ])
        else:
            self.swin_blocks = None
        
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
        
        # Residual scaling factor for stability (1/sqrt(2) prevents accumulation)
        self.residual_scale = 1.0 / math.sqrt(2)
        
        # Initialize last conv to zero for better training dynamics
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        r_emb: torch.Tensor,
        x_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            r_emb: Radius embedding of shape [B, radius_emb_dim]
            x_size: Optional (H, W) for Swin attention (uses actual size if None)
        
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
        # r_emb: [B, radius_emb_dim] -> [B, out_channels * 2]
        r_cond = self.radius_proj(r_emb)
        scale, shift = r_cond.chunk(2, dim=-1)
        scale = scale[:, :, None, None]  # [B, C, 1, 1]
        shift = shift[:, :, None, None]
        h = h * (1 + scale) + shift
        
        # Second conv
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # Optional Swin attention
        if self.swin_blocks is not None:
            # Reshape for Swin: [B, C, H, W] -> [B, H*W, C]
            h_flat = h.flatten(2).transpose(1, 2)
            current_size = x_size if x_size is not None else (H, W)
            
            for swin_block in self.swin_blocks:
                h_flat = swin_block(h_flat, current_size)
            
            # Reshape back: [B, H*W, C] -> [B, C, H, W]
            h = h_flat.transpose(1, 2).view(B, -1, H, W)
        
        # Residual with SFB (global features) - scaled for stability
        return self.residual_scale * (self.residual(skip) + h)


class Downsample(nn.Module):
    """
    Downsampling layer.
    
    Uses strided convolution for learnable downsampling.
    """
    
    def __init__(self, channels: int, out_channels: Optional[int] = None):
        super().__init__()
        out_channels = out_channels or channels
        self.conv = nn.Conv2d(channels, out_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """
    Upsampling layer.
    
    Uses nearest-neighbor interpolation followed by convolution.
    """
    
    def __init__(self, channels: int, out_channels: Optional[int] = None):
        super().__init__()
        out_channels = out_channels or channels
        self.conv = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)
