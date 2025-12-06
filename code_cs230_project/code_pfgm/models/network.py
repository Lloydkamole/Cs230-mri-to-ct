"""
SwinFIR UNet for PFGM++ (PFGM-FIR).

Main network architecture that combines:
- UNet encoder-decoder structure (from Diffusion_model_transformer)
- SwinFIR blocks with SFB for global receptive field
- PFGM++ radius conditioning

Architecture:
    Input: [noisy_CT, MRI] concatenated (2 channels)
    ↓
    Initial Conv → embed_dim
    ↓
    Encoder stages (with downsampling):
    - FIRResBlock with Swin + SFB + radius conditioning
    - Downsample at each stage
    ↓
    Bottleneck:
    - FIRResBlock with attention
    ↓
    Decoder stages (with upsampling):
    - Upsample
    - Concatenate with skip connection
    - FIRResBlock
    ↓
    Final Conv → output_channels
    ↓
    Output: Predicted vector field (1 channel for 2D)

The network predicts the vector field direction for PFGM++ denoising.
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import RadiusEmbedding
from .fir_blocks import FIRResBlock, Downsample, Upsample, SFB


class SwinFIRUNet(nn.Module):
    """
    SwinFIR-based UNet for PFGM++.
    
    Combines SwinFIR attention and SFB blocks with UNet architecture
    for MRI-to-CT synthesis with PFGM++ training.
    
    Args:
        in_channels: Number of input channels (2: noisy_CT + MRI condition).
        out_channels: Number of output channels (1: predicted vector field).
        base_channels: Base channel dimension (multiplied at each stage).
        channel_mults: Channel multipliers for each stage.
        num_res_blocks: Number of residual blocks per stage.
        radius_emb_dim: Dimension of radius embedding.
        attention_resolutions: Resolutions (relative to base) where attention is applied.
        num_heads: Number of attention heads.
        window_size: Window size for Swin attention.
        mlp_ratio: MLP expansion ratio in Swin blocks.
        use_sfb: Whether to use SFB (Spatial-Frequency Block).
        dropout: Dropout rate.
        drop_path_rate: Stochastic depth rate.
    """
    
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        radius_emb_dim: int = 256,
        attention_resolutions: Tuple[int, ...] = (1, 2, 4),  # relative to base
        num_heads: int = 8,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        use_sfb: bool = True,
        dropout: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.num_res_blocks = num_res_blocks
        self.window_size = window_size
        
        # Calculate stochastic depth rates
        num_stages = len(channel_mults)
        total_blocks = num_stages * num_res_blocks * 2  # encoder + decoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        
        # Radius embedding (PFGM++ conditioning)
        # Produces embeddings of shape [B, radius_emb_dim]
        # Each FIRResBlock has its own projection from radius_emb_dim to its output channels
        self.radius_emb = RadiusEmbedding(
            embed_dim=radius_emb_dim,
        )
        
        # Initial convolution
        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Encoder
        self.encoder_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        current_channels = base_channels
        block_idx = 0
        
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            use_attention = (mult in attention_resolutions)
            
            stage_blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                stage_blocks.append(
                    FIRResBlock(
                        in_channels=current_channels if j == 0 else out_ch,
                        out_channels=out_ch,
                        radius_emb_dim=radius_emb_dim,
                        use_swin=use_attention,
                        use_sfb=use_sfb,
                        num_heads=num_heads,
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        drop_path=dpr[block_idx],
                        dropout=dropout,
                    )
                )
                current_channels = out_ch
                block_idx += 1
            
            self.encoder_stages.append(stage_blocks)
            
            # Downsample (except last stage)
            if i < num_stages - 1:
                self.downsamples.append(Downsample(current_channels))
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([
            FIRResBlock(
                in_channels=current_channels,
                out_channels=current_channels,
                radius_emb_dim=radius_emb_dim,
                use_swin=True,
                use_sfb=use_sfb,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path_rate,
                dropout=dropout,
            ),
            FIRResBlock(
                in_channels=current_channels,
                out_channels=current_channels,
                radius_emb_dim=radius_emb_dim,
                use_swin=True,
                use_sfb=use_sfb,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path_rate,
                dropout=dropout,
            ),
        ])
        
        # Decoder (reverse order)
        self.decoder_stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        
        for i in reversed(range(num_stages)):
            mult = channel_mults[i]
            prev_mult = channel_mults[i + 1] if i < num_stages - 1 else channel_mults[-1]
            out_ch = base_channels * mult
            in_ch = base_channels * prev_mult
            use_attention = (mult in attention_resolutions)
            
            # Upsample (except first decoder stage)
            if i < num_stages - 1:
                self.upsamples.append(Upsample(in_ch, out_ch))
            
            # Skip connection conv (to match channels after concatenation)
            # After concat: out_ch (from upsample) + out_ch (from encoder) = 2 * out_ch
            skip_in_ch = out_ch * 2 if i < num_stages - 1 else in_ch
            self.skip_convs.append(nn.Conv2d(skip_in_ch, out_ch, kernel_size=1))
            
            stage_blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                stage_blocks.append(
                    FIRResBlock(
                        in_channels=out_ch,
                        out_channels=out_ch,
                        radius_emb_dim=radius_emb_dim,
                        use_swin=use_attention,
                        use_sfb=use_sfb,
                        num_heads=num_heads,
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        drop_path=dpr[block_idx % len(dpr)],
                        dropout=dropout,
                    )
                )
                block_idx += 1
            
            self.decoder_stages.append(stage_blocks)
        
        # Final output - use flexible group count
        def get_num_groups(channels, target=32):
            if channels >= target and channels % target == 0:
                return target
            for g in [32, 16, 8, 4, 2, 1]:
                if channels % g == 0:
                    return g
            return 1
        
        self.final_norm = nn.GroupNorm(get_num_groups(base_channels), base_channels)
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        
        # Initialize output conv to small values
        nn.init.zeros_(self.final_conv.weight)
        if self.final_conv.bias is not None:
            nn.init.zeros_(self.final_conv.bias)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with proper scaling."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if hasattr(m, '_is_output'):
                    continue
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _pad_to_window_size(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Pad input to be divisible by window size at all resolutions.
        
        For a 4-stage UNet with window_size=8, we need input divisible by:
        window_size * 2^(num_stages-1) = 8 * 8 = 64
        """
        _, _, H, W = x.shape
        pad_size = self.window_size * (2 ** (len(self.channel_mults) - 1))
        
        pad_h = (pad_size - H % pad_size) % pad_size
        pad_w = (pad_size - W % pad_size) % pad_size
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        
        return x, (H, W)
    
    def forward(
        self,
        x: torch.Tensor,
        mri: torch.Tensor,
        r: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Noisy CT image of shape [B, 1, H, W]
            mri: MRI condition image of shape [B, 1, H, W]
            r: PFGM++ radius of shape [B] or [B, 1]
        
        Returns:
            Predicted vector field of shape [B, 1, H, W]
        """
        # Ensure all inputs are float32 for cuDNN compatibility
        x = x.float()
        mri = mri.float()
        r = r.float()
        
        # Ensure r is 1D
        if r.dim() > 1:
            r = r.squeeze(-1)
        
        # Concatenate noisy CT with MRI condition
        # Input: [noisy_CT, MRI] -> 2 channels
        x = torch.cat([x, mri], dim=1)
        
        # Pad to window size
        x, orig_size = self._pad_to_window_size(x)
        B, _, H, W = x.shape
        
        # Radius embedding
        r_emb = self.radius_emb(r)  # [B, radius_emb_dim]
        
        # Initial conv
        h = self.input_conv(x)  # [B, base_channels, H, W]
        
        # Encoder with skip connections
        skips = []
        current_size = (H, W)
        
        for i, (stage, downsample) in enumerate(zip(
            self.encoder_stages,
            self.downsamples + [None]  # No downsample after last stage
        )):
            for block in stage:
                h = block(h, r_emb, x_size=current_size)
            
            skips.append(h)
            
            if downsample is not None:
                h = downsample(h)
                current_size = (current_size[0] // 2, current_size[1] // 2)
        
        # Bottleneck
        for block in self.bottleneck:
            h = block(h, r_emb, x_size=current_size)
        
        # Decoder with skip connections
        for i, (stage, skip_conv) in enumerate(zip(
            self.decoder_stages,
            self.skip_convs
        )):
            # Upsample (except first decoder stage)
            if i > 0:
                h = self.upsamples[i - 1](h)
                current_size = (current_size[0] * 2, current_size[1] * 2)
                
                # Concatenate with skip connection
                skip = skips[-(i + 1)]
                h = torch.cat([h, skip], dim=1)
            
            # Process skip concatenation
            h = skip_conv(h)
            
            # Apply blocks
            for block in stage:
                h = block(h, r_emb, x_size=current_size)
        
        # Final output
        h = self.final_norm(h)
        h = F.silu(h)
        h = self.final_conv(h)
        
        # Remove padding
        orig_h, orig_w = orig_size
        h = h[:, :, :orig_h, :orig_w]
        
        return h


class SwinFIRUNetConfig:
    """Configuration class for SwinFIRUNet."""
    
    # Small model (~15M params)
    SMALL = dict(
        base_channels=48,
        channel_mults=(1, 2, 4),
        num_res_blocks=1,
        attention_resolutions=(4,),
        num_heads=4,
        mlp_ratio=4.0,
        window_size=8,
    )
    
    # Base model (~100M params)
    BASE = dict(
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_resolutions=(4, 8),
        num_heads=8,
        mlp_ratio=4.0,
        window_size=8,
    )
    
    # Large model (~200M+ params)
    LARGE = dict(
        base_channels=96,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=3,
        attention_resolutions=(4, 8),
        num_heads=12,
        mlp_ratio=4.0,
        window_size=8,
    )


def create_swin_fir_unet(
    config: str = 'base',
    in_channels: int = 2,
    out_channels: int = 1,
    # Custom architecture parameters (used when config='custom')
    base_channels: int = None,
    channel_mults: tuple = None,
    num_res_blocks: int = None,
    attention_resolutions: tuple = None,
    num_heads: int = None,
    mlp_ratio: float = None,
    **kwargs
) -> SwinFIRUNet:
    """
    Create a SwinFIRUNet with predefined or custom configuration.
    
    Args:
        config: Model configuration ('small', 'base', 'large', 'custom')
        in_channels: Number of input channels
        out_channels: Number of output channels
        
        # Custom parameters (only used when config='custom'):
        base_channels: Base channel count
        channel_mults: Channel multipliers per stage
        num_res_blocks: Number of residual blocks per stage
        attention_resolutions: Which channel mults get Swin attention
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        
        **kwargs: Additional arguments (radius_emb_dim, window_size, use_sfb, etc.)
    
    Returns:
        SwinFIRUNet model
    """
    configs = {
        'small': SwinFIRUNetConfig.SMALL,
        'base': SwinFIRUNetConfig.BASE,
        'large': SwinFIRUNetConfig.LARGE,
    }
    
    if config.lower() == 'custom':
        # Use provided custom parameters
        model_kwargs = {}
        if base_channels is not None:
            model_kwargs['base_channels'] = base_channels
        if channel_mults is not None:
            model_kwargs['channel_mults'] = tuple(channel_mults) if isinstance(channel_mults, list) else channel_mults
        if num_res_blocks is not None:
            model_kwargs['num_res_blocks'] = num_res_blocks
        if attention_resolutions is not None:
            model_kwargs['attention_resolutions'] = tuple(attention_resolutions) if isinstance(attention_resolutions, list) else attention_resolutions
        if num_heads is not None:
            model_kwargs['num_heads'] = num_heads
        if mlp_ratio is not None:
            model_kwargs['mlp_ratio'] = mlp_ratio
    elif config.lower() in configs:
        model_kwargs = configs[config.lower()].copy()
    else:
        raise ValueError(f"Unknown config: {config}. Choose from {list(configs.keys()) + ['custom']}")
    
    # Override with any additional kwargs
    model_kwargs.update(kwargs)
    
    return SwinFIRUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        **model_kwargs
    )
