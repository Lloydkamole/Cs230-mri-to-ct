"""
SwinFIR UNet v2 with Cross-Attention for PFGM++ (PFGM-FIR).

This is a new architecture that uses:
- Separate MRI encoder (instead of channel concatenation)
- Multi-scale cross-attention in decoder (MRI conditions CT generation)
- Flash Attention for efficient attention computation

Architecture:
    MRI Input (B, 1, H, W)
    ↓
    MRIEncoder → Multi-scale context tokens [ctx_0, ctx_1, ctx_2, ctx_3]
    
    CT Input (noisy_CT) (B, 1, H, W)
    ↓
    Initial Conv → embed_dim
    ↓
    Encoder stages (with downsampling):
    - FIRResBlockV2 (self-attention only, no cross-attn)
    - Downsample at each stage
    ↓
    Bottleneck:
    - FIRResBlockV2 with cross-attention (ctx_3)
    ↓
    Decoder stages (with upsampling):
    - DecoderBlockV2 with cross-attention (ctx_i)
    - Skip connections from encoder
    ↓
    Final Conv → output_channels
    ↓
    Output: Predicted CT (1 channel)

The separate MRI encoder produces context tokens that the CT decoder
attends to via cross-attention, allowing dynamic spatial conditioning.
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import RadiusEmbedding
from .fir_blocks_v2 import FIRResBlockV2, DecoderBlockV2, Downsample, SFB, get_num_groups
from .cross_attention import MRIEncoderWithPos, CrossAttentionBlock


class SwinFIRUNetCrossAttn(nn.Module):
    """
    SwinFIR-based UNet with Cross-Attention for PFGM++.
    
    Key differences from v1 (SwinFIRUNet):
    - Separate MRI encoder produces context tokens
    - CT input is 1 channel (not concatenated with MRI)
    - Decoder uses cross-attention to condition on MRI
    - More expressive and spatially adaptive conditioning
    
    Args:
        in_channels: Number of input channels (1: noisy_CT only).
        out_channels: Number of output channels (1: predicted CT).
        base_channels: Base channel dimension.
        channel_mults: Channel multipliers for each stage.
        num_res_blocks: Number of residual blocks per stage.
        radius_emb_dim: Dimension of radius embedding.
        attention_resolutions: Resolutions where Swin attention is applied.
        cross_attn_resolutions: Resolutions where cross-attention is applied.
        num_heads: Number of attention heads.
        cross_attn_heads: Number of cross-attention heads.
        window_size: Window size for Swin attention.
        mlp_ratio: MLP expansion ratio in Swin blocks.
        use_sfb: Whether to use SFB (Spatial-Frequency Block).
        dropout: Dropout rate.
        drop_path_rate: Stochastic depth rate.
        mri_encoder_blocks: Number of blocks per stage in MRI encoder.
    """
    
    def __init__(
        self,
        in_channels: int = 1,  # Now just noisy CT
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        radius_emb_dim: int = 256,
        attention_resolutions: Tuple[int, ...] = (2, 4, 8),
        cross_attn_resolutions: Tuple[int, ...] = (1, 2, 4, 8),  # All decoder stages
        num_heads: int = 8,
        cross_attn_heads: int = 8,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        use_sfb: bool = True,
        dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        mri_encoder_blocks: int = 2,
        max_resolution: int = 256,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.num_res_blocks = num_res_blocks
        self.window_size = window_size
        self.cross_attn_resolutions = cross_attn_resolutions
        
        num_stages = len(channel_mults)
        
        # Calculate stochastic depth rates
        total_blocks = num_stages * num_res_blocks * 2
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        
        # =====================================================================
        # MRI Encoder (separate from CT path)
        # =====================================================================
        self.mri_encoder = MRIEncoderWithPos(
            in_channels=1,  # MRI is single channel
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_blocks_per_stage=mri_encoder_blocks,
            max_resolution=max_resolution,
        )
        self.mri_context_dims = self.mri_encoder.get_output_channels()
        
        # =====================================================================
        # Radius embedding (PFGM++ conditioning)
        # =====================================================================
        self.radius_emb = RadiusEmbedding(embed_dim=radius_emb_dim)
        
        # =====================================================================
        # CT Encoder (processes noisy CT)
        # =====================================================================
        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
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
                    FIRResBlockV2(
                        in_channels=current_channels if j == 0 else out_ch,
                        out_channels=out_ch,
                        radius_emb_dim=radius_emb_dim,
                        use_swin=use_attention,
                        use_sfb=use_sfb,
                        use_cross_attn=False,  # No cross-attn in encoder
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
        
        # =====================================================================
        # Bottleneck (with cross-attention)
        # =====================================================================
        bottleneck_context_dim = self.mri_context_dims[-1]  # Deepest MRI features
        
        self.bottleneck = nn.ModuleList([
            FIRResBlockV2(
                in_channels=current_channels,
                out_channels=current_channels,
                radius_emb_dim=radius_emb_dim,
                use_swin=True,
                use_sfb=use_sfb,
                use_cross_attn=True,  # Cross-attention at bottleneck
                context_dim=bottleneck_context_dim,
                cross_attn_heads=cross_attn_heads,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path_rate,
                dropout=dropout,
            ),
            FIRResBlockV2(
                in_channels=current_channels,
                out_channels=current_channels,
                radius_emb_dim=radius_emb_dim,
                use_swin=True,
                use_sfb=use_sfb,
                use_cross_attn=True,
                context_dim=bottleneck_context_dim,
                cross_attn_heads=cross_attn_heads,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path_rate,
                dropout=dropout,
            ),
        ])
        
        # =====================================================================
        # Decoder (with cross-attention at each stage)
        # =====================================================================
        self.decoder_stages = nn.ModuleList()
        
        # Decoder goes from deepest to shallowest
        for i in reversed(range(num_stages)):
            mult = channel_mults[i]
            prev_mult = channel_mults[i + 1] if i < num_stages - 1 else channel_mults[-1]
            out_ch = base_channels * mult
            in_ch = base_channels * prev_mult
            skip_ch = out_ch  # Skip connection has same channels as output
            use_attention = (mult in attention_resolutions)
            use_cross_attn = (mult in cross_attn_resolutions)
            
            # Context dimension for this level
            # MRI encoder outputs are indexed 0 (full res) to N-1 (lowest res)
            # Decoder stages are indexed 0 (lowest res) to N-1 (full res)
            ctx_idx = i  # Match encoder stage index
            context_dim = self.mri_context_dims[ctx_idx] if use_cross_attn else None
            
            if i < num_stages - 1:
                # Normal decoder stage with upsample
                self.decoder_stages.append(
                    DecoderBlockV2(
                        in_channels=in_ch,
                        skip_channels=skip_ch,
                        out_channels=out_ch,
                        radius_emb_dim=radius_emb_dim,
                        context_dim=context_dim,
                        use_cross_attn=use_cross_attn,
                        num_res_blocks=num_res_blocks,
                        use_swin=use_attention,
                        use_sfb=use_sfb,
                        cross_attn_heads=cross_attn_heads,
                        num_heads=num_heads,
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        drop_path=dpr[block_idx % len(dpr)],
                        dropout=dropout,
                    )
                )
            else:
                # First decoder stage (no upsample, just process bottleneck)
                # This connects bottleneck to the first skip connection
                stage_blocks = nn.ModuleList()
                for j in range(num_res_blocks):
                    stage_blocks.append(
                        FIRResBlockV2(
                            in_channels=in_ch,
                            out_channels=out_ch,
                            radius_emb_dim=radius_emb_dim,
                            use_swin=use_attention,
                            use_sfb=use_sfb,
                            use_cross_attn=use_cross_attn if j == 0 else False,
                            context_dim=context_dim,
                            cross_attn_heads=cross_attn_heads,
                            num_heads=num_heads,
                            window_size=window_size,
                            mlp_ratio=mlp_ratio,
                            drop_path=dpr[block_idx % len(dpr)],
                            dropout=dropout,
                        )
                    )
                    block_idx += 1
                self.decoder_stages.append(stage_blocks)
        
        # =====================================================================
        # Final output
        # =====================================================================
        self.final_norm = nn.GroupNorm(get_num_groups(base_channels), base_channels)
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        
        # Initialize output conv to zero
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
        """Pad input to be divisible by window size at all resolutions."""
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
            Predicted CT of shape [B, 1, H, W]
        """
        # Ensure all inputs are float32
        x = x.float()
        mri = mri.float()
        r = r.float()
        
        # Ensure r is 1D
        if r.dim() > 1:
            r = r.squeeze(-1)
        
        # Pad inputs to window size
        x, orig_size = self._pad_to_window_size(x)
        mri, _ = self._pad_to_window_size(mri)
        B, _, H, W = x.shape
        
        # =====================================================================
        # Encode MRI to multi-scale context tokens
        # =====================================================================
        # Returns list of context tensors: [ctx_0, ctx_1, ctx_2, ctx_3]
        # ctx_i has shape (B, H_i*W_i, C_i) where resolution halves each level
        mri_contexts = self.mri_encoder(mri)
        
        # =====================================================================
        # Radius embedding
        # =====================================================================
        r_emb = self.radius_emb(r)  # [B, radius_emb_dim]
        
        # =====================================================================
        # CT Encoder
        # =====================================================================
        h = self.input_conv(x)  # [B, base_channels, H, W]
        
        # Encoder with skip connections
        skips = []
        current_size = (H, W)
        
        for i, (stage, downsample) in enumerate(zip(
            self.encoder_stages,
            self.downsamples + [None]
        )):
            for block in stage:
                h = block(h, r_emb, context=None, x_size=current_size)
            
            skips.append(h)
            
            if downsample is not None:
                h = downsample(h)
                current_size = (current_size[0] // 2, current_size[1] // 2)
        
        # =====================================================================
        # Bottleneck (with cross-attention to deepest MRI context)
        # =====================================================================
        bottleneck_context = mri_contexts[-1]  # Deepest MRI features
        
        for block in self.bottleneck:
            h = block(h, r_emb, context=bottleneck_context, x_size=current_size)
        
        # =====================================================================
        # Decoder (with multi-scale cross-attention)
        # =====================================================================
        num_stages = len(self.channel_mults)
        
        for i, decoder_stage in enumerate(self.decoder_stages):
            # Determine which encoder skip to use
            # decoder_stages[0] corresponds to deepest level (channel_mults[-1])
            # decoder_stages[-1] corresponds to shallowest level (channel_mults[0])
            skip_idx = num_stages - 1 - i
            
            # Get context for this level
            # mri_contexts[0] is full resolution, mri_contexts[-1] is lowest
            ctx_level = skip_idx
            context = mri_contexts[ctx_level]
            
            if i == 0:
                # First decoder stage: just residual blocks on bottleneck output
                for block in decoder_stage:
                    h = block(h, r_emb, context=context, x_size=current_size)
            else:
                # DecoderBlockV2: upsample + skip + blocks with cross-attn
                skip = skips[skip_idx]
                h = decoder_stage(h, skip, r_emb, context=context, x_size=current_size)
                current_size = (current_size[0] * 2, current_size[1] * 2)
        
        # =====================================================================
        # Final output
        # =====================================================================
        h = self.final_norm(h)
        h = F.silu(h)
        h = self.final_conv(h)
        
        # Remove padding
        orig_h, orig_w = orig_size
        h = h[:, :, :orig_h, :orig_w]
        
        return h


class SwinFIRUNetCrossAttnConfig:
    """Configuration presets for SwinFIRUNetCrossAttn."""
    
    # Small model (~25M params)
    SMALL = dict(
        base_channels=48,
        channel_mults=(1, 2, 4),
        num_res_blocks=1,
        attention_resolutions=(4,),
        cross_attn_resolutions=(2, 4),
        num_heads=4,
        cross_attn_heads=4,
        mlp_ratio=4.0,
        window_size=8,
        mri_encoder_blocks=1,
    )
    
    # Base model (~150M params)
    BASE = dict(
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_resolutions=(4, 8),
        cross_attn_resolutions=(2, 4, 8),
        num_heads=8,
        cross_attn_heads=8,
        mlp_ratio=4.0,
        window_size=8,
        mri_encoder_blocks=2,
    )
    
    # Large model (~300M+ params)
    LARGE = dict(
        base_channels=96,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=3,
        attention_resolutions=(4, 8),
        cross_attn_resolutions=(1, 2, 4, 8),
        num_heads=12,
        cross_attn_heads=12,
        mlp_ratio=4.0,
        window_size=8,
        mri_encoder_blocks=2,
    )


def create_swin_fir_unet_cross_attn(
    config: str = 'base',
    in_channels: int = 1,
    out_channels: int = 1,
    # Custom architecture parameters
    base_channels: int = None,
    channel_mults: tuple = None,
    num_res_blocks: int = None,
    attention_resolutions: tuple = None,
    cross_attn_resolutions: tuple = None,
    num_heads: int = None,
    cross_attn_heads: int = None,
    mlp_ratio: float = None,
    window_size: int = None,
    mri_encoder_blocks: int = None,
    **kwargs
) -> SwinFIRUNetCrossAttn:
    """
    Create a SwinFIRUNetCrossAttn with predefined or custom configuration.
    
    Args:
        config: Model configuration ('small', 'base', 'large', 'custom')
        in_channels: Number of input channels (1 for noisy CT)
        out_channels: Number of output channels (1 for CT)
        **kwargs: Additional arguments for the model
    
    Returns:
        SwinFIRUNetCrossAttn model
    """
    configs = {
        'small': SwinFIRUNetCrossAttnConfig.SMALL,
        'base': SwinFIRUNetCrossAttnConfig.BASE,
        'large': SwinFIRUNetCrossAttnConfig.LARGE,
    }
    
    if config.lower() == 'custom':
        model_kwargs = {}
        if base_channels is not None:
            model_kwargs['base_channels'] = base_channels
        if channel_mults is not None:
            model_kwargs['channel_mults'] = tuple(channel_mults) if isinstance(channel_mults, list) else channel_mults
        if num_res_blocks is not None:
            model_kwargs['num_res_blocks'] = num_res_blocks
        if attention_resolutions is not None:
            model_kwargs['attention_resolutions'] = tuple(attention_resolutions) if isinstance(attention_resolutions, list) else attention_resolutions
        if cross_attn_resolutions is not None:
            model_kwargs['cross_attn_resolutions'] = tuple(cross_attn_resolutions) if isinstance(cross_attn_resolutions, list) else cross_attn_resolutions
        if num_heads is not None:
            model_kwargs['num_heads'] = num_heads
        if cross_attn_heads is not None:
            model_kwargs['cross_attn_heads'] = cross_attn_heads
        if mlp_ratio is not None:
            model_kwargs['mlp_ratio'] = mlp_ratio
        if window_size is not None:
            model_kwargs['window_size'] = window_size
        if mri_encoder_blocks is not None:
            model_kwargs['mri_encoder_blocks'] = mri_encoder_blocks
    elif config.lower() in configs:
        model_kwargs = configs[config.lower()].copy()
    else:
        raise ValueError(f"Unknown config: {config}. Choose from {list(configs.keys()) + ['custom']}")
    
    # Override with additional kwargs
    model_kwargs.update(kwargs)
    
    return SwinFIRUNetCrossAttn(
        in_channels=in_channels,
        out_channels=out_channels,
        **model_kwargs
    )
