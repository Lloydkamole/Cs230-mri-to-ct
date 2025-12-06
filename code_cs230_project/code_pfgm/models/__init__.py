"""
PFGM-FIR Models: SwinFIR-based architecture for PFGM++ MRI-to-CT synthesis.

This module contains the neural network components for PFGM-FIR:
- SwinFIRUNet: Main UNet model with SwinFIR blocks
- FIRResBlock: Core building block with Swin attention + SFB
- RadiusEmbedding: PFGM++ radius conditioning
- SFB: Spatial-Frequency Block for global receptive field
- FourierUnit: FFT-based processing
"""

from .embeddings import RadiusEmbedding, FourierEmbedding, TimestepEmbedding, timestep_embedding
from .swin_attention import (
    SwinTransformerBlock,
    WindowAttention,
    window_partition,
    window_reverse,
)
from .fir_blocks import (
    FourierUnit,
    SpectralTransform,
    ResB,
    SFB,
    CAB,
    ChannelAttention,
    FIRResBlock,
    Downsample,
    Upsample,
)
from .network import SwinFIRUNet, SwinFIRUNetConfig, create_swin_fir_unet

__all__ = [
    # Embeddings
    'RadiusEmbedding',
    'FourierEmbedding',
    'TimestepEmbedding',
    'timestep_embedding',
    # Swin attention
    'SwinTransformerBlock',
    'WindowAttention',
    'window_partition',
    'window_reverse',
    # FIR blocks
    'FourierUnit',
    'SpectralTransform',
    'ResB',
    'SFB',
    'CAB',
    'ChannelAttention',
    'FIRResBlock',
    'Downsample',
    'Upsample',
    # Main network
    'SwinFIRUNet',
    'SwinFIRUNetConfig',
    'create_swin_fir_unet',
]
