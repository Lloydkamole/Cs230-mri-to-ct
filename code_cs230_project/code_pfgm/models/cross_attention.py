"""
Cross-Attention and MRI Encoder modules for PFGM-FIR v2.

This module provides:
- CrossAttention: Multi-head cross-attention with Flash Attention
- MRIEncoder: Separate CNN encoder for MRI that produces multi-scale context tokens

The cross-attention mechanism allows the CT decoder to dynamically attend to
relevant MRI regions at each spatial location, providing more expressive
conditioning than simple concatenation.
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try Flash Attention, fall back to PyTorch's SDPA if not available
FLASH_ATTN_AVAILABLE = False
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
    print("Flash Attention available and loaded.")
except ImportError as e:
    print(f"Flash Attention not available ({e}), using PyTorch SDPA fallback.")
    flash_attn_func = None

from einops import rearrange


class CrossAttention(nn.Module):
    """
    Multi-head Cross-Attention with Flash Attention.
    
    Allows CT features (queries) to attend to MRI context (keys/values).
    Each CT spatial location can dynamically weight different MRI regions
    based on content similarity.
    
    Args:
        query_dim: Dimension of query features (CT decoder channels)
        context_dim: Dimension of context features (MRI encoder channels)
        heads: Number of attention heads
        dim_head: Dimension per head (default: query_dim // heads)
        dropout: Dropout rate for attention output
    
    Shapes:
        query: (B, C_q, H, W) - CT decoder features
        context: (B, N_ctx, C_ctx) - MRI context tokens
        output: (B, C_q, H, W) - attended features (same shape as query)
    """
    
    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        context_dim = context_dim or query_dim
        dim_head = dim_head or (query_dim // heads)
        inner_dim = dim_head * heads
        
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5  # For non-flash fallback
        
        # Projections
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )
        
        # Layer norm for stability (QK-norm from ViT-22B)
        self.q_norm = nn.LayerNorm(dim_head)
        self.k_norm = nn.LayerNorm(dim_head)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with small weights for residual-friendly start."""
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        nn.init.zeros_(self.to_out[0].weight)
        nn.init.zeros_(self.to_out[0].bias)
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Query features from CT decoder, shape (B, C, H, W)
            context: MRI context tokens, shape (B, N_ctx, C_ctx)
        
        Returns:
            Attended features, shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Flatten spatial dims: (B, C, H, W) -> (B, H*W, C)
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        
        # Project to Q, K, V
        q = self.to_q(x_flat)  # (B, H*W, inner_dim)
        k = self.to_k(context)  # (B, N_ctx, inner_dim)
        v = self.to_v(context)  # (B, N_ctx, inner_dim)
        
        # Reshape for multi-head: (B, N, heads, dim_head)
        q = rearrange(q, 'b n (h d) -> b n h d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b n h d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b n h d', h=self.heads)
        
        # QK-norm for stability
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Use Flash Attention if available, otherwise PyTorch SDPA
        if FLASH_ATTN_AVAILABLE and flash_attn_func is not None:
            # Flash Attention (expects bfloat16/float16 for best performance)
            # flash_attn_func expects: (B, seqlen, nheads, headdim)
            dtype = q.dtype
            if dtype == torch.float32:
                q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
            
            out = flash_attn_func(q, k, v, causal=False)
            out = out.to(dtype)
        else:
            # PyTorch SDPA fallback (still efficient with memory-efficient attention)
            # SDPA expects: (B, nheads, seqlen, headdim)
            q = rearrange(q, 'b n h d -> b h n d')
            k = rearrange(k, 'b n h d -> b h n d')
            v = rearrange(v, 'b n h d -> b h n d')
            
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            out = rearrange(out, 'b h n d -> b n h d')
        
        # Reshape back: (B, H*W, heads, dim_head) -> (B, H*W, inner_dim)
        out = rearrange(out, 'b n h d -> b n (h d)')
        
        # Output projection
        out = self.to_out(out)
        
        # Reshape to spatial: (B, H*W, C) -> (B, C, H, W)
        out = rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)
        
        return out


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block with pre-norm and residual connection.
    
    Architecture:
        x -> LayerNorm -> CrossAttention(x, context) -> + x (residual)
    
    Args:
        dim: Feature dimension
        context_dim: Context dimension
        heads: Number of attention heads
        dim_head: Dimension per head
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        dim: int,
        context_dim: int,
        heads: int = 8,
        dim_head: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Use GroupNorm for spatial features (like in UNet)
        num_groups = min(32, dim // 4) if dim >= 4 else 1
        while dim % num_groups != 0:
            num_groups -= 1
        self.norm = nn.GroupNorm(num_groups, dim)
        
        self.cross_attn = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) features
            context: (B, N, C_ctx) context tokens
        Returns:
            (B, C, H, W) features with cross-attention applied
        """
        # Pre-norm + cross-attention + residual
        return x + self.cross_attn(self.norm(x), context)


class MRIEncoderBlock(nn.Module):
    """
    Single encoder block: Conv -> GroupNorm -> SiLU -> Conv -> GroupNorm -> SiLU
    With residual connection.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        num_groups_in = min(32, in_channels) if in_channels >= 32 else in_channels
        while in_channels % num_groups_in != 0:
            num_groups_in -= 1
        
        num_groups_out = min(32, out_channels) if out_channels >= 32 else out_channels
        while out_channels % num_groups_out != 0:
            num_groups_out -= 1
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups_out, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups_out, out_channels)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        return h + self.skip(x)


class MRIEncoder(nn.Module):
    """
    Separate CNN encoder for MRI conditioning.
    
    Produces multi-scale context tokens for cross-attention at each decoder stage.
    
    Architecture:
        MRI (B, 1, H, W)
        ↓ Conv stem → (B, base_ch, H, W)
        ↓ Block + Downsample → (B, base_ch*2, H/2, W/2) → context level 0
        ↓ Block + Downsample → (B, base_ch*4, H/4, W/4) → context level 1
        ↓ Block + Downsample → (B, base_ch*8, H/8, W/8) → context level 2
        ↓ Block → (B, base_ch*8, H/8, W/8) → context level 3 (bottleneck)
    
    Each level's features are flattened to tokens: (B, H_i*W_i, C_i)
    
    Args:
        in_channels: Input channels (1 for MRI)
        base_channels: Base channel count
        channel_mults: Channel multipliers for each stage
        num_blocks_per_stage: Number of conv blocks per stage
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_blocks_per_stage: int = 2,
    ):
        super().__init__()
        
        self.channel_mults = channel_mults
        self.num_stages = len(channel_mults)
        
        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
        )
        
        # Encoder stages
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        current_ch = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            
            # Blocks for this stage
            blocks = nn.ModuleList()
            for j in range(num_blocks_per_stage):
                in_ch = current_ch if j == 0 else out_ch
                blocks.append(MRIEncoderBlock(in_ch, out_ch))
            self.stages.append(blocks)
            current_ch = out_ch
            
            # Downsample (except last stage)
            if i < len(channel_mults) - 1:
                self.downsamples.append(
                    nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
                )
            else:
                self.downsamples.append(nn.Identity())
        
        # Store output channel dims for decoder to know context dimensions
        self.output_channels = [base_channels * m for m in channel_mults]
    
    def forward(self, mri: torch.Tensor) -> List[torch.Tensor]:
        """
        Encode MRI to multi-scale context tokens.
        
        Args:
            mri: MRI image, shape (B, 1, H, W)
        
        Returns:
            List of context tensors, one per stage.
            Each has shape (B, N_i, C_i) where N_i = H_i * W_i
            Ordered from high resolution to low resolution.
        """
        contexts = []
        
        h = self.stem(mri)
        
        for i, (stage_blocks, downsample) in enumerate(zip(self.stages, self.downsamples)):
            # Apply blocks
            for block in stage_blocks:
                h = block(h)
            
            # Flatten to tokens: (B, C, H, W) -> (B, H*W, C)
            B, C, H_i, W_i = h.shape
            tokens = rearrange(h, 'b c h w -> b (h w) c')
            contexts.append(tokens)
            
            # Downsample for next stage
            h = downsample(h)
        
        return contexts
    
    def get_output_channels(self) -> List[int]:
        """Return the channel dimension for each context level."""
        return self.output_channels


class PositionalEncoding2D(nn.Module):
    """
    Learnable 2D positional encoding for context tokens.
    
    Adds position information so cross-attention knows spatial layout.
    """
    
    def __init__(self, dim: int, max_h: int = 256, max_w: int = 256):
        super().__init__()
        self.dim = dim
        self.max_h = max_h
        self.max_w = max_w
        
        # Learnable position embeddings
        self.h_embed = nn.Embedding(max_h, dim // 2)
        self.w_embed = nn.Embedding(max_w, dim // 2)
    
    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Add positional encoding to tokens.
        
        Args:
            x: Tokens of shape (B, H*W, C)
            h: Height of the spatial grid
            w: Width of the spatial grid
        
        Returns:
            Tokens with position encoding added, same shape
        """
        device = x.device
        
        # Create position indices
        h_pos = torch.arange(h, device=device)
        w_pos = torch.arange(w, device=device)
        
        # Get embeddings
        h_emb = self.h_embed(h_pos)  # (H, C//2)
        w_emb = self.w_embed(w_pos)  # (W, C//2)
        
        # Create 2D grid of positions
        h_emb = h_emb.unsqueeze(1).expand(-1, w, -1)  # (H, W, C//2)
        w_emb = w_emb.unsqueeze(0).expand(h, -1, -1)  # (H, W, C//2)
        
        # Concatenate h and w embeddings
        pos_emb = torch.cat([h_emb, w_emb], dim=-1)  # (H, W, C)
        pos_emb = rearrange(pos_emb, 'h w c -> (h w) c')  # (H*W, C)
        
        # Add to tokens (broadcast over batch)
        return x + pos_emb.unsqueeze(0)


class MRIEncoderWithPos(nn.Module):
    """
    MRI Encoder with positional encodings added to context tokens.
    
    Wraps MRIEncoder and adds learnable 2D position embeddings.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_blocks_per_stage: int = 2,
        max_resolution: int = 256,
    ):
        super().__init__()
        
        self.encoder = MRIEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_blocks_per_stage=num_blocks_per_stage,
        )
        
        # Positional encodings for each scale
        self.pos_encodings = nn.ModuleList()
        for i, mult in enumerate(channel_mults):
            ch = base_channels * mult
            # Resolution halves at each stage
            max_res = max_resolution // (2 ** i)
            self.pos_encodings.append(
                PositionalEncoding2D(ch, max_h=max_res, max_w=max_res)
            )
    
    def forward(self, mri: torch.Tensor) -> List[torch.Tensor]:
        """
        Encode MRI with positional encodings.
        
        Args:
            mri: (B, 1, H, W)
        
        Returns:
            List of context tensors with position info, shape (B, N_i, C_i)
        """
        contexts = self.encoder(mri)
        
        # Add positional encodings
        B = mri.shape[0]
        H, W = mri.shape[2], mri.shape[3]
        
        for i, (ctx, pos_enc) in enumerate(zip(contexts, self.pos_encodings)):
            # Compute spatial size at this level
            h_i = H // (2 ** i)
            w_i = W // (2 ** i)
            contexts[i] = pos_enc(ctx, h_i, w_i)
        
        return contexts
    
    def get_output_channels(self) -> List[int]:
        return self.encoder.get_output_channels()
