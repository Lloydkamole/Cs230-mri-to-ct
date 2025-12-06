"""
Embedding modules for PFGM-FIR.

RadiusEmbedding: Encodes the PFGM++ radius parameter r using Fourier features.
Similar to timestep embedding in standard diffusion models.
"""

import math
import torch
import torch.nn as nn


class FourierEmbedding(nn.Module):
    """
    Fourier feature embedding for continuous values (radius or timestep).
    
    Maps a scalar value r to a high-dimensional vector using sinusoidal functions:
    [sin(2π·r·f₁), cos(2π·r·f₁), sin(2π·r·f₂), cos(2π·r·f₂), ...]
    
    This is the standard approach from "Attention Is All You Need" and 
    "Denoising Diffusion Probabilistic Models".
    """
    
    def __init__(self, embed_dim: int, max_freq: float = 10000.0):
        """
        Args:
            embed_dim: Output embedding dimension (must be even).
            max_freq: Maximum frequency for the Fourier features.
        """
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even"
        
        self.embed_dim = embed_dim
        self.max_freq = max_freq
        
        # Precompute frequency bands
        half_dim = embed_dim // 2
        freqs = torch.exp(
            -math.log(max_freq) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer('freqs', freqs)
    
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Args:
            r: Tensor of shape [B] containing radius values.
        
        Returns:
            Tensor of shape [B, embed_dim] containing Fourier embeddings.
        """
        # r: [B] -> [B, 1]
        if r.dim() == 0:
            r = r.unsqueeze(0)
        if r.dim() == 1:
            r = r.unsqueeze(-1)
        
        # Compute arguments: [B, half_dim]
        args = r * self.freqs.unsqueeze(0) * 2 * math.pi
        
        # Compute sin and cos: [B, embed_dim]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        return embedding


class RadiusEmbedding(nn.Module):
    """
    Full radius embedding module for PFGM++.
    
    Combines Fourier features with an MLP to produce the final embedding
    that will be used for conditioning the network.
    
    Architecture:
        r -> FourierEmbedding -> Linear -> SiLU -> Linear -> embedding
    """
    
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = None,
        max_freq: float = 10000.0,
    ):
        """
        Args:
            embed_dim: Output embedding dimension.
            hidden_dim: Hidden dimension for MLP (default: 4 * embed_dim).
            max_freq: Maximum frequency for Fourier features.
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        hidden_dim = hidden_dim or embed_dim * 4
        
        # Fourier features
        self.fourier = FourierEmbedding(embed_dim, max_freq)
        
        # MLP projection
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
    
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Args:
            r: Tensor of shape [B] containing radius values.
        
        Returns:
            Tensor of shape [B, embed_dim] containing radius embeddings.
        """
        # Fourier features
        emb = self.fourier(r)
        
        # MLP projection
        emb = self.mlp(emb)
        
        return emb


class TimestepEmbedding(nn.Module):
    """
    Standard timestep embedding (alias for RadiusEmbedding).
    
    Included for compatibility with diffusion-style code that uses timesteps
    instead of radius.
    """
    
    def __init__(self, embed_dim: int, hidden_dim: int = None):
        super().__init__()
        self.embedding = RadiusEmbedding(embed_dim, hidden_dim)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.embedding(t)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """
    Functional interface for timestep/radius embedding.
    
    This matches the interface used in the original Diffusion_model_transformer.py.
    
    Args:
        t: Tensor of shape [B] containing timestep/radius values.
        dim: Embedding dimension.
        max_period: Maximum period for sinusoidal embedding.
    
    Returns:
        Tensor of shape [B, dim] containing embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    
    return embedding
