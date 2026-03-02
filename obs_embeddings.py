"""
Observation Embeddings for GEX / SC-VAE.

Goal:
  Make TransitionSCVAE env-agnostic by pushing all observation-format
  handling into this file.

Contract:
  embedding(obs) -> x where x is a float tensor shaped (B, C, H, W)
  embedding.out_channels -> C

Supported:
  - CategoricalGridEmbedding: obs (B,H,W,3) integer ids per cell (obj,color,state)
  - PixelCNNEmbedding:        obs (B,H,W,C) uint8 pixels or float in [0,1]
  - VectorToMapEmbedding:     obs (B,D) vectors -> (B,C,1,1) or (B,C,H,W) if desired
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Base interface
# ============================================================

class ObservationEmbedding(nn.Module):
    """
    Base class for observation embeddings.

    Must implement:
      forward(obs) -> (B,C,H,W) float tensor
      out_channels property

    Optional:
      obs_shape property (H,W) for spatial embeddings
    """

    @property
    def out_channels(self) -> int:
        raise NotImplementedError

    @property
    def obs_shape(self) -> Optional[Tuple[int, int]]:
        return None


# ============================================================
# 1) Categorical grid embedding: (B,H,W,3) ints -> (B,C,H,W)
# ============================================================

@dataclass
class CategoricalGridSpec:
    n_object_types: int
    n_colors: int
    n_states: int
    embed_per_channel: int = 4  # per (obj/color/state)


class CategoricalGridEmbedding(ObservationEmbedding):
    """
    MultiGrid-style categorical obs:
      obs[...,0] = object_type id
      obs[...,1] = color id
      obs[...,2] = state id

    Input:
      obs: (B,H,W,3) integer-like
    Output:
      x:   (B, 3*embed_per_channel, H, W) float
    """

    def __init__(self, spec: CategoricalGridSpec) -> None:
        super().__init__()
        self.spec = spec

        e = spec.embed_per_channel
        self.obj = nn.Embedding(spec.n_object_types, e)
        self.col = nn.Embedding(spec.n_colors, e)
        self.sta = nn.Embedding(spec.n_states, e)

        self._out_channels = 3 * e

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B,H,W,3) integer-like tensor
        Returns:
            x: (B,C,H,W) float32
        """
        if obs.dim() != 4 or obs.size(-1) != 3:
            raise ValueError(f"CategoricalGridEmbedding expects (B,H,W,3), got {tuple(obs.shape)}")

        obs = obs.long()
        o = self.obj(obs[..., 0])
        c = self.col(obs[..., 1])
        s = self.sta(obs[..., 2])

        # (B,H,W,3*e) -> (B,C,H,W)
        x = torch.cat([o, c, s], dim=-1).permute(0, 3, 1, 2).contiguous()
        return x.float()


# ============================================================
# 2) Pixel embedding: (B,H,W,C) -> (B,C',H,W)
# ============================================================

class PixelCNNEmbedding(ObservationEmbedding):
    """
    Pixel input (Atari/Crafter/etc). This is intentionally minimal:
    - Convert to float in [0,1] if uint8
    - Permute to channels-first

    Optionally add a 1x1 conv "stem" to set channels to a desired width.

    Input:
      obs: (B,H,W,C) uint8 or float
    Output:
      x:   (B,out_channels,H,W) float
    """

    def __init__(self, in_channels: int, out_channels: Optional[int] = None) -> None:
        super().__init__()
        self.in_channels = in_channels
        self._out_channels = out_channels or in_channels

        self.stem: Optional[nn.Module]
        if self._out_channels == in_channels:
            self.stem = None
        else:
            self.stem = nn.Conv2d(in_channels, self._out_channels, kernel_size=1)

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B,H,W,C)
        Returns:
            x: (B,C,H,W)
        """
        if obs.dim() != 4:
            raise ValueError(f"PixelCNNEmbedding expects (B,H,W,C), got {tuple(obs.shape)}")

        # uint8 -> float [0,1]
        if obs.dtype == torch.uint8:
            x = obs.float() / 255.0
        else:
            x = obs.float()

        x = x.permute(0, 3, 1, 2).contiguous()  # (B,C,H,W)

        if x.size(1) != self.in_channels:
            raise ValueError(
                f"PixelCNNEmbedding expected in_channels={self.in_channels}, got {x.size(1)}"
            )

        if self.stem is not None:
            x = self.stem(x)

        return x


# ============================================================
# 3) Vector embedding: (B,D) -> (B,C,1,1)
# ============================================================

class VectorToMapEmbedding(ObservationEmbedding):
    """
    Vector observations -> a "feature map" suitable for conv encoder.

    Input:
      obs: (B,D)
    Output:
      x:   (B,out_channels,1,1)

    This is the minimal option; later you can switch to an MLP encoder
    and skip conv entirely, but this keeps SC-VAE encoder uniform.
    """

    def __init__(self, in_dim: int, out_channels: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self._out_channels = out_channels
        self.proj = nn.Linear(in_dim, out_channels)

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def obs_shape(self) -> Optional[Tuple[int, int]]:
        return (1, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B,D)
        Returns:
            x: (B,C,1,1)
        """
        if obs.dim() != 2 or obs.size(-1) != self.in_dim:
            raise ValueError(f"VectorToMapEmbedding expects (B,{self.in_dim}), got {tuple(obs.shape)}")

        x = self.proj(obs.float())  # (B,C)
        return x.unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)


# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":
    B, H, W = 4, 5, 5

    # Categorical grid
    spec = CategoricalGridSpec(n_object_types=11, n_colors=6, n_states=4, embed_per_channel=4)
    emb = CategoricalGridEmbedding(spec)
    obs = torch.randint(0, 4, (B, H, W, 3))
    x = emb(obs)
    assert x.shape == (B, emb.out_channels, H, W)
    print("CategoricalGridEmbedding OK:", x.shape)

    # Pixels
    p = PixelCNNEmbedding(in_channels=3, out_channels=16)
    pix = torch.randint(0, 255, (B, 84, 84, 3), dtype=torch.uint8)
    x2 = p(pix)
    assert x2.shape == (B, 16, 84, 84)
    print("PixelCNNEmbedding OK:", x2.shape)

    # Vector
    v = VectorToMapEmbedding(in_dim=12, out_channels=32)
    vec = torch.randn(B, 12)
    x3 = v(vec)
    assert x3.shape == (B, 32, 1, 1)
    print("VectorToMapEmbedding OK:", x3.shape)

    print("ALL EMBEDDINGS TESTS PASSED")