from __future__ import annotations
from typing import Optional, Tuple

import torch as th
import torch.nn as nn


# ============================================================
# Descriptor
# ============================================================

@th.jit.script
class EmbeddingMeta:
    def __init__(self, out_channels: int, obs_h: int, obs_w: int, is_spatial: bool) -> None:
        self.out_channels = out_channels
        self.obs_h        = obs_h
        self.obs_w        = obs_w
        self.is_spatial   = is_spatial

    def spatial_shape(self) -> Tuple[int, int]:
        return (self.obs_h, self.obs_w)


# ============================================================
# Interface
# ============================================================

@th.jit.interface
class EmbeddingInterface:
    def forward(self, obs: th.Tensor) -> th.Tensor:
        pass

    def meta(self) -> EmbeddingMeta:
        pass


# ============================================================
# Implementations
# ============================================================

class CategoricalGridEmbedding(nn.Module):
    """(B,H,W,3) integer ids -> (B, 3*embed_per_channel, H, W)"""

    def __init__(
        self,
        n_object_types: int,
        n_colors: int,
        n_states: int,
        obs_h: int,
        obs_w: int,
        embed_per_channel: int = 4,
    ) -> None:
        super().__init__()
        e = embed_per_channel
        self.obj   = nn.Embedding(n_object_types, e)
        self.col   = nn.Embedding(n_colors, e)
        self.sta   = nn.Embedding(n_states, e)
        self._meta = EmbeddingMeta(3 * e, obs_h, obs_w, is_spatial=True)

    def meta(self) -> EmbeddingMeta:
        return self._meta

    def forward(self, obs: th.Tensor) -> th.Tensor:
        if obs.dim() != 4 or obs.size(-1) != 3:
            raise ValueError(f"Expected (B,H,W,3), got {tuple(obs.shape)}")
        obs = obs.long()
        x = th.cat([self.obj(obs[..., 0]),
                    self.col(obs[..., 1]),
                    self.sta(obs[..., 2])], dim=-1)
        return x.permute(0, 3, 1, 2).contiguous().float()


class PixelCNNEmbedding(nn.Module):
    """(B,H,W,C) uint8 or float -> (B, out_channels, H, W)"""

    def __init__(
        self,
        in_channels: int,
        obs_h: int,
        obs_w: int,
        out_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        _out      = out_channels or in_channels
        self.stem = nn.Conv2d(in_channels, _out, kernel_size=1) if _out != in_channels else None
        self._meta = EmbeddingMeta(_out, obs_h, obs_w, is_spatial=True)

    def meta(self) -> EmbeddingMeta:
        return self._meta

    def forward(self, obs: th.Tensor) -> th.Tensor:
        if obs.dim() != 4:
            raise ValueError(f"Expected (B,H,W,C), got {tuple(obs.shape)}")
        x = (obs.float() / 255.0) if obs.dtype == th.uint8 else obs.float()
        x = x.permute(0, 3, 1, 2).contiguous()
        if self.stem is not None:
            x = self.stem(x)
        return x
