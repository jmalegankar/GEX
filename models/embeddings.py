from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn


# ============================================================
# Metadata container
# ============================================================

@torch.jit.script
class EmbeddingMeta:
    """Carries static information about an embedding's output format."""

    def __init__(self, out_channels: int, obs_h: int, obs_w: int, is_spatial: bool) -> None:
        self.out_channels = out_channels
        self.obs_h = obs_h
        self.obs_w = obs_w
        self.is_spatial = is_spatial

    def spatial_shape(self) -> Tuple[int, int]:
        return (self.obs_h, self.obs_w)


# ============================================================
# Interface
# ============================================================

@torch.jit.interface
class EmbeddingInterface:
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        pass

    def meta(self) -> EmbeddingMeta:
        pass


# ============================================================
# 1) Categorical grid embedding
# ============================================================

# Example: MiniGrid obs (B, H, W, 3) with (object_type, color, state) integers per cell.
class CategoricalGridEmbedding(nn.Module):
    """
    Input:  (B, H, W, 3) integer tensor
    Output: (B, 3 * embed_per_channel, H, W)
    """

    def __init__(
        self,
        n_object_types: int,
        n_colors: int,
        n_states: int,
        obs_h: int,
        obs_w: int,
        embed_per_channel: int = 4,
    ):
        super().__init__()

        e = embed_per_channel

        self.obj = nn.Embedding(n_object_types, e)
        self.col = nn.Embedding(n_colors, e)
        self.sta = nn.Embedding(n_states, e)

        self._meta = EmbeddingMeta(3 * e, obs_h, obs_w, True)

    def meta(self) -> EmbeddingMeta:
        return self._meta

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.long()

        x = torch.cat(
            [
                self.obj(obs[..., 0]),
                self.col(obs[..., 1]),
                self.sta(obs[..., 2]),
            ],
            dim=-1,
        )

        return x.permute(0, 3, 1, 2).contiguous().float()


# ============================================================
# 2) Pixel embedding
# ============================================================

# Example: Atari obs (B, H, W, 3) uint8 RGB images.
class PixelCNNEmbedding(nn.Module):
    """
    Input:  (B, H, W, C)
    Output: (B, out_channels, H, W)
    """

    def __init__(self, in_channels: int, out_channels: int = None):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels

        if out_channels != in_channels:
            self.stem = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.stem = nn.Identity()

        self._meta = EmbeddingMeta(out_channels, 0, 0, True)

    def meta(self) -> EmbeddingMeta:
        return self._meta

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.float()

        if obs.dtype == torch.uint8:
            x = x / 255.0

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.stem(x)

        return x


# ============================================================
# 3) Crafter CNN embedding (64×64 RGB)
# ============================================================

class CrafterCNNEmbedding(nn.Module):
    """
    3-layer CNN for 64×64 RGB pixel observations (Crafter).

    Input:  (B, 3, 64, 64) float — SB3 auto-transposes image obs via VecTransposeImage
    Output: (B, 64, 4, 4)

    Spatial reduction: 64 → 15 → 6 → 4
    The output feeds into ConvEncoder for further processing.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 64):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4),  # 64 → 15
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2),           # 15 → 6
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, stride=1),  # 6 → 4
            nn.ReLU(inplace=True),
        )

        self._meta = EmbeddingMeta(out_channels, 4, 4, True)

    def meta(self) -> EmbeddingMeta:
        return self._meta

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # SB3 VecTransposeImage already converts (B, H, W, C) → (B, C, H, W)
        x = obs.float()
        if x.max() > 1.0:
            x = x / 255.0
        return self.cnn(x)


# ============================================================
# 4) Vector embedding
# ============================================================

# Example: Box2D obs (B, D) with D continuous variables.
class VectorToMapEmbedding(nn.Module):
    """
    Input:  (B, D)
    Output: (B, out_channels, 1, 1)
    """

    def __init__(self, in_dim: int, out_channels: int):
        super().__init__()

        self.in_dim = in_dim
        self.out_channels = out_channels
        self.proj = nn.Linear(in_dim, out_channels)

        self._meta = EmbeddingMeta(out_channels, 1, 1, False)

    def meta(self) -> EmbeddingMeta:
        return self._meta

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.proj(obs.float())
        return x.unsqueeze(-1).unsqueeze(-1)


# ============================================================
# 5) Categorical grid + direction embedding (DoorButtonEnv)
# ============================================================

# Input: (B, H, W, 4) int tensor — channels 0-2 are (object_type, color, state),
#        channel 3 is the agent's direction (0-3), constant across all cells.
class CategoricalGridWithDirEmbedding(nn.Module):
    def __init__(
        self,
        n_object_types: int,
        n_colors: int,
        n_states: int,
        obs_h: int,
        obs_w: int,
        embed_per_channel: int = 4,
        n_dirs: int = 4,
        dir_embed_dim: int = 4,
    ):
        super().__init__()

        e = embed_per_channel
        self.dir_embed_dim = dir_embed_dim

        self.obj = nn.Embedding(n_object_types, e)
        self.col = nn.Embedding(n_colors, e)
        self.sta = nn.Embedding(n_states, e)
        self.dir = nn.Embedding(n_dirs, dir_embed_dim)

        out_ch = 3 * e + dir_embed_dim
        self._meta = EmbeddingMeta(out_ch, obs_h, obs_w, True)

    def meta(self) -> EmbeddingMeta:
        return self._meta

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.long()                              # (B, H, W, 4)

        # Embed the three image channels → each (B, H, W, e)
        img_emb = torch.cat(
            [
                self.obj(obs[..., 0]),
                self.col(obs[..., 1]),
                self.sta(obs[..., 2]),
            ],
            dim=-1,
        )                                             # (B, H, W, 3*e)
        img_feat = img_emb.permute(0, 3, 1, 2).contiguous().float()  # (B, 3*e, H, W)

        # Direction is constant per sample — read from any cell (0, 0)
        direction = obs[:, 0, 0, 3]                  # (B,)
        dir_emb = self.dir(direction).float()         # (B, dir_embed_dim)
        H, W = obs.shape[1], obs.shape[2]
        dir_feat = dir_emb[:, :, None, None].expand(-1, -1, H, W)  # (B, dir_embed_dim, H, W)

        return torch.cat([img_feat, dir_feat], dim=1)  # (B, 3*e + dir_embed_dim, H, W)