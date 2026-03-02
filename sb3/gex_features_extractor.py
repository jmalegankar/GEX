"""
GEXFeaturesExtractor — SB3 CNN features extractor for GEX environments.

Architecture mirrors the SC-VAE encoder front-end:
    embedding(obs) → (B, C, H, W)
    ConvBlocks     → flatten
    Linear         → (B, features_dim)

This gives the policy a meaningful representation of the grid obs,
avoiding the raw-int-to-MLP problem while being independent of the
SC-VAE weights (policy learns its own representation).

Scaling:
    DoorButton / MiniGrid → CategoricalGridEmbedding
    Atari / Crafter       → PixelCNNEmbedding
    Just swap the embedding at construction time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from models.obs_embeddings import ObservationEmbedding


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GEXFeaturesExtractor(BaseFeaturesExtractor):
    """
    CNN features extractor parameterised by an ObservationEmbedding.

    Args:
        observation_space: raw env obs space (e.g. Box(5,5,3) int32)
        embedding:         ObservationEmbedding instance (CategoricalGrid / PixelCNN)
        conv_channels:     channel widths for conv blocks
        features_dim:      output dimension fed to PPO actor-critic heads
        sample_obs_shape:  shape of a single obs (without batch dim),
                           used to infer conv output size
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        embedding: ObservationEmbedding,
        conv_channels: tuple[int, ...] = (32, 64),
        features_dim: int = 256,
        sample_obs_shape: tuple[int, ...] | None = None,
    ):
        # BaseFeaturesExtractor stores features_dim; we'll set it properly below
        # after we know the actual flattened conv output size.
        super().__init__(observation_space, features_dim=features_dim)

        self.embedding = embedding

        layers = []
        in_ch = embedding.out_channels
        for out_ch in conv_channels:
            layers.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

        # Infer flattened conv output size with a dummy forward pass.
        shape = sample_obs_shape or observation_space.shape
        with torch.no_grad():
            dummy     = torch.zeros(1, *shape)
            feat_map  = self.conv(self.embedding(dummy))
            flat_dim  = feat_map.flatten(1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flat_dim, features_dim),
            nn.ReLU(),
        )

        # Overwrite the features_dim stored by BaseFeaturesExtractor.
        self._features_dim = features_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, *obs_shape)  — raw env observation
        returns: (B, features_dim)
        """
        x = self.conv(self.embedding(obs))   # (B, C, H, W)
        x = x.flatten(start_dim=1)           # (B, flat_dim)
        return self.fc(x)                    # (B, features_dim)