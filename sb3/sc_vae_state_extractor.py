"""
SCVAEStateExtractor — SB3 features extractor backed by sc_vae_target.

Policy sees h_s = sc_vae_target.encode_state(obs) instead of a separate CNN.

Benefits:
  - Policy representation improves as SC-VAE trains online
  - EMA target gives stable features (no gradient noise)
  - No separate CNN to train — one set of conv weights does both jobs
  - Scales to Atari: swap CategoricalGridEmbedding → PixelCNNEmbedding
    in SC-VAE config and the policy automatically gets pixel features

Lifecycle:
  1. PPOGEX.__init__ creates extractor pointing at online sc_vae
     (target doesn't exist yet — SB3 calls _setup_model inside super().__init__)
  2. PPOGEX._setup_model calls extractor.set_encoder(sc_vae_target)
     to swap to the EMA copy
  3. From that point on, all policy forward passes use the frozen target
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SCVAEStateExtractor(BaseFeaturesExtractor):
    """
    Thin wrapper: obs → sc_vae_target.encode_state(obs) → (B, feat_dim)

    No trainable parameters of its own.
    The underlying sc_vae_target is frozen (EMA-updated externally).
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        sc_vae,               # initially online encoder; swapped to target in _setup_model
    ):
        feat_dim = sc_vae._feat_dim
        super().__init__(observation_space, features_dim=feat_dim)

        # Mutable reference — swapped to sc_vae_target after SB3 setup
        self._encoder = sc_vae

        # No parameters of our own; encoder weights managed externally
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def set_encoder(self, encoder) -> None:
        """
        Called by PPOGEX._setup_model after sc_vae_target is created.
        Swaps the reference so the policy uses the EMA-stable target.
        """
        self._encoder = encoder
        self._features_dim = encoder._feat_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, *obs_shape) — raw env observation
        returns: (B, feat_dim) — flattened conv features from target encoder
        """
        with torch.no_grad():
            return self._encoder.encode_state(obs)