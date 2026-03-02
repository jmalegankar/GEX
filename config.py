from dataclasses import dataclass
from typing import Tuple


@dataclass
class SCVAEConfig:
    # Conv architecture (after embedding)
    conv_channels: Tuple[int, ...] = (32, 64)
    hidden_dim: int = 256

    # Latent
    latent_dim: int = 32
    rho_min: float = 0.001
    rho_max: float = 0.999

    # Action embedding
    n_actions: int = 7
    action_embed_dim: int = 8

    # KL
    kl_max_terms: int = 128