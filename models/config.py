from dataclasses import dataclass
from typing import Tuple

@dataclass
class SCVAEConfig:
    conv_channels: Tuple[int, ...] = (32, 64)
    hidden_dim: int = 256

    latent_dim: int = 32
    rho_min: float = 0.001
    rho_max: float = 0.999

    n_actions: int = 7
    action_embed_dim: int = 8

    kl_max_terms: int = 128

    no_op_action: int = 0

    # VAE training
    lr: float = 1e-4
    beta: float = 0.005

    # Uniformity loss
    uniformity_t: float = 2.0
    alpha_uniform: float = 0.05