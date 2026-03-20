from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# Vocabulary
# Shared by MiniGrid and MultiGrid observation encoders.

N_OBJECT_TYPES: int = 12
N_COLORS:       int = 6
N_STATES:       int = 3
N_DIRS:         int = 4


# Environment

@dataclass
class EnvConfig:
    env:       str = "door_button"   # "door_button" or a MiniGrid gym id
    view_size: int = 5               # agent partial-obs radius
    env_size:  int = 10              # grid width/height (door_button only)
    max_steps: int = 200             # episode step limit

    # door_button reward values
    button_reward: float = 1.0
    goal_reward:   float = 10.0


# Embedding

@dataclass
class EmbeddingConfig:
    embed_per_channel: int = 4   # embedding dim per categorical channel (obj/color/state)
    dir_embed_dim:     int = 4   # embedding dim for agent direction


# VAE (TransitionSCVAE / SCVAEConfig)

@dataclass
class VAEConfig:
    # Architecture
    conv_channels:    List[int] = field(default_factory=lambda: [32, 64, 128])
    hidden_dim:       int       = 256
    latent_dim:       int       = 32
    action_embed_dim: int       = 32

    # Spherical Cauchy KL
    kl_quad_points: int   = 64    # Gauss-Legendre quadrature points
    beta:           float = 0.5   # KL weight in standalone training
    gamma:          float = 0.001 # uniformity weight
    uniformity_t:   float = 2.0   # uniformity kernel bandwidth

    # Standalone training (not used during PPO)
    lr:               float = 3e-4
    weight_decay:     float = 1e-4
    batch_size:       int   = 256
    epochs:           int   = 100
    kl_warmup_epochs: int   = 0
    kl_ramp_epochs:   int   = 0
    beta_target:      float = 0.05

    # PPO loss coefficients
    recon_coef: float = 1.0
    kl_coef:    float = 0.01


# PPO

@dataclass
class PPOConfig:
    # Rollout
    n_steps:    int   = 512
    n_envs:     int   = 4
    batch_size: int   = 256
    n_epochs:   int   = 4

    # Optimisation
    learning_rate:        float = 3e-4
    gamma:                float = 0.99
    gae_lambda:           float = 0.95
    clip_range:           float = 0.2
    clip_range_vf:        Optional[float] = None
    normalize_advantage:  bool  = True
    ent_coef:             float = 0.01
    vf_coef:              float = 0.5
    max_grad_norm:        float = 0.5
    target_kl:            Optional[float] = None

    # Intrinsic reward
    intrinsic_scale: float = 1.0

    # SDE
    use_sde:         bool = False
    sde_sample_freq: int  = -1

    # Misc
    stats_window_size: int = 100
    verbose:           int = 1


# Policy network

@dataclass
class PolicyConfig:
    pi_layers: List[int] = field(default_factory=lambda: [256, 256])
    vf_layers: List[int] = field(default_factory=lambda: [256, 256])
    ortho_init:          bool  = True
    log_std_init:        float = 0.0
    share_features_extractor: bool = True
    normalize_images:    bool  = True


# Training run

@dataclass
class RunConfig:
    total_timesteps: int            = 500_000
    seed:            int            = 0
    device:          str            = "auto"
    tensorboard_log: Optional[str]  = None


# Master config

@dataclass
class Config:
    env:    EnvConfig    = field(default_factory=EnvConfig)
    embed:  EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vae:    VAEConfig    = field(default_factory=VAEConfig)
    ppo:    PPOConfig    = field(default_factory=PPOConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    run:    RunConfig    = field(default_factory=RunConfig)


# Default instance
# Import and override fields as needed, e.g.:
#   from config import cfg
#   cfg.ppo.n_envs = 8

cfg = Config()
