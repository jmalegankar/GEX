from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class SCVAEConfig:
    # ── Action ─────────────────────────────────────────────────────
    act_dim:          int        = 4

    # ── Action Embedding ───────────────────────────────────────────
    action_embed_dim: int        = 32

    # ── Encoder ────────────────────────────────────────────────────
    conv_channels:    List[int]  = field(default_factory=lambda: [32, 64, 128])

    # ── Encoder / Decoder trunk ────────────────────────────────────
    hidden_dim:       int        = 256
    latent_dim:       int        = 32

    # ── Spherical Cauchy ───────────────────────────────────────────
    kl_quad_points:   int        = 64   # Gauss-Legendre points for quadrature KL

    # ── Loss weights ───────────────────────────────────────────────
    beta:             float      = 0.5    # KL weight
    gamma:            float      = 0.001  # uniformity weight
    uniformity_t:     float      = 2.0    # bandwidth for uniformity kernel

    # ── Training ───────────────────────────────────────────────────
    lr:               float      = 3e-4
    weight_decay:     float      = 1e-4
    batch_size:       int        = 256
    epochs:           int        = 100

    # ── Free bits ──────────────────────────────────────────────────
    free_bits:        float      = 0.5   # min KL per sample (nats); 0 disables

    # ── KL annealing ───────────────────────────────────────────────
    kl_warmup_epochs: int        = 0     # beta=0 for this many epochs
    kl_ramp_epochs:   int        = 0     # linear ramp 0 → beta_target
    beta_target:      float      = 0.05  # final KL weight after ramp


@dataclass
class GaussianVAEConfig:
    # ── Action ─────────────────────────────────────────────────────
    act_dim:          int        = 4

    # ── Action Embedding ───────────────────────────────────────────
    action_embed_dim: int        = 32

    # ── Encoder ────────────────────────────────────────────────────
    conv_channels:    List[int]  = field(default_factory=lambda: [32, 64, 128])

    # ── Encoder / Decoder trunk ────────────────────────────────────
    hidden_dim:       int        = 256
    latent_dim:       int        = 32

    # ── Free bits ──────────────────────────────────────────────────
    free_bits:        float      = 0.5   # min KL per dimension (nats); 0 disables
