"""
WynerVAE — Temporal Common Information VAE for HSWVIME.

Architecture (Phase 1 — Slow Path only; Fast Path added in Phase 3):

    h_t      = SlowPath(sg(mu_t), h_{t-1})          # GRU episodic memory
    context_t = h_t                                   # Phase 3: GatedFusion(fast, slow)

    TRAINING (mu_{t+1} available from replay buffer):
        mu_p, logvar_p = PriorNet(context_t)
        mu_q, logvar_q = PosteriorNet(context_t, sg(mu_t), sg(mu_{t+1}))
        z_t            = reparameterize(mu_q, logvar_q)
        recon_past     = DecoderPast(z_t)             # target: sg(mu_t)
        recon_future   = DecoderFuture(z_t)           # target: sg(mu_{t+1})

    ROLLOUT (mu_{t+1} unavailable):
        mu_p, logvar_p = PriorNet(context_t)
        z_t            = reparameterize(mu_p, logvar_p)

    LOSS:
        L = KL[N(mu_q,logvar_q) || N(mu_p,logvar_p)]
          + λ_past   * MSE(recon_past,   sg(mu_t))
          + λ_future * MSE(recon_future, sg(mu_{t+1}))

    INTRINSIC REWARD (per sample, detached):
        r_int = α * KL[q || p]

Gradient flow (critical):
    - WynerVAE parameters  ← gradients from L  ✓
    - TransitionSCVAE      ← NO gradients       ✓  (all mu inputs are sg'd)
    - mu_t  in PosteriorNet is sg'd: WynerVAE reads the SCVAE repr but cannot shape it
    - mu_tp1 in PosteriorNet is sg'd: same reason, plus it's a future target
    - reconstruction targets are sg'd: stops decoder from collapsing onto live SCVAE graph

Why decoders receive ONLY z_t (no context):
    If context_t entered the decoders, they could reconstruct mu_t / mu_{t+1} from
    context alone, leaving z_t unused. KL minimisation pressure would then drive z_t
    to zero (posterior ≈ prior) — posterior collapse. Self-sufficiency is non-negotiable.

Why PosteriorNet sees mu_{t+1}:
    Wyner common information C(X;Y) = inf_{p(w|x,y): X⊥Y|W} I(X,Y;W).
    To find the minimising W, the encoder must observe BOTH X=mu_t and Y=mu_{t+1}.
    Without mu_{t+1} at training time the posterior cannot locate the Wyner bottleneck;
    it degenerates into a standard VAE posterior.
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WynerConfig:
    mu_dim:           int   # TransitionSCVAE latent_dim  (input to WynerVAE)
    wyner_dim:        int   = 64    # dimension of Wyner latent z
    context_dim:      int   = 256   # SlowPath GRU hidden dim
    hidden_dim:       int   = 256   # MLP hidden dim (Prior / Posterior / Decoders)
    lambda_past:      float = 1.0   # weight for past  reconstruction loss
    lambda_future:    float = 2.0   # weight for future reconstruction loss
                                    #   (future is harder / more stochastic → upweight)
    alpha_intrinsic:  float = 1.0   # scalar applied to per-sample KL → intrinsic reward
    free_nats:        float = 0.0   # KL floor (clip below this before mean); 0 = off


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

class WynerOutput:
    """Holds everything produced by forward_train; consumed by loss()."""
    __slots__ = (
        "mu_q", "logvar_q", "mu_p", "logvar_p",
        "z_t", "recon_past", "recon_future",
        "target_past", "target_future", "h_slow",
    )

    def __init__(
        self,
        mu_q:         th.Tensor,  # (B, wyner_dim)  posterior mean
        logvar_q:     th.Tensor,  # (B, wyner_dim)  posterior log-variance
        mu_p:         th.Tensor,  # (B, wyner_dim)  prior mean
        logvar_p:     th.Tensor,  # (B, wyner_dim)  prior log-variance
        z_t:          th.Tensor,  # (B, wyner_dim)  sampled Wyner latent
        recon_past:   th.Tensor,  # (B, mu_dim)     DecoderPast(z_t)
        recon_future: th.Tensor,  # (B, mu_dim)     DecoderFuture(z_t)
        target_past:  th.Tensor,  # (B, mu_dim)     sg(mu_t)
        target_future:th.Tensor,  # (B, mu_dim)     sg(mu_{t+1})
        h_slow:       th.Tensor,  # (B, context_dim) updated slow-path hidden state
    ):
        self.mu_q          = mu_q
        self.logvar_q      = logvar_q
        self.mu_p          = mu_p
        self.logvar_p      = logvar_p
        self.z_t           = z_t
        self.recon_past    = recon_past
        self.recon_future  = recon_future
        self.target_past   = target_past
        self.target_future = target_future
        self.h_slow        = h_slow


class WynerLoss:
    """Holds all scalar loss components and the per-sample intrinsic reward."""
    __slots__ = (
        "kl_loss", "recon_past_loss", "recon_future_loss",
        "total_loss", "intrinsic_reward",
    )

    def __init__(
        self,
        kl_loss:           th.Tensor,  # scalar
        recon_past_loss:   th.Tensor,  # scalar
        recon_future_loss: th.Tensor,  # scalar
        total_loss:        th.Tensor,  # scalar
        intrinsic_reward:  th.Tensor,  # (B,)  detached — no grad
    ):
        self.kl_loss           = kl_loss
        self.recon_past_loss   = recon_past_loss
        self.recon_future_loss = recon_future_loss
        self.total_loss        = total_loss
        self.intrinsic_reward  = intrinsic_reward


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_kl(
    mu_q:     th.Tensor,   # (B, D)
    logvar_q: th.Tensor,   # (B, D)
    mu_p:     th.Tensor,   # (B, D)
    logvar_p: th.Tensor,   # (B, D)
) -> th.Tensor:            # (B,)   — summed over D, NOT averaged
    """
    Analytical KL divergence KL( N(mu_q, exp(logvar_q)) || N(mu_p, exp(logvar_p)) ).

    KL = 0.5 * Σ_i [ exp(lv_q - lv_p) + (mu_p - mu_q)² / exp(lv_p) - 1 + lv_p - lv_q ]

    Returns per-sample scalar (B,); caller decides whether to .mean() for the loss
    or use raw values as per-step intrinsic rewards.
    """
    kl = 0.5 * (
          (logvar_q - logvar_p).exp()            # var_q / var_p
        + (mu_p - mu_q).pow(2) / logvar_p.exp() # (mu_p - mu_q)² / var_p
        - 1.0
        + logvar_p - logvar_q
    )
    return kl.sum(dim=-1)  # (B,)


def reparameterize(mu: th.Tensor, logvar: th.Tensor) -> th.Tensor:
    """
    z = mu + ε · exp(0.5 · logvar),  ε ~ N(0, I).
    Returns mu directly when grad is disabled (rollout / eval).
    """
    if not th.is_grad_enabled():
        return mu
    std = (0.5 * logvar).exp()
    return mu + th.randn_like(std) * std


# ─────────────────────────────────────────────────────────────────────────────
# Modules
# ─────────────────────────────────────────────────────────────────────────────

class SlowPath(nn.Module):
    """
    GRU-based episodic memory.

    h_t = GRU( sg(mu_t), h_{t-1} )

    Design choices:
    - Stateless: caller owns h and passes it in; enables correct episode resets.
    - sg(mu_t) inside forward: gradients from WynerVAE loss cannot reach SCVAE.
    - Phase 3 TODO: swap GRU for Mamba for O(N) scaling over long episodes.
    """

    def __init__(self, mu_dim: int, context_dim: int):
        super().__init__()
        self.context_dim = context_dim
        self.gru = nn.GRUCell(mu_dim, context_dim)

    def forward(
        self,
        mu_t:   th.Tensor,  # (B, mu_dim)     — SCVAE latent at time t
        h_prev: th.Tensor,  # (B, context_dim) — previous hidden state
    ) -> th.Tensor:          # (B, context_dim) — new hidden state h_t
        # sg here: WynerVAE loss never modifies SCVAE weights through the slow path
        return self.gru(mu_t.detach(), h_prev)

    def init_hidden(self, batch_size: int, device: th.device) -> th.Tensor:
        """Zero-initialise hidden state. Call at episode start."""
        return th.zeros(batch_size, self.context_dim, device=device)


class PriorNet(nn.Module):
    """
    Transition prior: p_θ(z_t | context_t).

    Context-only — no access to mu_t or mu_{t+1}.
    Used at ROLLOUT time (when mu_{t+1} is unavailable) to sample z_t.
    The KL between prior and posterior becomes the intrinsic reward.

    Zero-init on logvar bias: prior starts as N(0, I) which is a stable
    initialisation and matches the standard VAE prior.
    """

    def __init__(self, context_dim: int, wyner_dim: int, hidden_dim: int):
        super().__init__()
        self.wyner_dim = wyner_dim
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, wyner_dim * 2),   # [mu_p | logvar_p]
        )
        # Zero-init logvar head: prior initialises as N(0, I)
        nn.init.zeros_(self.net[-1].bias[wyner_dim:])

    def forward(
        self, context_t: th.Tensor  # (B, context_dim)
    ) -> Tuple[th.Tensor, th.Tensor]:
        out = self.net(context_t)
        mu_p, logvar_p = out.chunk(2, dim=-1)  # each (B, wyner_dim)
        return mu_p, logvar_p


class PosteriorNet(nn.Module):
    """
    Wyner posterior: q_φ(z_t | context_t, sg(mu_t), sg(mu_{t+1})).

    Sees BOTH mu_t and mu_{t+1} — required by Wyner's formulation:
    the minimal W such that X ⊥ Y | W can only be inferred by an encoder
    that observes both X and Y.

    Both mu inputs are sg'd at the call site in forward_train:
    - sg(mu_t)   → WynerVAE loss shapes posterior, not SCVAE encoder
    - sg(mu_{t+1}) → future is a target, not a learnable output
    """

    def __init__(
        self,
        context_dim: int,
        mu_dim:      int,
        wyner_dim:   int,
        hidden_dim:  int,
    ):
        super().__init__()
        self.wyner_dim = wyner_dim
        self.net = nn.Sequential(
            nn.Linear(context_dim + 2 * mu_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, wyner_dim * 2),   # [mu_q | logvar_q]
        )
        nn.init.zeros_(self.net[-1].bias[wyner_dim:])

    def forward(
        self,
        context_t: th.Tensor,   # (B, context_dim)
        mu_t:      th.Tensor,   # (B, mu_dim)  — sg applied by caller
        mu_tp1:    th.Tensor,   # (B, mu_dim)  — sg applied by caller
    ) -> Tuple[th.Tensor, th.Tensor]:
        x   = th.cat([context_t, mu_t, mu_tp1], dim=-1)
        out = self.net(x)
        mu_q, logvar_q = out.chunk(2, dim=-1)       # each (B, wyner_dim)
        return mu_q, logvar_q


class WynerDecoder(nn.Module):
    """
    Self-sufficient decoder: reconstructs a mu vector from z_t ONLY.

    Why no context allowed:
        If context_t were available, the decoder could reconstruct mu_t / mu_{t+1}
        from context alone and simply ignore z_t.  KL minimisation would then
        collapse z_t to the prior (posterior collapse).  Self-sufficiency forces
        ALL transition information into z_t, which is the Wyner bottleneck.

    Used twice: once as DecoderPast (target: sg(mu_t)),
                once as DecoderFuture (target: sg(mu_{t+1})).
    """

    def __init__(self, wyner_dim: int, mu_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(wyner_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, mu_dim),
        )

    def forward(self, z_t: th.Tensor) -> th.Tensor:  # (B, mu_dim)
        return self.net(z_t)


# ─────────────────────────────────────────────────────────────────────────────
# Main module
# ─────────────────────────────────────────────────────────────────────────────

class WynerVAE(nn.Module):
    """
    Wyner Common Information VAE for temporal transition representation.

    Phase 3 TODO: add FastPath (K=64 sliding-window cross-attention) and
    replace `context_t = h_slow` with `context_t = GatedFusion(fast, slow)`.
    """

    def __init__(self, cfg: WynerConfig):
        super().__init__()
        self.cfg = cfg

        self.slow_path      = SlowPath(cfg.mu_dim, cfg.context_dim)
        self.prior          = PriorNet(cfg.context_dim, cfg.wyner_dim, cfg.hidden_dim)
        self.posterior      = PosteriorNet(
            cfg.context_dim, cfg.mu_dim, cfg.wyner_dim, cfg.hidden_dim
        )
        self.decoder_past   = WynerDecoder(cfg.wyner_dim, cfg.mu_dim, cfg.hidden_dim)
        self.decoder_future = WynerDecoder(cfg.wyner_dim, cfg.mu_dim, cfg.hidden_dim)

    # ── FORWARD ──────────────────────────────────────────────────────────────

    def forward_train(
        self,
        mu_t:   th.Tensor,   # (B, mu_dim) — SCVAE latent at t
        mu_tp1: th.Tensor,   # (B, mu_dim) — SCVAE latent at t+1
        h_prev: th.Tensor,   # (B, context_dim)
    ) -> WynerOutput:
        """
        Training forward pass. Both mu inputs are sg'd here so that
        WynerVAE loss cannot modify TransitionSCVAE weights.

        Step-by-step:
          1. Slow path updates context  (sg(mu_t) → GRU)
          2. Prior predicts z_t from context only
          3. Posterior infers z_t from context + both mu's (Wyner condition)
          4. Sample z_t via reparameterisation
          5. Self-sufficient decoders reconstruct past and future from z_t alone
        """
        # Step 1: context (sg inside SlowPath.forward, but also explicit here for clarity)
        h_t = self.slow_path(mu_t, h_prev)           # h_t ≡ context_t (Phase 1)

        # Step 2: prior (context only)
        mu_p, logvar_p = self.prior(h_t)

        # Step 3: posterior (context + both mu's, both sg'd)
        mu_q, logvar_q = self.posterior(
            h_t,
            mu_t.detach(),    # sg(mu_t):   WynerVAE does not shape SCVAE repr
            mu_tp1.detach(),  # sg(mu_tp1): future is a fixed target
        )

        # Step 4: sample
        z_t = reparameterize(mu_q, logvar_q)

        # Step 5: self-sufficient decoders — z_t ONLY, no context
        recon_past   = self.decoder_past(z_t)
        recon_future = self.decoder_future(z_t)

        return WynerOutput(
            mu_q=mu_q,           logvar_q=logvar_q,
            mu_p=mu_p,           logvar_p=logvar_p,
            z_t=z_t,
            recon_past=recon_past,
            recon_future=recon_future,
            target_past=mu_t.detach(),     # sg(mu_t)
            target_future=mu_tp1.detach(), # sg(mu_{t+1})
            h_slow=h_t,
        )

    def forward_rollout(
        self,
        mu_t:   th.Tensor,   # (B, mu_dim)
        h_prev: th.Tensor,   # (B, context_dim)
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Rollout forward pass. mu_{t+1} unavailable → sample from prior.
        Returns (z_t, h_t).  Intrinsic reward is computed during the
        separate training pass on the same transitions from the replay buffer.
        """
        h_t = self.slow_path(mu_t, h_prev)
        mu_p, logvar_p = self.prior(h_t)
        z_t = reparameterize(mu_p, logvar_p)
        return z_t, h_t

    # ── LOSS ─────────────────────────────────────────────────────────────────

    def loss(self, out: WynerOutput) -> WynerLoss:
        """
        Three-term loss:
          (A) KL[q || p]           — minimality / information bottleneck
          (B) λ_past  * MSE(past)  — Wyner ← (past conditional independence)
          (C) λ_future * MSE(future) — Wyner → (future conditional independence)

        Per-sample KL also serves as the intrinsic reward (detached).
        """
        # (A) KL — per sample (B,), then optionally floor at free_nats
        kl_per_sample = gaussian_kl(
            out.mu_q, out.logvar_q,
            out.mu_p, out.logvar_p,
        )
        if self.cfg.free_nats > 0.0:
            kl_per_sample = th.clamp(kl_per_sample, min=self.cfg.free_nats)

        kl_loss = kl_per_sample.mean()

        # (B) Past reconstruction
        recon_past_loss = F.mse_loss(out.recon_past, out.target_past)

        # (C) Future reconstruction
        recon_future_loss = F.mse_loss(out.recon_future, out.target_future)

        total = (
            kl_loss
            + self.cfg.lambda_past   * recon_past_loss
            + self.cfg.lambda_future * recon_future_loss
        )

        # Intrinsic reward: per-sample, detached — no gradient flows through reward
        intrinsic_reward = self.cfg.alpha_intrinsic * kl_per_sample.detach()

        return WynerLoss(
            kl_loss=kl_loss,
            recon_past_loss=recon_past_loss,
            recon_future_loss=recon_future_loss,
            total_loss=total,
            intrinsic_reward=intrinsic_reward,
        )

    # ── HELPERS ──────────────────────────────────────────────────────────────

    def init_hidden(self, batch_size: int, device: th.device) -> th.Tensor:
        """Convenience wrapper; delegates to SlowPath."""
        return self.slow_path.init_hidden(batch_size, device)