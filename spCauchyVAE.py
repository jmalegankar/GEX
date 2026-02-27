"""
Transition Spherical Cauchy VAE.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT THIS MODULE DOES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Compresses a full transition (s_t, a_t, s_{t+1}) into a single unit
vector mu ∈ S^{d-1}. This compressed encoding is the primary currency
of the rest of the system — Wyner consumes mu to compute surprise,
and the novelty memory hashes mu to detect repeated states.

Encoding transitions (rather than raw observations) is deliberate:
  - A "state" is ambiguous — the same room looks different from different
    angles. A "transition" encodes what the agent DID and what CHANGED,
    which is a more stable and meaningful unit for exploration.
  - The action is embedded and fused at the bottleneck, so mu reflects
    both the context AND the choice that led to the new state.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENCODER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
s_t and s_{t+1} are processed by the SAME conv weights (shared/siamese).
This is correct: both are observations from the same environment, just
at different times. Using shared weights:
  1. Halves the parameters for the observation encoder.
  2. Ensures the features extracted from s_t and s_{t+1} live in the
     same feature space, making their combination at the bottleneck
     meaningful rather than arbitrary.

The outputs are two flat feature vectors, concatenated with the action
embedding, and passed to the transition bottleneck.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SPHERICAL CAUCHY LATENT SPACE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The latent space is S^{d-1} (a hypersphere), not R^d. This is critical.

Standard Gaussian VAEs in high dimensions suffer posterior collapse:
the KL term KL(N(mu,sig) ∥ N(0,I)) has magnitude ~d, which overwhelms
the reconstruction loss and drives the posterior to the prior regardless
of input. The encoder stops being informative.

Spherical Cauchy VAEs avoid this failure mode differently:
  - The posterior is spCauchy(mu, rho) and the prior is Uniform(S^{d-1}).
  - The KL(spCauchy ∥ Uniform) is independent of mu (direction is not
    directly penalized), and depends only on rho (concentration).
  - However, the KL is NOT the simple -(d-1)/2 * log(1-rho^2).
    For the Möbius-transformed spherical Cauchy used here, the KL has
    a rapidly convergent series form (Theorem 1 in arXiv:2506.21278).

rho is bounded to [rho_min, rho_max] to prevent numerical issues
and overly concentrated posteriors.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DECODER DESIGN — TWO SEPARATE STATE HEADS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The decoder must reconstruct [emb(s_t).flat, a_emb, emb(s_{t+1}).flat]
from a single z. The key constraint: s_t and s_{t+1} are generally
DIFFERENT observations, so the decoder must produce different outputs
for each half of the target.

The trunk (z → h) is shared — z encodes the full transition. But then:
  state_t_head(h)    → emb(s_t).flat
  action_head(h)     → a_emb
  state_next_head(h) → emb(s_{t+1}).flat

These are three independent linear layers. Using a shared state head
(called twice with the same h) would produce identical outputs for
s_t and s_{t+1}, which is wrong: the loss would push toward the
average of both embeddings rather than either one correctly.

Note: the ENCODER uses shared siamese weights because its inputs differ
(s_t ≠ s_{t+1}). The DECODER uses separate heads because its input is
the same (h from z) but outputs must differ. The symmetry is intentional
but the implementations are different for a good reason.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Public API
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TransitionSCVAE.encode(s, a, s')            → mu
TransitionSCVAE.decode(z)                   → recon
TransitionSCVAE.forward(s, a, s')           → SCVAEForwardOutput
TransitionSCVAE.loss(out)                   → (l_recon, l_kl)
TransitionSCVAE.transition_target(s, a, s') → flat target (detached)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================================================================
# Output type
# ===================================================================

class SCVAEForwardOutput(NamedTuple):
    recon:        torch.Tensor  # (B, transition_target_dim)
    mu:           torch.Tensor  # (B, d) unit vector on S^{d-1}
    rho:          torch.Tensor  # (B, 1) concentration in [rho_min, rho_max]
    z:            torch.Tensor  # (B, d) sampled latent (= mu at eval)
    recon_target: torch.Tensor  # (B, transition_target_dim) detached


# ===================================================================
# Spherical Cauchy helpers (private)
# ===================================================================

def _mobius_add(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Möbius addition in the Poincaré ball.

    Maps a (inside unit ball) and x (on unit sphere) to a point on the sphere.
    This is the core operation enabling reparameterized sampling from spCauchy:
    we add a scaled version of mu (rho*mu, inside the ball) to a uniform
    sphere sample, producing a sample concentrated around mu.
    """
    a_sq = (a * a).sum(-1, keepdim=True)
    x_sq = (x * x).sum(-1, keepdim=True)
    ax   = (a * x).sum(-1, keepdim=True)
    num  = (1 + 2 * ax + x_sq) * a + (1 - a_sq) * x
    den  = 1 + 2 * ax + a_sq * x_sq
    return F.normalize(num / (den + 1e-8), p=2, dim=-1)


def _sc_sample(mu: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """
    Reparameterized sample from spCauchy(mu, rho).

    1. xi ~ Uniform(S^{d-1})   — base noise
    2. a  = rho * mu            — shift point inside ball (concentration control)
    3. z  = Möbius(a, xi)       — map to sphere, concentrated around mu

    Higher rho → samples cluster tightly around mu.
    rho = 0    → uniform on sphere (= prior).
    """
    xi = F.normalize(torch.randn_like(mu), p=2, dim=-1)
    return _mobius_add(rho * mu, xi)


def _sc_z_of_rho(rho: torch.Tensor) -> torch.Tensor:
    """
    z(rho) = 4 rho / (1 + rho)^2, used by the KL series (arXiv:2506.21278, Thm 1).
    rho: (B,1) or (B,)
    """
    return 4.0 * rho / (1.0 + rho).pow(2)


def _sc_kl_uniform(
    rho: torch.Tensor,
    dim: int,
    *,
    max_terms: int = 256,
    tol: float = 1e-10,
) -> torch.Tensor:
    r"""
    KL(spCauchy_d(·|mu,rho) || Uniform(S^{d-1})) for the Möbius spCauchy used here.

    Uses Theorem 1 (arXiv:2506.21278) series:

      z(rho) = 4 rho / (1 + rho)^2

      KL = (d-1) log((1-rho)/(1+rho))
           + (d-1) * ((1-rho)/(1+rho))^(d-1)
             * sum_{k=0}^\infty [ (( (d-1)/2 )_k / k! ) * z(rho)^k
                                 * (psi(d-1+k) - psi(d-1)) ]

    Notes:
    - Independent of mu (direction not directly penalized), depends on rho only.
    - Series converges for rho in [0,1). We still cap terms for speed.
    - For rho very close to 1, convergence slows; keep rho_max < 1 (you do),
      and/or raise max_terms.

    Args:
        rho: (B,1) concentration in (0,1)
        dim: latent dimension d
        max_terms: max k to sum
        tol: early-stop threshold on max |term| across batch

    Returns:
        kl: (B,1) tensor
    """
    if dim < 2:
        raise ValueError(f"spCauchy KL requires dim>=2, got dim={dim}")

    # Ensure shape (B,1) and float
    rho = rho.to(dtype=torch.float32)
    if rho.dim() == 1:
        rho = rho.unsqueeze(-1)

    # Clamp away from exactly 0/1 for stability (grad-friendly-ish)
    eps = 1e-7
    rho = rho.clamp(min=0.0, max=1.0 - eps)

    d_minus_1 = float(dim - 1)
    device = rho.device
    dtype = rho.dtype

    # First term: (d-1) * log((1-rho)/(1+rho))
    # Use log1p for numerical stability.
    log_ratio = torch.log1p(-rho) - torch.log1p(rho)  # (B,1), negative
    term1 = d_minus_1 * log_ratio

    # Prefactor: (d-1) * ((1-rho)/(1+rho))^(d-1)
    ratio = (1.0 - rho) / (1.0 + rho)
    pref = d_minus_1 * ratio.pow(d_minus_1)  # (B,1)

    # Series: sum_{k>=0} coeff_k * z^k * (psi(d-1+k)-psi(d-1))
    z = _sc_z_of_rho(rho)  # (B,1)

    a = (d_minus_1 / 2.0)  # (d-1)/2
    psi_base = torch.digamma(torch.tensor(d_minus_1, device=device, dtype=dtype))

    # k=0 term is zero because psi(d-1+0)-psi(d-1)=0, so we start at k=1.
    # We'll build coeff_k = (a)_k / k! via log-gamma each iteration.
    series = torch.zeros_like(rho)

    for k in range(1, max_terms + 1):
        k_t = torch.tensor(float(k), device=device, dtype=dtype)

        # log((a)_k / k!) = lgamma(a+k) - lgamma(a) - lgamma(k+1)
        log_coeff = (
            torch.lgamma(torch.tensor(a, device=device, dtype=dtype) + k_t)
            - torch.lgamma(torch.tensor(a, device=device, dtype=dtype))
            - torch.lgamma(k_t + 1.0)
        )
        coeff = torch.exp(log_coeff)  # scalar

        psi_diff = torch.digamma(torch.tensor(d_minus_1 + float(k), device=device, dtype=dtype)) - psi_base  # scalar
        term = coeff * (z.pow(k)) * psi_diff  # (B,1)

        series = series + term

        # Early stopping if the newest term is tiny everywhere
        if float(term.abs().max().detach().cpu()) < tol:
            break

    kl = term1 + pref * series  # (B,1)
    # KL should be >=0. Numerical tiny negatives can happen at rho~0; clamp.
    return kl.clamp_min(0.0)


# ===================================================================
# Categorical embedding
# ===================================================================

class CategoricalEmbedding(nn.Module):
    """
    (B, H, W, 3) categorical IDs → (B, C, H, W) continuous embeddings.

    MultiGrid observations encode each cell as (object_type, color, state).
    Each channel is embedded independently into a learned continuous vector,
    then all three are concatenated. This gives the conv encoder a continuous
    input surface to work with rather than raw integer codes.
    """

    def __init__(self, n_obj: int, n_color: int, n_state: int, embed_per_ch: int) -> None:
        super().__init__()
        self.obj   = nn.Embedding(n_obj, embed_per_ch)
        self.color = nn.Embedding(n_color, embed_per_ch)
        self.state = nn.Embedding(n_state, embed_per_ch)
        self.out_channels: int = 3 * embed_per_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.obj(x[..., 0].long())
        c = self.color(x[..., 1].long())
        s = self.state(x[..., 2].long())
        return torch.cat([o, c, s], dim=-1).permute(0, 3, 1, 2)  # (B, C, H, W)


# ===================================================================
# Conv block
# ===================================================================

class ConvBlock(nn.Module):
    """
    Conv3x3 → ReLU → Conv3x3 → ReLU. No spatial downsampling.

    At view_size=5, spatial downsampling would reduce the feature map
    too aggressively. We keep spatial resolution and use channel expansion
    to increase capacity instead.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ===================================================================
# State encoder — siamese shared weights
# ===================================================================

class StateEncoder(nn.Module):
    """
    Shared-weight conv encoder called independently for s_t and s_{t+1}.

    Returns flat conv features only. Does not return the categorical
    embedding — transition_target() calls self.embedding directly to
    build reconstruction targets without redundant conv passes.

    The same weights process both observations. This is correct because
    s_t and s_{t+1} come from the same environment (same feature semantics),
    and we want the features to be comparable at the bottleneck.
    """

    def __init__(self, cfg: SCVAEConfig) -> None:
        super().__init__()
        self.embedding = CategoricalEmbedding(
            n_obj=cfg.n_object_types,
            n_color=cfg.n_colors,
            n_state=cfg.n_states,
            embed_per_ch=cfg.embed_per_ch,
        )
        ch = [cfg.embed_channels] + list(cfg.conv_channels)
        self.blocks = nn.ModuleList(
            [ConvBlock(ch[i], ch[i + 1]) for i in range(len(ch) - 1)]
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, H, W, 3) long categorical
        Returns:
            flat: (B, top_ch * H * W)
        """
        x = self.embedding(obs)
        for block in self.blocks:
            x = block(x)
        return x.flatten(start_dim=1)


# ===================================================================
# Transition bottleneck
# ===================================================================

class TransitionBottleneck(nn.Module):
    """
    [h_s, a_emb, h_s'] → (mu, rho) on S^{d-1}.

    Fuses the two observation features and the action embedding into
    a single transition code. The FC layer learns which combination of
    (what I saw, what I did, what resulted) is most informative for
    distinguishing this transition from others.

    rho is clamped to [rho_min, rho_max]:
      rho_min > 0: prevents numerical issues
      rho_max < 1: prevents a delta distribution (infinitely concentrated)
    """

    def __init__(self, feat_dim: int, action_dim: int, cfg: SCVAEConfig) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2 * feat_dim + action_dim, cfg.hidden_dim), nn.ReLU(),
        )
        self.fc_mu  = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.fc_rho = nn.Linear(cfg.hidden_dim, 1)
        self.rho_min: float = cfg.rho_min
        self.rho_max: float = cfg.rho_max
        nn.init.zeros_(self.fc_rho.bias)  # initial rho ≈ midpoint of [rho_min, rho_max]

    def forward(
        self,
        h_s: torch.Tensor,
        a_emb: torch.Tensor,
        h_sn: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h   = self.fc(torch.cat([h_s, a_emb, h_sn], dim=-1))
        mu  = F.normalize(self.fc_mu(h), p=2, dim=-1)
        rho = self.rho_min + torch.sigmoid(self.fc_rho(h)) * (self.rho_max - self.rho_min)
        return mu, rho


# ===================================================================
# Flat decoder — three independent heads
# ===================================================================

class FlatDecoder(nn.Module):
    """
    z → [emb(s_t).flat, a_emb, emb(s_{t+1}).flat].

    The trunk maps z to a shared hidden representation h, then three
    independent linear heads produce the three parts of the target.
    Independent heads are required because s_t and s_{t+1} are different
    observations — calling the same head twice with the same h would
    produce identical outputs for both, making the decoder unable to
    reconstruct the before/after distinction.
    """

    def __init__(self, cfg: SCVAEConfig) -> None:
        super().__init__()
        obs_flat: int = cfg.embed_channels * cfg.view_size ** 2
        self.trunk = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim), nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU(),
        )
        self.state_t_head    = nn.Linear(cfg.hidden_dim, obs_flat)
        self.state_next_head = nn.Linear(cfg.hidden_dim, obs_flat)
        self.action_head     = nn.Linear(cfg.hidden_dim, cfg.action_embed_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.trunk(z)
        return torch.cat([
            self.state_t_head(h),
            self.action_head(h),
            self.state_next_head(h),
        ], dim=-1)


# ===================================================================
# TransitionSCVAE — public API
# ===================================================================

class TransitionSCVAE(nn.Module):
    """
    Siamese Spherical Cauchy VAE for (s_t, a_t, s_{t+1}) transitions.

    Produces mu ∈ S^{d-1}: a compressed, direction-only encoding of a
    full transition.

    Training is i.i.d. over a flat transition buffer — no episode
    structure required. The trainer owns the beta schedule and calls
    loss() to get the two components separately.

    Attributes:
        latent_dim: dimensionality d of the hypersphere S^{d-1}
    """

    def __init__(self, cfg: SCVAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.latent_dim: int = cfg.latent_dim

        self._encoder      = StateEncoder(cfg)
        self._action_embed = nn.Embedding(cfg.n_actions, cfg.action_embed_dim)
        self._bottleneck   = TransitionBottleneck(cfg.conv_feature_dim, cfg.action_embed_dim, cfg)
        self._decoder      = FlatDecoder(cfg)

    def encode(self, s_t: torch.Tensor, a_t: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        """
        Transition → mu on S^{d-1}. Used at inference time.

        Returns only mu (not rho). rho is only needed during training for
        the KL loss. At inference, mu is the sufficient statistic.
        """
        h_s  = self._encoder(s_t)
        h_sn = self._encoder(s_next)
        a_emb = self._action_embed(a_t.long())
        mu, _ = self._bottleneck(h_s, a_emb, h_sn)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Latent sample → reconstructed flat transition. Exposed for diagnostics."""
        return self._decoder(z)

    def transition_target(
        self, s_t: torch.Tensor, a_t: torch.Tensor, s_next: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build the flat reconstruction target. Detached — no gradients flow back.

        Returns:
            (B, transition_target_dim) = [emb(s_t).flat, a_emb, emb(s_next).flat]
        """
        with torch.no_grad():
            emb_s  = self._encoder.embedding(s_t)
            emb_sn = self._encoder.embedding(s_next)
            a_emb  = self._action_embed(a_t.long())
        return torch.cat([emb_s.flatten(1), a_emb, emb_sn.flatten(1)], dim=-1)

    def forward(
        self, s_t: torch.Tensor, a_t: torch.Tensor, s_next: torch.Tensor,
    ) -> SCVAEForwardOutput:
        """
        Full training pass.

        At train time, z is sampled from spCauchy(mu, rho) via reparameterization.
        At eval time, z = mu (the mode of the distribution, no noise).
        """
        h_s  = self._encoder(s_t)
        h_sn = self._encoder(s_next)
        a_emb = self._action_embed(a_t.long())

        mu, rho = self._bottleneck(h_s, a_emb, h_sn)
        z = _sc_sample(mu, rho) if self.training else mu
        recon = self._decoder(z)
        recon_target = self.transition_target(s_t, a_t, s_next)

        return SCVAEForwardOutput(recon, mu, rho, z, recon_target)

    def loss(self, out: SCVAEForwardOutput) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruction and KL losses returned separately. Caller applies beta weighting.

        Returns:
            l_recon: MSE(recon, target) — scalar, with grad
            l_kl:    KL(spCauchy ∥ Uniform) — scalar, with grad
        """
        l_recon = F.mse_loss(out.recon, out.recon_target)
        l_kl    = _sc_kl_uniform(out.rho, self.latent_dim).mean()
        return l_recon, l_kl


# ===================================================================
# Smoke test
# ===================================================================

if __name__ == "__main__":
    from models.config import SCVAEConfig

    cfg   = SCVAEConfig()
    model = TransitionSCVAE(cfg)
    B     = 4

    s_t    = torch.randint(0, 5, (B, cfg.view_size, cfg.view_size, 3))
    a_t    = torch.randint(0, cfg.n_actions, (B,))
    s_next = torch.randint(0, 5, (B, cfg.view_size, cfg.view_size, 3))

    # encode
    mu = model.encode(s_t, a_t, s_next)
    assert mu.shape == (B, cfg.latent_dim)
    assert torch.allclose(mu.norm(dim=-1), torch.ones(B), atol=1e-5), "mu not unit"
    print(f"encode OK: {mu.shape}")

    # decode
    z = F.normalize(torch.randn(B, cfg.latent_dim), p=2, dim=-1)
    assert model.decode(z).shape == (B, cfg.transition_target_dim)
    print(f"decode OK: {(B, cfg.transition_target_dim)}")

    # transition_target: s_t and s_next portions must differ
    target = model.transition_target(s_t, a_t, s_next)
    assert target.shape == (B, cfg.transition_target_dim)
    assert not target.requires_grad
    obs_flat    = cfg.embed_channels * cfg.view_size ** 2
    s_t_part    = target[:, :obs_flat]
    s_next_part = target[:, obs_flat + cfg.action_embed_dim:]
    assert not torch.allclose(s_t_part, s_next_part), \
        "s_t and s_next portions of target are identical — embedding bug"
    print(f"transition_target OK: {target.shape}, s_t≠s_next ✓")

    # forward + loss
    model.train()
    out = model(s_t, a_t, s_next)
    assert isinstance(out, SCVAEForwardOutput)
    assert not out.recon_target.requires_grad

    # decoder must produce distinct s_t and s_next outputs
    s_t_recon    = out.recon[:, :obs_flat]
    s_next_recon = out.recon[:, obs_flat + cfg.action_embed_dim:]
    assert not torch.allclose(s_t_recon, s_next_recon), \
        "decoder produced identical s_t and s_next outputs — shared head bug"
    print(f"forward OK, s_t_recon≠s_next_recon ✓")

    l_recon, l_kl = model.loss(out)
    assert l_recon.requires_grad and l_kl.requires_grad
    (l_recon + 0.005 * l_kl).backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0
    print(f"loss OK: l_recon={l_recon.item():.4f}, l_kl={l_kl.item():.4f}, grad_norm={grad_norm:.4f}")

    print("\nALL SC-VAE TESTS PASSED")