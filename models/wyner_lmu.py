from __future__ import annotations

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import cont2discrete

from typing import Optional, List, Tuple


# ═══════════════════════════════════════════════════════════════════════
# LMU matrices A, B for given memory size d and window length theta.
# ═══════════════════════════════════════════════════════════════════════

def get_AB(d: int, theta: float = 1.0):
    Q = np.arange(d, dtype=float)
    R = (2 * Q + 1)[:, None]
    j, i = np.meshgrid(Q, Q)
    A = R * np.where(i < j, -1.0, (-1.0) ** (i - j + 1)) / theta
    B = R * ((-1.0) ** Q)[:, None] / theta
    C = np.zeros((1, d))
    D = np.zeros((1,))
    A_d, B_d, _, _, _ = cont2discrete((A, B, C, D), dt=1.0, method="zoh")
    return A_d.astype(np.float32), B_d.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# LMU Cell
# ═══════════════════════════════════════════════════════════════════════

class LMUCell(nn.Module):
    """
    Single-step LMU. State = (h, m).
      u_t = e_x(x_t) + e_h(h_{t-1}) + e_m(m_{t-1})   scalar
      m_t = A m_{t-1} + B u_t                          linear memory
      h_t = tanh(W_x x_t + W_h h_{t-1} + W_m m_t)     nonlinear hidden
    """

    def __init__(self, input_size: int, hidden_size: int, memory_size: int, theta: float):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size

        A, B = get_AB(memory_size, theta)
        self.register_buffer("A", th.from_numpy(A))  # (d, d)
        self.register_buffer("B", th.from_numpy(B))  # (d, 1)

        self.e_x = nn.Linear(input_size, 1, bias=False)
        self.e_h = nn.Linear(hidden_size, 1, bias=False)
        self.e_m = nn.Linear(memory_size, 1, bias=False)

        self.W_x = nn.Linear(input_size, hidden_size, bias=True)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_m = nn.Linear(memory_size, hidden_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.e_m.weight)
        for layer in (self.W_x, self.W_h, self.W_m):
            nn.init.xavier_normal_(layer.weight)
        for layer in (self.e_x, self.e_h):
            fan_in = layer.weight.shape[1]
            nn.init.uniform_(layer.weight, -fan_in**-0.5, fan_in**-0.5)

    def forward(self, x: th.Tensor, h: th.Tensor, m: th.Tensor):
        u = self.e_x(x) + self.e_h(h) + self.e_m(m)       # (B, 1)
        m_new = m @ self.A.T + u * self.B.T                # (B, d)
        h_new = th.tanh(self.W_x(x) + self.W_h(h) + self.W_m(m_new))
        return h_new, m_new


# ═══════════════════════════════════════════════════════════════════════
# Prior network: conv1d + attention over (h_{t-1}, m_{t-1})
# ═══════════════════════════════════════════════════════════════════════

class LMUPriorNetwork(nn.Module):
    """
    Learned prior p(z_t | h_{t-1}, m_{t-1}).

    m_{t-1} ∈ R^{memory_size} is treated as a 1D sequence of Legendre
    coefficients. A small conv1d extracts multi-scale features, then
    cross-attention (query=h_{t-1}, kv=conv features) produces a summary
    that is projected to (prior_mu, prior_logvar).
    """

    def __init__(
        self,
        hidden_size: int,
        memory_size: int,
        latent_dim: int,
        conv_channels: int = 64,
        n_conv_layers: int = 2,
        kernel_size: int = 3,
        n_attn_heads: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size

        # Conv1d over Legendre coefficients: (B, 1, memory_size) → (B, C, L')
        layers = []
        in_ch = 1
        for _ in range(n_conv_layers):
            layers.extend([
                nn.Conv1d(in_ch, conv_channels, kernel_size, padding=kernel_size // 2),
                nn.GroupNorm(min(8, conv_channels), conv_channels),
                nn.ReLU(inplace=True),
            ])
            in_ch = conv_channels
        self.conv = nn.Sequential(*layers)

        # Cross-attention: query from h, kv from conv features
        self.q_proj = nn.Linear(hidden_size, conv_channels)
        self.attn = nn.MultiheadAttention(
            conv_channels, num_heads=n_attn_heads, batch_first=True,
        )

        # Output heads
        self.fc = nn.Sequential(
            nn.Linear(conv_channels + hidden_size, 128),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        nn.init.zeros_(self.fc_logvar.bias)

    def forward(self, h_prev: th.Tensor, m_prev: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        h_prev: (B, hidden_size)
        m_prev: (B, memory_size)
        Returns: (prior_mu, prior_logvar) each (B, latent_dim)
        """
        # Conv over Legendre coefficients
        m_seq = m_prev.unsqueeze(1)           # (B, 1, memory_size)
        conv_out = self.conv(m_seq)           # (B, C, L')
        kv = conv_out.permute(0, 2, 1)       # (B, L', C)

        # Cross-attention: query = h_prev
        q = self.q_proj(h_prev).unsqueeze(1)  # (B, 1, C)
        attn_out, _ = self.attn(q, kv, kv)    # (B, 1, C)
        attn_out = attn_out.squeeze(1)         # (B, C)

        # Combine with h_prev for final projection
        combined = th.cat([attn_out, h_prev], dim=-1)
        feat = self.fc(combined)
        return self.fc_mean(feat), self.fc_logvar(feat)


# ═══════════════════════════════════════════════════════════════════════
# Legendre reconstruction decoder
# ═══════════════════════════════════════════════════════════════════════

class LegendreReconDecoder(nn.Module):
    """
    Decode (s, a, s') from LMU state.

    Step 1: Reconstruct mu_x from m_t via a learned linear readout of
            Legendre coefficients (exploiting that m_t encodes a windowed
            history of the scalar input signal).
    Step 2: Combine recon_mu_x with h_t through a gated residual:
            out = f(recon_mu_x, h_t) + recon_mu_x
    Step 3: MLP → (s, a, s') reconstruction.
    """

    def __init__(
        self,
        hidden_size: int,
        memory_size: int,
        mu_dim: int,
        recon_dim: int,
        decode_hidden: int = 128,
    ):
        super().__init__()
        self.mu_dim = mu_dim

        # Legendre readout: m_t → reconstructed mu_x
        self.legendre_readout = nn.Linear(memory_size, mu_dim)

        # Gated residual: f(recon_mu_x, h_t) + recon_mu_x
        self.gate_net = nn.Sequential(
            nn.Linear(mu_dim + hidden_size, decode_hidden),
            nn.ReLU(),
            nn.Linear(decode_hidden, mu_dim),
        )
        self.gate = nn.Linear(mu_dim + hidden_size, mu_dim)

        # Final MLP → recon_dim
        self.mlp = nn.Sequential(
            nn.Linear(mu_dim, decode_hidden),
            nn.ReLU(),
            nn.Linear(decode_hidden, decode_hidden),
            nn.ReLU(),
            nn.Linear(decode_hidden, recon_dim),
        )

    def forward(self, h: th.Tensor, m: th.Tensor) -> th.Tensor:
        """
        h: (B, hidden_size)  — nonlinear hidden state
        m: (B, memory_size)  — Legendre coefficients
        Returns: (B, recon_dim)
        """
        recon_mu = self.legendre_readout(m)                       # (B, mu_dim)
        combined = th.cat([recon_mu, h], dim=-1)                  # (B, mu_dim + hidden)
        g = th.sigmoid(self.gate(combined))                       # (B, mu_dim)
        residual = self.gate_net(combined)                        # (B, mu_dim)
        fused = g * residual + recon_mu                           # gated residual
        return self.mlp(fused)                                    # (B, recon_dim)


# ═══════════════════════════════════════════════════════════════════════
# Conv1d feature extractor for action network
# ═══════════════════════════════════════════════════════════════════════

class Conv1DHead(nn.Module):
    """Small conv1d + global avg pool over a 1D signal."""

    def __init__(self, in_dim: int, out_channels: int, kernel_size: int = 3, n_layers: int = 2):
        super().__init__()
        layers = []
        in_ch = 1
        for i in range(n_layers):
            out_ch = out_channels
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.out_dim = out_channels

    def forward(self, x: th.Tensor) -> th.Tensor:
        """x: (B, D) → (B, out_channels) via conv1d + global avg pool."""
        x = x.unsqueeze(1)            # (B, 1, D)
        x = self.conv(x)              # (B, C, D)
        return x.mean(dim=-1)         # (B, C) global avg pool


class LMUActionFeatures(nn.Module):
    """
    Action feature extractor using separate conv1d heads for m_t, h_t,
    and mu_{t-1}, merged via concatenation.

    Output dim = 3 * conv_channels (one per head).
    This replaces HSWVIMEFeaturesExtractor for LMU-based policies.
    """

    def __init__(
        self,
        hidden_size: int,
        memory_size: int,
        mu_dim: int,
        conv_channels: int = 32,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.m_head = Conv1DHead(memory_size, conv_channels, kernel_size)
        self.h_head = Conv1DHead(hidden_size, conv_channels, kernel_size)
        self.mu_head = Conv1DHead(mu_dim, conv_channels, kernel_size)
        self._features_dim = 3 * conv_channels

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, h: th.Tensor, m: th.Tensor, mu: th.Tensor) -> th.Tensor:
        """
        h:  (B, hidden_size)
        m:  (B, memory_size)
        mu: (B, mu_dim)
        Returns: (B, 3 * conv_channels)
        """
        return th.cat([
            self.m_head(m),
            self.h_head(h),
            self.mu_head(mu),
        ], dim=-1)


# ═══════════════════════════════════════════════════════════════════════
# WynerOutput / WynerLoss
# ═══════════════════════════════════════════════════════════════════════

@th.jit.script
class WynerLMUOutput:
    def __init__(
        self,
        w: th.Tensor,              # posterior mean (= h_t)
        logvar: th.Tensor,
        recon: th.Tensor,
        h: th.Tensor,              # LMU hidden state
        m: th.Tensor,              # LMU memory (Legendre coefficients)
        recon_next: Optional[th.Tensor] = None,
        prior_mu: Optional[th.Tensor] = None,
        prior_logvar: Optional[th.Tensor] = None,
    ):
        self.w = w
        self.logvar = logvar
        self.recon = recon
        self.h = h
        self.m = m
        self.recon_next = recon_next
        self.prior_mu = prior_mu
        self.prior_logvar = prior_logvar


@th.jit.script
class WynerLMULoss:
    def __init__(
        self,
        kl_loss: th.Tensor,
        recon_loss: Optional[th.Tensor] = None,
        recon_next_loss: Optional[th.Tensor] = None,
    ):
        self.kl_loss = kl_loss
        self.recon_loss = recon_loss
        self.recon_next_loss = recon_next_loss


# ═══════════════════════════════════════════════════════════════════════
# WynerLMUVAE
# ═══════════════════════════════════════════════════════════════════════

class WynerLMUVAE(nn.Module):
    """
    Wyner VAE with LMU backbone.

    State convention: the "memory" tensor passed between steps is
    (B, 1, hidden_size + memory_size) = packed (h, m). This replaces
    the GRU hidden / Mamba flat state from other Wyner variants.

    Encode:
      - LMU step: (mu_t, h_{t-1}, m_{t-1}) → (h_t, m_t)
      - Posterior mean: h_t (no extra projection)
      - Posterior logvar: Linear([h_t, m_t])

    Prior:
      - Conv1d + cross-attention over (h_{t-1}, m_{t-1})

    Decode:
      - Legendre readout of m_t → recon_mu_x
      - Gated residual with h_t → MLP → (s, a, s')
    """

    def __init__(
        self,
        recon_dim: int,
        mu_dim: int,
        latent_dim: int,          # = hidden_size of LMU
        memory_size: int = 64,    # = number of Legendre coefficients
        theta: float = 100.0,     # LMU window length
        decode_hidden: int = 128,
        free_bits: float = 0.0,
        prior_conv_channels: int = 64,
        prior_n_attn_heads: int = 4,
        # Unused but kept for interface compat with train.py
        latent_tokens: int = 1,
        state_dim: int = 0,
        state_tokens: int = 0,
        pos_embed_dim: int = 16,  # ignored — LMU handles time natively
    ):
        super().__init__()
        self.latent_dim = latent_dim      # = hidden_size
        self.memory_size = memory_size
        self.mu_dim = mu_dim
        self.recon_dim = recon_dim
        self.free_bits = free_bits
        self.latent_tokens = latent_tokens
        self.state_dim = state_dim
        self.state_tokens = state_tokens
        self.pos_embed_dim = pos_embed_dim  # stored but unused

        # ── LMU cell ─────────────────────────────────────────
        self.lmu = LMUCell(
            input_size=mu_dim,
            hidden_size=latent_dim,
            memory_size=memory_size,
            theta=theta,
        )

        # ── Posterior ─────────────────────────────────────────
        # Mean = h_t directly → no projection needed (dim = latent_dim)
        # Logvar from [h_t, m_t]
        self.fc_logvar = nn.Linear(latent_dim + memory_size, latent_dim)
        nn.init.zeros_(self.fc_logvar.bias)

        # ── Prior ─────────────────────────────────────────────
        self.prior_net = LMUPriorNetwork(
            hidden_size=latent_dim,
            memory_size=memory_size,
            latent_dim=latent_dim,
            conv_channels=prior_conv_channels,
            n_attn_heads=prior_n_attn_heads,
        )

        # ── Decoder ───────────────────────────────────────────
        self.decoder = LegendreReconDecoder(
            hidden_size=latent_dim,
            memory_size=memory_size,
            mu_dim=mu_dim,
            recon_dim=recon_dim,
            decode_hidden=decode_hidden,
        )

        # Packed state dim for storage: h + m
        self.packed_state_dim = latent_dim + memory_size

    # ── State packing ─────────────────────────────────────────

    def pack_state(self, h: th.Tensor, m: th.Tensor) -> th.Tensor:
        """(B, hidden), (B, memory) → (B, hidden + memory)"""
        return th.cat([h, m], dim=-1)

    def unpack_state(self, packed: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """(B, hidden + memory) → (h, m)"""
        h = packed[..., :self.latent_dim]
        m = packed[..., self.latent_dim:]
        return h, m

    # ── Encode ────────────────────────────────────────────────

    def encode(
        self,
        w: th.Tensor,
        mu: th.Tensor,
        skips: Optional[List[th.Tensor]] = None,
        timestep: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        w: (B, 1, packed_state_dim) or (B, packed_state_dim) — packed (h_{t-1}, m_{t-1})
        mu: (B, mu_dim) — VAE encoding of current transition
        Returns: (z_mu, z_logvar) each (B, latent_dim)
        """
        w_flat = w.squeeze(1) if w.dim() == 3 else w
        h_prev, m_prev = self.unpack_state(w_flat)

        # LMU step
        h_t, m_t = self.lmu(mu, h_prev, m_prev)

        # Posterior
        z_mu = h_t                                                # no projection
        z_logvar = self.fc_logvar(th.cat([h_t, m_t], dim=-1))

        return z_mu, z_logvar

    # ── Decode ────────────────────────────────────────────────

    def decode(
            self,
            z: th.Tensor,
            mu: th.Tensor,
            timestep: Optional[th.Tensor] = None,
        ) -> th.Tensor:
            """
            Decode (s, a, s') from latent.

            z can be either:
            - (B, packed_state_dim) = packed (h, m) from forward()'s w output
            - (B, latent_dim)       = a raw latent sample
            Automatically detects and unpacks when needed.
            """
            if z.shape[-1] == self.packed_state_dim:
                h, m = self.unpack_state(z)
            else:
                h = z
                m = th.zeros(z.size(0), self.memory_size, device=z.device)
            return self.decoder(h, m)

    def _decode_from_state(self, h: th.Tensor, m: th.Tensor) -> th.Tensor:
        """Full decode using both h and m."""
        return self.decoder(h, m)

    # ── Forward ───────────────────────────────────────────────

    def forward(
        self,
        w: th.Tensor,
        mu: th.Tensor,
        mu_next: Optional[th.Tensor] = None,
        skips: Optional[List[th.Tensor]] = None,
        timestep: Optional[th.Tensor] = None,
    ) -> WynerLMUOutput:
        w_flat = w.squeeze(1) if w.dim() == 3 else w
        h_prev, m_prev = self.unpack_state(w_flat)

        # Prior from previous state (BEFORE LMU update)
        prior_mu, prior_logvar = self.prior_net(h_prev, m_prev)

        # LMU step
        h_t, m_t = self.lmu(mu, h_prev, m_prev)

        # Posterior
        z_mu = h_t
        z_logvar = self.fc_logvar(th.cat([h_t, m_t], dim=-1))

        # Sample
        std = th.exp(0.5 * z_logvar)
        z = z_mu + th.randn_like(std) * std

        # Decode current
        recon = self._decode_from_state(z, m_t)

        # Decode next (if mu_next provided, step LMU again)
        recon_next = None
        if mu_next is not None:
            h_tp1, m_tp1 = self.lmu(mu_next, h_t, m_t)
            recon_next = self._decode_from_state(h_tp1, m_tp1)

        # Pack new state as w for storage: use z_mu (= h_t) and m_t
        new_w = self.pack_state(z_mu, m_t)

        return WynerLMUOutput(
            w=new_w,
            logvar=z_logvar,
            recon=recon,
            h=h_t,
            m=m_t,
            recon_next=recon_next,
            prior_mu=prior_mu,
            prior_logvar=prior_logvar,
        )

    # ── Loss ──────────────────────────────────────────────────

    def loss(
        self,
        output: WynerLMUOutput,
        recon_target: Optional[th.Tensor] = None,
        recon_next_target: Optional[th.Tensor] = None,
    ) -> WynerLMULoss:
        # Extract posterior mean from packed w
        z_mu = output.w[..., :self.latent_dim]

        # KL(q || prior)
        if output.prior_mu is not None and output.prior_logvar is not None:
            kl_per_dim = 0.5 * (
                output.prior_logvar - output.logvar
                + (output.logvar.exp() + (z_mu - output.prior_mu).pow(2))
                  / output.prior_logvar.exp()
                - 1.0
            )
        else:
            kl_per_dim = -0.5 * (1 + output.logvar - z_mu.pow(2) - output.logvar.exp())

        kl_loss = kl_per_dim.clamp_min(self.free_bits).mean(dim=-1)

        recon_loss = (
            F.mse_loss(output.recon, recon_target, reduction='none').mean(dim=-1)
            if recon_target is not None else None
        )
        recon_next_loss = (
            F.mse_loss(output.recon_next, recon_next_target, reduction='none').mean(dim=-1)
            if recon_next_target is not None and output.recon_next is not None else None
        )

        return WynerLMULoss(
            kl_loss=kl_loss,
            recon_loss=recon_loss,
            recon_next_loss=recon_next_loss,
        )

    def step(self, w: th.Tensor, mu: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """For rollout: run LMU, return (z_mu, packed_new_state, m_t)."""
        w_flat = w.squeeze(1) if w.dim() == 3 else w
        h_prev, m_prev = self.unpack_state(w_flat)
        h_t, m_t = self.lmu(mu, h_prev, m_prev)
        packed = self.pack_state(h_t, m_t)
        return h_t, m_t, packed


# ═══════════════════════════════════════════════════════════════════════
# Quick smoke test
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    th.manual_seed(0)
    B = 4
    mu_dim = 32
    latent_dim = 64
    memory_size = 64
    recon_dim = 2 * 128 + 32  # typical: 2*conv_out + action_embed

    model = WynerLMUVAE(
        recon_dim=recon_dim,
        mu_dim=mu_dim,
        latent_dim=latent_dim,
        memory_size=memory_size,
        theta=100.0,
        decode_hidden=128,
    )

    # Initial packed state: zeros
    w = th.zeros(B, 1, model.packed_state_dim)
    mu = th.randn(B, mu_dim)
    mu_next = th.randn(B, mu_dim)

    out = model(w, mu, mu_next)
    loss = model.loss(out, recon_target=th.randn(B, recon_dim),
                      recon_next_target=th.randn(B, recon_dim))

    print(f"w shape:          {out.w.shape}")           # (B, packed_state_dim)
    print(f"logvar shape:     {out.logvar.shape}")       # (B, latent_dim)
    print(f"recon shape:      {out.recon.shape}")        # (B, recon_dim)
    print(f"h shape:          {out.h.shape}")            # (B, latent_dim)
    print(f"m shape:          {out.m.shape}")            # (B, memory_size)
    print(f"kl_loss shape:    {loss.kl_loss.shape}")     # (B,)
    print(f"recon_loss shape: {loss.recon_loss.shape}")   # (B,)
    print(f"prior_mu shape:   {out.prior_mu.shape}")     # (B, latent_dim)

    # Action features
    feat_ext = LMUActionFeatures(latent_dim, memory_size, mu_dim, conv_channels=32)
    feats = feat_ext(out.h, out.m, mu)
    print(f"action features:  {feats.shape}")            # (B, 96)

    print("\nAll checks passed.")