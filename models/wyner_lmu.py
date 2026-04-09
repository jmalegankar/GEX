"""
WynerLMU — Wyner VAE with multi-input LMU backbone (HiPPO-style).

The original LMU bottlenecks through a scalar u_t — fine for 1D signals,
but mu ∈ R^{mu_dim} needs multiple channels. Following HiPPO/S4, we run
p independent Legendre channels sharing the same (A, B) matrices.
State: h ∈ R^n, m ∈ R^{p × d}.
"""

from __future__ import annotations

import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import cont2discrete

from typing import Optional, List, Tuple


# ═══════════════════════════════════════════════════════════════════════
# LMU discretized matrices
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


def _build_legendre_C(d: int) -> np.ndarray:
    """
    Shifted Legendre polynomial coefficient matrix C ∈ R^{d × d}.
    C[i, j] encodes the j-th power coefficient of P_i(r), so that
    P_i(r) = sum_j C[i,j] * r^j.
    """
    C = np.zeros((d, d), dtype=np.float32)
    for i in range(d):
        for j in range(i + 1):
            C[i, j] = ((-1) ** (i + j)) * math.comb(i, j) * math.comb(i + j, j)
    return C


# ═══════════════════════════════════════════════════════════════════════
# Multi-Input LMU Cell (HiPPO-style)
# ═══════════════════════════════════════════════════════════════════════

class MultiInputLMUCell(nn.Module):
    """
    Multi-input LMU: p independent channels, each with d Legendre coefficients.

    State: h ∈ R^{hidden_size}, m ∈ R^{p × d}

    Each channel independently tracks the history of one projected input
    dimension via the shared (A, B) Legendre dynamics. The nonlinear
    hidden state h reads from all channels simultaneously.

    This is the standard HiPPO/S4 factorization applied to the LMU:
    same (A, B) broadcast across p channels.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        memory_size: int,
        n_channels: int,
        theta: float,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.n_channels = n_channels
        self.theta = theta

        # ── Fixed Legendre dynamics (shared across channels) ──
        A, B = get_AB(memory_size, theta)
        self.register_buffer("A", th.from_numpy(A))       # (d, d)
        self.register_buffer("B", th.from_numpy(B))       # (d, 1)

        # ── Shifted Legendre polynomial coefficients for reconstruction ──
        C = _build_legendre_C(memory_size)
        self.register_buffer("C", th.from_numpy(C))       # (d, d)

        # ── Encoding: input → p channels ──
        # Each channel receives a learned scalar signal
        self.e_x = nn.Linear(input_size, n_channels, bias=False)
        self.e_h = nn.Linear(hidden_size, n_channels, bias=False)
        self.e_m = nn.Linear(n_channels * memory_size, n_channels, bias=False)

        # ── Hidden state kernels (read from full flattened memory) ──
        self.W_x = nn.Linear(input_size, hidden_size, bias=True)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_m = nn.Linear(n_channels * memory_size, hidden_size, bias=False)

        self._init_weights()

    @property
    def flat_memory_dim(self) -> int:
        """Flattened memory dimension p*d, for packing into state vectors."""
        return self.n_channels * self.memory_size

    def _init_weights(self):
        nn.init.zeros_(self.e_m.weight)
        for layer in (self.W_x, self.W_h, self.W_m):
            nn.init.xavier_normal_(layer.weight)
        for layer in (self.e_x, self.e_h):
            fan_in = layer.weight.shape[1]
            nn.init.uniform_(layer.weight, -fan_in ** -0.5, fan_in ** -0.5)

    def forward(
        self,
        x: th.Tensor,    # (B, input_size)
        h: th.Tensor,    # (B, hidden_size)
        m: th.Tensor,    # (B, p, d)
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Returns: (h_new, m_new) where h_new: (B, hidden_size), m_new: (B, p, d)
        """
        B = x.size(0)
        m_flat = m.reshape(B, -1)                                  # (B, p*d)

        # p-channel input signal
        u = self.e_x(x) + self.e_h(h) + self.e_m(m_flat)          # (B, p)

        # Batched memory update: A shared across all p channels
        # m @ A^T: (B, p, d) @ (d, d) → (B, p, d)
        # u[..., None] * B^T: (B, p, 1) * (1, d) → (B, p, d)
        m_new = th.matmul(m, self.A.T) + u.unsqueeze(-1) * self.B.T  # (B, p, d)

        m_new_flat = m_new.reshape(B, -1)                          # (B, p*d)
        h_new = th.tanh(self.W_x(x) + self.W_h(h) + self.W_m(m_new_flat))

        return h_new, m_new

    def recon_data(
        self,
        m: th.Tensor,           # (B, p, d)
        timestep: th.Tensor,    # (B,) or (B, 1)
    ) -> th.Tensor:
        """
        Reconstruct stored signal at given timestep using eq. 3.

        Returns: (B, p, d) — per-channel reconstruction weighted by
                 Legendre polynomials evaluated at the query point.
        """
        r = (timestep.float() / self.theta).frac()                 # (B,) or (B,1)
        if r.dim() > 1:
            r = r.squeeze(-1)                                      # (B,)

        j_idx = th.arange(self.memory_size, device=m.device)       # (d,)

        # r^j: (B, 1) ** (1, d) → (B, d)
        R = r.unsqueeze(1).pow(j_idx.unsqueeze(0))                 # (B, d)

        # P_i(r) = sum_j C[i,j] * r^j  →  P: (B, d)
        P = th.matmul(R, self.C.T)                                 # (B, d)

        # Broadcast across channels: P (B, 1, d) * m (B, p, d) → (B, p, d)
        return P.unsqueeze(1) * m                                  # (B, p, d)


# ═══════════════════════════════════════════════════════════════════════
# Prior network: conv1d + attention over (h_{t-1}, m_{t-1})
# ═══════════════════════════════════════════════════════════════════════

class LMUPriorNetwork(nn.Module):
    """
    Learned prior p(z_t | h_{t-1}, m_{t-1}).

    m_{t-1} ∈ R^{p × d} is reshaped to (B, p, d) and treated as a
    multi-channel 1D sequence. Conv1d extracts features across the
    Legendre coefficient axis, then cross-attention (query=h, kv=conv)
    produces (prior_mu, prior_logvar).
    """

    def __init__(
        self,
        hidden_size: int,
        n_channels: int,
        memory_size: int,
        latent_dim: int,
        conv_channels: int = 64,
        n_conv_layers: int = 2,
        kernel_size: int = 3,
        n_attn_heads: int = 4,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.memory_size = memory_size

        # Conv1d over Legendre coefficients: (B, p, d) → (B, C, d)
        layers: list[nn.Module] = []
        in_ch = n_channels
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

        _pool_out = 8
        _conv_ch = 32
        self.fc_logvar = nn.Sequential(
            nn.Conv1d(1, _conv_ch, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(_pool_out),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(_conv_ch * _pool_out, latent_dim),
        )
        nn.init.zeros_(self.fc_logvar[-1].bias)

    def forward(self, h_prev: th.Tensor, m_prev: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        h_prev: (B, hidden_size)
        m_prev: (B, p, d)
        Returns: (prior_mu, prior_logvar) each (B, latent_dim)
        """
        # Conv over Legendre coefficients — m_prev is already (B, p, d)
        conv_out = self.conv(m_prev)                       # (B, C, d)
        kv = conv_out.permute(0, 2, 1)                     # (B, d, C)

        # Cross-attention: query = h_prev
        q = self.q_proj(h_prev).unsqueeze(1)               # (B, 1, C)
        attn_out, _ = self.attn(q, kv, kv)                 # (B, 1, C)
        attn_out = attn_out.squeeze(1)                     # (B, C)

        combined = th.cat([attn_out, h_prev], dim=-1)
        feat = self.fc(combined)
        return self.fc_mean(feat), self.fc_logvar(feat.unsqueeze(1))


# ═══════════════════════════════════════════════════════════════════════
# Legendre reconstruction decoder
# ═══════════════════════════════════════════════════════════════════════

class LegendreReconDecoder(nn.Module):
    """
    Decode (s, a, s') from LMU state.

    Input u comes from LMUCell.recon_data — already Legendre-reconstructed.
    Shape: (B, p, d) flattened to (B, p*d).

    Step 1: Gated residual combining u with h_t
    Step 2: MLP → (s, a, s') reconstruction
    """

    def __init__(
        self,
        hidden_size: int,
        memory_flat_dim: int,     # p * d
        recon_dim: int,
        decode_hidden: int = 128,
    ):
        super().__init__()
        self.memory_flat_dim = memory_flat_dim

        # Gated residual: f(u, h) + u
        self.gate_net = nn.Sequential(
            nn.Linear(memory_flat_dim + hidden_size, decode_hidden),
            nn.ReLU(),
            nn.Linear(decode_hidden, memory_flat_dim),
        )

        # Final MLP → recon_dim
        self.mlp = nn.Sequential(
            nn.Linear(memory_flat_dim, decode_hidden),
            nn.ReLU(),
            nn.Linear(decode_hidden, decode_hidden),
            nn.ReLU(),
            nn.Linear(decode_hidden, recon_dim),
        )

    def forward(self, h: th.Tensor, u_flat: th.Tensor) -> th.Tensor:
        """
        h:      (B, hidden_size)
        u_flat: (B, p*d) — Legendre-reconstructed signal, flattened
        Returns: (B, recon_dim)
        """
        combined = th.cat([u_flat, h], dim=-1)
        residual = self.gate_net(combined)
        fused = residual + u_flat
        return self.mlp(fused)


# ═══════════════════════════════════════════════════════════════════════
# Conv1d feature extractor for action network
# ═══════════════════════════════════════════════════════════════════════

class Conv1DHead(nn.Module):
    """Small conv1d + global avg pool over a 1D signal."""

    def __init__(self, in_channels: int, seq_len: int, out_channels: int,
                 kernel_size: int = 3, n_layers: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = in_channels
        for _ in range(n_layers):
            layers.extend([
                nn.Conv1d(in_ch, out_channels, kernel_size, padding=kernel_size // 2),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_channels
        self.conv = nn.Sequential(*layers)
        self.out_dim = out_channels

    def forward(self, x: th.Tensor) -> th.Tensor:
        """x: (B, C_in, L) → (B, out_channels) via conv1d + global avg pool."""
        x = self.conv(x)              # (B, C_out, L)
        return x.mean(dim=-1)         # (B, C_out)


class LMUActionFeatures(nn.Module):
    """
    Action feature extractor for multi-input LMU.

    Separate conv1d heads for:
      - m_t: (B, p, d) — multi-channel Legendre memory
      - h_t: (B, 1, hidden_size) — hidden state as length-1 sequence
      - mu:  (B, 1, mu_dim) — VAE encoding as length-1 sequence

    Merged via concatenation.
    Output dim = 3 * conv_channels.
    """

    def __init__(
        self,
        observation_space=None,   # ignored — required by SB3
        *,
        hidden_size: int,
        memory_size: int,
        n_channels: int,
        mu_dim: int,
        conv_channels: int = 32,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.n_channels = n_channels

        # m_t: (B, p, d) — p channels, d-length sequences
        self.m_head = Conv1DHead(n_channels, memory_size, conv_channels, kernel_size)
        # h_t: (B, 1, hidden_size) — treat as 1-channel, hidden_size-length
        self.h_head = Conv1DHead(1, hidden_size, conv_channels, kernel_size)
        # mu: (B, 1, mu_dim)
        self.mu_head = Conv1DHead(1, mu_dim, conv_channels, kernel_size)

        self._features_dim = 3 * conv_channels

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, h: th.Tensor, m: th.Tensor, mu: th.Tensor) -> th.Tensor:
        """
        h:  (B, hidden_size)
        m:  (B, p, d)
        mu: (B, mu_dim)
        Returns: (B, 3 * conv_channels)
        """
        return th.cat([
            self.m_head(m),                      # m is already (B, p, d)
            self.h_head(h.unsqueeze(1)),          # (B, 1, hidden_size)
            self.mu_head(mu.unsqueeze(1)),         # (B, 1, mu_dim)
        ], dim=-1)


# ═══════════════════════════════════════════════════════════════════════
# WynerLMUOutput / WynerLMULoss
# ═══════════════════════════════════════════════════════════════════════

@th.jit.script
class WynerLMUOutput:
    def __init__(
        self,
        w: th.Tensor,
        logvar: th.Tensor,
        recon: th.Tensor,
        recon_next: Optional[th.Tensor] = None,
        prior_mu: Optional[th.Tensor] = None,
        prior_logvar: Optional[th.Tensor] = None,
    ):
        self.w = w
        self.logvar = logvar
        self.recon = recon
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
# WynerLMUVAE — main model
# ═══════════════════════════════════════════════════════════════════════

class WynerLMUVAE(nn.Module):
    """
    Wyner VAE with multi-input LMU backbone.

    Packed state = (h, m_flat) where m_flat = m.reshape(B, p*d).
    packed_state_dim = hidden_size + n_channels * memory_size.

    Encode:
      - LMU step: (mu_t, h_{t-1}, m_{t-1}) → (h_t, m_t)
      - Posterior mean: pack(h_t, m_t)
      - Posterior logvar: conv1d over packed state

    Prior:
      - Conv1d + cross-attention over (h_{t-1}, m_{t-1})

    Decode:
      - LMU.recon_data(m_t, timestep) → Legendre reconstruction
      - Gated residual with h_t → MLP → (s, a, s')
    """

    def __init__(
        self,
        recon_dim: int,
        mu_dim: int,
        latent_dim: int,              # = hidden_size of LMU
        memory_size: int = 64,        # = d, Legendre coefficients per channel
        n_channels: int = 8,          # = p, number of input channels
        theta: float = 100.0,
        decode_hidden: int = 128,
        free_bits: float = 0.0,
        prior_conv_channels: int = 64,
        prior_n_attn_heads: int = 4,
        # Interface compat (unused)
        latent_tokens: int = 1,
        state_dim: int = 0,
        state_tokens: int = 0,
        pos_embed_dim: int = 16,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.memory_size = memory_size
        self.n_channels = n_channels
        self.mu_dim = mu_dim
        self.recon_dim = recon_dim
        self.free_bits = free_bits
        self.latent_tokens = latent_tokens
        self.state_dim = state_dim
        self.state_tokens = state_tokens
        self.pos_embed_dim = pos_embed_dim

        # ── LMU cell ─────────────────────────────────────────
        self.lmu = MultiInputLMUCell(
            input_size=mu_dim,
            hidden_size=latent_dim,
            memory_size=memory_size,
            n_channels=n_channels,
            theta=theta,
        )

        # Flat memory dim for packing
        self._flat_mem_dim = n_channels * memory_size
        self.packed_state_dim = latent_dim + self._flat_mem_dim

        # ── Posterior ─────────────────────────────────────────
        _post_pool_out = 8
        _post_conv_ch = 32
        self.fc_logvar = nn.Sequential(
            nn.Conv1d(1, _post_conv_ch, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(_post_pool_out),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(_post_conv_ch * _post_pool_out, latent_dim),
        )
        nn.init.zeros_(self.fc_logvar[-1].bias)

        # ── Prior ─────────────────────────────────────────────
        self.prior_net = LMUPriorNetwork(
            hidden_size=latent_dim,
            n_channels=n_channels,
            memory_size=memory_size,
            latent_dim=latent_dim,
            conv_channels=prior_conv_channels,
            n_attn_heads=prior_n_attn_heads,
        )

        # ── Decoder ───────────────────────────────────────────
        self.decoder = LegendreReconDecoder(
            hidden_size=latent_dim,
            memory_flat_dim=self._flat_mem_dim,
            recon_dim=recon_dim,
            decode_hidden=decode_hidden,
        )

    # ── State packing ─────────────────────────────────────────

    def pack_state(self, h: th.Tensor, m: th.Tensor) -> th.Tensor:
        """
        h: (B, hidden_size), m: (B, p, d) → (B, packed_state_dim)
        """
        return th.cat([h, m.reshape(h.size(0), -1)], dim=-1)

    def unpack_state(self, packed: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        (B, packed_state_dim) → h: (B, hidden_size), m: (B, p, d)
        """
        h = packed[..., :self.latent_dim]
        m_flat = packed[..., self.latent_dim:]
        m = m_flat.reshape(-1, self.n_channels, self.memory_size)
        return h, m


    # ── Encode ────────────────────────────────────────────────

    def encode(
        self,
        w: th.Tensor,
        mu: th.Tensor,
        skips: Optional[List[th.Tensor]] = None,
        timestep: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor]:
        w_flat = w.squeeze(1) if w.dim() == 3 else w
        h_prev, m_prev = self.unpack_state(w_flat)

        h_t, m_t = self.lmu(mu, h_prev, m_prev)

        z_mu = self.pack_state(h_t, m_t)                          # (B, packed_state_dim)
        z_logvar = self.fc_logvar(z_mu.unsqueeze(1))               # (B, latent_dim)

        return z_mu, z_logvar

    # ── Decode ────────────────────────────────────────────────

    def decode(
        self,
        z: th.Tensor,
        mu: th.Tensor,
        timestep: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """
        z: (B, packed_state_dim) — packed (h, m)
        timestep: (B,) — for Legendre reconstruction query point
        """
        h, m = self.unpack_state(z)

        if timestep is not None:
            u = self.lmu.recon_data(m, timestep)                   # (B, p, d)
        else:
            # Fallback: use raw memory (no polynomial evaluation)
            u = m

        u_flat = u.reshape(u.size(0), -1)                         # (B, p*d)
        return self.decoder(h, u_flat)

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

        # LMU encode
        z_mu, z_logvar = self.encode(w, mu, skips, timestep)

        # Sample (reparameterize only the latent slice; remainder of packed state is deterministic)
        std = th.exp(0.5 * z_logvar)
        z = z_mu.clone()
        z[:, :std.size(1)] = z_mu[:, :std.size(1)] + th.randn_like(std) * std

        # Decode current
        recon = self.decode(z, mu, timestep)

        # Decode next
        recon_next = None
        if mu_next is not None:
            ts_next = (timestep + 1) if timestep is not None else None
            recon_next = self.decode(z, mu_next, ts_next)

        return WynerLMUOutput(
            w=z,
            logvar=z_logvar,
            recon=recon,
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
        z_mu = output.w[..., :self.latent_dim]

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


# ═══════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    th.manual_seed(0)
    B = 4
    mu_dim = 32
    latent_dim = 64
    memory_size = 64
    n_channels = 8
    recon_dim = 2 * 128 + 32

    model = WynerLMUVAE(
        recon_dim=recon_dim,
        mu_dim=mu_dim,
        latent_dim=latent_dim,
        memory_size=memory_size,
        n_channels=n_channels,
        theta=100.0,
        decode_hidden=128,
    )

    print(f"packed_state_dim: {model.packed_state_dim}")
    print(f"  = hidden({latent_dim}) + channels({n_channels}) * memory({memory_size})")
    print(f"  = {latent_dim} + {n_channels * memory_size} = {model.packed_state_dim}")

    # Initial packed state: zeros
    w = th.zeros(B, 1, model.packed_state_dim)
    mu = th.randn(B, mu_dim)
    mu_next = th.randn(B, mu_dim)
    timestep = th.randint(0, 100, (B,))


    # Test forward (training path)
    out = model(w, mu, mu_next, timestep=timestep)
    loss = model.loss(out, recon_target=th.randn(B, recon_dim),
                      recon_next_target=th.randn(B, recon_dim))

    print(f"\nforward() outputs:")
    print(f"  w:          {out.w.shape}")
    print(f"  logvar:     {out.logvar.shape}")
    print(f"  recon:      {out.recon.shape}")
    print(f"  prior_mu:   {out.prior_mu.shape}")
    print(f"  kl_loss:    {loss.kl_loss.shape}")
    print(f"  recon_loss: {loss.recon_loss.shape}")

    # Test action features
    feat_ext = LMUActionFeatures(
        hidden_size=latent_dim,
        memory_size=memory_size,
        n_channels=n_channels,
        mu_dim=mu_dim,
        conv_channels=32,
    )
    h, m = model.unpack_state(w.squeeze(1))
    features = feat_ext(h, m, mu)
    print(f"\nAction features shape: {features.shape}")

    # Verify A, B are frozen
    assert not model.lmu.A.requires_grad
    assert not model.lmu.B.requires_grad
    assert not model.lmu.C.requires_grad
    print("\nAll checks passed.")