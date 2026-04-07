"""
MambaCell: a single recurrent step of Mamba-2 (SSD) that mirrors the
nn.GRUCell API — takes (input, hidden) and returns new_hidden.

During rollout collection: called step-by-step, exactly like GRUCell.
During training: can optionally use the parallel scan over the full
rollout sequence for better gradient flow (see MambaSequence below).

Dependencies:
    pip install mamba-ssm causal-conv1d
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional

try:
    from mamba_ssm import Mamba2
    _MAMBA_AVAILABLE = True
except ImportError:
    _MAMBA_AVAILABLE = False


class MambaCell(nn.Module):
    """
    Single-step Mamba wrapper with a GRUCell-compatible interface.

        h_new = MambaCell(input_dim, hidden_dim)(x, h)

    The Mamba hidden state is a (B, n_heads, d_head, d_state) tensor.
    We flatten/unflatten it transparently so the caller just sees
    a (B, hidden_dim) vector, same as GRUCell.

    Args:
        input_dim:   dimensionality of x  (= mu_dim + pos_embed_dim in Wyner)
        hidden_dim:  dimensionality of h  (= latent_dim in Wyner)
        d_state:     SSM state size (default 64, Mamba-2 paper recommendation)
        d_conv:      local conv width (default 4)
        expand:      inner expansion factor (default 2)
        n_heads:     number of SSD heads (default 1 keeps hidden_dim clean)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        n_heads: int = 1,
    ):
        super().__init__()
        assert _MAMBA_AVAILABLE, (
            "mamba-ssm is not installed. Run: pip install mamba-ssm causal-conv1d"
        )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Project input to hidden_dim so Mamba sees a uniform-width stream
        self.input_proj = nn.Linear(input_dim, hidden_dim, bias=False)

        # Mamba2 block — operates on (B, L, d_model) sequences
        # We always pass L=1 for single-step recurrent use
        self.mamba = Mamba2(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=hidden_dim // n_heads,
        )

        # The Mamba2 ssm_state shape: (B, n_heads, d_head, d_state)
        # We compute this once and cache it
        self.n_heads = n_heads
        self.d_head = hidden_dim // n_heads
        self.d_state = d_state
        self._ssm_state_shape = (n_heads, self.d_head, d_state)
        # flat size stored in "hidden" vector
        self._flat_state_dim = n_heads * self.d_head * d_state  # = hidden_dim * d_state

    # ------------------------------------------------------------------
    # Helpers to pack/unpack the SSM state into the flat (B, H) vector
    # that the rest of the codebase expects as "memory".
    # ------------------------------------------------------------------

    def pack_state(self, ssm_state: torch.Tensor) -> torch.Tensor:
        """(B, n_heads, d_head, d_state) → (B, flat_state_dim)"""
        return ssm_state.reshape(ssm_state.size(0), -1)

    def unpack_state(self, h: torch.Tensor) -> torch.Tensor:
        """(B, flat_state_dim) → (B, n_heads, d_head, d_state)"""
        B = h.size(0)
        return h.reshape(B, *self._ssm_state_shape)

    # ------------------------------------------------------------------
    # GRUCell-compatible forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim)
            h: (B, flat_state_dim)  — packed SSM state
        Returns:
            h_new: (B, flat_state_dim)
        """
        x_proj = self.input_proj(x)                    # (B, hidden_dim)
        x_seq = x_proj.unsqueeze(1)                    # (B, 1, hidden_dim)

        ssm_state = self.unpack_state(h).to(dtype=x.dtype)  # (B, n_heads, d_head, d_state)

        # Mamba2.step: single-step recurrent forward
        # Returns (y, new_ssm_state) where y: (B, 1, d_model)
        _, new_ssm_state = self.mamba.step(x_seq, ssm_state)

        return self.pack_state(new_ssm_state)           # (B, flat_state_dim)

    @property
    def flat_state_dim(self) -> int:
        return self._flat_state_dim


class MambaSequence(nn.Module):
    """
    Parallel (training-time) Mamba scan over a full sequence.

    Use this in train() to get proper gradient flow over the entire rollout
    instead of stepping cell-by-cell.

        y = MambaSequence(cell)(x_seq)   # x_seq: (B, T, input_dim)

    The hidden state is NOT threaded through here — Mamba initialises
    from zeros internally, which is fine since the rollout buffer stores
    flattened (B*T, ...) samples, not full sequences. Gradient flow
    through the parallel scan still beats BPTT-truncated GRU.
    """

    def __init__(self, cell: MambaCell):
        super().__init__()
        self.cell = cell

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq: (B, T, input_dim)
        Returns:
            y_seq: (B, T, hidden_dim)  — Mamba output at each step
        """
        x_proj = self.cell.input_proj(x_seq)           # (B, T, hidden_dim)
        y_seq = self.cell.mamba(x_proj)                # (B, T, hidden_dim) — parallel scan
        return y_seq