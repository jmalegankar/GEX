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

    The Mamba2 hidden state consists of TWO parts that must both be
    preserved across steps:
      - conv_state: (B, conv_channels, d_conv)  — causal-conv buffer
      - ssm_state:  (B, n_heads, d_head, d_state) — SSM recurrent state

    We flatten both into a single (B, flat_state_dim) vector so the caller
    just sees a GRUCell-like interface.

    Args:
        input_dim:   dimensionality of x  (= mu_dim + pos_embed_dim in Wyner)
        hidden_dim:  d_model for the Mamba2 block
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
        self.d_state = d_state
        self.d_conv = d_conv
        self.n_heads = n_heads
        self.d_head = hidden_dim // n_heads

        # Project input to hidden_dim so Mamba sees a uniform-width stream
        self.input_proj = nn.Linear(input_dim, hidden_dim, bias=False)

        # Mamba2 block — operates on (B, L, d_model) sequences
        # We always pass L=1 for single-step recurrent use
        self.mamba = Mamba2(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=self.d_head,
        )

        # Derive conv_state shape from the instantiated module.
        # conv1d output channels = d_ssm + 2*ngroups*d_state (set by Mamba2 internally).
        # allocate_inference_cache returns conv_state of shape (B, conv_channels, d_conv).
        _dummy_conv, _dummy_ssm = self.mamba.allocate_inference_cache(1, 1)
        self._conv_state_shape = _dummy_conv.shape[1:]   # (conv_channels, d_conv)
        self._ssm_state_shape  = _dummy_ssm.shape[1:]   # (n_heads, d_head, d_state)
        self._flat_conv_dim = int(torch.tensor(self._conv_state_shape).prod().item())
        self._flat_ssm_dim  = int(torch.tensor(self._ssm_state_shape).prod().item())
        self._flat_state_dim = self._flat_conv_dim + self._flat_ssm_dim

    # ------------------------------------------------------------------
    # Helpers to pack/unpack the full hidden state into/from a flat vector
    # ------------------------------------------------------------------

    def pack_state(self, conv_state: torch.Tensor, ssm_state: torch.Tensor) -> torch.Tensor:
        """
        (B, conv_channels, d_conv) + (B, n_heads, d_head, d_state) → (B, flat_state_dim)
        """
        B = conv_state.size(0)
        return torch.cat([conv_state.reshape(B, -1), ssm_state.reshape(B, -1)], dim=-1)

    def unpack_state(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        (B, flat_state_dim) → (B, conv_channels, d_conv), (B, n_heads, d_head, d_state)
        """
        B = h.size(0)
        conv_flat = h[:, :self._flat_conv_dim]
        ssm_flat  = h[:, self._flat_conv_dim:]
        conv_state = conv_flat.reshape(B, *self._conv_state_shape)
        ssm_state  = ssm_flat.reshape(B, *self._ssm_state_shape)
        return conv_state, ssm_state

    # ------------------------------------------------------------------
    # GRUCell-compatible forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim)
            h: (B, flat_state_dim)  — packed (conv_state ++ ssm_state)
        Returns:
            h_new: (B, flat_state_dim)
        """
        x_proj = self.input_proj(x)                        # (B, hidden_dim)
        x_seq  = x_proj.unsqueeze(1)                       # (B, 1, hidden_dim)

        conv_state, ssm_state = self.unpack_state(h)
        conv_state = conv_state.to(dtype=x.dtype).contiguous()
        ssm_state  = ssm_state.to(dtype=x.dtype).contiguous()

        # Mamba2.step: (B,1,d_model), conv_state, ssm_state → (y, new_conv, new_ssm)
        _, new_conv_state, new_ssm_state = self.mamba.step(x_seq, conv_state, ssm_state)

        return self.pack_state(new_conv_state, new_ssm_state)  # (B, flat_state_dim)

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
