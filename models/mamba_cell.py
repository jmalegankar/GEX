"""
MambaCell: a single recurrent step of Mamba-2 (SSD) with a GRUCell-compatible
interface — takes (input, hidden) and returns (output, new_hidden).

Key change from previous version:
    forward() now returns (y, h_new) instead of just h_new.
    y: (B, hidden_dim)     — Mamba output, used to compute z_mu/z_logvar
    h_new: (B, flat_state_dim) — full SSM state, stored as memory in the buffer

Dependencies:
    https://github.com/state-spaces/mamba
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Tuple

try:
    from mamba_ssm import Mamba2
    _MAMBA_AVAILABLE = True
except ImportError:
    _MAMBA_AVAILABLE = False


class MambaCell(nn.Module):
    """
    Single-step Mamba-2 wrapper.

        y, h_new = MambaCell(input_dim, hidden_dim)(x, h)

    The Mamba2 hidden state has two parts that must both be preserved:
      conv_state: (B, conv_channels, d_conv)
      ssm_state:  (B, n_heads, d_head, d_state)

    Both are flattened into a single (B, flat_state_dim) vector so the caller
    sees a clean interface. flat_state_dim >> hidden_dim — keep d_state and
    d_conv small (16, 2) to control memory usage.

    Args:
        input_dim:  dimensionality of x
        hidden_dim: d_model for the Mamba2 block; also the dim of y
        d_state:    SSM state size (recommend 16 for RL, not 64)
        d_conv:     local conv width (recommend 2 for RL, not 4)
        expand:     inner expansion factor
        n_heads:    number of SSD heads
    """

    def __init__(
        self,
        input_dim:  int,
        hidden_dim: int,
        d_state:    int = 16,
        d_conv:     int = 2,
        expand:     int = 2,
        n_heads:    int = 1,
    ):
        super().__init__()
        assert _MAMBA_AVAILABLE, (
            "mamba-ssm is not installed. Run: pip install mamba-ssm causal-conv1d"
        )

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.d_state    = d_state
        self.d_conv     = d_conv
        self.n_heads    = n_heads
        self.d_head     = hidden_dim // n_heads

        self.input_proj = nn.Linear(input_dim, hidden_dim, bias=False)

        self.mamba = Mamba2(
            d_model  = hidden_dim,
            d_state  = d_state,
            d_conv   = d_conv,
            expand   = expand,
            headdim  = self.d_head,
        )

        # Derive state shapes from a live allocation — don't hardcode.
        _dummy_conv, _dummy_ssm = self.mamba.allocate_inference_cache(1, 1)
        self._conv_state_shape = _dummy_conv.shape[1:]   # (conv_channels, d_conv)
        self._ssm_state_shape  = _dummy_ssm.shape[1:]   # (n_heads, d_head, d_state)
        self._flat_conv_dim    = int(torch.tensor(self._conv_state_shape).prod().item())
        self._flat_ssm_dim     = int(torch.tensor(self._ssm_state_shape).prod().item())
        self._flat_state_dim   = self._flat_conv_dim + self._flat_ssm_dim

    # ── State packing ─────────────────────────────────────────────────────────

    def pack_state(self, conv_state: torch.Tensor, ssm_state: torch.Tensor) -> torch.Tensor:
        """(B, conv_channels, d_conv) + (B, n_heads, d_head, d_state) → (B, flat_state_dim)"""
        B = conv_state.size(0)
        return torch.cat([conv_state.reshape(B, -1), ssm_state.reshape(B, -1)], dim=-1)

    def unpack_state(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(B, flat_state_dim) → (B, conv_channels, d_conv), (B, n_heads, d_head, d_state)"""
        B = h.size(0)
        return (
            h[:, :self._flat_conv_dim].reshape(B, *self._conv_state_shape),
            h[:, self._flat_conv_dim:].reshape(B, *self._ssm_state_shape),
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, input_dim)
            h: (B, flat_state_dim)  — packed SSM state from previous step

        Returns:
            y:     (B, hidden_dim)      — Mamba output; use this to compute z
            h_new: (B, flat_state_dim)  — new SSM state; store this in the buffer
        """
        x_proj = self.input_proj(x).unsqueeze(1)          # (B, 1, hidden_dim)

        conv_state, ssm_state = self.unpack_state(h)
        conv_state = conv_state.to(dtype=x.dtype).contiguous()
        ssm_state  = ssm_state.to(dtype=x.dtype).contiguous()

        y, new_conv, new_ssm = self.mamba.step(x_proj, conv_state, ssm_state)
        y = y.squeeze(1)                                   # (B, hidden_dim)

        return y, self.pack_state(new_conv, new_ssm)

    @property
    def flat_state_dim(self) -> int:
        return self._flat_state_dim


class MambaSequence(nn.Module):
    """
    Parallel Mamba scan over a full sequence (training-time use only).

    Use this if you reconstruct episode sequences from the buffer for
    better gradient flow. Not currently wired into the training loop.

        y_seq = MambaSequence(cell)(x_seq)   # x_seq: (B, T, input_dim)

    Note: hidden state is not threaded through here — Mamba initialises
    from zeros. This is fine when sequences are reconstructed independently
    per rollout rather than being truly contiguous.
    """

    def __init__(self, cell: MambaCell):
        super().__init__()
        self.cell = cell

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq: (B, T, input_dim)
        Returns:
            y_seq: (B, T, hidden_dim)
        """
        x_proj = self.cell.input_proj(x_seq)   # (B, T, hidden_dim)
        return self.cell.mamba(x_proj)          # (B, T, hidden_dim) — parallel scan