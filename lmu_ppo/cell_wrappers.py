"""
Memory cell wrappers and dispatch factory.

The policy reads each cell through a uniform interface:
    forward(x, h_prev, m_prev, t=None) → (h, m, r_intr, gate, innov, u_x)
    read_state(h, m)                   → (B, head_input_size) tensor
    head_input_size                    → int  (size of read_state output)
    initial_state(n, device)           → (h, m)

Implementations live in:
    LMUCell         (lmu_t.py)   — current LegT gated cell
    LegSCell        (lmu_s.py)   — LegS variant
    GRUCellWrapper  (here)       — POPGym baseline
    LSTMCellWrapper (here)       — POPGym baseline

The vanilla-LMU ablation is configured LMUCell:
    gate_type='none', residual_scale=0.0, read_head='first_coef'

The `read_head='first_coef'` arg requires the next-batch update to lmu_t.py
and lmu_s.py. Until that lands, calling make_cell('vanilla_lmu', ...) will
raise TypeError on the unknown read_head kwarg — gated_lmu / gru / lstm work
immediately.

LSTM state packing:
    LSTM has two states (h, c). To avoid touching LMURolloutBuffer (which
    only stores h and m), we pack the cell state c into the first hidden_size
    slots of m's flattened view. Requires memory_size * input_size >= hidden_size,
    asserted at init. The packing uses torch.cat (functional, fully
    differentiable) rather than in-place setitem — see _pack_c.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# GRU
# ─────────────────────────────────────────────────────────────────────────────

class GRUCellWrapper(nn.Module):
    """
    Wraps nn.GRUCell to match the LMUCell interface.

    GRU has only one recurrent state h. The m tensor is allocated to keep
    buffer shapes uniform, but it stays at zero — no information is stored
    or read from it. Diagnostic returns (gate, innov, u_x, r_intr) are zeros.

    head_input_size = hidden_size. The actor/critic heads see only h.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        memory_size: int,        # only used to size the dummy m buffer
        **kwargs,                # absorb LMU-specific args (theta, gate_type, ...)
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size

        self.cell = nn.GRUCell(input_size, hidden_size)
        nn.init.orthogonal_(self.cell.weight_hh)
        nn.init.xavier_normal_(self.cell.weight_ih)
        if self.cell.bias_hh is not None:
            nn.init.zeros_(self.cell.bias_hh)
        if self.cell.bias_ih is not None:
            nn.init.zeros_(self.cell.bias_ih)

    @property
    def head_input_size(self) -> int:
        return self.hidden_size

    def read_state(self, h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        # Memory is zero-pad — only h carries information.
        return h

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        m_prev: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        h_new = self.cell(x, h_prev)
        m_new = m_prev   # pass-through (zeros)

        B = x.shape[0]
        zeros_C = torch.zeros_like(x)
        r_intr = torch.zeros(B, device=x.device, dtype=x.dtype)
        return h_new, m_new, r_intr, zeros_C, zeros_C, zeros_C

    def initial_state(
        self, n: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(n, self.hidden_size, device=device)
        m = torch.zeros(n, self.memory_size, self.input_size, device=device)
        return h, m


# ─────────────────────────────────────────────────────────────────────────────
# LSTM
# ─────────────────────────────────────────────────────────────────────────────

class LSTMCellWrapper(nn.Module):
    """
    Wraps nn.LSTMCell to match the LMUCell interface.

    LSTM has two states: h and c (cell state). We pack c into the first
    hidden_size flat slots of m via torch.cat with zero padding:
        m.reshape(B, -1) = cat([c, zeros(flat_dim - hidden_size)], dim=-1)

    This avoids touching LMURolloutBuffer's storage layout. Requires
    memory_size * input_size >= hidden_size — asserted at init. With our
    typical config (memory_size=32, input_size=64, hidden_size=128) this
    is 2048 >= 128, comfortably satisfied.

    Why torch.cat instead of in-place slice assignment:
        torch.zeros_like does not propagate requires_grad. An in-place
        setitem like `m_new_flat[:, :h] = c_new` on a non-grad leaf is
        ambiguous w.r.t. autograd graph construction — it may or may not
        connect c_new's gradient back through to LSTM weights depending on
        PyTorch internals. torch.cat is unambiguously differentiable. Cost:
        one extra zero allocation per forward, negligible at our scale.

    head_input_size = hidden_size. Cell state stays internal — actor/critic
    heads see only h, matching standard LSTM-policy practice.

    Forget-gate bias is initialized to 1.0 (Jozefowicz 2015 — encourages
    long-range memory at the start of training). PyTorch's default is 0.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        memory_size: int,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size

        flat_dim = memory_size * input_size
        assert flat_dim >= hidden_size, (
            f"LSTM cell state needs memory_size*input_size ({flat_dim}) "
            f">= hidden_size ({hidden_size}). Increase memory_size or input_size."
        )
        self._flat_dim = flat_dim
        self._pad_size = flat_dim - hidden_size

        self.cell = nn.LSTMCell(input_size, hidden_size)
        nn.init.orthogonal_(self.cell.weight_hh)
        nn.init.xavier_normal_(self.cell.weight_ih)
        if self.cell.bias_hh is not None:
            nn.init.zeros_(self.cell.bias_hh)
        if self.cell.bias_ih is not None:
            nn.init.zeros_(self.cell.bias_ih)
            # Forget-gate bias = 1.0. nn.LSTMCell.bias_ih has shape (4*hidden,)
            # ordered [input, forget, cell, output]; forget slice is [hidden:2*hidden].
            with torch.no_grad():
                self.cell.bias_ih[hidden_size:2 * hidden_size].fill_(1.0)

    @property
    def head_input_size(self) -> int:
        return self.hidden_size

    def read_state(self, h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        # Cell state is internal to the LSTM; heads see only h.
        return h

    def _unpack_c(self, m: torch.Tensor) -> torch.Tensor:
        """Read LSTM cell state from m's first hidden_size flat slots."""
        B = m.shape[0]
        return m.reshape(B, -1)[:, :self.hidden_size].contiguous()

    def _pack_c(self, c_new: torch.Tensor, m_template: torch.Tensor) -> torch.Tensor:
        """
        Pack LSTM cell state into a fresh m tensor via torch.cat.
        Differentiable through c_new — see class docstring.
        """
        B = c_new.shape[0]
        if self._pad_size > 0:
            padding = torch.zeros(
                B, self._pad_size, device=c_new.device, dtype=c_new.dtype
            )
            flat = torch.cat([c_new, padding], dim=-1)
        else:
            flat = c_new
        return flat.reshape(B, m_template.shape[1], m_template.shape[2])

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        m_prev: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        c_prev = self._unpack_c(m_prev)
        h_new, c_new = self.cell(x, (h_prev, c_prev))
        m_new = self._pack_c(c_new, m_prev)

        B = x.shape[0]
        zeros_C = torch.zeros_like(x)
        r_intr = torch.zeros(B, device=x.device, dtype=x.dtype)
        return h_new, m_new, r_intr, zeros_C, zeros_C, zeros_C

    def initial_state(
        self, n: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(n, self.hidden_size, device=device)
        m = torch.zeros(n, self.memory_size, self.input_size, device=device)
        return h, m


# ─────────────────────────────────────────────────────────────────────────────
# Cell factory
# ─────────────────────────────────────────────────────────────────────────────

CELL_TYPES = ('gated_lmu', 'vanilla_lmu', 'gru', 'lstm')


def make_cell(
    cell_type: str,
    *,
    input_size: int,
    hidden_size: int,
    memory_size: int,
    theta: float = 100.0,
    measure: str = 'LegT',
    gate_type: str = 'softsign_sum',
    residual_scale: float = 0.05,
    read_head: str = 'dynamic',
) -> nn.Module:
    """
    Construct a memory cell of the requested type.

    cell_type:
        'gated_lmu'   — full gated LMU with W_pre, dynamic W_query readout.
                        Uses gate_type/residual_scale/read_head as passed.
        'vanilla_lmu' — POPGym-style LMU. gate_type/residual_scale/read_head
                        are FORCED to ('none', 0.0, 'first_coef') regardless
                        of what's passed. theta and measure are honored.
        'gru'         — nn.GRUCell, h-only state, m as zero-pad.
        'lstm'        — nn.LSTMCell with forget-gate bias=1, c packed into m.

    measure='LegS' is honored only for {gated_lmu, vanilla_lmu}; ignored for
    GRU/LSTM (they have no theta or LegT/LegS distinction).
    """
    if cell_type not in CELL_TYPES:
        raise ValueError(
            f"cell_type must be one of {CELL_TYPES}; got {cell_type!r}"
        )

    if cell_type == 'gru':
        return GRUCellWrapper(
            input_size=input_size,
            hidden_size=hidden_size,
            memory_size=memory_size,
        )

    if cell_type == 'lstm':
        return LSTMCellWrapper(
            input_size=input_size,
            hidden_size=hidden_size,
            memory_size=memory_size,
        )

    # LMU variants — deferred import to avoid loading scipy etc. at module load.
    from lmu_t import LMUCell
    from lmu_s import LegSCell

    if cell_type == 'vanilla_lmu':
        gate_type_eff = 'none'
        residual_eff = 0.0
        read_head_eff = 'first_coef'
    else:  # gated_lmu
        gate_type_eff = gate_type
        residual_eff = residual_scale
        read_head_eff = read_head

    if measure == 'LegS':
        return LegSCell(
            input_size=input_size,
            hidden_size=hidden_size,
            memory_size=memory_size,
            gate_type=gate_type_eff,
            residual_scale=residual_eff,
            read_head=read_head_eff,
        )

    return LMUCell(
        input_size=input_size,
        hidden_size=hidden_size,
        memory_size=memory_size,
        theta=theta,
        gate_type=gate_type_eff,
        residual_scale=residual_eff,
        read_head=read_head_eff,
    )

