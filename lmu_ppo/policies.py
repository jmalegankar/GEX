"""
LMU Actor-Critic Policy — supports gated_lmu / vanilla_lmu / gru / lstm cells.

Cell dispatch (delegated to lmu_ppo.cell_wrappers.make_cell):
    'gated_lmu'   — current default. LMUCell with full innovations.
    'vanilla_lmu' — LMUCell with gate=none, residual=0, read_head=first_coef.
                    Mimics POPGym's published LMU baseline.
    'gru'         — GRUCellWrapper (POPGym baseline).
    'lstm'        — LSTMCellWrapper (POPGym baseline).

Encoder dispatch (delegated to lmu_ppo.popgym_encoders.make_encoder):
    Single-key Dict({obs: Discrete | MultiDiscrete | Box}) → POPGym* encoder.

Head sizing:
    Actor/critic input size from self.lmu_cell.head_input_size.
    Gated LMU (read_head='dynamic'):     hidden + encoder_dim
    Vanilla LMU (read_head='first_coef'): hidden
    GRU / LSTM:                           hidden

Optimizer exclusion:
    LMU cells expose a W_pre (OrthoLayer) updated via Riemannian steps in
    the train loop. Excluded from Adam here. GRU/LSTM have no W_pre, so all
    their params go through Adam.

LegS dispatch:
    Only fires when self.cell_type ∈ {gated_lmu, vanilla_lmu} AND
    self.measure == 'LegS'. For GRU/LSTM, t is never threaded.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
from gymnasium import spaces
from typing import Dict, Optional, Tuple

from .popgym_encoders import make_encoder
from .cell_wrappers import make_cell


class LMUActorCriticPolicy(nn.Module):
    """
    Encoder → memory cell → actor + critic.

    Cell type, encoder type, and head sizes are all dispatched at construction.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space:      spaces.Space,
        lr:                float = 3e-4,
        encoder_dim:       int = 64,
        hidden_size:       int = 64,
        memory_size:       int = 32,
        theta:             float = 100.0,
        measure:           str = 'LegT',
        gate_type:         str = 'softsign_sum',
        residual_scale:    float = 0.05,
        read_head:         str = 'dynamic',
        cell_type:         str = 'gated_lmu',
    ):
        super().__init__()
        assert isinstance(action_space, spaces.Discrete)

        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.encoder_dim = encoder_dim
        self.measure     = measure
        self.cell_type   = cell_type
        self.is_lmu      = cell_type in ('gated_lmu', 'vanilla_lmu')
        n_actions = action_space.n

        # Encoder dispatch — single-key Dict POPGym obs.
        self.encoder = make_encoder(observation_space, out_dim=encoder_dim)

        # Cell dispatch.
        self.lmu_cell = make_cell(
            cell_type,
            input_size=encoder_dim,
            hidden_size=hidden_size,
            memory_size=memory_size,
            theta=theta,
            measure=measure,
            gate_type=gate_type,
            residual_scale=residual_scale,
            read_head=read_head,
        )

        # Actor / critic heads sized from cell.head_input_size.
        head_in = self.lmu_cell.head_input_size

        self.actor = nn.Linear(head_in, n_actions)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

        self.critic = nn.Sequential(
            nn.Linear(head_in, head_in // 2),
            nn.Tanh(),
            nn.Linear(head_in // 2, 1),
        )
        nn.init.orthogonal_(self.critic[0].weight, gain=1.0)
        nn.init.zeros_(self.critic[0].bias)
        nn.init.orthogonal_(self.critic[2].weight, gain=1.0)
        nn.init.zeros_(self.critic[2].bias)

        # Optimizer: exclude W_pre (Riemannian-updated externally) only if
        # the cell has it (LMU cells do, GRU/LSTM don't).
        if hasattr(self.lmu_cell, 'W_pre'):
            ortho_params  = set(self.lmu_cell.W_pre.parameters())
            main_params   = [p for p in self.parameters() if p not in ortho_params]
        else:
            main_params = list(self.parameters())
        self.optimizer = torch.optim.Adam(main_params, lr=lr, eps=1e-5)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _critic_input(self, h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Read the actor/critic head input from the current cell state."""
        return self.lmu_cell.read_state(h, m)

    # ── rollout (single step) ─────────────────────────────────────────────────

    def forward(
        self,
        obs:    Dict[str, torch.Tensor],
        h_prev: torch.Tensor,
        m_prev: torch.Tensor,
        t:      Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns: action, value, log_prob, h, m, logits, r_intr, gate, innov, u_x"""
        x = self.encoder(obs)

        if self.measure == 'LegS' and self.is_lmu:
            h, m, r_intr, gate, innov, u_x = self.lmu_cell(
                x, h_prev, m_prev, t.float()
            )
        else:
            h, m, r_intr, gate, innov, u_x = self.lmu_cell(x, h_prev, m_prev)

        logits   = self.actor(self._critic_input(h, m))
        dist     = Categorical(logits=logits)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        value    = self.critic(self._critic_input(h, m)).squeeze(-1)

        return action, value, log_prob, h, m, logits, r_intr, gate, innov, u_x

    # ── PPO update (K-step unroll) ────────────────────────────────────────────

    def evaluate_actions(
        self,
        obs_seq:        Dict[str, torch.Tensor],
        lmu_h:          torch.Tensor,
        lmu_m:          torch.Tensor,
        episode_starts: torch.Tensor,
        actions_seq:    torch.Tensor,
        lmu_t:          Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unrolls K steps with full gradient. Returns flattened (B*K) tensors."""
        B, K = episode_starts.shape
        h, m = lmu_h, lmu_m

        all_values, all_log_probs, all_entropy, all_r_intrs = [], [], [], []

        uses_legs = (self.measure == 'LegS' and self.is_lmu)

        for k in range(K):
            reset = episode_starts[:, k:k+1]
            h = h * (1.0 - reset)
            m = m * (1.0 - reset.unsqueeze(-1))

            obs_k = {key: obs_seq[key][:, k] for key in obs_seq}
            x     = self.encoder(obs_k)

            if uses_legs:
                t_k = lmu_t[:, k].float()
                h, m, r_intr, _, _, _ = self.lmu_cell(x, h, m, t_k)
            else:
                h, m, r_intr, _, _, _ = self.lmu_cell(x, h, m)

            head   = self._critic_input(h, m)
            logits = self.actor(head)
            dist   = Categorical(logits=logits)
            all_log_probs.append(dist.log_prob(actions_seq[:, k]))
            all_entropy.append(dist.entropy())
            all_values.append(self.critic(head).squeeze(-1))
            all_r_intrs.append(r_intr)

        values    = torch.stack(all_values,    dim=1).reshape(B * K)
        log_probs = torch.stack(all_log_probs, dim=1).reshape(B * K)
        entropy   = torch.stack(all_entropy,   dim=1).reshape(B * K)
        r_intrs   = torch.stack(all_r_intrs,   dim=1).reshape(B * K)

        return values, log_probs, entropy, r_intrs

    # ── GAE bootstrap ─────────────────────────────────────────────────────────

    def predict_values(
        self,
        obs:   Dict[str, torch.Tensor],
        lmu_h: torch.Tensor,
        lmu_m: torch.Tensor,
        t:     Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.encoder(obs)
        if self.measure == 'LegS' and self.is_lmu:
            h, m, _, _, _, _ = self.lmu_cell(x, lmu_h, lmu_m, t.float())
        else:
            h, m, _, _, _, _ = self.lmu_cell(x, lmu_h, lmu_m)
        return self.critic(self._critic_input(h, m)).squeeze(-1)

    # ── state helpers ─────────────────────────────────────────────────────────

    def initial_state(
        self, n_envs: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.lmu_cell.initial_state(n_envs, device)

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)

