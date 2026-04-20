"""
LMU Actor-Critic Policy — Gated Write variant.

Changes from baseline:
──────────────────────
1. LMUCell.forward now returns (h, m, r_intr).
   All three call sites updated:
     forward()          → returns r_intr as 7th value (Option A)
     evaluate_actions() → accumulates r_intrs per step, returns as 4th value
     predict_values()   → discards r_intr with _

2. Optimizer excludes W_pre from Adam (see __init__).
   W_pre uses its own Riemannian update (ortho_update) in the training loop.

3. evaluate_actions returns r_intrs at β=0 (Step 1) for diagnostic logging.
   The values are not zero — the computation runs every step — but they are
   NOT in the loss until Step 2.  This is intentional: Step 1 validation
   requires observing r_intr trends to confirm the gated write is behaving
   before any gradient from it is introduced.

Old lines are commented with  # [OLD]  and kept for diff/reversion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from gymnasium import spaces
from typing import Dict, Tuple

from lmu import LMUCell


class MinigridEncoder(nn.Module):
    """Unchanged from baseline."""

    N_OBJECTS    = 11
    N_COLORS     = 6
    N_STATES     = 3
    N_DIRECTIONS = 4

    def __init__(
        self,
        obs_space:   spaces.Dict,
        out_dim:     int = 64,
        obj_emb_dim: int = 8,
        col_emb_dim: int = 4,
        sta_emb_dim: int = 2,
        dir_emb_dim: int = 4,
    ):
        super().__init__()
        C, H, W = obs_space['image'].shape

        self.obj_emb = nn.Embedding(self.N_OBJECTS,    obj_emb_dim)
        self.col_emb = nn.Embedding(self.N_COLORS,     col_emb_dim)
        self.sta_emb = nn.Embedding(self.N_STATES,     sta_emb_dim)
        self.dir_emb = nn.Embedding(self.N_DIRECTIONS, dir_emb_dim)

        tile_dim = obj_emb_dim + col_emb_dim + sta_emb_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(tile_dim, 32, kernel_size=2), nn.ReLU(),
            nn.Conv2d(32,       64, kernel_size=2), nn.ReLU(),
            nn.Conv2d(64,       64, kernel_size=2), nn.ReLU(),
            nn.Flatten(),
        )
        cnn_out = 64 * (H - 3) * (W - 3)
        self.proj = nn.Sequential(
            nn.Linear(cnn_out + dir_emb_dim, out_dim),
            nn.ReLU(),
        )
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        img  = obs['image'].long()
        dir_ = obs['direction'].long().view(-1)
        obj  = self.obj_emb(img[:, 0])
        col  = self.col_emb(img[:, 1])
        sta  = self.sta_emb(img[:, 2])
        x    = torch.cat([obj, col, sta], dim=-1).permute(0, 3, 1, 2).contiguous()
        return self.proj(torch.cat([self.cnn(x), self.dir_emb(dir_)], dim=-1))


class LMUActorCriticPolicy(nn.Module):
    """
    MinigridEncoder → LMUCell (gated write) → actor + critic.

    Optimizer note:
        W_pre must be excluded from Adam — it is updated via Riemannian steps
        in the training loop (lmu_ppo.py train()).  The split is done here in
        __init__ so the optimizer is constructed correctly from the start.

        self.optimizer covers all params EXCEPT lmu_cell.W_pre.
        lmu_ppo.train() must call self.policy.lmu_cell.W_pre.ortho_update(lr)
        after every loss.backward() and before optimizer.step().
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
    ):
        super().__init__()
        assert isinstance(action_space, spaces.Discrete)

        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.encoder_dim = encoder_dim
        n_actions = action_space.n

        self.encoder  = MinigridEncoder(observation_space, encoder_dim)
        self.lmu_cell = LMUCell(encoder_dim, hidden_size, memory_size, theta)

        head_in = hidden_size + encoder_dim

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

        # [NEW] Exclude W_pre from Adam.
        # W_pre has its own Riemannian update (OrthoLayer.ortho_update).
        # If W_pre were included, Adam would apply stale momentum updates
        # after ortho_update zeros the grad, corrupting orthogonality.
        ortho_params  = set(self.lmu_cell.W_pre.parameters())
        main_params   = [p for p in self.parameters() if p not in ortho_params]
        # [OLD] self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-5)
        self.optimizer = torch.optim.Adam(main_params, lr=lr, eps=1e-5)

    # ── helpers (unchanged) ───────────────────────────────────────────────────

    def _critic_input(self, h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        m_pooled = m.mean(dim=1)   # (B, d, C) → (B, C)
        return torch.cat([h, m_pooled], dim=-1)

    # ── rollout (single step) ─────────────────────────────────────────────────

    def forward(
        self,
        obs:    Dict[str, torch.Tensor],
        h_prev: torch.Tensor,
        m_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        # [OLD] Returns: action, value, log_prob, h, m, logits        (6 values)
        Returns:         action, value, log_prob, h, m, logits, r_intr (7 values)

        r_intr : (B,) — Step 1: caller logs this, does NOT add to rewards.
        """
        x = self.encoder(obs)

        # [OLD] h, m    = self.lmu_cell(x, h_prev, m_prev)
        h, m, r_intr = self.lmu_cell(x, h_prev, m_prev)

        logits   = self.actor(self._critic_input(h, m))
        dist     = Categorical(logits=logits)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        value    = self.critic(self._critic_input(h, m)).squeeze(-1)

        # [OLD] return action, value, log_prob, h, m, logits
        return action, value, log_prob, h, m, logits, r_intr

    # ── PPO update (K-step unroll) ────────────────────────────────────────────

    def evaluate_actions(
        self,
        obs_seq:        Dict[str, torch.Tensor],
        lmu_h:          torch.Tensor,
        lmu_m:          torch.Tensor,
        episode_starts: torch.Tensor,
        actions_seq:    torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Unrolls K LMU steps with full gradient.

        # [OLD] Returns (flattened over B*K): values, log_probs, entropy
        Returns (flattened over B*K):          values, log_probs, entropy, r_intrs

        r_intrs : (B*K,)
          Step 1 (β=0): returned for logging only.  NOT in the PPO loss.
          Step 2: lmu_ppo.train() adds η * r_intrs.mean() to the loss.

        NOTE on r_intrs vs collect_rollouts r_intr:
          These are recomputed with gradient during the PPO update pass.
          The values will differ slightly from the no-grad rollout values
          because the policy weights have been updated between rollout and
          update. This is the same situation as log_probs vs old_log_prob —
          expected and correct.  Do not use evaluate_actions r_intrs for the
          reward shaping signal; use the rollout values for that.
        """
        B, K = episode_starts.shape
        h, m = lmu_h, lmu_m

        all_values, all_log_probs, all_entropy, all_r_intrs = [], [], [], []

        for k in range(K):
            reset = episode_starts[:, k:k+1]          # (B, 1)
            h = h * (1.0 - reset)                     # (B, n)
            m = m * (1.0 - reset.unsqueeze(-1))       # (B, d, C)

            obs_k = {key: obs_seq[key][:, k] for key in obs_seq}
            x     = self.encoder(obs_k)               # (B, C)

            # [OLD] h, m  = self.lmu_cell(x, h, m)
            h, m, r_intr = self.lmu_cell(x, h, m)    # (B,n), (B,d,C), (B,)

            head   = self._critic_input(h, m)
            logits = self.actor(head)
            dist   = Categorical(logits=logits)
            all_log_probs.append(dist.log_prob(actions_seq[:, k]))
            all_entropy.append(dist.entropy())
            all_values.append(self.critic(head).squeeze(-1))
            all_r_intrs.append(r_intr)                # [NEW]

        values    = torch.stack(all_values,    dim=1).reshape(B * K)
        log_probs = torch.stack(all_log_probs, dim=1).reshape(B * K)
        entropy   = torch.stack(all_entropy,   dim=1).reshape(B * K)
        r_intrs   = torch.stack(all_r_intrs,   dim=1).reshape(B * K)  # [NEW]

        # [OLD] return values, log_probs, entropy
        return values, log_probs, entropy, r_intrs

    # ── GAE bootstrap ─────────────────────────────────────────────────────────

    def predict_values(
        self,
        obs:   Dict[str, torch.Tensor],
        lmu_h: torch.Tensor,
        lmu_m: torch.Tensor,
    ) -> torch.Tensor:
        x = self.encoder(obs)
        # [OLD] h, m = self.lmu_cell(x, lmu_h, lmu_m)
        h, m, _ = self.lmu_cell(x, lmu_h, lmu_m)   # r_intr not needed here
        return self.critic(self._critic_input(h, m)).squeeze(-1)

    # ── state helpers (unchanged) ─────────────────────────────────────────────

    def initial_state(
        self, n_envs: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.lmu_cell.initial_state(n_envs, device)

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)