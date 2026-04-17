"""
LMU Actor-Critic Policy — TBPTT version.

The only method that changes meaningfully is evaluate_actions.
Previously it ran 1 LMU step per sample.  Now it unrolls K steps,
with gradient flowing back through the entire chunk.

Episode boundary handling:
    episode_starts[b, k] == 1 means obs[b, k] is the first obs of a new episode.
    Before processing step k, we zero h and m for any env where this is True.
    This is equivalent to h = h * (1 - reset), m = m * (1 - reset).
    Gradient does NOT flow through a reset (the multiplication kills it), so
    episodes are independent in the backward pass — correct TBPTT behaviour.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
from gymnasium import spaces
from typing import Dict, Tuple

from lmu import LMUCell


class MinigridEncoder(nn.Module):
    """
    Encoder for MiniGrid Dict obs (unchanged from previous version).
    image: (B, 3, H, W) categorical channels after VecTransposeImage
    direction: (B,) Discrete(4)
    Output: (B, out_dim)
    """

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
    MinigridEncoder → LMUCell → actor + critic.

    State convention (unchanged):
        h : (B, n)      stored BEFORE processing current obs
        m : (B, d, C)   stored BEFORE processing current obs
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

        # Both actor and critic receive cat([h, m_pooled]) directly.
        #
        # Why actor also needs m_pooled:
        #   The LMU's C_proj readout (m → y → W_m → h) collapses during
        #   training because C_proj and W_m are coupled — if C_proj is small,
        #   W_m sees noise and doesn't learn; Adam then shrinks C_proj further.
        #   In practice h ends up dominated by W_x(x) + W_h(h) (current obs
        #   + recurrence) with almost no memory signal. The actor acting on h
        #   alone is effectively memoryless.
        #
        #   Giving both heads direct access to m.mean(d) → (B, C) bypasses
        #   the C_proj bottleneck entirely. C_proj + W_m remain as an auxiliary
        #   pathway that can still learn, but the primary memory signal is
        #   direct. The actor can now condition on ball identity from step 1
        #   even before C_proj has learned anything useful.
        #
        # head_in = hidden_size + encoder_dim  (h concat m_pooled)
        head_in = hidden_size + encoder_dim

        # Actor: linear over head_in → n_actions.
        # Small orthogonal gain (0.01) keeps initial policy near-uniform.
        self.actor = nn.Linear(head_in, n_actions)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

        # Critic: two-layer MLP — nonlinearity lets it threshold on memory
        # content ("did we see the ball?") not just linearly combine features.
        self.critic = nn.Sequential(
            nn.Linear(head_in, head_in // 2),
            nn.Tanh(),
            nn.Linear(head_in // 2, 1),
        )
        nn.init.orthogonal_(self.critic[0].weight, gain=1.0)
        nn.init.zeros_(self.critic[0].bias)
        nn.init.orthogonal_(self.critic[2].weight, gain=1.0)
        nn.init.zeros_(self.critic[2].bias)

        # Re-initialise C_proj larger so the W_m pathway isn't dead from the
        # start. Previously ±1/√d ≈ ±0.177 shrank to ~0.04 during training.
        # Orthogonal init preserves gradient magnitude through the readout.
        with torch.no_grad():
            tmp = torch.empty(1, self.lmu_cell.memory_size)
            nn.init.orthogonal_(tmp)
            self.lmu_cell.C_proj.data.copy_(tmp.squeeze(0))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-5)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _critic_input(self, h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        Concatenate h and mean-pooled m for the critic.

        h : (B, n)
        m : (B, d, C)
        → (B, n + C)
        """
        m_pooled = m.mean(dim=1)          # (B, d, C) → (B, C), pool over Legendre dim
        return torch.cat([h, m_pooled], dim=-1)

    # ── rollout (single step, no grad needed) ────────────────────────────────

    def forward(
        self,
        obs:    Dict[str, torch.Tensor],
        h_prev: torch.Tensor,   # (B, n)
        m_prev: torch.Tensor,   # (B, d, C)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.encoder(obs)
        h, m     = self.lmu_cell(x, h_prev, m_prev)
        logits   = self.actor(self._critic_input(h, m))   # actor now sees memory too
        dist     = Categorical(logits=logits)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        value    = self.critic(self._critic_input(h, m)).squeeze(-1)
        return action, value, log_prob, h, m, logits

    # ── PPO update (K-step unroll, full gradient) ─────────────────────────────

    def evaluate_actions(
        self,
        obs_seq:        Dict[str, torch.Tensor],  # each (B, K, ...)
        lmu_h:          torch.Tensor,             # (B, n)       chunk-start state
        lmu_m:          torch.Tensor,             # (B, d, C)    chunk-start state
        episode_starts: torch.Tensor,             # (B, K)  float32
        actions_seq:    torch.Tensor,             # (B, K)  long
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Unrolls K LMU steps with full gradient.

        Episode boundary handling:
            Before step k, zero h and m wherever episode_starts[:, k] == 1.
            Multiplication by (1 - reset) is differentiable but kills gradient
            at the boundary — correct: no grad flows between episodes.

            reset_h : (B, 1)     broadcasts over hidden dim
            reset_m : (B, 1, 1)  broadcasts over (d, C)

        Returns (flattened over B*K):
            values   : (B*K,)
            log_probs: (B*K,)
            entropy  : (B*K,)
        """
        B, K = episode_starts.shape
        h, m = lmu_h, lmu_m

        all_values, all_log_probs, all_entropy = [], [], []

        for k in range(K):
            # Zero state at episode boundaries (no grad through reset)
            reset    = episode_starts[:, k:k+1]              # (B, 1)
            h = h * (1.0 - reset)                            # (B, n)
            m = m * (1.0 - reset.unsqueeze(-1))              # (B, d, C)

            obs_k = {key: obs_seq[key][:, k] for key in obs_seq}
            x     = self.encoder(obs_k)                      # (B, C)
            h, m  = self.lmu_cell(x, h, m)                  # (B,n), (B,d,C)

            head    = self._critic_input(h, m)             # (B, n+C) — shared input
            logits  = self.actor(head)
            dist    = Categorical(logits=logits)
            all_log_probs.append(dist.log_prob(actions_seq[:, k]))
            all_entropy.append(dist.entropy())
            all_values.append(self.critic(head).squeeze(-1))

        # (B, K) → (B*K,)
        values    = torch.stack(all_values,    dim=1).reshape(B * K)
        log_probs = torch.stack(all_log_probs, dim=1).reshape(B * K)
        entropy   = torch.stack(all_entropy,   dim=1).reshape(B * K)

        return values, log_probs, entropy

    # ── GAE bootstrap (single step, no grad needed) ──────────────────────────

    def predict_values(
        self,
        obs:   Dict[str, torch.Tensor],
        lmu_h: torch.Tensor,
        lmu_m: torch.Tensor,
    ) -> torch.Tensor:
        x = self.encoder(obs)
        h, m = self.lmu_cell(x, lmu_h, lmu_m)
        return self.critic(self._critic_input(h, m)).squeeze(-1)

    def initial_state(
        self, n_envs: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.lmu_cell.initial_state(n_envs, device)

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)