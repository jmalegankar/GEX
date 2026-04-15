"""
LMU Actor-Critic Policy for MiniGrid.

State shapes (multichannel LMU):
    h : (B, hidden_size)              — flat hidden, fed directly to actor/critic
    m : (B, memory_size, encoder_dim) — d Legendre coeffs × C channels

The actor/critic heads are unchanged — they still read from h (B, n).
The only architectural difference is that m is now 3D and initial_state
returns a 3D tensor for m.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
from gymnasium import spaces
from typing import Tuple

from lmu import LMUCell


class MinigridEncoder(nn.Module):
    """
    Encoder for MiniGrid Dict observation space.

    'image'    : (C=3, H, W) after VecTransposeImage — three categorical channels:
                   ch0 = OBJECT_IDX (0–10)
                   ch1 = COLOR_IDX  (0–5)
                   ch2 = STATE      (0–2)
    'direction': Discrete(4)

    Output: (B, out_dim)  — fed as x_t to LMUCell each step.
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

    def forward(self, obs: dict) -> torch.Tensor:
        img = obs['image'].long()                    # (B, 3, H, W)
        dir_ = obs['direction'].long().view(-1)      # (B,)

        obj = self.obj_emb(img[:, 0])               # (B, H, W, obj_emb_dim)
        col = self.col_emb(img[:, 1])
        sta = self.sta_emb(img[:, 2])

        x = torch.cat([obj, col, sta], dim=-1)       # (B, H, W, tile_dim)
        x = x.permute(0, 3, 1, 2).contiguous()      # (B, tile_dim, H, W)
        x = self.cnn(x)                              # (B, cnn_out)

        d = self.dir_emb(dir_)                       # (B, dir_emb_dim)
        return self.proj(torch.cat([x, d], dim=-1))  # (B, out_dim=C)


class LMUActorCriticPolicy(nn.Module):
    """
    MinigridEncoder → LMUCell → actor head + critic head.

    LMU state convention:
        h : (B, n)      — hidden state BEFORE processing current obs
        m : (B, d, C)   — memory BEFORE processing current obs

    Storing pre-step states allows re-running the LMU with gradient during
    the PPO update (exact same computation, gradient flows through the cell).

    Recommended hyperparameters:
        encoder_dim (C) : 64
        hidden_size (n) : 64–128
        memory_size (d) : 32–64
        theta           : 100 for MemoryS7, 200 for MemoryS13
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space:      spaces.Space,
        lr:                float = 3e-4,
        encoder_dim:       int = 64,    # C — LMU channel dimension
        hidden_size:       int = 64,    # n
        memory_size:       int = 32,    # d
        theta:             float = 100.0,
    ):
        super().__init__()
        assert isinstance(action_space, spaces.Discrete)

        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.encoder_dim = encoder_dim  # needed by buffer for m shape
        n_actions = action_space.n

        self.encoder  = MinigridEncoder(observation_space, encoder_dim)
        self.lmu_cell = LMUCell(encoder_dim, hidden_size, memory_size, theta)
        #                        C             n            d

        # Actor and critic heads read from h: (B, n) — unchanged from before.
        # The multichannel LMU doesn't change what goes into these heads.
        self.actor  = nn.Linear(hidden_size, n_actions)
        self.critic = nn.Linear(hidden_size, 1)
        nn.init.orthogonal_(self.actor.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.actor.bias)
        nn.init.zeros_(self.critic.bias)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-5)

    def forward(
        self,
        obs:    dict,
        h_prev: torch.Tensor,   # (B, n)
        m_prev: torch.Tensor,   # (B, d, C)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Used during rollout collection (no gradient needed).

        Returns:
            action   : (B,)
            value    : (B,)
            log_prob : (B,)
            h_new    : (B, n)
            m_new    : (B, d, C)
            logits   : (B, n_actions)  — kept for optional KL logging
        """
        x = self.encoder(obs)                       # (B, C)
        h, m = self.lmu_cell(x, h_prev, m_prev)     # (B, n),  (B, d, C)

        logits   = self.actor(h)                    # (B, n_actions)
        dist     = Categorical(logits=logits)
        action   = dist.sample()                    # (B,)
        log_prob = dist.log_prob(action)            # (B,)
        value    = self.critic(h).squeeze(-1)       # (B,)

        return action, value, log_prob, h, m, logits

    def evaluate_actions(
        self,
        obs:     dict,
        lmu_h:   torch.Tensor,   # (B, n)    stored h_{t-1} from buffer
        lmu_m:   torch.Tensor,   # (B, d, C) stored m_{t-1} from buffer
        actions: torch.Tensor,   # (B,) long
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-runs the LMU step with gradient for the PPO update.
        The stored (h, m) are the states BEFORE obs_t, so this exactly
        replicates the rollout computation.

        Returns: value (B,), log_prob (B,), entropy (B,)
        """
        x = self.encoder(obs)                        # (B, C)
        h, _ = self.lmu_cell(x, lmu_h, lmu_m)       # (B, n)  — m_new unused here

        logits   = self.actor(h)                     # (B, n_actions)
        dist     = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)            # (B,)
        entropy  = dist.entropy()                    # (B,)
        value    = self.critic(h).squeeze(-1)        # (B,)

        return value, log_prob, entropy

    def predict_values(
        self,
        obs:   dict,
        lmu_h: torch.Tensor,   # (B, n)
        lmu_m: torch.Tensor,   # (B, d, C)
    ) -> torch.Tensor:
        """GAE bootstrap at end of rollout."""
        x = self.encoder(obs)
        h, _ = self.lmu_cell(x, lmu_h, lmu_m)
        return self.critic(h).squeeze(-1)

    def initial_state(
        self, n_envs: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns zeroed (h, m).
            h : (n_envs, hidden_size)
            m : (n_envs, memory_size, encoder_dim)   ← 3D now
        """
        return self.lmu_cell.initial_state(n_envs, device)

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)