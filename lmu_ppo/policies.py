"""
LMU Actor-Critic Policy for MiniGrid.

Deliberately not subclassing SB3's ActorCriticPolicy — that class assumes
a specific (obs → features → actor/critic) pipeline that doesn't compose
cleanly with recurrent state. We expose the interface LMUPPO needs:
  - forward()         used during rollout collection
  - evaluate_actions()used during PPO update
  - predict_values()  used for GAE bootstrap
"""

import torch as th
import torch.nn as nn
from torch.distributions import Categorical
from gymnasium import spaces
from typing import Tuple

from lmu import LMUCell  # your lmu.py


# ---------------------------------------------------------------------------
# Observation encoder
# ---------------------------------------------------------------------------

class MinigridEncoder(nn.Module):
    """
    Encoder for MiniGrid's Dict observation space:
        'image'    : Box(0, 255, (H, W, 3), uint8)
                     Despite the Box declaration, values are categorical integers:
                       channel 0 — OBJECT_IDX (0–10, 11 types)
                       channel 1 — COLOR_IDX  (0–5,  6 colors)
                       channel 2 — STATE      (0–2,  door: open/closed/locked)
        'direction': Discrete(4)  — agent facing (0=right,1=down,2=left,3=up)
        'mission'  : ignored

    Pipeline:
        image  → embed each channel → (B, H, W, emb_dim)
                 → CNN → (B, cnn_out)
        direction → embedding → (B, dir_emb_dim)
        concat + linear → (B, out_dim)
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
        # After SB3's VecTransposeImage the image shape is (C, H, W) = (3, 7, 7)
        # C=3 are the three categorical channels, not RGB
        C, H, W = obs_space["image"].shape

        # Image channel embeddings
        self.obj_emb = nn.Embedding(self.N_OBJECTS,    obj_emb_dim)
        self.col_emb = nn.Embedding(self.N_COLORS,     col_emb_dim)
        self.sta_emb = nn.Embedding(self.N_STATES,     sta_emb_dim)
        self.dir_emb = nn.Embedding(self.N_DIRECTIONS, dir_emb_dim)

        tile_dim = obj_emb_dim + col_emb_dim + sta_emb_dim

        # Spatial CNN: 3 × (2×2 conv) → H,W: 7→6→5→4 for default 7×7
        self.cnn = nn.Sequential(
            nn.Conv2d(tile_dim, 32, kernel_size=2), nn.ReLU(),
            nn.Conv2d(32,       64, kernel_size=2), nn.ReLU(),
            nn.Conv2d(64,       64, kernel_size=2), nn.ReLU(),
            nn.Flatten(),
        )
        cnn_out_dim = 64 * (H - 3) * (W - 3)

        self.proj = nn.Sequential(
            nn.Linear(cnn_out_dim + dir_emb_dim, out_dim),
            nn.ReLU(),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs: dict) -> th.Tensor:
        # After VecTransposeImage: image is (B, C, H, W) = (B, 3, H, W)
        # Channel 0: OBJECT_IDX, channel 1: COLOR_IDX, channel 2: STATE
        img = obs["image"].long()            # (B, 3, H, W)
        direction = obs["direction"].long().view(-1)  # (B,)

        obj = self.obj_emb(img[:, 0])       # (B, H, W, obj_emb_dim)
        col = self.col_emb(img[:, 1])       # (B, H, W, col_emb_dim)
        sta = self.sta_emb(img[:, 2])       # (B, H, W, sta_emb_dim)

        x = th.cat([obj, col, sta], dim=-1)         # (B, H, W, tile_dim)
        x = x.permute(0, 3, 1, 2).contiguous()     # (B, tile_dim, H, W)
        x = self.cnn(x)                             # (B, cnn_out_dim)

        d = self.dir_emb(direction)                 # (B, dir_emb_dim)
        return self.proj(th.cat([x, d], dim=-1))    # (B, out_dim)


# ---------------------------------------------------------------------------
# LMU Actor-Critic Policy
# ---------------------------------------------------------------------------

class LMUActorCriticPolicy(nn.Module):
    """
    Encoder → LMUCell → actor head + critic head.

    State convention: (h, m) are the LMU states BEFORE processing the
    current observation, so that forward() can re-run the LMU step with
    gradient during the PPO update (matching what's stored in the buffer).
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space:      spaces.Space,
        lr:                float = 3e-4,
        encoder_dim:       int = 64,
        hidden_size:       int = 64,   # LMU n  — nonlinear hidden units
        memory_size:       int = 32,   # LMU d  — Legendre coefficients
        theta:             float = 50.0,  # set to expected episode memory horizon
    ):
        super().__init__()
        assert isinstance(action_space, spaces.Discrete), \
            "LMUActorCriticPolicy currently supports Discrete action spaces only."

        self.hidden_size = hidden_size
        self.memory_size = memory_size
        n_actions = action_space.n

        self.encoder  = MinigridEncoder(observation_space, encoder_dim)
        self.lmu_cell = LMUCell(encoder_dim, hidden_size, memory_size, theta)

        # Actor and critic heads (orthogonal init, small gain for actor)
        self.actor  = nn.Linear(hidden_size, n_actions)
        self.critic = nn.Linear(hidden_size, 1)
        nn.init.orthogonal_(self.actor.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.actor.bias)
        nn.init.zeros_(self.critic.bias)

        self.optimizer = th.optim.Adam(self.parameters(), lr=lr, eps=1e-5)

    # ------------------------------------------------------------------
    # Core forward (used during rollout collection — no gradient needed)
    # ------------------------------------------------------------------

    def forward(
        self,
        obs: th.Tensor,
        h_prev: th.Tensor,
        m_prev: th.Tensor,
    ):
        x = self.encoder(obs)
        h, m, u = self.lmu_cell(x, h_prev, m_prev)

        logits = self.actor(h)
        dist   = Categorical(logits=logits)
        action = dist.sample()

        value    = self.critic(h).squeeze(-1)
        log_prob = dist.log_prob(action)

        return action, value, log_prob, h, m, u, logits

    # ------------------------------------------------------------------
    # Evaluate stored actions (used during PPO update — needs gradient)
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        obs:     th.Tensor,   # (B, H, W, C)
        lmu_h:   th.Tensor,   # (B, hidden_size)  stored h_{t-1}
        lmu_m:   th.Tensor,   # (B, memory_size)  stored m_{t-1}
        actions: th.Tensor,   # (B,) long
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Returns: value (B,), log_prob (B,), entropy (B,)
        """
        x = self.encoder(obs)
        h, _, _ = self.lmu_cell(x, lmu_h, lmu_m)

        logits   = self.actor(h)
        dist     = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy  = dist.entropy()
        value    = self.critic(h).squeeze(-1)

        return value, log_prob, entropy

    # ------------------------------------------------------------------
    # Value-only (used for GAE bootstrap at end of rollout)
    # ------------------------------------------------------------------

    def predict_values(
        self,
        obs:    th.Tensor,
        lmu_h:  th.Tensor,
        lmu_m:  th.Tensor,
    ) -> th.Tensor:
        x = self.encoder(obs)
        h, _, _= self.lmu_cell(x, lmu_h, lmu_m)
        return self.critic(h).squeeze(-1)

    def initial_state(
        self, n_envs: int, device: th.device
    ) -> Tuple[th.Tensor, th.Tensor]:
        h = th.zeros(n_envs, self.hidden_size, device=device)
        m = th.zeros(n_envs, self.memory_size,  device=device)
        return h, m

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)