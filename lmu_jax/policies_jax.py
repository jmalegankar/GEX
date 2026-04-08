"""
JAX/Flax LMU Actor-Critic Policy for MiniGrid.
Ports policies.py — same design decisions, Flax-native API.

Key difference from PyTorch version:
  - No VecTransposeImage: encoder expects NHWC image (B, H, W, 3).
  - __call__  == evaluate_actions (used in PPO update, needs gradient)
  - act()     == forward for rollout (returns logits; caller samples)
  - Both paths share _encode_and_step so params are the same pytree.
"""

import math
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from lmu_jax import LMUCell


# ---------------------------------------------------------------------------
# Observation encoder
# ---------------------------------------------------------------------------

class MinigridEncoder(nn.Module):
    """
    Encodes MiniGrid Dict obs *without* VecTransposeImage.
    Expects image shape (B, H, W, 3) — channels are categorical integers:
        channel 0 : OBJECT_IDX  (0–10, 11 types)
        channel 1 : COLOR_IDX   (0–5,  6 colors)
        channel 2 : STATE       (0–2,  door state)
    direction : (B,)  Discrete(4)

    Works for any grid size — CNN output size is inferred at init.
    For 7×7: 3 × (2×2 valid conv) → (4, 4, 64) → 1024-d flat.
    """
    out_dim:     int = 64
    obj_emb_dim: int = 8
    col_emb_dim: int = 4
    sta_emb_dim: int = 2
    dir_emb_dim: int = 4

    @nn.compact
    def __call__(self, obs: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        img       = obs["image"].astype(jnp.int32)           # (B, H, W, 3)
        direction = obs["direction"].astype(jnp.int32).reshape(-1)  # (B,)

        # Per-channel embeddings  →  (B, H, W, emb_dim)
        obj = nn.Embed(11, self.obj_emb_dim, name="obj_emb")(img[..., 0])
        col = nn.Embed(6,  self.col_emb_dim, name="col_emb")(img[..., 1])
        sta = nn.Embed(3,  self.sta_emb_dim, name="sta_emb")(img[..., 2])
        d   = nn.Embed(4,  self.dir_emb_dim, name="dir_emb")(direction)   # (B, dir_emb_dim)

        # Spatial CNN  (NHWC — Flax default)
        # 3 × (2×2, valid padding, stride 1): H,W: 7→6→5→4
        _sqrt2 = math.sqrt(2.0)
        x = jnp.concatenate([obj, col, sta], axis=-1)        # (B, H, W, tile_dim)
        x = nn.relu(nn.Conv(32, (2, 2), padding="VALID",
                            kernel_init=nn.initializers.orthogonal(_sqrt2),
                            name="conv1")(x))
        x = nn.relu(nn.Conv(64, (2, 2), padding="VALID",
                            kernel_init=nn.initializers.orthogonal(_sqrt2),
                            name="conv2")(x))
        x = nn.relu(nn.Conv(64, (2, 2), padding="VALID",
                            kernel_init=nn.initializers.orthogonal(_sqrt2),
                            name="conv3")(x))
        x = x.reshape(x.shape[0], -1)                        # (B, cnn_out_dim)

        # Concat direction embedding and project
        x = jnp.concatenate([x, d], axis=-1)
        x = nn.relu(nn.Dense(self.out_dim,
                             kernel_init=nn.initializers.orthogonal(_sqrt2),
                             name="proj")(x))
        return x    # (B, out_dim)


# ---------------------------------------------------------------------------
# LMU Actor-Critic Policy
# ---------------------------------------------------------------------------

class LMUActorCriticPolicy(nn.Module):
    """
    Encoder → LMUCell → actor head + critic head.

    Two public methods:

    act(obs, h, m)
        Used during rollout.  Returns (logits, value, h_new, m_new).
        Caller is responsible for sampling: jax.random.categorical(key, logits).

    __call__(obs, h, m, actions)   [= evaluate_actions]
        Used during PPO update.  Re-runs LMU step with stored (h, m) so that
        gradients flow through the same computation as at collection time.
        Returns (value, log_prob, entropy) — all shape (B,).
    """
    n_actions:   int
    encoder_dim: int = 64
    hidden_size: int = 64
    memory_size: int = 32
    theta:       float = 50.0

    def setup(self):
        self.encoder  = MinigridEncoder(out_dim=self.encoder_dim)
        self.lmu_cell = LMUCell(
            input_size  = self.encoder_dim,
            hidden_size = self.hidden_size,
            memory_size = self.memory_size,
            theta       = self.theta,
        )
        # Small orthogonal gain for actor (per SB3 defaults)
        self.actor  = nn.Dense(self.n_actions,
                               kernel_init=nn.initializers.orthogonal(0.01),
                               bias_init=nn.initializers.zeros)
        self.critic = nn.Dense(1,
                               kernel_init=nn.initializers.orthogonal(1.0),
                               bias_init=nn.initializers.zeros)

    # ------------------------------------------------------------------
    # Shared trunk
    # ------------------------------------------------------------------

    def _encode_and_step(
        self,
        obs:    Dict[str, jnp.ndarray],
        h_prev: jnp.ndarray,
        m_prev: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x               = self.encoder(obs)
        h_new, m_new    = self.lmu_cell(x, h_prev, m_prev)
        logits          = self.actor(h_new)
        value           = self.critic(h_new).squeeze(-1)    # (B,)
        return logits, value, h_new, m_new

    # ------------------------------------------------------------------
    # Rollout forward  (no action sampling — pure, no rng needed)
    # ------------------------------------------------------------------

    def act(
        self,
        obs:    Dict[str, jnp.ndarray],
        h_prev: jnp.ndarray,    # (B, hidden_size)
        m_prev: jnp.ndarray,    # (B, memory_size)
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Returns (logits, value, h_new, m_new)."""
        return self._encode_and_step(obs, h_prev, m_prev)

    # ------------------------------------------------------------------
    # PPO update  (= evaluate_actions)
    # ------------------------------------------------------------------

    def __call__(
        self,
        obs:     Dict[str, jnp.ndarray],
        h_prev:  jnp.ndarray,   # (B, hidden_size) — stored from rollout
        m_prev:  jnp.ndarray,   # (B, memory_size) — stored from rollout
        actions: jnp.ndarray,   # (B,) int32
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Returns (value, log_prob, entropy) — all (B,)."""
        logits, value, _, _ = self._encode_and_step(obs, h_prev, m_prev)
        log_probs = jax.nn.log_softmax(logits)                      # (B, n_actions)
        log_prob  = log_probs[jnp.arange(actions.shape[0]), actions] # (B,)
        probs     = jax.nn.softmax(logits)
        entropy   = -jnp.sum(probs * log_probs, axis=-1)            # (B,)
        return value, log_prob, entropy