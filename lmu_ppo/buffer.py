"""
TBPTT rollout buffer for LMU-PPO.

Key change from single-step buffer:
    get() yields (B, K, ...) sequence chunks instead of (B, ...) flat transitions.
    Only the LMU state at each chunk's START is stored — the cell is re-run for
    K steps with gradient during evaluate_actions, so intra-chunk states are
    recomputed on the fly.

    episode_starts is included in each sample so evaluate_actions can zero-out
    state at episode boundaries within a chunk (no gradient bleeds across episodes).

Chunk construction:
    buffer stores (T, n_envs, ...) arrays.
    get() enumerates all valid chunk starts (t, env) where t+K <= T,
    shuffles them, then yields batches of B chunks as (B, K, ...) tensors.
"""

from typing import Dict, Generator, List, NamedTuple, Optional, Tuple

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize


class LMURolloutBufferSamples(NamedTuple):
    observations:  Dict[str, th.Tensor]  # each (B, K, ...)
    actions:       th.Tensor             # (B, K)       long
    old_values:    th.Tensor             # (B*K,)
    old_log_prob:  th.Tensor             # (B*K,)
    advantages:    th.Tensor             # (B*K,)
    returns:       th.Tensor             # (B*K,)
    lmu_h:         th.Tensor             # (B, hidden_size)              — chunk-start state
    lmu_m:         th.Tensor             # (B, memory_size, encoder_dim) — chunk-start state
    episode_starts: th.Tensor            # (B, K)  float32  1=new episode


class LMURolloutBuffer(DictRolloutBuffer):
    """
    DictRolloutBuffer extended with:
      - per-step LMU state storage (lmu_h, lmu_m)
      - episode_starts tracking (already in parent as self.episode_starts)
      - chunk-based get() for TBPTT

    Arrays are kept in (T, n_envs, ...) layout throughout — we never call
    swap_and_flatten so the sequential structure is preserved for chunk indexing.
    """

    lmu_h: np.ndarray   # (T, n_envs, hidden_size)
    lmu_m: np.ndarray   # (T, n_envs, memory_size, encoder_dim)

    def __init__(
        self,
        buffer_size:       int,
        observation_space: spaces.Space,
        action_space:      spaces.Space,
        hidden_size:       int,
        memory_size:       int,
        encoder_dim:       int,
        chunk_len:         int,
        device:            str = "auto",
        gamma:             float = 0.99,
        gae_lambda:        float = 0.95,
        n_envs:            int = 1,
    ):
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.encoder_dim = encoder_dim
        self.chunk_len   = chunk_len

        assert buffer_size % chunk_len == 0, (
            f"buffer_size ({buffer_size}) must be divisible by chunk_len ({chunk_len})"
        )

        super().__init__(
            buffer_size, observation_space, action_space,
            device, gamma, gae_lambda, n_envs,
        )

    def reset(self) -> None:
        self.lmu_h = np.zeros(
            (self.buffer_size, self.n_envs, self.hidden_size),
            dtype=np.float32,
        )
        self.lmu_m = np.zeros(
            (self.buffer_size, self.n_envs, self.memory_size, self.encoder_dim),
            dtype=np.float32,
        )
        super().reset()

    def add(
        self,
        obs:           np.ndarray,
        action:        np.ndarray,
        reward:        np.ndarray,
        episode_start: np.ndarray,
        value:         th.Tensor,
        log_prob:      th.Tensor,
        lmu_h:         th.Tensor,   # (n_envs, hidden_size)
        lmu_m:         th.Tensor,   # (n_envs, memory_size, encoder_dim)
    ) -> None:
        self.lmu_h[self.pos] = lmu_h.cpu().numpy()
        self.lmu_m[self.pos] = lmu_m.cpu().numpy()
        super().add(obs, action, reward, episode_start, value, log_prob)

    def get(
        self, n_chunks: Optional[int] = None
    ) -> Generator[LMURolloutBufferSamples, None, None]:
        """
        Yield batches of sequence chunks for TBPTT.

        n_chunks : number of chunks per yielded batch.
                   None = one giant batch (all chunks at once).
                   Typical value: (buffer_size // chunk_len * n_envs) // 4

        Each yielded batch contains n_chunks independent sequences of
        length chunk_len, drawn from random (env, time) positions.
        The effective number of transitions per batch = n_chunks * chunk_len.
        """
        assert self.full, "Buffer must be full before sampling."
        K = self.chunk_len
        n_chunks_per_env = self.buffer_size // K  # non-overlapping chunks per env

        # All valid (chunk_start_timestep, env_idx) pairs.
        # Each pair identifies the start of one independent K-step sequence.
        all_chunks = [
            (t * K, e)
            for e in range(self.n_envs)
            for t in range(n_chunks_per_env)
        ]
        np.random.shuffle(all_chunks)

        # Default: one batch containing every chunk (full-buffer update)
        n_chunks = n_chunks or len(all_chunks)
        for start in range(0, len(all_chunks), n_chunks):
            yield self._get_samples(all_chunks[start : start + n_chunks])

    def _get_samples(
        self,
        chunks: List[Tuple[int, int]],
        env: Optional[VecNormalize] = None,
    ) -> LMURolloutBufferSamples:
        """
        chunks : list of (t_start, env_idx) tuples
        Returns a batch where every sequential field has shape (B, K, ...).
        """
        K = self.chunk_len
        B = len(chunks)

        t_starts = np.array([c[0] for c in chunks], dtype=np.int64)  # (B,)
        envs     = np.array([c[1] for c in chunks], dtype=np.int64)  # (B,)

        # Time indices for all steps in each chunk: (B, K)
        # t_idx[b, k] = t_starts[b] + k
        t_idx = t_starts[:, None] + np.arange(K, dtype=np.int64)[None, :]

        # ── observations  dict of (B, K, ...) ────────────────────────────
        # self.observations[key]: (T, n_envs, ...)
        # Fancy index with (B, K) for time and (B, 1)→(B, K) broadcast for env.
        obs = {
            key: self.to_torch(self.observations[key][t_idx, envs[:, None]])
            for key in self.observations
        }

        # ── initial LMU state at chunk start  (B, ...) ───────────────────
        lmu_h = self.to_torch(self.lmu_h[t_starts, envs])       # (B, n)
        lmu_m = self.to_torch(self.lmu_m[t_starts, envs])       # (B, d, C)

        # ── episode_starts  (B, K)  float32 ──────────────────────────────
        # episode_starts[b, k] = 1 if obs[b, k] begins a new episode → zero h,m
        ep_starts = self.to_torch(
            self.episode_starts[t_idx, envs[:, None]].astype(np.float32)
        )

        # ── per-step scalars  (B, K) → flatten to (B*K,) for PPO losses ──
        actions    = self.to_torch(self.actions[t_idx, envs[:, None]])   # (B,K,1)
        advantages = self.to_torch(self.advantages[t_idx, envs[:, None]])
        returns    = self.to_torch(self.returns[t_idx, envs[:, None]])
        old_vals   = self.to_torch(self.values[t_idx, envs[:, None]])
        old_lp     = self.to_torch(self.log_probs[t_idx, envs[:, None]])

        return LMURolloutBufferSamples(
            observations=obs,
            actions=actions.long().squeeze(-1),          # (B, K)
            old_values=old_vals.reshape(B * K),
            old_log_prob=old_lp.reshape(B * K),
            advantages=advantages.reshape(B * K),
            returns=returns.reshape(B * K),
            lmu_h=lmu_h,
            lmu_m=lmu_m,
            episode_starts=ep_starts,                    # (B, K)
        )