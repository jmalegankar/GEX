"""
Minimal recurrent rollout buffer for LMU-PPO.
Stores (obs, action, reward, done, value, log_prob, h_{t-1}, m_{t-1}).
h and m are the LMU states BEFORE processing obs_t, so we can re-run
the LMU step during training with gradient flow.
"""

from typing import Dict, Generator, NamedTuple, Optional, Tuple

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize


class LMURolloutBufferSamples(NamedTuple):
    observations: Dict[str, th.Tensor]  # {'image': (B,C,H,W), 'direction': (B,)}
    actions:      th.Tensor             # (B, action_dim)
    old_values:   th.Tensor             # (B,)
    old_log_prob: th.Tensor             # (B,)
    advantages:   th.Tensor             # (B,)
    returns:      th.Tensor             # (B,)
    lmu_h:        th.Tensor             # (B, hidden_size)
    lmu_m:        th.Tensor             # (B, memory_size)


class LMURolloutBuffer(DictRolloutBuffer):
    """
    Extends SB3's DictRolloutBuffer (required for Dict obs spaces) with
    LMU state storage. DictRolloutBuffer handles the per-key obs arrays;
    we add lmu_h and lmu_m on top.
    """

    lmu_h: np.ndarray   # (buffer_size, n_envs, hidden_size)
    lmu_m: np.ndarray   # (buffer_size, n_envs, memory_size)

    def __init__(
        self,
        buffer_size:       int,
        observation_space: spaces.Space,
        action_space:      spaces.Space,
        hidden_size:       int,
        memory_size:       int,
        device:            str = "auto",
        gamma:             float = 0.99,
        gae_lambda:        float = 0.95,
        n_envs:            int = 1,
    ):
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        super().__init__(
            buffer_size, observation_space, action_space,
            device, gamma, gae_lambda, n_envs,
        )

    def reset(self) -> None:
        self.lmu_h = np.zeros(
            (self.buffer_size, self.n_envs, self.hidden_size), dtype=np.float32
        )
        self.lmu_m = np.zeros(
            (self.buffer_size, self.n_envs, self.memory_size), dtype=np.float32
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
        lmu_m:         th.Tensor,   # (n_envs, memory_size)
    ) -> None:
        self.lmu_h[self.pos] = lmu_h.cpu().numpy()
        self.lmu_m[self.pos] = lmu_m.cpu().numpy()
        super().add(obs, action, reward, episode_start, value, log_prob)

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[LMURolloutBufferSamples, None, None]:
        assert self.full, "Buffer must be full before sampling."
        indices = np.random.permutation(self.buffer_size * self.n_envs)

        if not self.generator_ready:
            # Flatten dict observations (DictRolloutBuffer stores them per-key)
            for key in self.observations:
                self.observations[key] = self.swap_and_flatten(self.observations[key])
            for tensor in ["actions", "values", "log_probs", "advantages", "returns",
                           "lmu_h", "lmu_m"]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        batch_size = batch_size or (self.buffer_size * self.n_envs)
        start = 0
        while start < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start : start + batch_size])
            start += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> LMURolloutBufferSamples:
        obs = {key: self.to_torch(self.observations[key][batch_inds])
               for key in self.observations}
        return LMURolloutBufferSamples(
            observations=obs,
            actions=self.to_torch(self.actions[batch_inds]),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
            lmu_h=self.to_torch(self.lmu_h[batch_inds]),
            lmu_m=self.to_torch(self.lmu_m[batch_inds]),
        )