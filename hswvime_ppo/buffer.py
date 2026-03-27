import numpy as np
import torch as th
from stable_baselines3.common.buffers import RolloutBuffer
from gymnasium import spaces

from typing import NamedTuple, Tuple, Optional, Generator
from stable_baselines3.common.vec_env import VecNormalize


class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    prev_observations: th.Tensor
    next_observations: th.Tensor
    memories: th.Tensor
    wyner_h: th.Tensor
    slot_memories: th.Tensor
    actions: th.Tensor
    prev_actions: th.Tensor
    timesteps: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor

class TransitionRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer that explicitly stores (s_t, a_t, s_{t+1}) transitions
    along with Wyner memory and slot memory states.
    """
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    memories: np.ndarray
    wyner_h: np.ndarray
    slot_memories: np.ndarray
    prev_actions: np.ndarray
    next_observations: np.ndarray
    prev_observations: np.ndarray
    timesteps: np.ndarray
    intrinsic_rewards: np.ndarray
    _last_values: np.ndarray
    _dones: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: str = "auto",
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        n_envs: int = 1,
        memory_shape: Tuple[int, ...] = (1, 64),
        slot_memory_shape: Tuple[int, ...] = (8, 64),
        # Legacy kwargs accepted but ignored
        answer_dim: int = 0,
        num_qa: int = 0,
    ):
        self.memory_shape = memory_shape
        self.slot_memory_shape = slot_memory_shape

        # Will be initialized in reset()
        self.next_observations = None
        self.prev_observations = None
        self.timesteps = None
        self.intrinsic_rewards = None
        self.memories = None
        self.wyner_h = None
        self.slot_memories = None
        self.prev_actions = None

        self._last_values = None
        self._dones = None
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            gamma,
            gae_lambda,
            n_envs,
        )

    def reset(self) -> None:
        """Reset the buffer and re-initialise extra storage arrays."""
        self.next_observations = np.zeros(
            (self.buffer_size, self.n_envs, *self.obs_shape),
            dtype=np.float32,
        )
        self.prev_observations = np.zeros(
            (self.buffer_size, self.n_envs, *self.obs_shape),
            dtype=np.float32,
        )
        self.memories = np.zeros(
            (self.buffer_size, self.n_envs, *self.memory_shape),
            dtype=np.float32,
        )
        self.wyner_h = np.zeros(
            (self.buffer_size, self.n_envs, *self.memory_shape),
            dtype=np.float32,
        )
        self.slot_memories = np.zeros(
            (self.buffer_size, self.n_envs, *self.slot_memory_shape),
            dtype=np.float32,
        )
        self.timesteps = np.zeros(
            (self.buffer_size, self.n_envs),
            dtype=np.int64,
        )
        self.intrinsic_rewards = np.zeros(
            (self.buffer_size, self.n_envs),
            dtype=np.float32,
        )
        self.prev_actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim),
            dtype=self.action_space.dtype,
        )
        super().reset()

    def compute_returns_and_advantage(self, last_values: np.ndarray, dones: np.ndarray) -> None:
        self._last_values = last_values
        self._dones = dones

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.intrinsic_rewards[step] + self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values


    def add(
        self,
        obs: np.ndarray,
        prev_obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        memory: th.Tensor,
        wyner_h: th.Tensor,
        slot_memory: th.Tensor,
        prev_action: np.ndarray,
        timestep: Optional[np.ndarray] = None,
        intrinsic_reward: Optional[np.ndarray] = None,
    ) -> None:
        if isinstance(self.observation_space, spaces.Discrete):
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))
            prev_obs = prev_obs.reshape((self.n_envs, *self.obs_shape))

        self.next_observations[self.pos] = np.array(next_obs)
        self.prev_observations[self.pos] = np.array(prev_obs)
        self.memories[self.pos] = memory.clone().cpu().numpy()
        self.wyner_h[self.pos] = wyner_h.clone().cpu().numpy()
        self.slot_memories[self.pos] = slot_memory.clone().cpu().numpy()
        if timestep is not None:
            self.timesteps[self.pos] = np.array(timestep)
        if intrinsic_reward is not None:
            self.intrinsic_rewards[self.pos] = intrinsic_reward

        prev_action = prev_action.reshape((self.n_envs, self.action_dim))
        self.prev_actions[self.pos] = np.array(prev_action)

        super().add(obs, action, reward, episode_start, value, log_prob)

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "prev_observations",
                "next_observations",
                "memories",
                "wyner_h",
                "slot_memories",
                "actions",
                "prev_actions",
                "timesteps",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.prev_observations[batch_inds],
            self.next_observations[batch_inds],
            self.memories[batch_inds],
            self.wyner_h[batch_inds],
            self.slot_memories[batch_inds],
            self.actions[batch_inds].astype(np.float32, copy=False),
            self.prev_actions[batch_inds].astype(np.float32, copy=False),
            self.timesteps[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))
