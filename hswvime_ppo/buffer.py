import numpy as np
import torch as th
from stable_baselines3.common.buffers import RolloutBuffer
from gymnasium import spaces

from typing import NamedTuple, Tuple, Optional, Generator
from stable_baselines3.common.vec_env import VecNormalize


class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    next_observations: th.Tensor
    prev_observations: th.Tensor
    actions: th.Tensor
    prev_actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    gru_hidden_states: th.Tensor  # (B, gru_hidden_dim) or (B, 0)


class TransitionRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer that stores (s_{t-1}, a_{t-1}, s_t, a_t, s_{t+1}) transitions,
    intrinsic rewards, and GRU hidden states for recurrent policy.
    """
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    prev_actions: np.ndarray
    next_observations: np.ndarray
    prev_observations: np.ndarray
    intrinsic_rewards: np.ndarray
    gru_hidden_states: np.ndarray
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
        gru_hidden_dim: int = 0,
    ):
        self.next_observations = None
        self.prev_observations = None
        self.intrinsic_rewards = None
        self.prev_actions = None
        self.gru_hidden_states = None
        self.gru_hidden_dim = gru_hidden_dim

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
        self.intrinsic_rewards = np.zeros(
            (self.buffer_size, self.n_envs),
            dtype=np.float32,
        )
        self.prev_actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim),
            dtype=self.action_space.dtype,
        )
        # GRU hidden states: (buffer_size, n_envs, gru_hidden_dim)
        self.gru_hidden_states = np.zeros(
            (self.buffer_size, self.n_envs, max(self.gru_hidden_dim, 1)),
            dtype=np.float32,
        )
        super().reset()

    def compute_returns_and_advantage(self, last_values: np.ndarray, dones: np.ndarray) -> None:
        """
        Compute lambda-return (TD(lambda)) and GAE(lambda) advantage.
        Uses intrinsic_rewards + rewards as the total reward signal.
        """
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
        prev_action: np.ndarray,
        intrinsic_reward: Optional[np.ndarray] = None,
        gru_hidden_state: Optional[np.ndarray] = None,
    ) -> None:
        """
        Store a full transition (s_{t-1}, a_{t-1}, s_t, a_t, s_{t+1}).
        """
        if isinstance(self.observation_space, spaces.Discrete):
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))
            prev_obs = prev_obs.reshape((self.n_envs, *self.obs_shape))

        self.next_observations[self.pos] = np.array(next_obs)
        self.prev_observations[self.pos] = np.array(prev_obs)
        if intrinsic_reward is not None:
            self.intrinsic_rewards[self.pos] = intrinsic_reward
        if gru_hidden_state is not None:
            self.gru_hidden_states[self.pos] = gru_hidden_state

        prev_action = prev_action.reshape((self.n_envs, self.action_dim))
        self.prev_actions[self.pos] = np.array(prev_action)

        super().add(obs, action, reward, episode_start, value, log_prob)

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "prev_observations",
                "next_observations",
                "actions",
                "prev_actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "gru_hidden_states",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

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
            self.next_observations[batch_inds],
            self.prev_observations[batch_inds],
            self.actions[batch_inds].astype(np.float32, copy=False),
            self.prev_actions[batch_inds].astype(np.float32, copy=False),
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.gru_hidden_states[batch_inds],
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))
