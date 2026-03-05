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
    memories: th.Tensor
    actions: th.Tensor
    prev_actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor

class TransitionRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer that explicitly stores (s_t, a_t, s_{t+1}) transitions
    along with embeddings and extrinsic rewards for per-epoch intrinsic
    reward recomputation.
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
    prev_actions: np.ndarray
    next_observations: np.ndarray
    prev_observations: np.ndarray
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
    ):
        self.memory_shape = memory_shape

        # Will be initialized in reset()
        self.next_observations = None
        self.prev_observations = None
        self.intrinsic_rewards = None
        self.memories = None
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
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
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
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
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
        prev_action: np.ndarray,
        intrinsic_reward: Optional[np.ndarray] = None,
    ) -> None:
        """
        Store a full transition (s_t, a_t, s_{t+1}, reward, memory, ...).

        :param obs: Current observation s_t
        :param next_obs: Next observation s_{t+1}
        :param action: Action taken
        :param reward: Combined reward (extrinsic + intrinsic)
        :param episode_start: Whether this is the first step of an episode
        :param value: Value estimate for s_t
        :param log_prob: Log probability of the action
        :param memory: Policy feature memory for intrinsic reward recomputation
        :param prev_action: Previous action taken (for intrinsic reward recomputation)
        :param intrinsic_reward: Optional intrinsic reward to store separately for GAE computation
        """
        # Let the base class store the rest (s_t, action, reward, ...)
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))
            prev_obs = prev_obs.reshape((self.n_envs, *self.obs_shape))
        
        self.next_observations[self.pos] = np.array(next_obs)
        self.prev_observations[self.pos] = np.array(prev_obs)
        self.memories[self.pos] = memory.clone().cpu().numpy()
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
                "actions",
                "prev_actions",
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
            # Cast to float32 (backward compatible), this would lead to RuntimeError for MultiBinary space
            self.actions[batch_inds].astype(np.float32, copy=False),
            self.prev_actions[batch_inds].astype(np.float32, copy=False),
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))
    
    # def squash_and_swap(self, array: np.ndarray) -> np.ndarray:
    #     """Inverse of swap_and_flatten: reshape from (buffer_size * n_envs, ...) to (buffer_size, n_envs, ...)."""
    #     return array.reshape((self.n_envs, self.buffer_size, *array.shape[1:])).swapaxes(0, 1)

    # def intrinsic_reward_updater(self, batch_size: int) -> Generator[Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor], np.ndarray, None]:
    #     """
    #     Create a Generator that returns batches of transitions for intrinsic reward recomputation.
    #     The generator yields batches of (observations, next_observations, memories, actions, prev_actions).
    #     The generator expects to receive a batch of intrinsic rewards computed from the yielded transitions,
    #     which it will store in the buffer for use in GAE computation.
    #     """
    #     assert self.full, "Buffer must be full before updating intrinsic rewards."
    #     start_idx = 0
    #     self.__dict__["intrinsic_rewards"] = self.swap_and_flatten(self.__dict__["intrinsic_rewards"]).flatten()
    #     if not self.generator_ready:
    #         self.__dict__["memories"] = self.swap_and_flatten(self.__dict__["memories"])
    #         self.__dict__["prev_actions"] = self.swap_and_flatten(self.__dict__["prev_actions"])
    #         self.__dict__["actions"] = self.swap_and_flatten(self.__dict__["actions"])
    #         self.__dict__["next_observations"] = self.swap_and_flatten(self.__dict__["next_observations"])
    #         self.__dict__["prev_observations"] = self.swap_and_flatten(self.__dict__["prev_observations"])
    #         self.__dict__["observations"] = self.swap_and_flatten(self.__dict__["observations"])
    #     while start_idx < self.buffer_size * self.n_envs:
    #         transitions = (
    #             self.observations[start_idx : start_idx + batch_size],
    #             self.next_observations[start_idx : start_idx + batch_size],
    #             self.memories[start_idx : start_idx + batch_size],
    #             self.prev_actions[start_idx : start_idx + batch_size].astype(np.float32, copy=False),
    #         )

    #         transitions = tuple(map(self.to_torch, transitions))

    #         # Yield the transitions and wait for the new intrinsic rewards
    #         new_intrinsic_rewards = yield transitions
    #         # Update the buffer with the new intrinsic rewards
    #         self.intrinsic_rewards[start_idx : start_idx + batch_size] = new_intrinsic_rewards
    #         start_idx += batch_size

    #     self.__dict__["intrinsic_rewards"] = self.squash_and_swap(self.__dict__["intrinsic_rewards"])

    #     if not self.generator_ready:
    #         self.__dict__["memories"] = self.squash_and_swap(self.__dict__["memories"])
    #         self.__dict__["prev_actions"] = self.squash_and_swap(self.__dict__["prev_actions"])
    #         self.__dict__["actions"] = self.squash_and_swap(self.__dict__["actions"])
    #         self.__dict__["next_observations"] = self.squash_and_swap(self.__dict__["next_observations"])
    #         self.__dict__["prev_observations"] = self.squash_and_swap(self.__dict__["prev_observations"])
    #         self.__dict__["observations"] = self.squash_and_swap(self.__dict__["observations"])
        
    #     if self.generator_ready:
    #         self.__dict__["advantages"] = self.squash_and_swap(self.__dict__["advantages"]).reshape((self.buffer_size, self.n_envs))
    #         self.__dict__["returns"] = self.squash_and_swap(self.__dict__["returns"]).reshape((self.buffer_size, self.n_envs))
    #         self.__dict__["values"] = self.squash_and_swap(self.__dict__["values"]).reshape((self.buffer_size, self.n_envs))

    #     self.compute_returns_and_advantage(self._last_values, self._dones)
        
    #     if self.generator_ready:
    #         self.__dict__["advantages"] = self.swap_and_flatten(self.__dict__["advantages"])
    #         self.__dict__["returns"] = self.swap_and_flatten(self.__dict__["returns"])
    #         self.__dict__["values"] = self.swap_and_flatten(self.__dict__["values"])

if __name__ == "__main__":
    # Test the TransitionRolloutBuffer
    from gymnasium.spaces import Box, Discrete

    buffer_size = 4
    n_envs = 2
    obs_space = Box(low=0, high=1, shape=(3,), dtype=np.float32)
    action_space = Discrete(2)

    buffer = TransitionRolloutBuffer(buffer_size, obs_space, action_space, n_envs=n_envs)

    for i in range(buffer_size):
        obs = np.random.rand(n_envs, 3).astype(np.float32)
        next_obs = np.random.rand(n_envs, 3).astype(np.float32)
        prev_obs = np.random.rand(n_envs, 3).astype(np.float32)
        action = np.random.randint(0, 2, size=(n_envs,))
        reward = np.random.rand(n_envs).astype(np.float32)
        episode_start = np.random.choice([True, False], size=(n_envs,))
        value = th.rand(n_envs)
        log_prob = th.rand(n_envs)
        memory = th.rand(n_envs, 64)
        prev_action = np.random.randint(0, 2, size=(n_envs,))
        intrinsic_reward = np.array([i]*n_envs).astype(np.float32)

        buffer.add(obs, prev_obs, next_obs, action, reward, episode_start, value, log_prob, memory, prev_action, intrinsic_reward)
    
    last_values = np.random.rand(n_envs).astype(np.float32)
    dones = np.random.choice([True, False], size=(n_envs,))
    buffer.compute_returns_and_advantage(last_values, dones)


    # Test the intrinsic reward updater generator
    # batch_size = 2
    # generator = buffer.intrinsic_reward_updater(batch_size)
    # try:
    #     int_rew = None
    #     while True:
    #         transitions = generator.send(int_rew)
    #         assert transitions[0].shape == (batch_size, 3)
    #         int_rew = np.zeros(batch_size).astype(np.float32)
    # except StopIteration:
    #     pass

    for samples in buffer.get():
        assert samples.observations.shape[0] == buffer_size * n_envs
        
    # generator = buffer.intrinsic_reward_updater(batch_size)
    # try:
    #     int_rew = None
    #     while True:
    #         transitions = generator.send(int_rew)
    #         int_rew = np.zeros(batch_size).astype(np.float32)
    # except StopIteration:
    #     pass

    # for samples in buffer.get():
    #     assert samples.observations.shape[0] == buffer_size * n_envs