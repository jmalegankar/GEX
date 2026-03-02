import numpy as np
import torch as th
from stable_baselines3.common.buffers import RolloutBuffer


class GEXRolloutBuffer(RolloutBuffer):
    """
    RolloutBuffer + extras needed by GEX:
      - next_observations: for online SC-VAE training
      - intrinsic_rewards: for diagnostics / logging
      - extrinsic_rewards: optional (useful for debugging)
    """

    def reset(self) -> None:
        super().reset()
        self.next_observations = np.zeros_like(self.observations)
        self.intrinsic_rewards = np.zeros_like(self.rewards)
        self.extrinsic_rewards = np.zeros_like(self.rewards)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        *,
        next_obs: np.ndarray = None,
        intrinsic_reward: np.ndarray = None,
        extrinsic_reward: np.ndarray = None,
    ) -> None:
        # store extras at current position BEFORE parent increments pos
        self.next_observations[self.pos] = next_obs if next_obs is not None else obs
        self.intrinsic_rewards[self.pos] = intrinsic_reward if intrinsic_reward is not None else 0.0
        self.extrinsic_rewards[self.pos] = extrinsic_reward if extrinsic_reward is not None else reward

        super().add(obs, action, reward, episode_start, value, log_prob)