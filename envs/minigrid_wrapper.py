"""
Minimal SB3 wrapper for MiniGrid environments.

MiniGrid returns obs as {"image": (H,W,3)} by default.
This wrapper unwraps the dict and ensures int32 dtype so
CategoricalGridEmbedding gets the right input.

Usage:
    env = MiniGridWrapper("MiniGrid-KeyCorridorS3R3-v0")
    env = MiniGridWrapper("MiniGrid-MultiRoom-N6-v0")
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from minigrid.wrappers import ImgObsWrapper


class MiniGridWrapper(gym.Wrapper):
    """
    Unwraps MiniGrid dict obs → (H, W, 3) int32.
    Everything else (action space, step, reset) passes through unchanged.
    """

    def __init__(self, env_id: str, **kwargs):
        env = gym.make(env_id, **kwargs)
        env = ImgObsWrapper(env)   # dict {"image": ...} → flat (H,W,3) array
        super().__init__(env)

        h, w, c = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(h, w, c),
            dtype=np.int32,
        )

    def observation(self, obs):
        return np.asarray(obs, dtype=np.int32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info