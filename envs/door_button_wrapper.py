"""
SB3-compatible single-agent wrapper for DoorButtonEnv.

MultiGrid returns:
  obs        — list[dict]  one per agent, each {"image": (H,W,3) uint8}
  rewards    — list[float]
  terminated — list[bool]
  truncated  — list[bool]
  infos      — list[dict]

This wrapper:
  - Extracts agent-0 image only
  - Merges terminated | truncated → done  (SB3 expects a single bool)
  - Passes TimeLimit.truncated through so ppo_gex bootstrap fires correctly
  - Exposes a flat Box obs space of shape (view_size, view_size, 3) int32
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.door_button import DoorButtonEnv


class DoorButtonSB3Env(gym.Env):
    """Single-agent, SB3-compatible wrapper around DoorButtonEnv."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, size: int = 10, view_size: int = 5, max_steps: int = 400):
        super().__init__()
        self._env = DoorButtonEnv(size=size, view_size=view_size, max_steps=max_steps)
        self._view_size = view_size

        # Integer ids in [0, max_id].  MultiGrid uses:
        #   channel 0: object type  (0-10, 11 types)
        #   channel 1: color        (0-5,   6 colors)
        #   channel 2: state        (0-3,   4 states)
        self.observation_space = spaces.Box(
            low=0,
            high=255,               # safe upper bound for all id channels
            shape=(view_size, view_size, 3),
            dtype=np.int32,
        )

        # MultiGrid default: 7 discrete actions
        self.action_space = spaces.Discrete(self._env.action_space[0].n)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_obs(self, raw_obs) -> np.ndarray:
        """
        MultiGrid obs[0] is either:
          dict  {"image": (H,W,3)}  — standard MultiGrid
          array (H,W,3)             — some wrappers unwrap it already
        """
        o = raw_obs[0] #if isinstance(raw_obs, (list, tuple)) else raw_obs
        if isinstance(o, dict):
            o = o["image"]
        return np.asarray(o, dtype=np.int32)

    # ------------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            obs_list, info_list = self._env.reset(seed=seed)
        else:
            obs_list, info_list = self._env.reset()

        obs  = self._extract_obs(obs_list)
        info = info_list[0] if isinstance(info_list, (list, tuple)) else info_list
        return obs, info

    def step(self, action: int):
        obs_list, rewards, terminated, truncated, infos = self._env.step([action])

        obs  = self._extract_obs(obs_list)
        rew  = float(rewards[0])
        term = bool(terminated[0])
        trunc = bool(truncated[0])
        info = infos[0] if isinstance(infos, (list, tuple)) else infos

        # SB3 expects a single done flag; pass truncated separately in info
        # so ppo_gex.py can apply the bootstrap correction.
        done = term or trunc
        if trunc and not term:
            info["TimeLimit.truncated"] = True

        return obs, rew, term, done, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()