import numpy as np
import gymnasium
from gymnasium import spaces
from envs.door_button import DoorButtonEnv


class DoorButtonTrainingWrapper(gymnasium.Env):
    """
    Single-agent gymnasium wrapper around DoorButtonEnv for SB3 training.

    Observation layout:
      MultiGrid returns obs[agent_idx] = {'image': (H, W, 3) int, 'direction': int}
      This wrapper packs both into a single (H, W, 4) int32 tensor:
        channels 0-2 : image  (object_type, color, state)
        channel  3   : agent direction (0-3), broadcast across all cells

    Action:
      Accepts a single integer and forwards it as [action] to the multi-agent env.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, size: int = 10, view_size: int = 5, max_steps: int = 200, **kwargs):
        super().__init__()
        self._env = DoorButtonEnv(size=size, view_size=view_size, max_steps=max_steps, **kwargs)
        self.view_size = view_size

        # (H, W, 4): image channels + direction channel
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(view_size, view_size, 4),
            dtype=np.int32,
        )
        # Single agent action space (Discrete)
        self.action_space = self._env.action_space[0]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pack_obs(self, raw_obs: dict) -> np.ndarray:
        """Convert {'image': (H,W,3), 'direction': int} → (H,W,4) int32."""
        img = raw_obs["image"].astype(np.int32)            # (H, W, 3)
        direction = int(raw_obs["direction"])
        dir_channel = np.full((*img.shape[:2], 1), direction, dtype=np.int32)  # (H, W, 1)
        return np.concatenate([img, dir_channel], axis=-1)  # (H, W, 4)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        obs_list, info_list = self._env.reset(seed=seed, options=options)
        return self._pack_obs(obs_list[0]), info_list[0]

    def step(self, action):
        obs_list, rewards, terminated, truncated, infos = self._env.step([action])
        obs = self._pack_obs(obs_list[0])
        return obs, float(rewards[0]), bool(terminated[0]), bool(truncated[0]), infos[0]

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()


class MiniGridTrainingWrapper(gymnasium.Wrapper):
    """
    Observation wrapper for any MiniGrid environment for SB3 training.

    Observation layout:
      MiniGrid returns {'image': (H, W, 3) uint8, 'direction': int, 'mission': str}
      Packed into a single (H, W, 4) int32 tensor:
        channels 0-2 : image  (object_type, color, state)
        channel  3   : agent direction (0-3), broadcast across all cells
    """

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        h, w, _ = env.observation_space["image"].shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(h, w, 4),
            dtype=np.int32,
        )

    def _pack_obs(self, raw_obs: dict) -> np.ndarray:
        """Convert {'image': (H,W,3), 'direction': int, ...} → (H,W,4) int32."""
        img = raw_obs["image"].astype(np.int32)
        direction = int(raw_obs["direction"])
        dir_channel = np.full((*img.shape[:2], 1), direction, dtype=np.int32)
        return np.concatenate([img, dir_channel], axis=-1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._pack_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._pack_obs(obs), reward, terminated, truncated, info


class CrafterTrainingWrapper(gymnasium.Env):
    """
    Gymnasium wrapper around Crafter for SB3 training.

    Observation: (64, 64, 3) uint8 RGB — passed through as-is.
    Action: Discrete(17).
    Tracks per-achievement success rates in info for achievement score reporting.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, max_steps: int = 10_000, **kwargs):
        super().__init__()
        import crafter
        self._env = crafter.Env()
        self._max_steps = max_steps
        self._step_count = 0

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(64, 64, 3), dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(17)

        # Track cumulative achievements across episode
        self._achievements = None

    def reset(self, *, seed=None, options=None):
        obs = self._env.reset()
        self._step_count = 0
        self._achievements = None
        return obs.astype(np.uint8), {}

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        self._step_count += 1

        # Track achievements (cumulative max per episode)
        if "achievements" in info:
            if self._achievements is None:
                self._achievements = dict(info["achievements"])
            else:
                for k, v in info["achievements"].items():
                    self._achievements[k] = max(self._achievements[k], v)

        truncated = self._step_count >= self._max_steps and not done
        terminated = done

        out_info = {}
        if self._achievements is not None:
            out_info["achievements"] = dict(self._achievements)

        return obs.astype(np.uint8), float(reward), terminated, truncated, out_info
