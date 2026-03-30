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


class MemorySignalVisibleWrapper(gymnasium.Wrapper):
    """
    Wrapper for MiniGrid Memory environments that auto-executes a turn-around
    sequence on reset so the agent always observes the signal object.

    At reset, the signal (key or ball) is BEHIND the agent.  This wrapper
    executes ``turn_steps`` left-turn actions, then ``turn_steps`` more to
    face forward again.  The intermediate observations are discarded — the
    agent receives the post-turn observation as its initial obs, with the
    remaining max_steps budget reduced accordingly.

    The wrapper also stores ``signal_obs`` (the backward-facing observation
    that contains the signal) so diagnostics can inspect it.
    """

    def __init__(self, env: gymnasium.Env, turn_steps: int = 2):
        super().__init__(env)
        self.turn_steps = turn_steps
        self.signal_obs = None  # set during reset
        self.turn_history = []  # [(obs, action), ...] for each turn step

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.turn_history = []

        prev_obs = obs
        # Turn left to face the signal
        for _ in range(self.turn_steps):
            self.turn_history.append((prev_obs.copy(), 0))
            obs, _, term, trunc, info = self.env.step(0)  # 0 = turn left
            prev_obs = obs
            if term or trunc:
                info["turn_history"] = self.turn_history
                return obs, info

        # Store the backward-facing observation containing the signal
        self.signal_obs = obs

        # Turn back to face forward
        for _ in range(self.turn_steps):
            self.turn_history.append((prev_obs.copy(), 0))
            obs, _, term, trunc, info = self.env.step(0)
            prev_obs = obs
            if term or trunc:
                info["turn_history"] = self.turn_history
                return obs, info

        info["turn_history"] = self.turn_history
        return obs, info

    def step(self, action):
        return self.env.step(action)


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
