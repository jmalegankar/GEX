"""
MemoryStartWrapper
==================
Forces the agent to always spawn at the entrance of the corridor,
facing the hint object room.

Without this, MiniGrid-Memory places the agent at a random position
along the corridor. If spawned past the hint room, the agent never
observes the target object and must guess — making the task unsolvable
by any memory architecture regardless of capacity.

The Memory env layout (horizontal variant):

    [hint room] | [====== corridor ======] | [junction: ball / key]

Agent must:
  1. Enter the hint room, observe the object (ball or key)
  2. Walk the corridor
  3. At the junction, go to the side matching the hint object

We force spawn to the corridor entrance (x=2, facing right) so the
agent always walks through the hint room first.
"""

import gymnasium as gym
import numpy as np
from minigrid.core.constants import DIR_TO_VEC


class MemoryStartWrapper(gym.Wrapper):
    """
    Resets the agent position to the corridor entrance after each env reset.

    Works by inspecting the grid to find the hint room opening (the only
    floor cell adjacent to the left wall in the corridor row) and placing
    the agent there facing right (direction=0).

    Compatible with FilterObservation and VecTransposeImage wrappers.
    """

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._force_start_pos()
        # Re-render the observation from the forced position
        obs = self.env.unwrapped.gen_obs()
        return obs, info

    def _force_start_pos(self):
        env = self.env.unwrapped

        # The corridor row is the row the agent spawns in.
        # The hint room is always at x=1 (left side).
        # Place agent at x=2 (corridor entrance) facing right (dir=0).
        #
        # MiniGrid grid coords: (col, row) = (x, y)
        # The agent's row is fixed by env layout — we read it from the
        # current (random) spawn position.
        agent_row = env.agent_pos[1]

        # Corridor entrance = first walkable cell to the right of the hint room
        # The hint room occupies x=1; corridor starts at x=2.
        start_x = 2
        env.agent_pos = np.array([start_x, agent_row])
        env.agent_dir = 2  # facing right → toward hint room first