"""
DoorButtonEnv — A sparse-reward environment for testing RND exploration.

Layout:
  Room A (large, right) — agent starts here, contains a button
  Room B (small, left)  — contains the goal

A locked door separates the rooms. The agent must:
  1. Find and toggle the button (unlocks + opens the door)
  2. Navigate through the door
  3. Reach the goal

Rewards are SPARSE by design — RND provides the exploration signal.
  - Button press: +1.0 (one-time)
  - Reach goal:   +10.0 (terminal)
  - Everything else: 0.0
"""
from multigrid.base import MultiGridEnv
from multigrid.core import Grid, WorldObj, Wall, Door, Goal
from multigrid.core.actions import Action
from multigrid.core.agent import Agent
import numpy as np


class Button(WorldObj):
    """A button that can be toggled once. Red=off, Green=on."""

    def __new__(cls):
        button = super().__new__(cls, color='red')
        button.is_pressed = False
        return button

    def can_overlap(self):
        return False

    def toggle(self, env, agent, pos):
        if self.is_pressed:
            return False
        self.is_pressed = True
        self.color = 'green'
        return True

    def encode(self):
        return (self[WorldObj.TYPE], self[WorldObj.COLOR], int(self.is_pressed))

    def render(self, img):
        from multigrid.utils.rendering import fill_coords, point_in_circle
        c = (0, 255, 0) if self.is_pressed else (255, 0, 0)
        fill_coords(img, point_in_circle(0.5, 0.5, 0.4), np.array(c))


class DoorButtonEnv(MultiGridEnv):
    """
    Single-agent grid world with sparse rewards.
    
    The agent must press a button to unlock a door, then reach a goal.
    Only two reward events exist — everything else is zero.
    """

    def __init__(self, size=10, view_size=5, max_steps=200, **kwargs):
        self.mission = "Press the button, go through the door, reach the goal."

        agents = [Agent(0, view_size=view_size)]
        agents[0].color = 'red'

        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            agents=agents,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Grid generation
    # ------------------------------------------------------------------
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        occupied = set()

        # Dividing wall at ~1/5 width (small goal room on left)
        wall_x = max(3, width // 5)
        for j in range(height):
            self.grid.set(wall_x, j, Wall())
            occupied.add((wall_x, j))

        # Locked door in the wall
        door_y = self.np_random.integers(1, height - 1)
        self.door = Door('yellow', is_locked=True)
        self.put_obj(self.door, wall_x, door_y)

        # Button in Room A (right side)
        button_x = self.np_random.integers(wall_x + 2, width - 1)
        button_y = self.np_random.integers(1, height - 1)
        self.button = Button()
        self.put_obj(self.button, button_x, button_y)
        occupied.add((button_x, button_y))

        # Goal in Room B (left side)
        goal_x = self.np_random.integers(1, wall_x)
        goal_y = self.np_random.integers(1, height - 1)
        self.goal = Goal()
        self.put_obj(self.goal, goal_x, goal_y)
        self.goal_pos = (goal_x, goal_y)
        occupied.add((goal_x, goal_y))

        # Place agent in Room A, away from button
        for _ in range(100):
            ax = self.np_random.integers(wall_x + 2, width - 1)
            ay = self.np_random.integers(1, height - 1)
            if (ax, ay) not in occupied:
                if abs(ax - button_x) + abs(ay - button_y) > 3:
                    self.agents[0].state.pos = (ax, ay)
                    self.agents[0].state.dir = self.np_random.integers(0, 4)
                    occupied.add((ax, ay))
                    break

        self._button_rewarded = False

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._button_rewarded = False
        return obs, info

    # ------------------------------------------------------------------
    # Step — SPARSE rewards only
    # ------------------------------------------------------------------
    def step(self, actions):
        obs, rewards, terminated, truncated, infos = super().step(actions)
        agent = self.agents[0]
        idx = 0

        rewards[idx] = 0.0

        infos[idx]['button_pressed'] = self.button.is_pressed
        infos[idx]['is_success'] = False

        # --- Reward 1: Button press (one-time, +1.0) ---
        if self.button.is_pressed and not self._button_rewarded:
            rewards[idx] += 1.0
            self._button_rewarded = True
            infos[idx]['button_pressed'] = True

        # --- Mechanical link: button opens the door ---
        if self.button.is_pressed and self.door.is_locked:
            self.door.is_locked = False
            self.door.is_open = True
            self.grid.update(*self.door.cur_pos)

        
        # --- Reward 2: Reach the goal (terminal, +10.0) ---
        if tuple(agent.state.pos) == self.goal_pos:
            
            rewards[idx] += 10.0
            terminated[idx] = True
            infos[idx]['is_success'] = True

        return obs, rewards, terminated, truncated, infos

