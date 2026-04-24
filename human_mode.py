"""
Play MiniGrid-Memory envs in human mode.

Controls:
    ← → : turn left / right
    ↑   : move forward
    space: toggle / pick up
    r   : reset episode
    q   : quit

Run:
    python play.py --env MemoryS11
    python play.py --env MemoryS7
"""

import argparse
import random
import gymnasium as gym
import minigrid  # noqa
from minigrid.manual_control import ManualControl
from mem_start import MemoryStartWrapper

ENV_IDS = {
    "MemoryS5":  "MiniGrid-MemoryS5-v0",
    "MemoryS7":  "MiniGrid-MemoryS7-v0",
    "MemoryS9":  "MiniGrid-MemoryS9-v0",
    "MemoryS11": "MiniGrid-MemoryS11-v0",
    "MemoryS13": "MiniGrid-MemoryS13-v0",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",  default="MemoryS13", choices=list(ENV_IDS))
    parser.add_argument("--seed", type=int, default=random.randint(0, 10000))
    parser.add_argument("--tile_size", type=int, default=40)
    args = parser.parse_args()

    env = gym.make(
        ENV_IDS[args.env],
        render_mode="human",
        tile_size=args.tile_size,
    )
    # env = MemoryStartWrapper(env)


    if args.seed is not None:
        env.reset(seed=args.seed)

    manual = ManualControl(env, seed=args.seed)
    manual.start()

if __name__ == "__main__":
    main()