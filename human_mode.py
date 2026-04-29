"""
Terminal-based human / inspection mode for POPGym envs.

POPGym tasks have abstract obs (Discrete int / MultiDiscrete vector) — there's
no 2D grid to render. Instead, this prints obs/action/reward streams to the
terminal. Useful for:
    - Sanity-checking task semantics before launching a sweep
    - Measuring per-task episode-length / reward distribution to inform
      chunk_len, gamma, and n_steps choices

Three modes:
    interactive  Type each action; see obs and reward after each step.
                 For Discrete actions, press 0-9. For MultiDiscrete, type
                 comma-separated ints and press Enter.
    random       Auto-step with a uniform-random policy, print the stream.
                 Useful for getting a feel for episode pacing.
    stats        Silent run of N episodes; print distribution summary.
                 Run this BEFORE picking chunk_len for a new task.

Usage:
    python play.py --env popgym-RepeatPreviousMedium-v0 --mode interactive
    python play.py --env popgym-ConcentrationMedium-v0 --mode random --max_steps 200
    python play.py --env popgym-CountRecallMedium-v0 --mode stats --n_episodes 100
"""

import argparse
import sys
import time
from collections import Counter

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from lmu_ppo.popgym_envs import (
    SingleKeyDictWrapper,
    TupleToMultiDiscreteWrapper,
)


# ─────────────────────────────────────────────────────────────────────────────
# Env construction (no Monitor wrapper — we want raw obs/reward access)
# ─────────────────────────────────────────────────────────────────────────────

def make_play_env(env_id: str, seed: int = 0) -> gym.Env:
    """
    Build a POPGym env with the same wrappers as training, minus Monitor.

    The Tuple→MultiDiscrete wrapper IS applied because we want the same obs
    semantics the model sees. SingleKeyDictWrapper is applied for the same
    reason — though for human inspection we'll un-key before printing.
    """
    import popgym  # noqa: F401  -- registers env IDs

    env = gym.make(env_id)
    if isinstance(env.observation_space, spaces.Tuple):
        env = TupleToMultiDiscreteWrapper(env)
    env = SingleKeyDictWrapper(env)
    env.reset(seed=seed)
    return env


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-printing helpers
# ─────────────────────────────────────────────────────────────────────────────

def fmt_obs(obs_dict: dict, key: str = 'obs') -> str:
    """Format the inner obs from a SingleKeyDictWrapper output."""
    inner = obs_dict[key]
    if isinstance(inner, (int, np.integer)):
        return f"obs={int(inner)}"
    if isinstance(inner, np.ndarray):
        # Truncate long arrays for readability.
        if inner.size > 16:
            return f"obs=array(shape={inner.shape}, head={inner.flatten()[:16].tolist()}...)"
        return f"obs={inner.tolist()}"
    return f"obs={inner!r}"


def fmt_action(action, action_space) -> str:
    if isinstance(action_space, spaces.Discrete):
        return f"action={int(action)}"
    if isinstance(action_space, spaces.MultiDiscrete):
        return f"action={list(action)}"
    return f"action={action!r}"


def describe_env(env: gym.Env, env_id: str) -> None:
    """Print env metadata once at startup."""
    inner_obs = env.observation_space.spaces['obs']

    if isinstance(inner_obs, spaces.Discrete):
        obs_desc = f"Discrete({inner_obs.n})"
    elif isinstance(inner_obs, spaces.MultiDiscrete):
        nvec = np.asarray(inner_obs.nvec).flatten()
        if (nvec == nvec[0]).all() and len(nvec) > 4:
            obs_desc = f"MultiDiscrete([{nvec[0]}]*{len(nvec)})"
        else:
            obs_desc = f"MultiDiscrete({nvec.tolist()})"
    elif isinstance(inner_obs, spaces.Box):
        obs_desc = f"Box{inner_obs.shape} {inner_obs.dtype}"
    else:
        obs_desc = repr(inner_obs)

    if isinstance(env.action_space, spaces.Discrete):
        act_desc = f"Discrete({env.action_space.n})"
    elif isinstance(env.action_space, spaces.MultiDiscrete):
        act_desc = f"MultiDiscrete({list(env.action_space.nvec)})"
    else:
        act_desc = repr(env.action_space)

    print(f"\nEnv: {env_id}")
    print(f"  obs (inner):  {obs_desc}")
    print(f"  action:       {act_desc}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Action parsing for interactive mode
# ─────────────────────────────────────────────────────────────────────────────

def prompt_action(action_space, default_msg: str = "") -> np.ndarray:
    """
    Prompt user for an action via stdin. Loops on parse failure.

    Discrete:      single int, e.g. "0", "1"
    MultiDiscrete: comma-separated ints, e.g. "0,1" or "0, 2, 1"

    Empty input returns a random action — convenient for "skip ahead."
    Returns whatever shape/dtype the env expects.
    """
    while True:
        try:
            raw = input(f"action {default_msg}> ").strip()
        except EOFError:
            print()
            sys.exit(0)

        if raw == "":
            return action_space.sample()
        if raw in ("q", "quit", "exit"):
            sys.exit(0)
        if raw == "r":
            return None   # signal to caller to reset

        try:
            if isinstance(action_space, spaces.Discrete):
                a = int(raw)
                if not (0 <= a < action_space.n):
                    print(f"  action out of range [0, {action_space.n})")
                    continue
                return a
            if isinstance(action_space, spaces.MultiDiscrete):
                parts = [int(p.strip()) for p in raw.split(",")]
                arr = np.asarray(parts, dtype=action_space.dtype)
                if arr.shape != action_space.nvec.shape:
                    print(f"  expected {action_space.nvec.shape[0]} ints, got {len(parts)}")
                    continue
                if not (arr < action_space.nvec).all() or (arr < 0).any():
                    print(f"  one or more components out of range")
                    continue
                return arr
            # Fallback for unsupported action spaces.
            return action_space.sample()
        except ValueError:
            print(f"  could not parse {raw!r} — type 'q' to quit, 'r' to reset, "
                  f"<Enter> for random")


# ─────────────────────────────────────────────────────────────────────────────
# Modes
# ─────────────────────────────────────────────────────────────────────────────

def run_interactive(env: gym.Env, env_id: str, max_steps: int) -> None:
    describe_env(env, env_id)
    print("Controls: type action and Enter. <Enter> alone = random. "
          "'r' = reset. 'q' = quit.\n")

    obs, info = env.reset()
    print(f"[reset] {fmt_obs(obs)}")

    step = 0
    ep_return = 0.0
    while step < max_steps:
        action = prompt_action(env.action_space)
        if action is None:
            obs, info = env.reset()
            print(f"[reset] {fmt_obs(obs)}  (manual)")
            ep_return = 0.0
            continue

        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += float(reward)
        flag = " TERM" if terminated else (" TRUNC" if truncated else "")
        print(f"  step {step:4d}  {fmt_action(action, env.action_space):20s}  "
              f"reward={float(reward):+.3f}  ep_return={ep_return:+.3f}  "
              f"{fmt_obs(obs)}{flag}")
        step += 1

        if terminated or truncated:
            print(f"  [episode done after {step} steps, return={ep_return:+.3f}]\n")
            obs, info = env.reset()
            print(f"[reset] {fmt_obs(obs)}")
            ep_return = 0.0


def run_random(env: gym.Env, env_id: str, max_steps: int, sleep: float) -> None:
    describe_env(env, env_id)
    print(f"Random policy, max_steps={max_steps}, sleep={sleep}s/step.\n")

    obs, info = env.reset()
    print(f"[reset] {fmt_obs(obs)}")

    ep_return = 0.0
    ep_len = 0
    for step in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += float(reward)
        ep_len += 1

        flag = " TERM" if terminated else (" TRUNC" if truncated else "")
        print(f"  step {step:4d}  {fmt_action(action, env.action_space):20s}  "
              f"reward={float(reward):+.3f}  ep_return={ep_return:+.3f}  "
              f"{fmt_obs(obs)}{flag}")

        if sleep > 0:
            time.sleep(sleep)

        if terminated or truncated:
            print(f"  [episode done after {ep_len} steps, return={ep_return:+.3f}]\n")
            obs, info = env.reset()
            print(f"[reset] {fmt_obs(obs)}")
            ep_return = 0.0
            ep_len = 0


def run_stats(env: gym.Env, env_id: str, n_episodes: int, max_steps_per_ep: int) -> None:
    """
    Silent run of N episodes with random policy. Print distribution summary.

    Output is structured for direct use in choosing chunk_len, n_steps, gamma.
    """
    describe_env(env, env_id)
    print(f"Running {n_episodes} episodes with random policy "
          f"(max_steps_per_ep={max_steps_per_ep})...\n")

    ep_lengths = []
    ep_returns = []
    reward_per_step = []  # all per-step rewards across all episodes

    for ep in range(n_episodes):
        obs, info = env.reset()
        ret = 0.0
        n = 0
        for _ in range(max_steps_per_ep):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ret += float(reward)
            reward_per_step.append(float(reward))
            n += 1
            if terminated or truncated:
                break
        ep_lengths.append(n)
        ep_returns.append(ret)

    lens = np.asarray(ep_lengths)
    rets = np.asarray(ep_returns)
    rps  = np.asarray(reward_per_step)

    def pct(arr, q):
        return float(np.percentile(arr, q))

    print(f"Episode length")
    print(f"  min / median / max          : {lens.min()} / {int(np.median(lens))} / {lens.max()}")
    print(f"  mean ± std                  : {lens.mean():.1f} ± {lens.std():.1f}")
    print(f"  p10 / p50 / p90             : {pct(lens, 10):.0f} / {pct(lens, 50):.0f} / {pct(lens, 90):.0f}")
    fixed = lens.std() < 0.5
    print(f"  fixed-length?                : {'YES' if fixed else 'no'}")

    print(f"\nEpisode return (random policy)")
    print(f"  min / median / max          : {rets.min():+.3f} / {np.median(rets):+.3f} / {rets.max():+.3f}")
    print(f"  mean ± std                  : {rets.mean():+.3f} ± {rets.std():.3f}")

    print(f"\nPer-step reward")
    print(f"  zero fraction                : {(rps == 0).mean():.2%}")
    print(f"  nonzero mean magnitude       : "
          f"{rps[rps != 0].mean() if (rps != 0).any() else 0.0:+.4f}")
    nonzero_count = int((rps != 0).sum())
    sparse = (rps != 0).mean() < 0.1
    print(f"  reward density               : {nonzero_count}/{len(rps)} steps ({(rps != 0).mean():.2%}) — "
          f"{'SPARSE' if sparse else 'dense'}")

    # Suggestions based on distribution.
    print(f"\nSuggested config knobs:")
    p90_len = int(pct(lens, 90))
    print(f"  chunk_len  = {min(32, max(8, p90_len // 4))}  "
          f"(rule of thumb: p90_episode_length / 4, clamped to [8, 32])")
    if sparse:
        print(f"  gamma      = 0.99    (sparse reward — long credit assignment)")
    else:
        print(f"  gamma      = 0.95    (dense reward — short credit assignment may suffice)")
    print(f"  n_steps    = {max(p90_len * 2, 128)}  "
          f"(2x p90 episode length, ≥128, ensures full episodes per rollout)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env", default="popgym-RepeatPreviousMedium-v0",
                        help="POPGym env ID")
    parser.add_argument("--mode", default="interactive",
                        choices=["interactive", "random", "stats"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=200,
                        help="Max steps for interactive/random modes.")
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="Sleep between steps in random mode.")
    parser.add_argument("--n_episodes", type=int, default=100,
                        help="Number of episodes for stats mode.")
    parser.add_argument("--max_steps_per_ep", type=int, default=2000,
                        help="Safety cap on steps-per-episode in stats mode.")
    args = parser.parse_args()

    env = make_play_env(args.env, seed=args.seed)
    try:
        if args.mode == "interactive":
            run_interactive(env, args.env, args.max_steps)
        elif args.mode == "random":
            run_random(env, args.env, args.max_steps, args.sleep)
        else:  # stats
            run_stats(env, args.env, args.n_episodes, args.max_steps_per_ep)
    finally:
        env.close()


if __name__ == "__main__":
    main()