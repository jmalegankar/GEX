"""
Memory validity tests for LMU-PPO.

Two tests, run independently:

TEST 1 — Memory Knockout
────────────────────────
At the step where the agent FIRST sees the decision junction (both ball and
key simultaneously visible), we zero out h and m before the policy acts.
If the agent genuinely used memory to encode ball vs key identity, its
success rate should drop toward 50% (random chance between two objects).
If it stays near 100%, the agent is reading the hint directly from its
current view at decision time — not from memory.

    python scripts/test_memory.py knockout \
        --checkpoint lmu_ppo_MemoryS11_s1.zip \
        --env MemoryS11 --n_episodes 200

TEST 2 — View Size Sensitivity Curve
──────────────────────────────────────
Evaluates a trained checkpoint at progressively smaller view sizes by
patching the env's view_size at load time. This is diagnostic only —
a model trained at view_size=7 evaluated at view_size=3 will almost
certainly degrade. The useful version is to compare TRAINING curves at
different view sizes, which is done via train.py --view_size.

This test answers: "Can the policy at all function with a smaller view?"
If it totally collapses at view_size=3, that suggests it was exploiting
view-specific visual cues (seeing the corridor end), not just memory.

    python scripts/test_memory.py viewsize \
        --checkpoint lmu_ppo_MemoryS11_s1.zip \
        --env MemoryS11 --view_sizes 3 5 7

Notes
─────
- The 'corridor can it see the end' geometry for MemoryS11 (grid 11x11,
  corridor ~9 cells): with view_size=7, the agent CAN see 7 cells ahead.
  The hint is at one end (step 0-1 visible), the junction at the other.
  With default MemoryStartWrapper the agent starts near the hint, so it
  sees the hint immediately in its 7-cell window. Whether it then STORES
  the hint in memory or can re-read it from the junction view depends on
  the junction layout. The knockout test is the only clean answer.

- view_size=3 is the minimal view where the agent can navigate (it can
  see one cell ahead and to the sides). At this size it is IMPOSSIBLE to
  see both the hint and the junction simultaneously for S11. If the agent
  succeeds here (when retrained), memory is unambiguously load-bearing.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
import minigrid  # noqa: F401
from gymnasium.wrappers import FilterObservation
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from lmu_ppo.lmu_ppo import LMUPPO
from mem_start import MemoryStartWrapper


ENV_IDS = {
    "MemoryS5":  "MiniGrid-MemoryS5-v0",
    "MemoryS7":  "MiniGrid-MemoryS7-v0",
    "MemoryS9":  "MiniGrid-MemoryS9-v0",
    "MemoryS11": "MiniGrid-MemoryS11-v0",
    "MemoryS13": "MiniGrid-MemoryS13-v0",
}

BALL_IDX = 6
KEY_IDX  = 5


def make_env(env_id: str, seed: int, use_wrapper: bool = True,
             view_size: Optional[int] = None):
    def _init():
        kwargs = {}
        if view_size is not None:
            kwargs['agent_view_size'] = view_size
        env = gym.make(env_id, **kwargs)
        if use_wrapper:
            env = MemoryStartWrapper(env)
        env = FilterObservation(env, filter_keys=["image", "direction"])
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def n_hint_objects_visible(obs_image: np.ndarray) -> int:
    """obs_image: (C, H, W). Channel 0 is object IDs."""
    obj = obs_image[0]
    return int((obj == BALL_IDX).sum() + (obj == KEY_IDX).sum())


# ──────────────────────────────────────────────────────────────────────────
# TEST 1: Memory Knockout
# ──────────────────────────────────────────────────────────────────────────

def run_knockout(
    model: LMUPPO,
    env,
    n_episodes: int,
    device: torch.device,
    knockout_at: str = 'junction',  # 'junction' | 'midcorridor' | 'always'
    seed: int = 9999,
) -> dict:
    """
    Evaluate policy with optional LMU state zeroing at specific moments.

    knockout_at:
        'junction'     : zero h,m at the first step where both objects visible
                         (classic memory knockout — does the agent know which
                         to pick, given it can no longer rely on stored hint?)
        'midcorridor'  : zero h,m at the midpoint of the episode
        'always'       : zero h,m at EVERY step (pure feedforward baseline —
                         gives the lower bound on what memory contributes)
        'never'        : normal eval, no zeroing (upper bound / ground truth)

    Returns dict with success_rate, mean_episode_length, and per-step stats.
    """
    policy = model.policy
    policy.set_training_mode(False)

    results = []
    per_step_success = defaultdict(list)  # step_of_knockout → success

    for ep_idx in range(n_episodes):
        obs = env.reset()
        h, m = policy.initial_state(n_envs=1, device=device)

        ep_reward = 0.0
        knockout_applied = False
        knockout_step = None

        for t in range(1000):
            obs_img = obs['image'][0]  # (C, H, W)
            n_objs = n_hint_objects_visible(obs_img)

            # Decide whether to knockout at this step
            zero_state = False
            if knockout_at == 'always':
                zero_state = True
            elif knockout_at == 'junction' and n_objs >= 2 and not knockout_applied:
                zero_state = True
                knockout_applied = True
                knockout_step = t
            elif knockout_at == 'midcorridor' and not knockout_applied:
                # Zero at step T//2 — determined dynamically by episode length
                # We use a heuristic: step 5 for S11 (half of typical ~10 ep)
                if t == 5:
                    zero_state = True
                    knockout_applied = True
                    knockout_step = t

            if zero_state:
                h = torch.zeros_like(h)
                m = torch.zeros_like(m)

            with torch.no_grad():
                obs_t = {k: torch.as_tensor(v, device=device)
                         for k, v in obs.items()}
                action, _, _, h_new, m_new, _, _, _, _, _ = policy.forward(
                    obs_t, h, m
                )

            action_np = action.cpu().numpy()
            obs, reward, done, info = env.step(action_np)
            ep_reward += float(reward[0])

            h = h_new
            m = m_new

            if done[0]:
                break

        success = ep_reward > 0.5
        results.append({
            'success': success,
            'knockout_applied': knockout_applied,
            'knockout_step': knockout_step,
            'ep_reward': ep_reward,
        })

        if knockout_step is not None:
            per_step_success[knockout_step].append(success)

    # Aggregate
    n_total = len(results)
    n_success = sum(r['success'] for r in results)
    n_knockout_applied = sum(r['knockout_applied'] for r in results)

    # Per-knockout-step success rate
    per_step = {
        str(step): {
            'n': len(slist),
            'success_rate': float(np.mean(slist)),
        }
        for step, slist in sorted(per_step_success.items())
    }

    return {
        'knockout_at': knockout_at,
        'n_episodes': n_total,
        'n_knockout_applied': n_knockout_applied,
        'success_rate': float(n_success / n_total),
        'success_rate_when_knocked_out':
            float(np.mean([r['success'] for r in results if r['knockout_applied']]))
            if n_knockout_applied > 0 else None,
        'success_rate_when_not_knocked_out':
            float(np.mean([r['success'] for r in results if not r['knockout_applied']]))
            if n_total - n_knockout_applied > 0 else None,
        'per_knockout_step_success': per_step,
    }


def test_knockout(args):
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    )
    env = VecTransposeImage(DummyVecEnv([
        make_env(ENV_IDS[args.env], args.seed, use_wrapper=not args.no_wrapper)
    ]))
    model = LMUPPO.load(args.checkpoint, env=env, device=device)
    print(f"Loaded {args.checkpoint}")
    print(f"  memory_size={model.policy.lmu_cell.memory_size} "
          f"hidden_size={model.policy.lmu_cell.hidden_size}")

    conditions = ['never', 'junction', 'midcorridor', 'always']
    all_results = {}

    for cond in conditions:
        print(f"\nRunning knockout_at='{cond}' ({args.n_episodes} eps)...")
        r = run_knockout(model, env, args.n_episodes, device,
                         knockout_at=cond, seed=args.seed)
        all_results[cond] = r
        print(f"  success_rate = {r['success_rate']:.3f} "
              f"  (knockout applied to {r['n_knockout_applied']}/{r['n_episodes']} eps)")
        if r.get('success_rate_when_knocked_out') is not None:
            print(f"  success_rate WHEN knocked out = "
                  f"{r['success_rate_when_knocked_out']:.3f}")

    # Summary table
    print("\n── Memory knockout summary ─────────────────────────────────")
    print(f"  {'Condition':<20} {'Success rate':>14} {'Knocked out eps':>16}")
    for cond in conditions:
        r = all_results[cond]
        sr_str = f"{r['success_rate']:.3f}"
        ko_str = f"{r['n_knockout_applied']}/{r['n_episodes']}"
        print(f"  {cond:<20} {sr_str:>14} {ko_str:>16}")

    print("\n── Interpretation ──────────────────────────────────────────")
    sr_normal = all_results['never']['success_rate']
    sr_junction = all_results['junction'].get('success_rate_when_knocked_out')
    sr_always = all_results['always']['success_rate']

    if sr_junction is not None:
        delta = sr_normal - sr_junction
        print(f"  Normal success rate:          {sr_normal:.3f}")
        print(f"  After junction knockout:      {sr_junction:.3f}")
        print(f"  Drop:                         {delta:.3f}")
        print()
        if delta > 0.35:
            print("  VERDICT: Agent IS using memory. Junction knockout causes "
                  f"large drop ({delta:.2%}). LMU is encoding hint identity "
                  "across the corridor successfully.")
        elif delta > 0.15:
            print("  VERDICT: Agent uses memory PARTIALLY. Some knockout episodes "
                  "succeed via visual fallback. Memory is helpful but the agent "
                  "has a partial vision-based strategy too.")
        else:
            print("  VERDICT: Agent may NOT be using memory. Junction knockout "
                  f"causes only {delta:.2%} drop. The agent can read the hint "
                  "from the junction view directly — check if view_size allows "
                  "both ends to be visible.")
    print(f"  Feedforward baseline (always):  {sr_always:.3f} "
          f"({'≈ random' if sr_always < 0.6 else 'above random — vision helps'})")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nWrote: {args.output}")

    env.close()


# ──────────────────────────────────────────────────────────────────────────
# TEST 2: View Size Sensitivity
# ──────────────────────────────────────────────────────────────────────────

def test_viewsize(args):
    """
    Evaluate a trained checkpoint at multiple view sizes.
    Note: a model TRAINED at view_size=7 evaluated at view_size=3 is expected
    to degrade. The useful comparison is training curves — use train.py
    --view_size for that. This test checks whether the POLICY STRUCTURE
    (not trained weights) can function at all with reduced vision.
    """
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    )

    all_results = {}
    for vs in args.view_sizes:
        print(f"\nEvaluating at view_size={vs}...")
        env = VecTransposeImage(DummyVecEnv([
            make_env(ENV_IDS[args.env], args.seed,
                     use_wrapper=not args.no_wrapper, view_size=vs)
        ]))
        model = LMUPPO.load(args.checkpoint, env=env, device=device)

        successes = []
        for ep in range(args.n_episodes):
            obs = env.reset()
            h, m = model.policy.initial_state(1, device)
            ep_reward = 0.0
            for _ in range(1000):
                with torch.no_grad():
                    obs_t = {k: torch.as_tensor(v, device=device)
                             for k, v in obs.items()}
                    action, _, _, h, m, _, _, _, _, _ = model.policy.forward(
                        obs_t, h, m
                    )
                obs, rew, done, _ = env.step(action.cpu().numpy())
                ep_reward += float(rew[0])
                if done[0]:
                    break
            successes.append(ep_reward > 0.5)

        sr = float(np.mean(successes))
        all_results[vs] = {'view_size': vs, 'success_rate': sr,
                           'n_episodes': args.n_episodes}
        print(f"  view_size={vs} → success_rate={sr:.3f}")
        env.close()

    print("\n── View size sensitivity summary ───────────────────────────")
    for vs, r in sorted(all_results.items()):
        bar = '█' * int(r['success_rate'] * 20)
        print(f"  view_size={vs:2d} │ {r['success_rate']:.3f} │ {bar}")

    print("\n── Interpretation ──────────────────────────────────────────")
    print("  These are cross-eval results (trained at one view, tested at others).")
    print("  For the definitive memory test, retrain at view_size=3 via:")
    print("    python train.py --env MemoryS11 --view_size 3 --seed 0 \\")
    print("                   --total_steps 5_000_000")
    print("  If the agent trains to solve it at view_size=3 with similar")
    print("  sample efficiency, memory is unambiguously load-bearing.")
    print("  If it fails or takes 3× longer, the wide view was doing work.")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nWrote: {args.output}")


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='test', required=True)

    # Common args
    def add_common(p):
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--env", default="MemoryS11", choices=list(ENV_IDS))
        p.add_argument("--n_episodes", type=int, default=200)
        p.add_argument("--seed", type=int, default=9999)
        p.add_argument("--device", default="auto")
        p.add_argument("--no_wrapper", action='store_true',
                       help="Evaluate without MemoryStartWrapper")
        p.add_argument("--output", default=None)

    # Knockout sub-command
    p_ko = sub.add_parser("knockout",
                           help="Memory knockout test (zero h,m at junction)")
    add_common(p_ko)

    # View size sub-command
    p_vs = sub.add_parser("viewsize",
                           help="View size sensitivity test")
    add_common(p_vs)
    p_vs.add_argument("--view_sizes", type=int, nargs='+', default=[3, 5, 7],
                      help="List of view sizes to evaluate at")

    args = parser.parse_args()

    if args.test == 'knockout':
        test_knockout(args)
    elif args.test == 'viewsize':
        test_viewsize(args)


if __name__ == "__main__":
    main()