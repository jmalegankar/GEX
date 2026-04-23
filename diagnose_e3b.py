"""
Phase 1 diagnostic: does the LMU's W_query readout y discriminate hint-room
states from corridor states under an elliptical (Mahalanobis) metric?

This is the go/no-go gate for the E3B-on-y approach. It runs on a CONVERGED
S11 checkpoint (no training, no gradient) and computes:

  ratio_b = E[b_t | hint_room] / E[b_t | corridor]

where b_t = y_t^T M_{t-1} y_t is the E3B elliptical bonus, M_{t-1} is the
running inverse second-moment of y across the current episode, and
hint_room / corridor are defined by ground truth from the observation.

Ground-truth classifier (DEFAULT):
  hint_room : observation contains a ball or key (MiniGrid object IDs 5, 6)
  corridor  : observation contains NO ball and NO key
  excluded  : step 0 (y=0 by construction, since m=0 at episode start)

R_intr-based classifier (FALLBACK, used only on non-converged checkpoints):
  hint_room : r_intr > mean + 1.5*std AND step < max_hint_steps
  corridor  : all other non-step-0 steps

Decision table (from plan.md Phase 1.4):
    ratio_b > 3    : proceed with y as phi directly
    1.5 <= r <= 3  : proceed but add inverse-dynamics auxiliary loss on y
    ratio_b < 1.5  : stop, investigate y collapse

Usage:
    python scripts/diagnose_e3b.py \
        --checkpoint path/to/lmu_ppo_MemoryS11_s0.zip \
        --env MemoryS11 \
        --n_episodes 50 \
        --classifier ground_truth \
        --output results/e3b_diag_s11.json
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import gymnasium as gym
import minigrid  # noqa: F401  register MiniGrid envs
from gymnasium.wrappers import FilterObservation
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from lmu_ppo.lmu_ppo import LMUPPO
from mem_start import MemoryStartWrapper


# ──────────────────────────────────────────────────────────────────────────
# Env construction
# ──────────────────────────────────────────────────────────────────────────

ENV_IDS = {
    "MemoryS5":  "MiniGrid-MemoryS5-v0",
    "MemoryS7":  "MiniGrid-MemoryS7-v0",
    "MemoryS9":  "MiniGrid-MemoryS9-v0",
    "MemoryS11": "MiniGrid-MemoryS11-v0",
    "MemoryS13": "MiniGrid-MemoryS13-v0",
}

# MiniGrid object IDs (from minigrid.core.constants.OBJECT_TO_IDX).
# The MiniGrid-Memory family uses ball (6) or key (5) as the hint object
# and the two choices at the junction.
BALL_IDX = 6
KEY_IDX  = 5
HINT_OBJECT_IDS = {BALL_IDX, KEY_IDX}


def make_env(env_id: str, seed: int, use_wrapper: bool):
    """Single-env factory. use_wrapper=True matches training conditions."""
    def _init():
        env = gym.make(env_id)
        if use_wrapper:
            env = MemoryStartWrapper(env)
        env = FilterObservation(env, filter_keys=["image", "direction"])
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


# ──────────────────────────────────────────────────────────────────────────
# E3B elliptical bonus (single-env version for diagnostic)
# ──────────────────────────────────────────────────────────────────────────

class E3BBuffer:
    """
    Single-env elliptical bonus buffer.
    Maintains M = Lambda^{-1} in R^{CxC}, reset to (1/lambda) I at each
    episode start. Updated via Sherman-Morrison.
    """
    def __init__(self, dim: int, lambda_reg: float, device: torch.device):
        self.dim = dim
        self.lam = lambda_reg
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.M = torch.eye(self.dim, device=self.device) / self.lam

    def bonus_and_update(self, phi: torch.Tensor) -> float:
        """
        phi: (C,) float tensor.
        Returns: scalar bonus computed with M_{t-1} (before update).
        Side effect: updates self.M to M_t via Sherman-Morrison.
        """
        Mphi = self.M @ phi                    # (C,)
        bonus = float(phi @ Mphi)              # scalar
        denom = 1.0 + bonus                    # scalar, always >= 1
        outer = Mphi.unsqueeze(-1) * Mphi.unsqueeze(-2)  # (C, C)
        self.M = self.M - outer / denom
        return bonus


# ──────────────────────────────────────────────────────────────────────────
# y extraction (readout of the LMU)
# ──────────────────────────────────────────────────────────────────────────

def compute_y(policy, h_prev: torch.Tensor, m_new: torch.Tensor) -> torch.Tensor:
    """
    y_t = <C_t, m_t> where C_t = normalize(W_query(h_{t-1})).

    At t=0 of an episode, h_prev=0 and m_new is the write from step 0 only.
    W_query(0) = 0 and F.normalize(0) = 0 numerically, so y_0 = 0.
    That's why we exclude step 0 from both hint and corridor buckets.

    h_prev: (1, hidden)   — the hidden state BEFORE this step
    m_new:  (1, d, C)     — the memory state AFTER this step's write
    """
    lmu = policy.lmu_cell
    C_t = F.normalize(lmu.W_query(h_prev), dim=-1)         # (1, d)
    y = torch.einsum('bd,bdc->bc', C_t, m_new).squeeze(0)  # (C,)
    return y


def observation_has_hint(obs_image: np.ndarray) -> bool:
    """
    Ground truth: does the agent's egocentric view contain a hint object?

    MiniGrid obs structure (after VecTransposeImage): (C=3, H, W)
      channel 0: object IDs
      channel 1: color
      channel 2: state
    """
    obj_channel = obs_image[0]
    return bool(np.isin(obj_channel, list(HINT_OBJECT_IDS)).any())


def count_hint_objects_visible(obs_image: np.ndarray) -> int:
    """
    Number of hint-object cells visible. At the hint room, typically 1.
    At the decision junction, the agent sees BOTH the ball and key — so 2.
    Separates hint-room (== 1) from decision-junction (>= 2).
    """
    obj_channel = obs_image[0]
    n_balls = int((obj_channel == BALL_IDX).sum())
    n_keys  = int((obj_channel == KEY_IDX).sum())
    return n_balls + n_keys


# ──────────────────────────────────────────────────────────────────────────
# Main rollout loop
# ──────────────────────────────────────────────────────────────────────────

def rollout_and_record(
    model: LMUPPO,
    env,
    n_episodes: int,
    lambda_reg: float,
    device: torch.device,
) -> List[Dict]:
    policy = model.policy
    policy.set_training_mode(False)

    obs0 = env.reset()
    C = policy.lmu_cell.input_size  # readout dimension (= encoder_dim)

    e3b = E3BBuffer(dim=C, lambda_reg=lambda_reg, device=device)
    episodes: List[Dict] = []

    for ep_idx in range(n_episodes):
        if ep_idx > 0:
            obs0 = env.reset()

        h, m = policy.initial_state(n_envs=1, device=device)
        e3b.reset()

        per_ep = defaultdict(list)
        last_obs = obs0
        last_episode_start = np.array([True])

        for t in range(1000):
            with torch.no_grad():
                obs_t = {k: torch.as_tensor(v, device=device)
                         for k, v in last_obs.items()}

                # policy.forward returns 10 values; discard gate/innov/u_x
                action, _, _, h_new, m_new, _, r_intr, _, _, _ = policy.forward(
                    obs_t, h, m
                )

                y_t = compute_y(policy, h, m_new)  # (C,)
                b_t = e3b.bonus_and_update(y_t)

                r_masked = 0.0 if last_episode_start[0] else float(r_intr.item())
                b_masked = 0.0 if last_episode_start[0] else float(b_t)

            action_np = action.cpu().numpy()
            new_obs, reward, done, info = env.step(action_np)

            obs_img = last_obs['image'][0].copy()  # (C, H, W)

            per_ep['step'].append(t)
            per_ep['r_intr'].append(r_masked)
            per_ep['b'].append(b_masked)
            per_ep['y'].append(y_t.cpu().numpy().tolist())
            per_ep['obs_image'].append(obs_img)
            per_ep['hint_visible'].append(observation_has_hint(obs_img))
            per_ep['n_hint_objects'].append(count_hint_objects_visible(obs_img))
            per_ep['done'].append(bool(done[0]))
            per_ep['ext_reward'].append(float(reward[0]))

            h = h_new
            m = m_new
            last_obs = new_obs
            last_episode_start = done

            if done[0]:
                break

        episodes.append(dict(per_ep))
        if (ep_idx + 1) % 10 == 0:
            n_hint_steps = sum(per_ep['hint_visible'])
            print(f"  rolled out {ep_idx + 1}/{n_episodes} episodes "
                  f"(last len={len(per_ep['step'])}, "
                  f"hint_visible_steps={n_hint_steps}, "
                  f"ext_sum={sum(per_ep['ext_reward']):.2f})")

    return episodes


# ──────────────────────────────────────────────────────────────────────────
# Classifiers
# ──────────────────────────────────────────────────────────────────────────

def classify_ground_truth(
    episodes: List[Dict],
    exclude_ambiguous: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    hint_room  : exactly 1 hint object visible
    corridor   : 0 hint objects visible
    excluded   : step 0, OR 2+ hint objects visible (decision junction)
    """
    hint_masks, corridor_masks = [], []
    for ep in episodes:
        n = len(ep['step'])
        if n == 0:
            hint_masks.append(np.array([], dtype=bool))
            corridor_masks.append(np.array([], dtype=bool))
            continue

        n_objects = np.asarray(ep['n_hint_objects'])
        hint_vis  = np.asarray(ep['hint_visible'])

        hint = hint_vis & (n_objects == 1)
        corridor = ~hint_vis

        if n > 0:
            hint[0] = False
            corridor[0] = False

        hint_masks.append(hint)
        corridor_masks.append(corridor)

    return hint_masks, corridor_masks


def classify_r_intr_proxy(
    episodes: List[Dict],
    z_threshold: float = 1.5,
    max_steps_for_hint: int = 10,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Legacy r_intr-based classifier. Fails on converged checkpoints."""
    hint_masks, corridor_masks = [], []
    for ep in episodes:
        n = len(ep['step'])
        if n == 0:
            hint_masks.append(np.array([], dtype=bool))
            corridor_masks.append(np.array([], dtype=bool))
            continue

        r = np.asarray(ep['r_intr'])
        mu, sigma = r.mean(), r.std() + 1e-8
        novelty_mask = r > (mu + z_threshold * sigma)
        step_mask = np.asarray(ep['step']) < max_steps_for_hint
        hint = novelty_mask & step_mask

        corridor = ~hint
        if n > 0:
            hint[0] = False
            corridor[0] = False

        hint_masks.append(hint)
        corridor_masks.append(corridor)

    return hint_masks, corridor_masks


# ──────────────────────────────────────────────────────────────────────────
# Discrimination analysis
# ──────────────────────────────────────────────────────────────────────────

def compute_discrimination_metrics(
    episodes: List[Dict],
    hint_masks: List[np.ndarray],
    corridor_masks: List[np.ndarray],
) -> Dict:
    b_hint_all, b_corridor_all = [], []
    r_hint_all, r_corridor_all = [], []

    for ep, h_mask, c_mask in zip(episodes, hint_masks, corridor_masks):
        if len(h_mask) == 0:
            continue
        if h_mask.sum() == 0 and c_mask.sum() == 0:
            continue
        b = np.asarray(ep['b'])
        r = np.asarray(ep['r_intr'])

        if h_mask.sum() > 0:
            b_hint_all.append(b[h_mask])
            r_hint_all.append(r[h_mask])
        if c_mask.sum() > 0:
            b_corridor_all.append(b[c_mask])
            r_corridor_all.append(r[c_mask])

    if not b_hint_all:
        return {
            'status': 'no_hint_detected',
            'note': 'Classifier found no hint-room steps.',
        }
    if not b_corridor_all:
        return {
            'status': 'no_corridor_detected',
            'note': 'Classifier found no corridor steps. With wrapper, '
                    'episodes may be too short for a pure-corridor phase.',
        }

    b_hint = np.concatenate(b_hint_all)
    b_corridor = np.concatenate(b_corridor_all)
    r_hint = np.concatenate(r_hint_all)
    r_corridor = np.concatenate(r_corridor_all)

    eps = 1e-8
    ratio_b = b_hint.mean() / (b_corridor.mean() + eps)
    ratio_r = r_hint.mean() / (r_corridor.mean() + eps)

    if ratio_b > 3.0:
        decision = 'PROCEED_with_y_as_phi'
    elif ratio_b > 1.5:
        decision = 'PROCEED_with_inverse_dynamics_aux'
    else:
        decision = 'STOP_investigate_y_collapse'

    return {
        'status': 'ok',
        'n_hint_steps': int(b_hint.size),
        'n_corridor_steps': int(b_corridor.size),
        'b_hint_mean': float(b_hint.mean()),
        'b_hint_std': float(b_hint.std()),
        'b_corridor_mean': float(b_corridor.mean()),
        'b_corridor_std': float(b_corridor.std()),
        'ratio_b': float(ratio_b),
        'r_intr_hint_mean': float(r_hint.mean()),
        'r_intr_corridor_mean': float(r_corridor.mean()),
        'ratio_r': float(ratio_r),
        'decision': decision,
    }


def compute_episode_profile(episodes: List[Dict]) -> Dict:
    b_start, b_mid, b_end = [], [], []
    b_start_skip0, b_mid_skip0, b_end_skip0 = [], [], []

    for ep in episodes:
        b = np.asarray(ep['b'])
        T = len(b)
        if T < 3:
            continue
        t1, t2 = T // 3, 2 * T // 3
        b_start.append(b[:t1].mean())
        b_mid.append(b[t1:t2].mean())
        b_end.append(b[t2:].mean())

        if T > 3 and t1 >= 2:
            b_start_skip0.append(b[1:t1].mean())
            b_mid_skip0.append(b[t1:t2].mean())
            b_end_skip0.append(b[t2:].mean())

    return {
        'b_start_third_mean': float(np.mean(b_start)) if b_start else 0.0,
        'b_middle_third_mean': float(np.mean(b_mid)) if b_mid else 0.0,
        'b_end_third_mean': float(np.mean(b_end)) if b_end else 0.0,
        'b_start_third_mean_skip0':
            float(np.mean(b_start_skip0)) if b_start_skip0 else 0.0,
        'b_middle_third_mean_skip0':
            float(np.mean(b_mid_skip0)) if b_mid_skip0 else 0.0,
        'b_end_third_mean_skip0':
            float(np.mean(b_end_skip0)) if b_end_skip0 else 0.0,
    }


def compute_hint_visibility_stats(episodes: List[Dict]) -> Dict:
    hint_step_ranges = []
    n_hint_steps_per_ep = []
    for ep in episodes:
        hv = np.asarray(ep['hint_visible'])
        if hv.any():
            idx = np.where(hv)[0]
            hint_step_ranges.append((int(idx.min()), int(idx.max())))
            n_hint_steps_per_ep.append(int(hv.sum()))
        else:
            n_hint_steps_per_ep.append(0)

    if not hint_step_ranges:
        return {'status': 'hint_never_visible'}

    first_steps = [r[0] for r in hint_step_ranges]
    last_steps  = [r[1] for r in hint_step_ranges]
    return {
        'n_episodes_with_hint': len(hint_step_ranges),
        'hint_first_step_mean': float(np.mean(first_steps)),
        'hint_last_step_mean':  float(np.mean(last_steps)),
        'hint_steps_per_episode_mean': float(np.mean(n_hint_steps_per_ep)),
    }


def compute_aliasing_check(episodes: List[Dict], step_idx: int = 30) -> Dict:
    ys_at_step = []
    for ep in episodes:
        y_arr = np.asarray(ep['y'])
        if y_arr.shape[0] > step_idx:
            ys_at_step.append(y_arr[step_idx])
    if len(ys_at_step) < 2:
        return {'status': 'insufficient_episode_length',
                'n_samples': len(ys_at_step),
                'note': f'Need > 2 episodes reaching step {step_idx}.'}

    Y = np.stack(ys_at_step, axis=0)
    norms = np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8
    Yn = Y / norms
    cos_matrix = Yn @ Yn.T
    mask = ~np.eye(len(Y), dtype=bool)
    cos_pairs = cos_matrix[mask]

    return {
        'step_idx': step_idx,
        'n_samples': int(len(Y)),
        'cos_mean': float(cos_pairs.mean()),
        'cos_median': float(np.median(cos_pairs)),
        'cos_p95': float(np.quantile(cos_pairs, 0.95)),
        'cos_min': float(cos_pairs.min()),
        'warning': 'y may be collapsed' if cos_pairs.mean() > 0.95 else 'ok',
    }


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="Path to saved LMUPPO .zip checkpoint")
    parser.add_argument("--env", default="MemoryS11", choices=list(ENV_IDS))
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=9999)
    parser.add_argument("--use_wrapper", action='store_true',
                        help="Include MemoryStartWrapper (matches training).")
    parser.add_argument("--classifier", default="ground_truth",
                        choices=["ground_truth", "r_intr"],
                        help="ground_truth uses ball/key visibility. "
                             "r_intr is the legacy proxy.")
    parser.add_argument("--success_only", action='store_true',
                        help="Restrict analysis to episodes where the agent "
                             "succeeded (ext_reward > 0.5). Strips failure-mode "
                             "artifacts when running wrapper-trained policy "
                             "without wrapper.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    parser.add_argument("--z_threshold", type=float, default=1.5)
    parser.add_argument("--max_hint_steps", type=int, default=10)
    parser.add_argument("--aliasing_step_idx", type=int, default=30)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )

    env = VecTransposeImage(DummyVecEnv([
        make_env(ENV_IDS[args.env], args.seed, use_wrapper=args.use_wrapper)
    ]))

    print(f"Loading checkpoint: {args.checkpoint}")
    model = LMUPPO.load(args.checkpoint, env=env, device=device)
    print(f"  encoder_dim={model.policy.lmu_cell.input_size}  "
          f"memory_size={model.policy.lmu_cell.memory_size}  "
          f"hidden_size={model.policy.lmu_cell.hidden_size}")

    ortho_err = model.policy.lmu_cell.W_pre.orthogonality_error()
    print(f"  W_pre ortho_err = {ortho_err:.2e} "
          f"{'(ok)' if ortho_err < 1e-3 else '(WARNING: drifted)'}")

    print(f"\nRolling out {args.n_episodes} episodes "
          f"(wrapper={args.use_wrapper}, classifier={args.classifier})...")
    episodes = rollout_and_record(
        model, env,
        n_episodes=args.n_episodes,
        lambda_reg=args.lambda_reg,
        device=device,
    )

    episodes = [e for e in episodes if len(e.get('step', [])) > 0]
    print(f"\nCollected {len(episodes)} non-empty episodes.")
    lens = [len(e['step']) for e in episodes]
    rewards = [sum(e['ext_reward']) for e in episodes]
    print(f"  Episode lengths: mean={np.mean(lens):.1f} min={min(lens)} max={max(lens)}")
    print(f"  Extrinsic return: mean={np.mean(rewards):.3f} "
          f"success_rate={np.mean([r > 0.5 for r in rewards]):.2%}")

    if args.success_only:
        before = len(episodes)
        episodes = [e for e in episodes if sum(e['ext_reward']) > 0.5]
        print(f"\n[--success_only] Filtered: {before} → {len(episodes)} episodes")
        if len(episodes) == 0:
            print("  No successful episodes — cannot continue analysis.")
            return
        lens_succ = [len(e['step']) for e in episodes]
        print(f"  Successful episode lengths: "
              f"mean={np.mean(lens_succ):.1f} min={min(lens_succ)} max={max(lens_succ)}")

    vis_stats = compute_hint_visibility_stats(episodes)
    print("\n── Hint visibility (ground truth from observation) ───────")
    for k, v in vis_stats.items():
        print(f"  {k}: {v}")

    print(f"\nClassifying with '{args.classifier}'...")
    if args.classifier == "ground_truth":
        hint_masks, corridor_masks = classify_ground_truth(episodes)
    else:
        hint_masks, corridor_masks = classify_r_intr_proxy(
            episodes,
            z_threshold=args.z_threshold,
            max_steps_for_hint=args.max_hint_steps,
        )

    metrics = compute_discrimination_metrics(episodes, hint_masks, corridor_masks)
    print("\n── E3B discrimination (go/no-go) ─────────────────────────")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    profile = compute_episode_profile(episodes)
    print("\n── Episode profile (bonus by episode third) ──────────────")
    for k, v in profile.items():
        print(f"  {k}: {v:.4f}")

    aliasing = compute_aliasing_check(episodes, step_idx=args.aliasing_step_idx)
    print("\n── Aliasing check (late-episode y collapse) ──────────────")
    for k, v in aliasing.items():
        print(f"  {k}: {v}")

    result = {
        'checkpoint': args.checkpoint,
        'env': args.env,
        'n_episodes': len(episodes),
        'lambda_reg': args.lambda_reg,
        'classifier': args.classifier,
        'use_wrapper': args.use_wrapper,
        'success_only': args.success_only,
        'W_pre_ortho_err': ortho_err,
        'episode_lengths': {
            'mean': float(np.mean(lens)),
            'min': int(min(lens)),
            'max': int(max(lens)),
        },
        'success_rate': float(np.mean([r > 0.5 for r in rewards])),
        'hint_visibility': vis_stats,
        'discrimination': metrics,
        'profile': profile,
        'aliasing': aliasing,
    }

    if args.output:
        Path(os.path.dirname(args.output) or '.').mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote: {args.output}")

    if metrics.get('status') == 'ok':
        print("\n" + "=" * 60)
        print(f"DECISION: {metrics['decision']}")
        print(f"  ratio_b = {metrics['ratio_b']:.3f}  "
              f"(b_hint={metrics['b_hint_mean']:.3f}, "
              f"b_corridor={metrics['b_corridor_mean']:.3f})")
        print(f"  ratio_r = {metrics['ratio_r']:.3f}  (r_intr comparison)")
        print(f"  n_hint={metrics['n_hint_steps']}  "
              f"n_corridor={metrics['n_corridor_steps']}")
        print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()


