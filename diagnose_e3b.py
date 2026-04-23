"""
scripts/diagnose_e3b.py

Phase 1 diagnostic: post-hoc episodic elliptical bonus (E3B) on a converged
LMU-PPO checkpoint.

Tests two things:
  1. Discrimination: does y_t = LMU dynamic readout distinguish hint_room steps
     from corridor steps? (ratio_b = b_hint / b_corridor, target > 3)
  2. Aliasing: does y at corridor step ~alias_step retain episode-specific
     information (which ball was in the hint room)? (cosine < 0.95 = good)

No training, no bonus in rewards. Pure diagnostic rollout.

Usage:
    python scripts/diagnose_e3b.py \\
        --checkpoint lmu_ppo_MemoryS11_s1 \\
        --env MemoryS11 \\
        --n_episodes 50 \\
        --ridge 0.1 \\
        --hint_steps 10 \\
        --alias_step 50 \\
        --out results/e3b_diag.pkl

Decision table (ratio_b):
    > 3:    Proceed to Phase 2 with y as φ.
    1.5–3:  Proceed to Phase 2 but add inverse-dynamics auxiliary loss on y.
    < 1.5:  STOP. Investigate y collapse (low-rank manifold, theta too short).

NOTE on m_norm ≈ 0 (your current run):
    If memory writes have collapsed, y ≈ C_t @ 0 ≈ 0 for all steps.
    In that case ratio_b will be near 1 regardless of hint structure —
    the diagnostic will correctly route you to the STOP branch and prompt
    investigation of the write gate / theta / encoder saturation before
    committing to Phase 2.
"""

import argparse
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
import minigrid  # noqa: F401
from gymnasium.wrappers import FilterObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.utils import obs_as_tensor

sys.path.insert(0, str(Path(__file__).parent.parent))
from lmu_ppo.lmu_ppo import LMUPPO
from mem_start import MemoryStartWrapper


# ── MiniGrid constants ────────────────────────────────────────────────────────
# OBJECT_TO_IDX: empty=1, wall=2, door=4, key=5, ball=6, box=7, goal=8
BALL_OBJ_IDX = 6
# COLOR_TO_IDX: red=0, green=1, blue=2, purple=3, yellow=4, grey=5
COLOR_NAMES  = {0: 'red', 1: 'green', 2: 'blue', 3: 'purple', 4: 'yellow', 5: 'grey'}

ENV_IDS = {
    "MemoryS7":  "MiniGrid-MemoryS7-v0",
    "MemoryS9":  "MiniGrid-MemoryS9-v0",
    "MemoryS11": "MiniGrid-MemoryS11-v0",
    "MemoryS13": "MiniGrid-MemoryS13-v0",
}


# ── Sherman-Morrison E3B ──────────────────────────────────────────────────────

def sm_init(d: int, ridge: float) -> np.ndarray:
    """
    M_0 = (λI)^{-1} = (1/λ) I_d.

    Initial bonus b_0 = φ^T M_0 φ = (1/λ) ‖φ‖².
    With normalize_phi=True, ‖φ‖ = 1 so b_0 = 1/λ for every episode.
    This is the correct high-novelty starting state — all features are new.
    """
    return np.eye(d, dtype=np.float64) / ridge


def sm_step(M: np.ndarray, phi: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute E3B bonus THEN update the inverse covariance.

    b_t  = φ^T M_{t-1} φ            (novelty; high if φ unseen this episode)
    M_t  = M_{t-1} - (M_{t-1}φ φ^T M_{t-1}) / (1 + φ^T M_{t-1}φ)   [SM rank-1]

    Returns (bonus: float, M_new: ndarray).

    Numerical note: denom = 1 + b_t ≥ 1 always (M is PSD, φ^T M φ ≥ 0),
    so no risk of division by zero or sign flip.
    """
    phi   = phi.astype(np.float64)
    Mphi  = M @ phi                          # (d,)
    bonus = float(phi @ Mphi)               # scalar, ≥ 0
    M_new = M - np.outer(Mphi, Mphi) / (1.0 + bonus)
    return bonus, M_new


# ── y extraction ─────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_y(policy, h_prev: torch.Tensor, m_new: torch.Tensor) -> np.ndarray:
    """
    Recompute LMU dynamic readout y from (h_prev, m_new).

    Mirrors exactly what LMUCell.forward computes internally:
        C_t = normalize(W_query(h_prev))         # (B, memory_size)
        y   = einsum('bd,bdc->bc', C_t, m_new)  # (B, encoder_dim)

    We call it externally using return values we already captured — no
    side-effect attributes, no race condition with EvalCallback.

    Returns: np.ndarray of shape (encoder_dim,), dtype float64.
    """
    C_t = F.normalize(policy.lmu_cell.W_query(h_prev), dim=-1)  # (1, d)
    y   = torch.einsum('bd,bdc->bc', C_t, m_new)                 # (1, C)
    return y.squeeze(0).cpu().numpy().astype(np.float64)


# ── hint detection ────────────────────────────────────────────────────────────

def detect_ball_color(obs_image: np.ndarray) -> int:
    """
    Detect hint ball color from a (3, H, W) image observation.

    Channel 0 = object type, channel 1 = color, channel 2 = state.
    VecTransposeImage has already moved channels first.

    Returns: color index ∈ [0, 5], or -1 if no ball found.
    """
    obj_mask = obs_image[0] == BALL_OBJ_IDX   # (H, W)
    if not obj_mask.any():
        return -1
    colors = obs_image[1][obj_mask]           # colors under ball pixels
    return int(np.bincount(colors).argmax())


# ── data containers ───────────────────────────────────────────────────────────

@dataclass
class StepRecord:
    step:         int
    r_intr:       float
    b_t:          float   # E3B bonus (before SM update for this step)
    y_t:          np.ndarray
    is_hint_room: bool    # step < hint_steps


@dataclass
class EpisodeData:
    hint_color:   int            # -1 if not detected
    steps:        List[StepRecord] = field(default_factory=list)
    total_reward: float = 0.0

    @property
    def length(self) -> int:
        return len(self.steps)


# ── main rollout ──────────────────────────────────────────────────────────────

def run_episodes(
    model:       LMUPPO,
    env,
    n_episodes:  int,
    hint_steps:  int,
    ridge:       float,
    normalize_phi: bool = True,
) -> List[EpisodeData]:
    """
    Roll out the policy for n_episodes with no training and no bonus in rewards.

    normalize_phi: L2-normalise y before the SM update.
        True  → b_t ∈ [0, 1/λ], directly comparable across episodes.
        False → b_t also reflects y magnitude; informative if m_norm is near 0
                (y ≈ 0 → b_t ≈ 0 for all steps → ratio_b ≈ 1 regardless of hint).
    """
    policy = model.policy
    policy.set_training_mode(False)
    device = model.device
    d      = model.encoder_dim   # y dimension

    episodes:   List[EpisodeData] = []
    obs         = env.reset()
    h, m        = policy.initial_state(1, device)
    M           = sm_init(d, ridge)
    step_in_ep  = 0
    ep          = EpisodeData(hint_color=detect_ball_color(obs['image'][0]))

    while len(episodes) < n_episodes:
        obs_t  = obs_as_tensor(obs, device)
        h_prev = h.clone()   # capture BEFORE forward — needed for y computation

        with torch.no_grad():
            actions, values, log_probs, h, m, logits, r_intr, gate, innov, u_x = \
                policy.forward(obs_t, h, m)
            # Extract y immediately, before any callback or env step can
            # trigger another policy.forward call with a different batch size.
            y_t = extract_y(policy, h_prev, m)   # (d,) — m is now m_new

        phi = y_t / (np.linalg.norm(y_t) + 1e-8) if normalize_phi else y_t
        b_t, M = sm_step(M, phi)

        ep.steps.append(StepRecord(
            step         = step_in_ep,
            r_intr       = float(r_intr[0].item()),
            b_t          = b_t,
            y_t          = y_t.copy(),
            is_hint_room = step_in_ep < hint_steps,
        ))

        obs, rewards, dones, infos = env.step(actions.cpu().numpy())
        ep.total_reward += float(rewards[0])
        step_in_ep += 1

        if dones[0]:
            episodes.append(ep)
            n = len(episodes)
            if n % 10 == 0 or n == n_episodes:
                succ = ep.total_reward > 0
                color_name = COLOR_NAMES.get(ep.hint_color, '?')
                print(f"  ep {n:3d}/{n_episodes}  len={ep.length:4d}  "
                      f"ret={ep.total_reward:.2f}  "
                      f"hint={color_name}  "
                      f"{'✓' if succ else '✗'}")

            if len(episodes) < n_episodes:
                h, m        = policy.initial_state(1, device)
                M           = sm_init(d, ridge)
                step_in_ep  = 0
                ep          = EpisodeData(hint_color=detect_ball_color(obs['image'][0]))

    return episodes


# ── discrimination metrics ────────────────────────────────────────────────────

def compute_discrimination(episodes: List[EpisodeData]) -> Dict:
    """
    Classify each step as hint_room (step < hint_steps) or corridor, then
    compute the ratio_b and ratio_r primary metrics.

    b_start: first 3 steps — should be the HIGHEST (most novel, episode just started)
    b_late:  last 20% of episode — should approach 0 (all y seen before)
    """
    b_hint, b_corr   = [], []
    r_hint, r_corr   = [], []
    b_start, b_late  = [], []
    y_norms          = []   # track if y is near-zero (m_norm collapse indicator)

    for ep in episodes:
        T          = ep.length
        late_start = max(1, int(0.8 * T))

        for s in ep.steps:
            y_norms.append(float(np.linalg.norm(s.y_t)))

            if s.is_hint_room:
                b_hint.append(s.b_t);  r_hint.append(s.r_intr)
            else:
                b_corr.append(s.b_t);  r_corr.append(s.r_intr)

            if s.step < 3:
                b_start.append(s.b_t)
            if s.step >= late_start:
                b_late.append(s.b_t)

    def _stats(xs):
        return (float(np.mean(xs)), float(np.std(xs))) if xs else (float('nan'), float('nan'))

    bh_mu, bh_sd = _stats(b_hint)
    bc_mu, bc_sd = _stats(b_corr)
    ratio_b = bh_mu / bc_mu if bc_mu > 0 else float('inf')
    ratio_r = (np.mean(r_hint) / np.mean(r_corr)) if r_corr and np.mean(r_corr) > 0 else float('inf')

    return {
        'b_hint_mean':      bh_mu,   'b_hint_std':      bh_sd,
        'b_corridor_mean':  bc_mu,   'b_corridor_std':  bc_sd,
        'b_start_mean':     _stats(b_start)[0],
        'b_late_mean':      _stats(b_late)[0],
        'r_hint_mean':      float(np.mean(r_hint))  if r_hint  else float('nan'),
        'r_corridor_mean':  float(np.mean(r_corr))  if r_corr  else float('nan'),
        'ratio_b':          ratio_b,
        'ratio_r':          float(ratio_r),
        'y_norm_mean':      float(np.mean(y_norms)),   # near 0 → memory collapsed
        'y_norm_std':       float(np.std(y_norms)),
        'n_hint_steps':     len(b_hint),
        'n_corridor_steps': len(b_corr),
    }


# ── aliasing check ────────────────────────────────────────────────────────────

def compute_aliasing(episodes: List[EpisodeData], alias_step: int) -> Dict:
    """
    1.5: Aliasing check at corridor step alias_step.

    For episodes long enough to reach alias_step:
      - Extract y at that step
      - Compute pairwise cosine similarity
      - Compute Mahalanobis distance using pool covariance
      - If hint colors known: compare within-color vs cross-color cosine

    Interpretation:
      mean_cosine > 0.95  → all y's look the same regardless of hint →
                             hint info has decayed (theta too short, or m_norm ≈ 0)
      mean_cosine < 0.95  → y retains episode-specific structure → good
      color_discrim > 0   → within-hint-color y's are MORE similar than
                             cross-color y's → memory encodes the hint → great
    """
    Y_list, colors = [], []

    for ep in episodes:
        if ep.length > alias_step:
            Y_list.append(ep.steps[alias_step].y_t)
            colors.append(ep.hint_color)

    N = len(Y_list)
    if N < 2:
        return {'error': f'Only {N} episode(s) reached step {alias_step}. '
                         f'Try a smaller --alias_step.'}

    Y = np.stack(Y_list, axis=0)   # (N, d)

    # ── cosine similarity ─────────────────────────────────────────────────────
    Y_n  = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
    C_mat = Y_n @ Y_n.T           # (N, N)
    mask  = ~np.eye(N, dtype=bool)
    mean_cos = float(C_mat[mask].mean())
    max_cos  = float(C_mat[mask].max())

    # ── Mahalanobis ───────────────────────────────────────────────────────────
    # Use pinv for robustness when d > N (likely here: d=64, N≤50)
    cov     = np.cov(Y.T) + 1e-6 * np.eye(Y.shape[1])
    cov_inv = np.linalg.pinv(cov)
    # Sample up to 10×10 pairs to keep computation cheap
    idx   = np.arange(min(N, 10))
    dists = []
    for i in idx:
        for j in idx:
            if i >= j:
                continue
            d_vec = Y[i] - Y[j]
            dists.append(float(np.sqrt(max(0, d_vec @ cov_inv @ d_vec))))
    mean_mahal = float(np.mean(dists)) if dists else float('nan')

    result = {
        'alias_step':               alias_step,
        'n_episodes':               N,
        'mean_cosine':              mean_cos,
        'max_cosine':               max_cos,
        'mean_mahalanobis':         mean_mahal,
        'y_mean_norm_at_step':      float(np.linalg.norm(Y, axis=1).mean()),
    }

    # ── per-color breakdown ───────────────────────────────────────────────────
    known   = [(i, c) for i, c in enumerate(colors) if c >= 0]
    if len(known) >= 4 and len(set(c for _, c in known)) >= 2:
        within, cross = [], []
        for a, (i, ci) in enumerate(known):
            for b_, (j, cj) in enumerate(known):
                if b_ <= a:
                    continue
                (within if ci == cj else cross).append(C_mat[i, j])
        result['within_color_cosine']  = float(np.mean(within)) if within else float('nan')
        result['cross_color_cosine']   = float(np.mean(cross))  if cross  else float('nan')
        # Positive = within-color y's are closer = hint is encoded in y
        result['color_discrimination'] = (
            result['within_color_cosine'] - result['cross_color_cosine']
        )

    return result


# ── report ────────────────────────────────────────────────────────────────────

def print_report(disc: Dict, alias: Dict, n_episodes: int,
                 mean_ep_len: float, mean_ret: float):
    W = 62
    print("\n" + "="*W)
    print("  PHASE 1 — E3B DIAGNOSTIC REPORT")
    print("="*W)
    print(f"  Episodes: {n_episodes}  |  mean_len: {mean_ep_len:.1f}  |  "
          f"mean_ret: {mean_ret:.3f}")
    print()

    print("[Discrimination]")
    print(f"  b_hint_mean     = {disc['b_hint_mean']:8.4f}  ± {disc['b_hint_std']:.4f}  "
          f"(n={disc['n_hint_steps']})")
    print(f"  b_corridor_mean = {disc['b_corridor_mean']:8.4f}  ± {disc['b_corridor_std']:.4f}  "
          f"(n={disc['n_corridor_steps']})")
    print(f"  b_start_mean    = {disc['b_start_mean']:8.4f}  (first 3 steps; should be highest)")
    print(f"  b_late_mean     = {disc['b_late_mean']:8.4f}  (last 20%; should be near 0)")
    print()
    print(f"  ratio_b = {disc['ratio_b']:.3f}   ← PRIMARY METRIC (target > 3)")
    print(f"  ratio_r = {disc['ratio_r']:.3f}   ← r_intr ratio (comparison)")
    print(f"  y_norm_mean = {disc['y_norm_mean']:.4f} ± {disc['y_norm_std']:.4f}"
          f"  ← near 0 means memory has collapsed (m_norm issue)")
    print()

    ratio = disc['ratio_b']
    print("[Decision]")
    if ratio > 3.0:
        print("  ✓  ratio_b > 3")
        print("     → Proceed to Phase 2 with y as φ (no auxiliary loss needed).")
    elif ratio >= 1.5:
        print("  ⚠  1.5 ≤ ratio_b ≤ 3")
        print("     → Proceed to Phase 2 with inverse-dynamics auxiliary loss on y.")
        print("       This strengthens the episodic structure before E3B scaling.")
    else:
        print("  ✗  ratio_b < 1.5")
        print("     → STOP. y is not discriminative. Investigate:")
        if disc['y_norm_mean'] < 0.01:
            print("       • y_norm ≈ 0: memory has collapsed (m_norm issue from TB).")
            print("         Fix encoder saturation (u_x_zero_frac=0.72) first.")
            print("         Replace proj ReLU with ELU or add pre-LMU LayerNorm.")
        else:
            print("       • y has nonzero norm but no episodic structure.")
            print("         Consider: theta too short, W_query not learning, or")
            print("         e_m=0 init preventing meaningful reads from memory.")
    print()

    print("[Aliasing check]")
    if 'error' in alias:
        print(f"  ERROR: {alias['error']}")
    else:
        print(f"  step={alias['alias_step']}  n_episodes={alias['n_episodes']}")
        print(f"  y_norm at step   = {alias['y_mean_norm_at_step']:.4f}")
        print(f"  mean_cosine      = {alias['mean_cosine']:.4f}  "
              f"({'✗ HIGH — hint decayed' if alias['mean_cosine'] > 0.95 else '✓ OK'})")
        print(f"  max_cosine       = {alias['max_cosine']:.4f}")
        print(f"  mean_mahalanobis = {alias['mean_mahalanobis']:.4f}")
        if 'color_discrimination' in alias:
            cd = alias['color_discrimination']
            print(f"  within_color_cos = {alias['within_color_cosine']:.4f}")
            print(f"  cross_color_cos  = {alias['cross_color_cosine']:.4f}")
            print(f"  color_discrim    = {cd:.4f}  "
                  f"({'✓ hint retained in y' if cd > 0 else '✗ hint NOT retained in y'})")
        if alias['mean_cosine'] > 0.95:
            print()
            print("  High cosine at corridor step → hint information has decayed.")
            print("  Likely causes: theta too short for episode length, or m_norm ≈ 0")
            print("  (write gate inactive → y is always the same low-rank output).")
    print("="*W + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Phase 1: post-hoc E3B diagnostic on a converged LMU-PPO checkpoint.'
    )
    p.add_argument('--checkpoint', required=True,
                   help='Path to saved LMUPPO model (without .zip)')
    p.add_argument('--env',        default='MemoryS11', choices=list(ENV_IDS))
    p.add_argument('--n_episodes', type=int,   default=50)
    p.add_argument('--ridge',      type=float, default=0.1,
                   help='Ridge λ for E3B inverse init (M_0 = (1/λ)·I)')
    p.add_argument('--hint_steps', type=int,   default=10,
                   help='Steps 0..hint_steps-1 classified as hint_room')
    p.add_argument('--alias_step', type=int,   default=50,
                   help='Corridor step used for aliasing check')
    p.add_argument('--no_normalize_phi', action='store_true',
                   help='Skip L2-normalising y before SM update '
                        '(useful to diagnose y magnitude collapse)')
    p.add_argument('--seed',   type=int, default=42)
    p.add_argument('--device', default='cpu',
                   help='Inference device (cpu recommended; diagnostics are offline)')
    p.add_argument('--out', default='results/e3b_diag.pkl',
                   help='Output path for full results dict (pickle)')
    return p.parse_args()


def make_env(env_id: str, seed: int):
    def _init():
        env = gym.make(env_id)
        env = MemoryStartWrapper(env)
        env = FilterObservation(env, filter_keys=['image', 'direction'])
        env.reset(seed=seed)
        return env
    return VecTransposeImage(DummyVecEnv([_init]))


def main():
    args = parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print(f"Env:        {ENV_IDS[args.env]}")
    print(f"Checkpoint: {args.checkpoint}")
    env = make_env(ENV_IDS[args.env], args.seed)

    model = LMUPPO.load(args.checkpoint, env=env, device=args.device)
    model.policy.set_training_mode(False)

    print(f"y_dim={model.encoder_dim}  "
          f"ridge={args.ridge}  "
          f"hint_steps={args.hint_steps}  "
          f"alias_step={args.alias_step}  "
          f"normalize_phi={not args.no_normalize_phi}")
    print(f"E3B matrix: ({model.encoder_dim}×{model.encoder_dim}) per episode\n")

    episodes = run_episodes(
        model, env,
        n_episodes    = args.n_episodes,
        hint_steps    = args.hint_steps,
        ridge         = args.ridge,
        normalize_phi = not args.no_normalize_phi,
    )

    disc  = compute_discrimination(episodes)
    alias = compute_aliasing(episodes, args.alias_step)

    ep_lens = [ep.length for ep in episodes]
    ep_rets = [ep.total_reward for ep in episodes]
    print_report(disc, alias, len(episodes),
                 mean_ep_len=float(np.mean(ep_lens)),
                 mean_ret=float(np.mean(ep_rets)))

    results = {
        'args':          vars(args),
        'disc':          disc,
        'alias':         alias,
        'ep_lengths':    np.array(ep_lens),
        'ep_returns':    np.array(ep_rets),
        'hint_colors':   np.array([ep.hint_color for ep in episodes]),
        # Raw per-step data for further analysis if needed
        'all_b':   np.array([s.b_t    for ep in episodes for s in ep.steps]),
        'all_rintr': np.array([s.r_intr for ep in episodes for s in ep.steps]),
        'all_step':  np.array([s.step   for ep in episodes for s in ep.steps]),
        'all_is_hint': np.array([s.is_hint_room for ep in episodes for s in ep.steps]),
    }

    with open(args.out, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved → {args.out}")

    env.close()


if __name__ == '__main__':
    main()