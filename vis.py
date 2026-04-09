"""
Phase 1 Surprise Signal Validation — Offline AUROC Analysis
============================================================

Loads per-rollout .npz files written by LMUPPO.collect_rollouts and computes:
  - AUROC per signal (S1/S3/S4/S5) vs ground-truth ball-visibility events
  - Pearson correlation with advantage_t (decision-relevance, not just event detection)
  - Early vs late training split (0–500K steps vs 1.5M–2M steps)
  - Per-signal calibration plot (precision-recall curve)

Ground truth:
  ball_visible (B, n_envs) — 1 when BALL (type 6) is in the agent's partial view.
  In MiniGrid-MemoryS11 this fires at t≈0 of each episode (the signal event).

Usage:
    python scripts/analyze_surprise_auroc.py --log_dir runs/surprise_logs/my_run
    python scripts/analyze_surprise_auroc.py --log_dir runs/surprise_logs/my_run \\
        --early_cutoff 500000 --late_start 1500000 --plot
"""

import argparse
import glob
import os
import sys

import numpy as np

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    from scipy.stats import pearsonr
except ImportError:
    sys.exit(
        "sklearn and scipy are required: pip install scikit-learn scipy"
    )

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


SIGNALS = ["s1", "s3", "s4", "s5"]
SIGNAL_LABELS = {
    "s1": "S1: LMU info-gain  ||B·u_t||²",
    "s3": "S3: Legendre delta  ||Δm_t||²/||m_{t-1}||²",
    "s4": "S4: Policy KL  KL(π_t||π_{t-1})",
    "s5": "S5: Value delta  |V_t - V_{t-1}|",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_npz_dir(log_dir: str):
    """
    Load all rollout .npz files from log_dir, sorted by filename (= time order).
    Returns dict of arrays concatenated across all rollouts.
    """
    files = sorted(glob.glob(os.path.join(log_dir, "rollout_*.npz")))
    if not files:
        sys.exit(f"No rollout_*.npz files found in {log_dir}")
    print(f"Loading {len(files)} rollout files from {log_dir}...")

    all_s1, all_s3, all_s4, all_s5, all_ball, all_ts = [], [], [], [], [], []

    for f in files:
        d = np.load(f)
        t0 = int(d["timestep"])
        T  = len(d["s1"])
        # Assign timestep to each step within the rollout (approx — ignores n_envs offset)
        ts = np.arange(t0, t0 + T, dtype=np.int64)

        all_s1.append(d["s1"])
        all_s3.append(d["s3"])
        all_s4.append(d["s4"])
        all_s5.append(d["s5"])
        all_ts.append(ts)

        # ball_visible shape: (T, n_envs) — flatten env dim, take OR across envs
        ball = d["ball_visible"]
        if ball.ndim == 2:
            ball = ball.max(axis=-1)   # (T,) — 1 if any env sees ball
        all_ball.append(ball)

    return {
        "s1":   np.concatenate(all_s1),
        "s3":   np.concatenate(all_s3),
        "s4":   np.concatenate(all_s4),
        "s5":   np.concatenate(all_s5),
        "ball": np.concatenate(all_ball),
        "ts":   np.concatenate(all_ts),
    }


# ---------------------------------------------------------------------------
# AUROC + Pearson computation
# ---------------------------------------------------------------------------

def compute_metrics(data: dict, label_key: str = "ball") -> dict:
    """
    For each signal, compute AUROC and Pearson r vs ground-truth labels.
    Returns dict of {signal: {auroc, pearson_r, pearson_p, ap}}.
    """
    labels = data[label_key].astype(int)
    if labels.sum() == 0:
        print(f"WARNING: no positive labels found for '{label_key}' — check ball detection.")
        return {}

    results = {}
    for sig in SIGNALS:
        scores = data[sig]
        # Handle NaN/inf from early steps
        valid = np.isfinite(scores)
        if valid.sum() < 100:
            print(f"  {sig}: too few valid steps ({valid.sum()}) — skipping")
            continue

        s_v = scores[valid]
        l_v = labels[valid]

        if l_v.sum() == 0 or (l_v == 0).sum() == 0:
            auroc = ap = float("nan")
        else:
            auroc = roc_auc_score(l_v, s_v)
            ap    = average_precision_score(l_v, s_v)

        r, p = pearsonr(s_v, l_v.astype(float))

        results[sig] = dict(auroc=auroc, ap=ap, pearson_r=r, pearson_p=p, n=valid.sum())

    return results


def print_results(results: dict, title: str = ""):
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    print(f"{'Signal':<8} {'AUROC':>8} {'Avg-Prec':>10} {'Pearson r':>10} {'p-value':>10}  {'n':>8}")
    print("-" * 60)
    for sig, m in results.items():
        flag = " ✓" if m["auroc"] > 0.70 and m["pearson_r"] > 0.15 else ""
        print(
            f"{sig:<8} {m['auroc']:>8.4f} {m['ap']:>10.4f} "
            f"{m['pearson_r']:>10.4f} {m['pearson_p']:>10.2e}  {m['n']:>8}{flag}"
        )
    print()
    print("Pass criterion: AUROC > 0.70  AND  Pearson r > 0.15  (✓)")


# ---------------------------------------------------------------------------
# Optional plot
# ---------------------------------------------------------------------------

def plot_signals(data: dict, results: dict, out_path: str = "surprise_signals.png"):
    if not HAS_MPL:
        print("matplotlib not installed — skipping plot")
        return

    ts = data["ts"]
    fig, axes = plt.subplots(len(SIGNALS) + 1, 1, figsize=(14, 3 * (len(SIGNALS) + 1)), sharex=True)

    # Ground truth
    axes[0].fill_between(ts, data["ball"], alpha=0.5, color="red", label="ball visible (GT)")
    axes[0].set_ylabel("GT event", fontsize=9)
    axes[0].legend(loc="upper right", fontsize=8)

    for i, sig in enumerate(SIGNALS):
        ax  = axes[i + 1]
        lbl = SIGNAL_LABELS[sig]
        m   = results.get(sig, {})
        extra = (
            f"  AUROC={m['auroc']:.3f}  r={m['pearson_r']:.3f}"
            if m else ""
        )
        ax.plot(ts, data[sig], lw=0.6, color=f"C{i}")
        ax.set_ylabel(sig, fontsize=9)
        ax.set_title(lbl + extra, fontsize=9, loc="left")

    axes[-1].set_xlabel("Training timestep")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir",      required=True, help="Directory with rollout_*.npz files")
    p.add_argument("--early_cutoff", type=int, default=500_000,
                   help="Upper timestep bound for 'early training' window")
    p.add_argument("--late_start",   type=int, default=1_500_000,
                   help="Lower timestep bound for 'late training' window")
    p.add_argument("--plot",         action="store_true", help="Save signal time-series plot")
    p.add_argument("--plot_out",     default="surprise_signals.png")
    return p.parse_args()


def main():
    args = parse_args()
    data = load_npz_dir(args.log_dir)
    ts   = data["ts"]

    print(f"\nTotal steps loaded : {len(ts):,}")
    print(f"Positive GT events : {data['ball'].sum():,.0f}  "
          f"({100*data['ball'].mean():.2f}% of steps)")

    # Full-run metrics
    res_all = compute_metrics(data)
    print_results(res_all, title="All timesteps")

    # Early training window
    mask_early = ts <= args.early_cutoff
    if mask_early.sum() > 1000:
        data_early = {k: v[mask_early] for k, v in data.items()}
        res_early  = compute_metrics(data_early)
        print_results(res_early, title=f"Early training  (steps ≤ {args.early_cutoff:,})")
    else:
        print(f"\n[skip early window — only {mask_early.sum()} steps]")

    # Late training window
    mask_late = ts >= args.late_start
    if mask_late.sum() > 1000:
        data_late = {k: v[mask_late] for k, v in data.items()}
        res_late  = compute_metrics(data_late)
        print_results(res_late, title=f"Late training  (steps ≥ {args.late_start:,})")
    else:
        print(f"\n[skip late window — only {mask_late.sum()} steps]")

    # Summary recommendation
    print("\n--- Signal ranking (full run, by AUROC) ---")
    ranked = sorted(
        [(s, m["auroc"]) for s, m in res_all.items() if np.isfinite(m["auroc"])],
        key=lambda x: -x[1],
    )
    for rank, (sig, auc) in enumerate(ranked, 1):
        print(f"  {rank}. {sig}: AUROC={auc:.4f}  {SIGNAL_LABELS[sig]}")

    if args.plot:
        plot_signals(data, res_all, args.plot_out)


if __name__ == "__main__":
    main()