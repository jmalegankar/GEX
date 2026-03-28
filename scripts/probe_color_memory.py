"""
Linear probe: can we recover the episode signal (key vs ball) from slot memory?

In MemoryS7, the signal is an object (key=5 or ball=6) BEHIND the agent
at reset.  This script forces a turn-around to observe it, then runs
the full VAE→Wyner→SlotMemory pipeline with a random policy.

Probes (logistic regression) on:
  - μ (VAE latent) at each timestep
  - π (policy projection) at each timestep
  - Flattened slot contents
  - h (Wyner GRU hidden state)
  - Each slot individually

Reports accuracy at early/mid/late buckets to test whether the signal
persists in memory across the episode.

Usage:
    python scripts/probe_color_memory.py [--checkpoint PATH] [--n_episodes 200] [--device cpu]
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch as th
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import gymnasium
import minigrid

from envs.wrappers import MiniGridTrainingWrapper
from models.embeddings import CategoricalGridWithDirEmbedding
from models.config import SCVAEConfig
from models.vae import TransitionSCVAE
from models.wyner import WynerLBSVAE
from models.slot_memory import SlotMemory

# MiniGrid constants
SIGNAL_KEY = 5
SIGNAL_BALL = 6
TYPE_NAMES = {5: "key", 6: "ball"}
VIEW_SIZE = 5  # must match training
MU_DIM = 32
WYNER_LATENT_DIM = 64
NUM_SLOTS = 8


def extract_signal_type(obs: np.ndarray) -> int:
    """Extract signal object type from observation. Returns 5 (key), 6 (ball), or -1."""
    obj_types = obs[:, :, 0]
    for r in range(obs.shape[0]):
        for c in range(obs.shape[1]):
            if obj_types[r, c] in (SIGNAL_KEY, SIGNAL_BALL):
                return int(obj_types[r, c])
    return -1


def turn_around_and_get_signal(env):
    """
    Turn agent around, walk forward to see signal, turn back.
    Returns (signal_type, obs_after_returning, prev_action, all step observations).
    """
    signal_type = -1
    history = []  # (obs, action) pairs

    # Turn left twice
    for _ in range(2):
        obs, _, term, trunc, _ = env.step(0)
        history.append((obs, 0))
        if signal_type < 0:
            signal_type = extract_signal_type(obs)

    # Walk forward up to 3 steps to find signal
    for _ in range(3):
        obs, _, term, trunc, _ = env.step(2)
        history.append((obs, 2))
        if signal_type < 0:
            signal_type = extract_signal_type(obs)
        if term or trunc:
            break

    # Turn back (2 left turns)
    for _ in range(2):
        obs, _, term, trunc, _ = env.step(0)
        history.append((obs, 0))
        if term or trunc:
            break

    return signal_type, history


def build_models(device="cpu"):
    """Build VAE, Wyner, SlotMemory matching training config."""
    embedding = CategoricalGridWithDirEmbedding(
        n_object_types=12, n_colors=6, n_states=3,
        obs_h=VIEW_SIZE, obs_w=VIEW_SIZE,
        embed_per_channel=4, n_dirs=4, dir_embed_dim=4,
    )
    conv_channels = [32, 64, 128]
    cfg = SCVAEConfig(
        act_dim=7, action_embed_dim=32,
        conv_channels=conv_channels, hidden_dim=256, latent_dim=MU_DIM,
    )
    vae = TransitionSCVAE(embedding=embedding, cfg=cfg).to(device)

    recon_dim = 2 * conv_channels[-1] + 32
    wyner = WynerLBSVAE(
        recon_dim=recon_dim, mu_dim=MU_DIM, latent_dim=WYNER_LATENT_DIM,
        latent_tokens=1, decode_hidden=128, pos_embed_dim=16, free_bits=0.0,
    ).to(device)

    slot_mem = SlotMemory(
        num_slots=NUM_SLOTS, slot_dim=WYNER_LATENT_DIM,
        gate_mode="detached", gate_scale=3.0, gate_threshold=2.0,
    ).to(device)

    return vae, wyner, slot_mem


def collect_data(env, vae, wyner, slot_mem, device, n_episodes=200):
    """Run episodes with turn-around + random policy, collecting all representations."""
    records = []
    vae.eval()
    wyner.eval()
    skipped = 0

    for ep in range(n_episodes):
        obs_init, _ = env.reset()
        signal_type, turn_history = turn_around_and_get_signal(env)

        if signal_type < 0:
            skipped += 1
            continue

        # Init memory state
        h = th.zeros(1, 1, WYNER_LATENT_DIM, device=device)
        slots = slot_mem.init_state(1, device)

        # Process turn-around steps through the pipeline too
        # (so memory accumulates from the very start)
        prev_obs = obs_init.copy()
        prev_action = np.array([0], dtype=np.float32)  # null action

        for turn_obs, turn_act in turn_history:
            with th.no_grad():
                s_tm1 = th.tensor(prev_obs, dtype=th.float32, device=device).unsqueeze(0)
                a_tm1 = th.tensor(prev_action, dtype=th.float32, device=device).unsqueeze(0)
                s_t = th.tensor(turn_obs, dtype=th.float32, device=device).unsqueeze(0)

                mu, pi, skips = vae.encode(s_tm1, a_tm1, s_t)
                ts_tensor = th.tensor([0], device=device, dtype=th.long)
                h_new, _ = wyner.encode(h, mu.detach(), skips, timestep=ts_tensor)
                h_new_3d = h_new.unsqueeze(1)

                wyner_out = wyner.forward(h, mu, None, skips, timestep=ts_tensor, slots=slots)
                kl = wyner.loss(wyner_out).kl_loss.view(-1)
                new_slots, gate = slot_mem.write(slots, wyner_out.posterior_mu.detach(), kl)

            h = h_new_3d
            slots = new_slots
            prev_obs = turn_obs.copy()
            prev_action = np.array([turn_act], dtype=np.float32)

        # Now walk forward with random policy
        # Get current obs (agent is facing forward again)
        obs = turn_history[-1][0].copy()
        done = False
        t = 0

        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)

            with th.no_grad():
                s_tm1 = th.tensor(prev_obs, dtype=th.float32, device=device).unsqueeze(0)
                a_tm1 = th.tensor(prev_action, dtype=th.float32, device=device).unsqueeze(0)
                s_t = th.tensor(obs, dtype=th.float32, device=device).unsqueeze(0)

                mu, pi, skips = vae.encode(s_tm1, a_tm1, s_t)
                ts_tensor = th.tensor([t + len(turn_history)], device=device, dtype=th.long)
                h_new, _ = wyner.encode(h, mu.detach(), skips, timestep=ts_tensor)
                h_new_3d = h_new.unsqueeze(1)

                wyner_out = wyner.forward(h, mu, None, skips, timestep=ts_tensor, slots=slots)
                kl = wyner.loss(wyner_out).kl_loss.view(-1)
                new_slots, gate = slot_mem.write(slots, wyner_out.posterior_mu.detach(), kl)

            records.append({
                "timestep": t,
                "signal_type": signal_type,
                "episode_id": ep,
                "mu": mu.cpu().numpy().squeeze(0).copy(),
                "pi": pi.cpu().numpy().squeeze(0).copy(),
                "slots": new_slots.cpu().numpy().squeeze(0).copy(),
                "h": h_new.cpu().numpy().squeeze(0).copy(),
                "gate": gate.item(),
                "kl": kl.item(),
            })

            h = h_new_3d
            slots = new_slots
            prev_obs = obs.copy()
            prev_action = np.array([action], dtype=np.float32)
            obs = next_obs
            done = terminated or truncated
            t += 1

        if (ep + 1) % 25 == 0:
            print(f"  Ep {ep+1}/{n_episodes}, signal={TYPE_NAMES.get(signal_type, '?')}, len={t}")

    print(f"  Skipped {skipped}/{n_episodes} episodes (signal not found)")
    return records


def run_probe(records, feature_key, feature_name, n_classes):
    """Logistic regression probe, reporting accuracy by timestep bucket."""
    timesteps = np.array([r["timestep"] for r in records])
    labels = np.array([r["signal_type"] for r in records])
    max_t = max(timesteps.max(), 1)

    mid = max(max_t // 2, 3)
    buckets = {
        "early (t=0-2)": (0, 2),
        f"mid (t=3-{mid})": (3, mid),
        f"late (t={mid+1}-{max_t})": (mid + 1, max_t),
        "all": (0, max_t),
    }

    print(f"\n{'='*60}")
    print(f"  Probe: {feature_name} → signal type")
    print(f"  Random baseline: {1.0/n_classes:.1%}")
    print(f"{'='*60}")

    results = {}
    for name, (lo, hi) in buckets.items():
        mask = (timesteps >= lo) & (timesteps <= hi)
        if mask.sum() < 20:
            print(f"  {name}: too few samples ({mask.sum()})")
            continue

        if feature_key == "slots":
            X = np.array([r["slots"].ravel() for r in np.array(records)[mask]])
        else:
            X = np.array([r[feature_key] for r in np.array(records)[mask]])

        y = labels[mask]
        unique = np.unique(y)
        if len(unique) < 2:
            print(f"  {name}: only 1 class")
            continue

        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        scores = cross_val_score(clf, X, y, cv=min(5, mask.sum() // 10), scoring="accuracy")
        acc = scores.mean()
        results[name] = acc
        marker = " <<<" if acc > 1.0 / n_classes + 0.1 else ""
        print(f"  {name:25s}: {acc:.1%} ± {scores.std():.1%}  (n={mask.sum()}){marker}")

    return results


def per_slot_probe(records, n_classes):
    """Probe each slot individually at late timesteps."""
    timesteps = np.array([r["timestep"] for r in records])
    labels = np.array([r["signal_type"] for r in records])

    max_t = timesteps.max()
    mid = max(max_t // 2, 3)
    mask = timesteps >= mid
    if mask.sum() < 20:
        print("  Too few late samples")
        return None

    records_late = np.array(records)[mask]
    y = labels[mask]
    unique = np.unique(y)
    if len(unique) < 2:
        print("  Only 1 class in late steps")
        return None

    print(f"\n{'='*60}")
    print(f"  Per-slot probe (t>={mid})")
    print(f"  Random baseline: {1.0/n_classes:.1%}")
    print(f"{'='*60}")

    accs = []
    for k in range(NUM_SLOTS):
        X = np.array([r["slots"][k] for r in records_late])
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        scores = cross_val_score(clf, X, y, cv=min(5, mask.sum() // 10), scoring="accuracy")
        acc = scores.mean()
        accs.append(acc)
        marker = " <<<" if acc > 1.0 / n_classes + 0.1 else ""
        print(f"  Slot {k}: {acc:.1%} ± {scores.std():.1%}{marker}")

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(NUM_SLOTS), accs, color="steelblue")
    ax.axhline(1.0 / n_classes, color="red", ls="--", label=f"random ({1.0/n_classes:.0%})")
    ax.set_xlabel("Slot index")
    ax.set_ylabel("Probe accuracy")
    ax.set_title("Per-slot probe: which slot retains signal type?")
    ax.legend()
    ax.set_ylim(0, 1)
    for i, v in enumerate(accs):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig("slot_probe.png", dpi=150)
    print(f"\nSaved slot_probe.png")
    return accs


def plot_gate_timeline(records, output="gate_timeline.png"):
    """Gate values and KL over time for sample episodes."""
    episodes = sorted(set(r["episode_id"] for r in records))
    sample_eps = episodes[:min(6, len(episodes))]

    fig, axes = plt.subplots(len(sample_eps), 1, figsize=(12, 2.5 * len(sample_eps)), sharex=False)
    if len(sample_eps) == 1:
        axes = [axes]

    for ax, ep_id in zip(axes, sample_eps):
        ep_recs = [r for r in records if r["episode_id"] == ep_id]
        ts = [r["timestep"] for r in ep_recs]
        gates = [r["gate"] for r in ep_recs]
        kls = [r["kl"] for r in ep_recs]
        sig = TYPE_NAMES.get(ep_recs[0]["signal_type"], "?")

        ax.plot(ts, gates, "b-o", markersize=3, label="gate", alpha=0.8)
        ax2 = ax.twinx()
        ax2.plot(ts, kls, "r-s", markersize=3, label="KL", alpha=0.5)
        ax.set_ylabel("gate", color="blue")
        ax2.set_ylabel("KL", color="red")
        ax.set_title(f"Ep {ep_id} (signal: {sig})")
        ax.set_ylim(-0.1, 1.1)

    axes[-1].set_xlabel("timestep")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved {output}")


def plot_accuracy_over_time(records, n_classes, output="accuracy_over_time.png"):
    """Probe accuracy as a function of timestep — shows how fast signal decays."""
    timesteps = np.array([r["timestep"] for r in records])
    labels = np.array([r["signal_type"] for r in records])
    max_t = min(timesteps.max(), 50)

    # Bin timesteps
    bin_size = max(1, max_t // 15)
    bins = range(0, max_t + 1, bin_size)

    results = {"μ": [], "π": [], "slots": [], "h": []}
    bin_centers = []

    for b_start in bins:
        b_end = b_start + bin_size
        mask = (timesteps >= b_start) & (timesteps < b_end)
        if mask.sum() < 20:
            continue

        y = labels[mask]
        if len(np.unique(y)) < 2:
            continue

        bin_centers.append(b_start + bin_size / 2)

        for feat_key, feat_name in [("mu", "μ"), ("pi", "π"), ("h", "h")]:
            X = np.array([r[feat_key] for r in np.array(records)[mask]])
            clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
            scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")
            results[feat_name].append(scores.mean())

        X_slots = np.array([r["slots"].ravel() for r in np.array(records)[mask]])
        clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
        scores = cross_val_score(clf, X_slots, y, cv=3, scoring="accuracy")
        results["slots"].append(scores.mean())

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, accs in results.items():
        if accs:
            ax.plot(bin_centers[:len(accs)], accs, "-o", label=name, markersize=4)
    ax.axhline(1.0 / n_classes, color="gray", ls="--", alpha=0.5, label="random")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Probe accuracy")
    ax.set_title("Signal retention over time: can we recover key vs ball?")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n_episodes", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print("Building environment (view_size=5)...")
    env = MiniGridTrainingWrapper(
        gymnasium.make("MiniGrid-MemoryS7-v0", agent_view_size=VIEW_SIZE)
    )

    print("Building models...")
    vae, wyner, slot_mem = build_models(args.device)

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        # SB3 checkpoints are zip files containing policy.pth
        import zipfile, io
        with zipfile.ZipFile(args.checkpoint, "r") as zf:
            with zf.open("policy.pth") as f:
                buf = io.BytesIO(f.read())
        sd = th.load(buf, map_location=args.device, weights_only=False)
        vae_keys = {k.replace("vae_feature_extractor.", ""): v
                    for k, v in sd.items() if "vae_feature_extractor" in k}
        wyner_keys = {k.replace("wyner_feature_extractor.", ""): v
                      for k, v in sd.items() if "wyner_feature_extractor" in k}
        if vae_keys:
            vae.load_state_dict(vae_keys)
            print(f"  Loaded {len(vae_keys)} VAE params")
        if wyner_keys:
            wyner.load_state_dict(wyner_keys)
            print(f"  Loaded {len(wyner_keys)} Wyner params")

    print(f"Collecting data ({args.n_episodes} episodes)...")
    records = collect_data(env, vae, wyner, slot_mem, args.device, args.n_episodes)
    env.close()

    if len(records) < 50:
        print("ERROR: Too few records")
        return

    labels = np.array([r["signal_type"] for r in records])
    unique = np.unique(labels)
    n_classes = len(unique)
    print(f"\n{len(records)} samples, {n_classes} classes: {[TYPE_NAMES.get(c, str(c)) for c in unique]}")

    # Run all probes
    run_probe(records, "mu", "μ (VAE latent)", n_classes)
    run_probe(records, "pi", "π (policy head)", n_classes)
    run_probe(records, "slots", "Slots (all, flattened)", n_classes)
    run_probe(records, "h", "h (Wyner GRU)", n_classes)

    # Per-slot analysis
    per_slot_probe(records, n_classes)

    # Accuracy over time
    plot_accuracy_over_time(records, n_classes)

    # Gate timeline
    plot_gate_timeline(records)

    print("\nDone! Outputs: slot_probe.png, accuracy_over_time.png, gate_timeline.png")


if __name__ == "__main__":
    main()
