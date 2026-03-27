"""
Latent space visualization for HSWVIME-PPO.

Runs N episodes of MiniGrid-MemoryS7-v0 with forced turn-around at start
(to see the signal object), then collects μ, π at every timestep and
produces t-SNE plots colored by:
  1. Signal type (key vs ball) seen at episode start
  2. Timestep within episode

In MemoryS7, the signal is BEHIND the agent at reset.  The agent must
turn around to see it, remember it (key=5 or ball=6), then walk forward
and pick the matching goal.  Both goals are always visible going forward.

Usage:
    python scripts/visualize_latents.py [--checkpoint PATH] [--n_episodes 50] [--device cpu]
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch as th
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import gymnasium
import minigrid

from envs.wrappers import MiniGridTrainingWrapper
from models.embeddings import CategoricalGridWithDirEmbedding
from models.config import SCVAEConfig
from models.vae import TransitionSCVAE

# MiniGrid constants
SIGNAL_KEY = 5
SIGNAL_BALL = 6
TYPE_NAMES = {5: "key", 6: "ball"}
VIEW_SIZE = 5  # must match training


def extract_signal_type(obs: np.ndarray) -> int:
    """
    Extract signal object type from a backward-facing observation.
    The signal is a key(5) or ball(6) visible when the agent turns around.
    Returns 5 (key), 6 (ball), or -1 if not found.
    """
    obj_types = obs[:, :, 0]
    for r in range(obs.shape[0]):
        for c in range(obs.shape[1]):
            if obj_types[r, c] in (SIGNAL_KEY, SIGNAL_BALL):
                return int(obj_types[r, c])
    return -1


def turn_around_and_get_signal(env):
    """
    Turn agent around (2 left turns), observe the signal, turn back.
    Returns (signal_type, list of (obs, action) pairs from the turn sequence).
    """
    turn_history = []
    signal_type = -1

    # Turn left twice to face backward
    for _ in range(2):
        obs, reward, term, trunc, info = env.step(0)  # 0 = turn left
        turn_history.append((obs, 0))
        if signal_type < 0:
            signal_type = extract_signal_type(obs)

    # Walk forward briefly to get closer to signal if needed
    for _ in range(2):
        obs, reward, term, trunc, info = env.step(2)  # 2 = forward
        turn_history.append((obs, 2))
        if signal_type < 0:
            signal_type = extract_signal_type(obs)

    # Turn back (2 more left turns)
    for _ in range(2):
        obs, reward, term, trunc, info = env.step(0)
        turn_history.append((obs, 0))

    return signal_type, turn_history


def build_vae(device="cpu"):
    """Build a TransitionSCVAE with MemoryS7 settings."""
    embedding = CategoricalGridWithDirEmbedding(
        n_object_types=12, n_colors=6, n_states=3,
        obs_h=VIEW_SIZE, obs_w=VIEW_SIZE,
        embed_per_channel=4, n_dirs=4, dir_embed_dim=4,
    )
    cfg = SCVAEConfig(
        act_dim=7,
        action_embed_dim=32,
        conv_channels=[32, 64, 128],
        hidden_dim=256,
        latent_dim=32,
    )
    return TransitionSCVAE(embedding=embedding, cfg=cfg).to(device)


def collect_episode_data(env, vae, device, n_episodes=50):
    """
    Run episodes: turn around to see signal, then random-walk forward.
    Collect μ, π at every step.
    """
    all_mu, all_pi = [], []
    all_timesteps, all_signals, all_episode_ids = [], [], []

    vae.eval()
    skipped = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        signal_type, turn_hist = turn_around_and_get_signal(env)

        if signal_type < 0:
            skipped += 1
            continue

        # Now agent is facing forward again.
        # Get the current obs after turning back
        prev_obs = turn_hist[-1][0].copy()
        prev_action = np.array([turn_hist[-1][1]], dtype=np.float32)

        # Step once to get current obs
        obs, _, term, trunc, _ = env.step(2)  # one forward step
        if term or trunc:
            continue

        done = False
        t = 0

        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)

            with th.no_grad():
                s_tm1 = th.tensor(prev_obs, dtype=th.float32, device=device).unsqueeze(0)
                a_tm1 = th.tensor(prev_action, dtype=th.float32, device=device).unsqueeze(0)
                s_t = th.tensor(obs, dtype=th.float32, device=device).unsqueeze(0)
                mu, pi, _ = vae.encode(s_tm1, a_tm1, s_t)

            all_mu.append(mu.cpu().numpy().squeeze(0))
            all_pi.append(pi.cpu().numpy().squeeze(0))
            all_timesteps.append(t)
            all_signals.append(signal_type)
            all_episode_ids.append(ep)

            prev_obs = obs.copy()
            prev_action = np.array([action], dtype=np.float32)
            obs = next_obs
            done = terminated or truncated
            t += 1

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{n_episodes}, signal={TYPE_NAMES.get(signal_type, '?')}, len={t}")

    print(f"  Skipped {skipped}/{n_episodes} episodes (signal not found)")

    return {
        "mu": np.array(all_mu),
        "pi": np.array(all_pi),
        "timestep": np.array(all_timesteps),
        "signal_type": np.array(all_signals),
        "episode_id": np.array(all_episode_ids),
    }


def plot_tsne(embeddings, labels, title, label_names=None, ax=None, cmap="tab10"):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(embeddings)

    unique_labels = np.unique(labels)
    colormap = plt.get_cmap(cmap)

    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = label_names.get(label, str(label)) if label_names else str(label)
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[colormap(i / max(len(unique_labels) - 1, 1))],
                   label=name, alpha=0.5, s=10)

    ax.set_title(title)
    ax.legend(markerscale=3, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_tsne_continuous(embeddings, values, title, ax=None, cmap="viridis"):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(embeddings)

    sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap=cmap, alpha=0.5, s=10)
    plt.colorbar(sc, ax=ax, label="timestep")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="latent_viz.png")
    args = parser.parse_args()

    print("Building environment (view_size=5 to match training)...")
    env = MiniGridTrainingWrapper(
        gymnasium.make("MiniGrid-MemoryS7-v0", agent_view_size=VIEW_SIZE)
    )

    print("Building VAE...")
    vae = build_vae(args.device)

    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        state_dict = th.load(args.checkpoint, map_location=args.device, weights_only=False)
        vae_keys = {k.replace("policy.vae_feature_extractor.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith("policy.vae_feature_extractor.")}
        if vae_keys:
            vae.load_state_dict(vae_keys)
            print(f"  Loaded {len(vae_keys)} VAE parameters")
        else:
            print("  WARNING: No VAE keys found, using random init")

    print(f"Collecting data from {args.n_episodes} episodes...")
    data = collect_episode_data(env, vae, args.device, args.n_episodes)
    env.close()

    if len(data["mu"]) < 50:
        print("ERROR: Too few samples collected")
        return

    mu = data["mu"]
    pi = data["pi"]
    sig = data["signal_type"]
    ts = data["timestep"]

    # Subsample if too many points (t-SNE is slow)
    if len(mu) > 5000:
        idx = np.random.RandomState(42).choice(len(mu), 5000, replace=False)
        mu, pi, sig, ts = mu[idx], pi[idx], sig[idx], ts[idx]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    print("Computing t-SNE for μ by signal type...")
    plot_tsne(mu, sig, "μ (VAE latent) by signal type (key vs ball)",
              label_names=TYPE_NAMES, ax=axes[0, 0])

    print("Computing t-SNE for μ by timestep...")
    plot_tsne_continuous(mu, ts, "μ (VAE latent) by timestep", ax=axes[0, 1])

    print("Computing t-SNE for π by signal type...")
    plot_tsne(pi, sig, "π (policy head) by signal type (key vs ball)",
              label_names=TYPE_NAMES, ax=axes[1, 0])

    print("Computing t-SNE for π by timestep...")
    plot_tsne_continuous(pi, ts, "π (policy head) by timestep", ax=axes[1, 1])

    fig.suptitle("Latent Space — MemoryS7 (signal = key vs ball)", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {args.output}")

    # Cosine similarity analysis
    print("\n--- Cosine similarity: same-signal vs different-signal ---")
    for name, vecs in [("μ", mu), ("π", pi)]:
        cos = cosine_similarity(vecs)
        same, diff = [], []
        n = len(vecs)
        # Subsample pairs for speed
        rng = np.random.RandomState(42)
        pairs = rng.choice(n, size=(min(10000, n * (n-1) // 2), 2), replace=True)
        for i, j in pairs:
            if i == j:
                continue
            if sig[i] == sig[j]:
                same.append(cos[i, j])
            else:
                diff.append(cos[i, j])
        if same and diff:
            gap = np.mean(same) - np.mean(diff)
            print(f"  {name}: same-signal cos = {np.mean(same):.3f} ± {np.std(same):.3f}")
            print(f"  {name}: diff-signal cos = {np.mean(diff):.3f} ± {np.std(diff):.3f}")
            print(f"  {name}: gap = {gap:.3f} ({'SIGNAL ENCODED' if gap > 0.03 else 'no separation'})")


if __name__ == "__main__":
    main()
