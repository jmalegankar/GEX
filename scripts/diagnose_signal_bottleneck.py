"""
Diagnose WHERE the key-vs-ball signal is lost in the VAE pipeline.

Tests linear separability at each stage:
  1. Raw observation (flattened)
  2. Embedding output (spatial)
  3. Conv encoder output (after global avg pool)
  4. fc_mu output (before L2 norm)
  5. mu (after L2 norm)

Uses random weights AND v6 checkpoint weights to isolate architecture vs training.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch as th
import gymnasium
import minigrid
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from envs.wrappers import MiniGridTrainingWrapper
from models.embeddings import CategoricalGridWithDirEmbedding
from models.config import SCVAEConfig
from models.vae import TransitionSCVAE
from models.encoders import ConvEncoder

VIEW_SIZE = 5
SIGNAL_KEY = 5
SIGNAL_BALL = 6


def collect_signal_obs(n_episodes=300):
    """Collect observations where the signal object is visible."""
    env = MiniGridTrainingWrapper(
        gymnasium.make("MiniGrid-MemoryS7-v0", agent_view_size=VIEW_SIZE)
    )

    signal_obs = []  # obs containing signal
    signal_labels = []  # 0=key, 1=ball

    for ep in range(n_episodes):
        obs, _ = env.reset()

        # Turn left twice to face signal
        for _ in range(2):
            obs, _, term, trunc, _ = env.step(0)
            if term or trunc:
                break

        # Check if signal is visible
        obj_types = obs[:, :, 0]
        for r in range(obs.shape[0]):
            for c in range(obs.shape[1]):
                if obj_types[r, c] == SIGNAL_KEY:
                    signal_obs.append(obs.copy())
                    signal_labels.append(0)
                    break
                elif obj_types[r, c] == SIGNAL_BALL:
                    signal_obs.append(obs.copy())
                    signal_labels.append(1)
                    break
            else:
                continue
            break

        # Also walk forward to try to find it
        if len(signal_obs) == ep:  # didn't find it yet
            for _ in range(3):
                obs, _, term, trunc, _ = env.step(2)
                if term or trunc:
                    break
                obj_types = obs[:, :, 0]
                found = False
                for r in range(obs.shape[0]):
                    for c in range(obs.shape[1]):
                        if obj_types[r, c] == SIGNAL_KEY:
                            signal_obs.append(obs.copy())
                            signal_labels.append(0)
                            found = True
                            break
                        elif obj_types[r, c] == SIGNAL_BALL:
                            signal_obs.append(obs.copy())
                            signal_labels.append(1)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break

    env.close()
    return np.array(signal_obs), np.array(signal_labels)


def build_vae(device="cpu"):
    embedding = CategoricalGridWithDirEmbedding(
        n_object_types=12, n_colors=6, n_states=3,
        obs_h=VIEW_SIZE, obs_w=VIEW_SIZE,
        embed_per_channel=4, n_dirs=4, dir_embed_dim=4,
    )
    cfg = SCVAEConfig(
        act_dim=7, action_embed_dim=32,
        conv_channels=[32, 64, 128], hidden_dim=256, latent_dim=32,
    )
    vae = TransitionSCVAE(embedding=embedding, cfg=cfg).to(device)
    return vae


def load_checkpoint_weights(vae, checkpoint_path, device="cpu"):
    import zipfile, io
    with zipfile.ZipFile(checkpoint_path, "r") as zf:
        with zf.open("policy.pth") as f:
            buf = io.BytesIO(f.read())
    sd = th.load(buf, map_location=device, weights_only=False)
    vae_keys = {k.replace("vae_feature_extractor.", ""): v
                for k, v in sd.items() if "vae_feature_extractor" in k}
    if vae_keys:
        vae.load_state_dict(vae_keys)
        print(f"  Loaded {len(vae_keys)} VAE params from checkpoint")
    else:
        print("  WARNING: No VAE keys found in checkpoint!")
    return vae


def probe_accuracy(X, y, name):
    """Run logistic regression probe and return accuracy."""
    if len(np.unique(y)) < 2:
        print(f"  {name:40s}: only 1 class")
        return 0.5
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    acc = scores.mean()
    marker = " <<<" if acc > 0.7 else ""
    print(f"  {name:40s}: {acc:.1%} ± {scores.std():.1%}{marker}")
    return acc


def run_diagnosis(vae, obs_batch, labels, tag=""):
    """Extract features at each pipeline stage and probe."""
    print(f"\n{'='*60}")
    print(f"  Pipeline diagnosis{' — ' + tag if tag else ''}")
    print(f"  N={len(labels)}, key={sum(labels==0)}, ball={sum(labels==1)}")
    print(f"{'='*60}")

    device = next(vae.parameters()).device
    obs_t = th.tensor(obs_batch, dtype=th.float32, device=device)

    with th.no_grad():
        # Stage 1: Raw obs
        X_raw = obs_batch.reshape(len(obs_batch), -1)
        probe_accuracy(X_raw, labels, "1. Raw obs (flattened)")

        # Stage 2: Embedding output (spatial)
        emb = vae.embedding(obs_t)  # (B, C, H, W)
        X_emb = emb.cpu().numpy().reshape(len(obs_batch), -1)
        probe_accuracy(X_emb, labels, "2. Embedding (spatial, flattened)")

        # Stage 2b: Embedding — just the signal cell
        # Signal should be near center of view when facing it
        # Check which cells differ between key/ball observations
        emb_spatial = emb.cpu().numpy()  # (B, C, H, W)
        key_mask = labels == 0
        ball_mask = labels == 1
        if key_mask.sum() > 0 and ball_mask.sum() > 0:
            key_mean = emb_spatial[key_mask].mean(axis=0)  # (C, H, W)
            ball_mean = emb_spatial[ball_mask].mean(axis=0)
            diff = np.abs(key_mean - ball_mean)
            # Find max-diff cell
            diff_per_cell = diff.sum(axis=0)  # (H, W)
            max_cell = np.unravel_index(diff_per_cell.argmax(), diff_per_cell.shape)
            print(f"     Max embedding diff at cell {max_cell}, diff={diff_per_cell[max_cell]:.3f}")
            print(f"     Diff per cell:\n{np.array2string(diff_per_cell, precision=2, suppress_small=True)}")

        # Stage 3: Conv encoder (after global avg pool)
        conv_out = vae.conv(emb)  # (B, 128)
        X_conv = conv_out.cpu().numpy()
        probe_accuracy(X_conv, labels, "3. Conv encoder (global avg pool)")

        # Stage 4: Full encoder trunk output (before mu/rho heads)
        # Need a dummy transition: use same obs for s_t and s_tp1
        null_a = th.zeros(len(obs_batch), dtype=th.float32, device=device)
        h_s = conv_out
        h_sn = conv_out  # same obs
        a_emb = vae.action_embed(null_a.long()).view(-1, vae.cfg.action_embed_dim)
        trunk_in = th.cat([h_s, a_emb, h_sn], dim=-1)
        trunk_out = vae.encoder_trunk(trunk_in)
        X_trunk = trunk_out.cpu().numpy()
        probe_accuracy(X_trunk, labels, "4. Encoder trunk output")

        # Stage 5: fc_mu (before L2 norm)
        mu_raw = vae.fc_mu(trunk_out)
        X_mu_raw = mu_raw.cpu().numpy()
        probe_accuracy(X_mu_raw, labels, "5. fc_mu (before L2 norm)")

        # Stage 6: mu (after L2 norm)
        import torch.nn.functional as F
        mu = F.normalize(mu_raw, p=2, dim=-1)
        X_mu = mu.cpu().numpy()
        probe_accuracy(X_mu, labels, "6. mu (after L2 norm)")

        # Stage 7: pi head
        pi = vae.fc_pi(trunk_out)
        X_pi = pi.cpu().numpy()
        probe_accuracy(X_pi, labels, "7. pi (policy head)")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n_episodes", type=int, default=300)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print("Collecting signal observations...")
    obs, labels = collect_signal_obs(args.n_episodes)
    print(f"Collected {len(obs)} signal observations (key={sum(labels==0)}, ball={sum(labels==1)})")

    if len(obs) < 50:
        print("ERROR: Too few observations")
        return

    # Test with random weights
    print("\nBuilding VAE (random weights)...")
    vae = build_vae(args.device)
    vae.eval()
    run_diagnosis(vae, obs, labels, tag="RANDOM WEIGHTS")

    # Test with trained weights
    if args.checkpoint:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        vae = build_vae(args.device)
        vae = load_checkpoint_weights(vae, args.checkpoint, args.device)
        vae.eval()
        run_diagnosis(vae, obs, labels, tag="TRAINED WEIGHTS (v6 100k)")


if __name__ == "__main__":
    main()
