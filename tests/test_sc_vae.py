import torch
import torch.nn as nn
import torch.nn.functional as F

from models.config import SCVAEConfig
from models.sc_vae import TransitionSCVAE


class DummyEmbedding(nn.Module):
    """
    Minimal embedding that converts integer obs to float channels.
    Input: (B,H,W,3) long
    Output: (B,C,H,W) float
    """
    def __init__(self, out_channels: int = 12):
        super().__init__()
        self.out_channels = out_channels
        self.lin = nn.Linear(3, out_channels)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B,H,W,3)
        x = obs.float()
        B, H, W, C = x.shape
        x = self.lin(x.view(B * H * W, C)).view(B, H, W, self.out_channels)
        return x.permute(0, 3, 1, 2).contiguous()


def test_shapes_and_unit_norm():
    cfg = SCVAEConfig(
        conv_channels=(16, 32),
        hidden_dim=64,
        latent_dim=32,
        n_actions=7,
        action_embed_dim=8,
        rho_min=0.001,
        rho_max=0.999,
        kl_max_terms=64,
    )
    emb = DummyEmbedding(out_channels=12)
    H = W = 5
    model = TransitionSCVAE(emb, cfg, sample_input_shape=(H, W, 3))

    B = 4
    s_t = torch.randint(0, 5, (B, H, W, 3))
    a_t = torch.randint(0, cfg.n_actions, (B,))
    s_n = torch.randint(0, 5, (B, H, W, 3))

    model.train()
    out = model(s_t, a_t, s_n)

    assert out.mu.shape == (B, cfg.latent_dim)
    assert torch.allclose(out.mu.norm(dim=-1), torch.ones(B), atol=1e-5)

    assert out.rho.shape == (B, 1)
    assert (out.rho >= cfg.rho_min - 1e-6).all()
    assert (out.rho <= cfg.rho_max + 1e-6).all()

    # recon = [feat, a_emb, feat]
    assert out.recon.shape[0] == B
    assert out.recon.shape == out.recon_target.shape
    assert not out.recon_target.requires_grad


def test_loss_backward_has_grads():
    cfg = SCVAEConfig(
        conv_channels=(16, 32),
        hidden_dim=64,
        latent_dim=16,
        n_actions=7,
        action_embed_dim=8,
        rho_min=0.001,
        rho_max=0.999,
        kl_max_terms=64,
    )
    emb = DummyEmbedding(out_channels=8)
    H = W = 5
    model = TransitionSCVAE(emb, cfg, sample_input_shape=(H, W, 3))

    B = 8
    s_t = torch.randint(0, 5, (B, H, W, 3))
    a_t = torch.randint(0, cfg.n_actions, (B,))
    s_n = torch.randint(0, 5, (B, H, W, 3))

    model.train()
    out = model(s_t, a_t, s_n)
    l_recon, l_kl = model.loss(out)

    loss = l_recon + 0.005 * l_kl
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_eval_mode_z_equals_mu():
    cfg = SCVAEConfig(
        conv_channels=(8,),
        hidden_dim=32,
        latent_dim=8,
        n_actions=7,
        action_embed_dim=4,
        rho_min=0.001,
        rho_max=0.999,
        kl_max_terms=32,
    )
    emb = DummyEmbedding(out_channels=4)
    H = W = 5
    model = TransitionSCVAE(emb, cfg, sample_input_shape=(H, W, 3))

    B = 2
    s_t = torch.randint(0, 5, (B, H, W, 3))
    a_t = torch.randint(0, cfg.n_actions, (B,))
    s_n = torch.randint(0, 5, (B, H, W, 3))

    model.eval()
    with torch.no_grad():
        out = model(s_t, a_t, s_n)

    assert torch.allclose(out.z, out.mu, atol=1e-6)