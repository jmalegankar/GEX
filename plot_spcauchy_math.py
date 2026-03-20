"""
Generate Workstream 4 spCauchy math verification plots.

Usage:
    python plot_spcauchy_math.py [--output PATH]

Produces 8 subplots:
  (a) KL divergence to uniform vs ρ for d ∈ {8,16,32,64}
  (b) Quadratic approximation near ρ=0 confirming curvature formula
  (c) Collapse curvature: paper's 2(d-1) vs correct 4(d-1)²/d
  (d) Paper underestimation factor 2(d-1)/d → 2
  (e) Möbius sample concentration at different ρ
  (f) Möbius sample norm deviation from 1 (ppm)
  (g) KL derivative (monotonicity check)
  (h) fc_rho bias initialization: σ(-2) ≈ 0.12
"""

import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from models.utils import sc_kl_uniform, sc_sample, _sc_kl_quadrature, _gauss_legendre_01


DIMS = [8, 16, 32, 64]
COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']


def plot_kl_vs_rho(ax):
    for dim, c in zip(DIMS, COLORS):
        rhos = torch.linspace(0.01, 0.98, 300)
        kls = sc_kl_uniform(rhos, dim).squeeze().detach().numpy()
        ax.plot(rhos.numpy(), kls, label=f'd={dim}', color=c, linewidth=2)
    ax.set_xlabel('ρ (concentration)', fontsize=12)
    ax.set_ylabel('KL(spCauchy ‖ Uniform)', fontsize=12)
    ax.set_title('(a) KL Divergence to Uniform vs ρ', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)


def plot_quadratic_regime(ax):
    for dim, c in zip(DIMS, COLORS):
        rhos = torch.linspace(0.001, 0.15, 200)
        kls = sc_kl_uniform(rhos, dim).squeeze().detach().numpy()
        rhos_np = rhos.numpy()
        curvature = 4 * (dim - 1)**2 / dim
        kl_theory = 0.5 * curvature * rhos_np**2
        ax.plot(rhos_np, kls, label=f'd={dim} (quadrature)', color=c, linewidth=2)
        ax.plot(rhos_np, kl_theory, '--', color=c, alpha=0.5, linewidth=1.5)
    ax.set_xlabel('ρ', fontsize=12)
    ax.set_ylabel('KL', fontsize=12)
    ax.set_title('(b) KL Near ρ=0: Solid=Actual, Dashed=½·4(d-1)²/d·ρ²', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_curvature(ax):
    gl = _gauss_legendre_01(2048, torch.device('cpu'), torch.float64)
    dims_range = list(range(4, 129, 2))
    curvatures = []
    for dim in dims_range:
        rho = torch.tensor([[1e-4]], dtype=torch.float64, requires_grad=True)
        kl = _sc_kl_quadrature(rho, dim, gl)
        g1 = torch.autograd.grad(kl, rho, create_graph=True)[0]
        g2 = torch.autograd.grad(g1, rho)[0]
        curvatures.append(g2.item())

    dims_arr = np.array(dims_range)
    ax.plot(dims_arr, curvatures, 'ko', markersize=3, label='Numerical (autograd)', zorder=3)
    ax.plot(dims_arr, 4*(dims_arr-1)**2/dims_arr, 'r-', linewidth=2, label=r'$4(d{-}1)^2/d$ (correct)', zorder=2)
    ax.plot(dims_arr, 2*(dims_arr-1), 'b--', linewidth=2, label=r'$2(d{-}1)$ (paper claim)', zorder=1)
    ax.set_xlabel('Dimension d', fontsize=12)
    ax.set_ylabel(r"$d^2 KL / d\rho^2$ at $\rho=0$", fontsize=12)
    ax.set_title('(c) Collapse Curvature: Paper vs Correct Formula', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)


def plot_underestimation(ax):
    dims_arr = np.arange(4, 129, 2)
    correct = 4 * (dims_arr - 1)**2 / dims_arr
    paper = 2 * (dims_arr - 1)
    ratio = correct / paper
    ax.plot(dims_arr, ratio, 'r-', linewidth=2.5, label=r'$2(d-1)/d$')
    ax.axhline(y=2, color='gray', linestyle=':', alpha=0.5, label='Asymptote = 2')
    ax.axhline(y=1, color='blue', linestyle=':', alpha=0.5, label='Paper claim = 1')
    for dim in DIMS:
        r = 2*(dim-1)/dim
        ax.plot(dim, r, 'ko', markersize=8, zorder=3)
        ax.annotate(f'd={dim}\n{r:.2f}x', (dim, r), textcoords="offset points",
                     xytext=(10, -5), fontsize=9)
    ax.set_xlabel('Dimension d', fontsize=12)
    ax.set_ylabel('Actual / Paper Claim', fontsize=12)
    ax.set_title('(d) Paper Underestimation Factor', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 2.5)


def plot_mobius_concentration(ax):
    torch.manual_seed(42)
    dim = 32
    mu = torch.zeros(1, dim)
    mu[0, 0] = 1.0
    rho_vals = [0.01, 0.3, 0.7, 0.95]
    for i, rho_val in enumerate(rho_vals):
        mu_batch = mu.expand(2000, -1)
        rho = torch.full((2000, 1), rho_val)
        z = sc_sample(mu_batch, rho)
        cosines = (z * mu_batch).sum(dim=-1).detach().numpy()
        ax.hist(cosines, bins=50, alpha=0.5, density=True, label=f'ρ={rho_val}',
                 color=COLORS[i], edgecolor='none')
    ax.set_xlabel('cos(θ) = ⟨z, μ⟩', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'(e) Möbius Samples: Concentration vs ρ (d={dim})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)


def plot_norm_deviation(ax):
    torch.manual_seed(0)
    for dim_plot, c in zip([8, 32, 64], COLORS[:3]):
        mu = torch.nn.functional.normalize(torch.randn(5000, dim_plot), p=2, dim=-1)
        rho = torch.full((5000, 1), 0.5)
        z = sc_sample(mu, rho)
        norm_devs = (z.norm(p=2, dim=-1) - 1.0).detach().numpy() * 1e6
        ax.hist(norm_devs, bins=50, alpha=0.6, density=True, label=f'd={dim_plot}',
                 color=c, edgecolor='none')
    ax.axvline(x=0, color='red', linewidth=2, linestyle='--', label='Perfect unit norm')
    ax.set_xlabel('||z||₂ - 1  (×10⁻⁶)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('(f) Möbius Norm Deviation from 1 (ppm)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)


def plot_monotonicity(ax):
    for dim, c in zip(DIMS, COLORS):
        rhos = torch.linspace(0.01, 0.98, 200)
        kls = sc_kl_uniform(rhos, dim).squeeze().detach().numpy()
        dkl = np.diff(kls) / np.diff(rhos.numpy())
        ax.plot(rhos.numpy()[1:], dkl, label=f'd={dim}', color=c, linewidth=2)
    ax.set_xlabel('ρ', fontsize=12)
    ax.set_ylabel('dKL/dρ', fontsize=12)
    ax.set_title('(g) KL Derivative (Monotonicity: should be > 0)', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)


def plot_sigmoid_init(ax):
    logits = torch.linspace(-6, 6, 300)
    rhos_sigmoid = torch.sigmoid(logits).numpy()
    ax.plot(logits.numpy(), rhos_sigmoid, 'b-', linewidth=2)
    ax.axvline(x=-2.0, color='red', linewidth=2, linestyle='--')
    init_rho = torch.sigmoid(torch.tensor(-2.0)).item()
    ax.plot(-2.0, init_rho, 'ro', markersize=12, zorder=5,
             label=f'ρ₀ = σ(-2) ≈ {init_rho:.3f}')
    ax.axhline(y=init_rho, color='red', linewidth=1, linestyle=':', alpha=0.4)
    ax.fill_between(logits.numpy(), 0, rhos_sigmoid, where=rhos_sigmoid < 0.2,
                     alpha=0.1, color='green', label='Near-uniform zone (ρ < 0.2)')
    ax.set_xlabel('fc_rho logit (pre-sigmoid)', fontsize=12)
    ax.set_ylabel('ρ = σ(logit)', fontsize=12)
    ax.set_title('(h) fc_rho Bias Init: Start Near Uniform', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)


def main():
    parser = argparse.ArgumentParser(description='Generate spCauchy math verification plots')
    parser.add_argument('--output', default='spcauchy_math_verification.png',
                        help='Output path (default: spcauchy_math_verification.png)')
    args = parser.parse_args()

    fig = plt.figure(figsize=(20, 24))
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)

    plot_kl_vs_rho(fig.add_subplot(gs[0, 0]))
    plot_quadratic_regime(fig.add_subplot(gs[0, 1]))
    plot_curvature(fig.add_subplot(gs[1, 0]))
    plot_underestimation(fig.add_subplot(gs[1, 1]))
    plot_mobius_concentration(fig.add_subplot(gs[2, 0]))
    plot_norm_deviation(fig.add_subplot(gs[2, 1]))
    plot_monotonicity(fig.add_subplot(gs[3, 0]))
    plot_sigmoid_init(fig.add_subplot(gs[3, 1]))

    fig.suptitle('Spherical Cauchy Distribution — Workstream 4 Math Verification',
                 fontsize=18, fontweight='bold', y=0.995)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
