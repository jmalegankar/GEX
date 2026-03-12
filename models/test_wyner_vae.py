"""
Unit tests for WynerVAE — Phase 0 validation.

Tests are ordered from primitive to composite:
  T01–T02  gaussian_kl utility
  T03–T05  SlowPath
  T06–T08  PriorNet
  T09–T11  PosteriorNet
  T12–T13  WynerDecoder
  T14–T17  WynerVAE (forward_train, forward_rollout, loss)
  T18      Gradient flow audit

Run with:
    python test_wyner_vae.py

Each test prints PASS or FAIL with a reason.
All tests must pass before proceeding to Phase 1 integration.
"""

import torch as th
import torch.nn as nn
import traceback

from wyner import (
    WynerConfig,
    WynerVAE,
    SlowPath,
    PriorNet,
    PosteriorNet,
    WynerDecoder,
    gaussian_kl,
    reparameterize,
)

# ─────────────────────────────────────────────────────────────────────────────
# Test harness
# ─────────────────────────────────────────────────────────────────────────────

_results = []

def test(name: str):
    """Decorator: runs function, catches exceptions, records PASS / FAIL."""
    def decorator(fn):
        try:
            fn()
            _results.append((name, True, ""))
            print(f"  PASS  {name}")
        except Exception as e:
            tb = traceback.format_exc()
            _results.append((name, False, str(e)))
            print(f"  FAIL  {name}\n        {e}\n{tb}")
        return fn
    return decorator

def assert_shape(tensor, expected, label="tensor"):
    assert tuple(tensor.shape) == tuple(expected), \
        f"{label}: expected shape {expected}, got {tuple(tensor.shape)}"

def assert_finite(tensor, label="tensor"):
    assert th.isfinite(tensor).all(), \
        f"{label} contains NaN or Inf"

def assert_no_grad(tensor, label="tensor"):
    assert not tensor.requires_grad, \
        f"{label} should have requires_grad=False (detached)"

def assert_has_grad(tensor, label="tensor"):
    assert tensor.grad_fn is not None or tensor.requires_grad, \
        f"{label} should have a gradient (not detached)"

# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

B          = 8     # batch size
MU_DIM     = 16    # TransitionSCVAE latent dim (small for tests)
WYNER_DIM  = 8
CONTEXT_DIM = 32
HIDDEN_DIM  = 32

CFG = WynerConfig(
    mu_dim=MU_DIM,
    wyner_dim=WYNER_DIM,
    context_dim=CONTEXT_DIM,
    hidden_dim=HIDDEN_DIM,
    lambda_past=1.0,
    lambda_future=2.0,
    alpha_intrinsic=1.0,
    free_nats=0.0,
)

DEVICE = th.device("cpu")


def make_mu():
    """Simulate L2-normalised SCVAE output."""
    return nn.functional.normalize(th.randn(B, MU_DIM), dim=-1)

def make_context():
    return th.randn(B, CONTEXT_DIM)

def make_h_prev():
    return th.zeros(B, CONTEXT_DIM)


# ─────────────────────────────────────────────────────────────────────────────
# T01 – gaussian_kl: zero when distributions are identical
# ─────────────────────────────────────────────────────────────────────────────

@test("T01 gaussian_kl is zero when q == p")
def _():
    mu      = th.randn(B, WYNER_DIM)
    logvar  = th.randn(B, WYNER_DIM)
    kl      = gaussian_kl(mu, logvar, mu, logvar)
    assert_shape(kl, (B,), "kl")
    assert (kl.abs() < 1e-4).all(), \
        f"KL should be ~0 when q==p; got max {kl.abs().max():.6f}"


# ─────────────────────────────────────────────────────────────────────────────
# T02 – gaussian_kl: positive when distributions differ
# ─────────────────────────────────────────────────────────────────────────────

@test("T02 gaussian_kl is positive and finite when q != p")
def _():
    mu_q     = th.randn(B, WYNER_DIM)
    logvar_q = th.zeros(B, WYNER_DIM)
    mu_p     = th.randn(B, WYNER_DIM) * 2.0
    logvar_p = th.ones(B, WYNER_DIM)
    kl       = gaussian_kl(mu_q, logvar_q, mu_p, logvar_p)
    assert_shape(kl, (B,), "kl")
    assert_finite(kl, "kl")
    assert (kl > 0).all(), "KL should be strictly positive when q != p"


# ─────────────────────────────────────────────────────────────────────────────
# T03 – SlowPath: output shape
# ─────────────────────────────────────────────────────────────────────────────

@test("T03 SlowPath output shape")
def _():
    sp  = SlowPath(MU_DIM, CONTEXT_DIM)
    h_t = sp(make_mu(), make_h_prev())
    assert_shape(h_t, (B, CONTEXT_DIM), "h_t")
    assert_finite(h_t, "h_t")


# ─────────────────────────────────────────────────────────────────────────────
# T04 – SlowPath: hidden state changes across steps
# ─────────────────────────────────────────────────────────────────────────────

@test("T04 SlowPath hidden state updates each step")
def _():
    sp    = SlowPath(MU_DIM, CONTEXT_DIM)
    h     = sp.init_hidden(B, DEVICE)
    h1    = sp(make_mu(), h)
    h2    = sp(make_mu(), h1)
    assert not th.allclose(h, h1),  "h0 == h1: hidden state did not update on step 1"
    assert not th.allclose(h1, h2), "h1 == h2: hidden state did not update on step 2"


# ─────────────────────────────────────────────────────────────────────────────
# T05 – SlowPath: stop-gradient on mu_t
#        Gradients from SlowPath output must NOT flow into mu_t.
# ─────────────────────────────────────────────────────────────────────────────

@test("T05 SlowPath stop-gradient: no grad flows into mu_t")
def _():
    sp    = SlowPath(MU_DIM, CONTEXT_DIM)
    mu_t  = make_mu().requires_grad_(True)
    h_prev = make_h_prev()
    h_t   = sp(mu_t, h_prev)
    h_t.sum().backward()
    assert mu_t.grad is None, \
        "mu_t.grad should be None — SlowPath must sg(mu_t) before GRU"


# ─────────────────────────────────────────────────────────────────────────────
# T06 – PriorNet: output shapes
# ─────────────────────────────────────────────────────────────────────────────

@test("T06 PriorNet output shapes")
def _():
    prior        = PriorNet(CONTEXT_DIM, WYNER_DIM, HIDDEN_DIM)
    mu_p, lv_p   = prior(make_context())
    assert_shape(mu_p, (B, WYNER_DIM), "mu_p")
    assert_shape(lv_p, (B, WYNER_DIM), "logvar_p")
    assert_finite(mu_p,  "mu_p")
    assert_finite(lv_p,  "logvar_p")


# ─────────────────────────────────────────────────────────────────────────────
# T07 – PriorNet: logvar bias initialised to zero (near-unit Gaussian at init)
# ─────────────────────────────────────────────────────────────────────────────

@test("T07 PriorNet logvar bias is zero at init")
def _():
    prior = PriorNet(CONTEXT_DIM, WYNER_DIM, HIDDEN_DIM)
    bias  = prior.net[-1].bias
    logvar_bias = bias[WYNER_DIM:]
    assert (logvar_bias.abs() < 1e-6).all(), \
        f"logvar bias not zero-init; max abs = {logvar_bias.abs().max():.6f}"


# ─────────────────────────────────────────────────────────────────────────────
# T08 – PriorNet: gradients flow through context into net parameters
# ─────────────────────────────────────────────────────────────────────────────

@test("T08 PriorNet gradients flow through context")
def _():
    prior   = PriorNet(CONTEXT_DIM, WYNER_DIM, HIDDEN_DIM)
    ctx     = make_context().requires_grad_(True)
    mu_p, _ = prior(ctx)
    mu_p.sum().backward()
    assert ctx.grad is not None, "No grad at context input to PriorNet"
    # all prior parameters should receive gradients
    for name, p in prior.named_parameters():
        assert p.grad is not None, f"PriorNet param '{name}' has no gradient"


# ─────────────────────────────────────────────────────────────────────────────
# T09 – PosteriorNet: output shapes
# ─────────────────────────────────────────────────────────────────────────────

@test("T09 PosteriorNet output shapes")
def _():
    post         = PosteriorNet(CONTEXT_DIM, MU_DIM, WYNER_DIM, HIDDEN_DIM)
    mu_q, lv_q   = post(make_context(), make_mu(), make_mu())
    assert_shape(mu_q, (B, WYNER_DIM), "mu_q")
    assert_shape(lv_q, (B, WYNER_DIM), "logvar_q")
    assert_finite(mu_q, "mu_q")
    assert_finite(lv_q, "logvar_q")


# ─────────────────────────────────────────────────────────────────────────────
# T10 – PosteriorNet: mu_{t+1} must be stop-gradiented by caller
#        Verify: when mu_tp1.requires_grad=True but sg'd before PosteriorNet,
#        mu_tp1.grad is None after backward.
# ─────────────────────────────────────────────────────────────────────────────

@test("T10 PosteriorNet: sg(mu_tp1) — grad does not flow to mu_tp1")
def _():
    post   = PosteriorNet(CONTEXT_DIM, MU_DIM, WYNER_DIM, HIDDEN_DIM)
    mu_tp1 = make_mu().requires_grad_(True)
    # Caller's responsibility: detach before passing in
    mu_q, _ = post(make_context(), make_mu(), mu_tp1.detach())
    mu_q.sum().backward()
    assert mu_tp1.grad is None, \
        "mu_tp1 should receive no gradient (caller must sg it before PosteriorNet)"


# ─────────────────────────────────────────────────────────────────────────────
# T11 – PosteriorNet: mu_t does NOT receive gradient when sg'd by caller
#        In WynerVAE.forward_train, mu_t is sg'd before PosteriorNet.
# ─────────────────────────────────────────────────────────────────────────────

@test("T11 PosteriorNet: sg(mu_t) — grad does not flow to mu_t (SCVAE protected)")
def _():
    post  = PosteriorNet(CONTEXT_DIM, MU_DIM, WYNER_DIM, HIDDEN_DIM)
    mu_t  = make_mu().requires_grad_(True)
    mu_q, _ = post(make_context(), mu_t.detach(), make_mu())
    mu_q.sum().backward()
    assert mu_t.grad is None, \
        "mu_t should receive no gradient when sg'd — SCVAE must not be modified by WynerVAE loss"


# ─────────────────────────────────────────────────────────────────────────────
# T12 – WynerDecoder: output shape
# ─────────────────────────────────────────────────────────────────────────────

@test("T12 WynerDecoder output shape")
def _():
    dec  = WynerDecoder(WYNER_DIM, MU_DIM, HIDDEN_DIM)
    z_t  = th.randn(B, WYNER_DIM)
    out  = dec(z_t)
    assert_shape(out, (B, MU_DIM), "recon")
    assert_finite(out, "recon")


# ─────────────────────────────────────────────────────────────────────────────
# T13 – WynerDecoder: self-sufficient (only accepts z_t)
#        This is a static interface check — the decoder signature must not
#        allow context to sneak in.  We verify by inspecting forward's signature.
# ─────────────────────────────────────────────────────────────────────────────

@test("T13 WynerDecoder is self-sufficient (forward takes only z_t)")
def _():
    import inspect
    dec    = WynerDecoder(WYNER_DIM, MU_DIM, HIDDEN_DIM)
    params = list(inspect.signature(dec.forward).parameters.keys())
    assert params == ["z_t"], \
        f"WynerDecoder.forward should accept only 'z_t'; got {params}"


# ─────────────────────────────────────────────────────────────────────────────
# T14 – WynerVAE: forward_train output shapes
# ─────────────────────────────────────────────────────────────────────────────

@test("T14 WynerVAE.forward_train output shapes")
def _():
    model = WynerVAE(CFG)
    out   = model.forward_train(make_mu(), make_mu(), make_h_prev())

    assert_shape(out.mu_q,          (B, WYNER_DIM),   "mu_q")
    assert_shape(out.logvar_q,      (B, WYNER_DIM),   "logvar_q")
    assert_shape(out.mu_p,          (B, WYNER_DIM),   "mu_p")
    assert_shape(out.logvar_p,      (B, WYNER_DIM),   "logvar_p")
    assert_shape(out.z_t,           (B, WYNER_DIM),   "z_t")
    assert_shape(out.recon_past,    (B, MU_DIM),       "recon_past")
    assert_shape(out.recon_future,  (B, MU_DIM),       "recon_future")
    assert_shape(out.target_past,   (B, MU_DIM),       "target_past")
    assert_shape(out.target_future, (B, MU_DIM),       "target_future")
    assert_shape(out.h_slow,        (B, CONTEXT_DIM),  "h_slow")

    for name in ("mu_q","logvar_q","mu_p","logvar_p","z_t",
                 "recon_past","recon_future"):
        assert_finite(getattr(out, name), name)


# ─────────────────────────────────────────────────────────────────────────────
# T15 – WynerVAE: targets must be detached (no grad)
# ─────────────────────────────────────────────────────────────────────────────

@test("T15 WynerVAE targets are detached (sg on reconstruction targets)")
def _():
    model = WynerVAE(CFG)
    out   = model.forward_train(make_mu(), make_mu(), make_h_prev())
    assert_no_grad(out.target_past,   "target_past")
    assert_no_grad(out.target_future, "target_future")


# ─────────────────────────────────────────────────────────────────────────────
# T16 – WynerVAE: loss values are finite and positive
# ─────────────────────────────────────────────────────────────────────────────

@test("T16 WynerVAE.loss components are finite")
def _():
    model = WynerVAE(CFG)
    out   = model.forward_train(make_mu(), make_mu(), make_h_prev())
    L     = model.loss(out)

    for name in ("kl_loss", "recon_past_loss", "recon_future_loss", "total_loss"):
        val = getattr(L, name)
        assert_finite(val, name)
        assert val.shape == th.Size([]), f"{name} should be a scalar"

    assert_shape(L.intrinsic_reward, (B,), "intrinsic_reward")
    assert_finite(L.intrinsic_reward, "intrinsic_reward")
    assert (L.intrinsic_reward >= 0).all(), "intrinsic_reward should be non-negative"


# ─────────────────────────────────────────────────────────────────────────────
# T17 – WynerVAE: intrinsic reward is detached
# ─────────────────────────────────────────────────────────────────────────────

@test("T17 WynerVAE intrinsic_reward is detached from computation graph")
def _():
    model = WynerVAE(CFG)
    out   = model.forward_train(make_mu(), make_mu(), make_h_prev())
    L     = model.loss(out)
    assert_no_grad(L.intrinsic_reward, "intrinsic_reward")


# ─────────────────────────────────────────────────────────────────────────────
# T18 – Gradient flow audit
#        After loss.backward():
#          ✓ All WynerVAE parameters have gradients
#          ✗ mu_t input (simulating SCVAE output) has NO gradient
#          ✗ mu_tp1 input has NO gradient
# ─────────────────────────────────────────────────────────────────────────────

@test("T18 Gradient flow: WynerVAE params update; SCVAE inputs are protected")
def _():
    model  = WynerVAE(CFG)

    # Simulate SCVAE outputs — leaf tensors with requires_grad to detect leakage
    mu_t   = make_mu().requires_grad_(True)
    mu_tp1 = make_mu().requires_grad_(True)
    h_prev = make_h_prev()

    out = model.forward_train(mu_t, mu_tp1, h_prev)
    L   = model.loss(out)
    L.total_loss.backward()

    # --- WynerVAE parameters must all have gradients ---
    for name, param in model.named_parameters():
        assert param.grad is not None, \
            f"WynerVAE parameter '{name}' has no gradient after backward"
        assert th.isfinite(param.grad).all(), \
            f"WynerVAE parameter '{name}' has non-finite gradient"

    # --- SCVAE inputs must NOT have gradients ---
    assert mu_t.grad is None, \
        "mu_t received a gradient — WynerVAE loss is leaking into SCVAE!"
    assert mu_tp1.grad is None, \
        "mu_tp1 received a gradient — WynerVAE loss is leaking into SCVAE!"


# ─────────────────────────────────────────────────────────────────────────────
# T19 – WynerVAE: forward_rollout (no mu_{t+1})
# ─────────────────────────────────────────────────────────────────────────────

@test("T19 WynerVAE.forward_rollout shapes and no grad context")
def _():
    model = WynerVAE(CFG)
    with th.no_grad():
        z_t, h_t = model.forward_rollout(make_mu(), make_h_prev())
    assert_shape(z_t, (B, WYNER_DIM),  "z_t (rollout)")
    assert_shape(h_t, (B, CONTEXT_DIM), "h_t (rollout)")
    assert_finite(z_t, "z_t")
    assert_finite(h_t, "h_t")


# ─────────────────────────────────────────────────────────────────────────────
# T20 – KL > 0 on random data (prior != posterior in general)
#        Sanity check: the KL intrinsic reward is actually active on random data.
# ─────────────────────────────────────────────────────────────────────────────

@test("T20 KL intrinsic reward is > 0 on random data (prior != posterior)")
def _():
    model = WynerVAE(CFG)
    out   = model.forward_train(make_mu(), make_mu(), make_h_prev())
    L     = model.loss(out)
    assert (L.intrinsic_reward > 0).all(), \
        "Intrinsic reward should be > 0 on random data (prior and posterior differ)"


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("WynerVAE Phase 0 Unit Tests")
    print("="*60)

    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, _ in _results if not ok)
    total  = len(_results)

    print(f"\nResults: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
        print("\nFailed tests:")
        for name, ok, reason in _results:
            if not ok:
                print(f"  • {name}: {reason}")
    else:
        print(" — all clear ✓")
        print("\nPhase 0 complete. Proceed to Phase 1 integration.")
    print()