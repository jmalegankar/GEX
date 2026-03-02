import torch
from reward_normalizer import RunningMeanStd


def test_rms_basic():
    rms = RunningMeanStd()

    x = torch.tensor([1.0, 2.0, 3.0])
    rms.update(x)

    norm = rms.normalize(x)

    assert torch.isfinite(norm).all()


def test_rms_updates():
    rms = RunningMeanStd()

    for _ in range(10):
        x = torch.randn(32)
        rms.update(x)

    assert rms.var > 0