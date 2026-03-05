import torch
import torch.nn as nn
from typing import List


# ── helpers ──────────────────────────────────────────────────────────

def _num_groups(channels: int) -> int:
    """Largest power-of-2 group count that evenly divides `channels`."""
    for g in [8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


# ── building blocks ───────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Two 3×3 convs with a GroupNorm + ReLU each, plus an identity skip.
    No spatial change — call after a stride-2 stem to refine features.
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()

        g = _num_groups(channels)
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=1, bias=False),
            nn.GroupNorm(g, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, padding=1, bias=False),
            nn.GroupNorm(g, channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class DownBlock(nn.Module):
    """
    Stride-2 conv for spatial downsampling, followed by a residual refinement.

    Input:  (B, in_ch, H,   W  )
    Output: (B, out_ch, H/2, W/2)
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()

        g = _num_groups(out_ch)
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=2, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
        )
        self.refine = ResidualBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.down(x))


# ── main encoder ──────────────────────────────────────────────────────

class ConvEncoder(nn.Module):
    """
    Progressive downsampling convolutional encoder.

    Each entry in `channels` halves the spatial resolution while deepening
    features. Global average pooling at the end yields a fixed-size vector
    regardless of input (H, W).

    Input:  (B, in_channels, H, W)
    Output: (B, channels[-1])
    """

    def __init__(self, in_channels: int, channels: List[int]):
        super().__init__()

        assert len(channels) >= 1, "channels must be non-empty"

        blocks: List[nn.Module] = []
        in_ch = in_channels
        for out_ch in channels:
            blocks.append(DownBlock(in_ch, out_ch))
            in_ch = out_ch

        self.net     = nn.Sequential(*blocks)
        self.out_dim = channels[-1] 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)                   # (B, C, H', W')
        return x.mean(dim=(-2, -1))       # global avg pool → (B, channels[-1])