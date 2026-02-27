from torch import nn
import torch
# ===================================================================
# Categorical embedding for MultiGrid observations.
# ===================================================================

class CategoricalEmbedding(nn.Module):
    """
    (B, H, W, 3) categorical IDs → (B, C, H, W) continuous embeddings.

    MultiGrid observations encode each cell as (object_type, color, state).
    Each channel is embedded independently into a learned continuous vector,
    then all three are concatenated. This gives the conv encoder a continuous
    input surface to work with rather than raw integer codes.
    """

    def __init__(self, n_obj: int, n_color: int, n_state: int, embed_per_ch: int) -> None:
        super().__init__()
        self.obj   = nn.Embedding(n_obj, embed_per_ch)
        self.color = nn.Embedding(n_color, embed_per_ch)
        self.state = nn.Embedding(n_state, embed_per_ch)
        self.out_channels: int = 3 * embed_per_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.obj(x[..., 0].long())
        c = self.color(x[..., 1].long())
        s = self.state(x[..., 2].long())
        return torch.cat([o, c, s], dim=-1).permute(0, 3, 1, 2)  # (B, C, H, W)
