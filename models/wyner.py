import torch as th

from typing import Optional, List, Tuple

@th.jit.script
class WynerOutput:
    def __init__(
        self,
        recon: th.Tensor,
        recon_next: th.Tensor,
        w: th.Tensor,
        logvar: th.Tensor,
    ):
        self.recon = recon
        self.recon_next = recon_next
        self.w = w
        self.logvar = logvar


@th.jit.script
class WynerLoss:
    def __init__(
        self,
        recon_loss: th.Tensor,
        recon_next_loss: th.Tensor,
        kl_loss: th.Tensor,
    ):
        self.recon_loss = recon_loss
        self.recon_next_loss = recon_next_loss
        self.kl_loss = kl_loss

@th.jit.interface
class WynerInterface:
    def encode(self, w: th.Tensor, mu: th.Tensor, skips: Optional[List[th.Tensor]]) -> Tuple[th.Tensor, th.Tensor]:
        pass

    def decode(self, z: th.Tensor, mu: th.Tensor) -> th.Tensor:
        pass

    def forward(self, w: th.Tensor, mu: th.Tensor, mu_next: th.Tensor, skips: Optional[List[th.Tensor]]) -> WynerOutput:
        pass

    def loss(self, output: WynerOutput, recon_target: th.Tensor, recon_next_target: th.Tensor) -> WynerLoss:
        pass