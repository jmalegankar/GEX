import torch
from lmu_ppo.episodic_bonus import EllipticalEpisodicBonus

eb = EllipticalEpisodicBonus(n_envs=1, dim=4, lambda_reg=1.0)

# Test 1: standard case
phi = torch.tensor([[1., 0., 0., 0.]])
b1 = eb.bonus_and_update(phi); print(f"b_1 = {b1.item():.4f}")  # 1.0
b2 = eb.bonus_and_update(phi); print(f"b_2 = {b2.item():.4f}")  # 0.5
b3 = eb.bonus_and_update(phi); print(f"b_3 = {b3.item():.4f}")  # ~0.333

# Test 2: large bonus should not corrupt M
eb.reset_all()
big_phi = torch.tensor([[10., 0., 0., 0.]])
b_big = eb.bonus_and_update(big_phi); print(f"b_big = {b_big.item():.4f}")  # 100
print(f"min eigval after big phi: {eb.min_eigenvalue():.6f}")              # > 0
b_again = eb.bonus_and_update(big_phi); print(f"b_again = {b_again.item():.4f}")  # ~0.99

# Test 3: M stays PSD
print(f"diag = {eb.M[0].diag()}")  # all entries > 0