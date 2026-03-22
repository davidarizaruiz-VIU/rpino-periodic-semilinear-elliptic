from __future__ import annotations
import numpy as np
import torch
from q1pde.pde import PeriodicNonlinearPoisson


class TorchCoarseBackbone:
    def __init__(self, pde: PeriodicNonlinearPoisson, coarse_modes: int, nonlinear_weight: float = 1.0):
        self.pde = pde
        self.coarse_modes = int(coarse_modes)
        self.nonlinear_weight = float(nonlinear_weight)

    def __call__(self, u: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        device = u.device
        u_np = u.detach().cpu().numpy()
        f_np = f.detach().cpu().numpy()
        out = []
        for uu, ff in zip(u_np, f_np):
            rhs = ff[0] - self.nonlinear_weight * uu[0] ** 3
            bb = self.pde.coarse_linear_solve(rhs, self.coarse_modes)
            out.append(bb[None, ...])
        arr = np.stack(out).astype(np.float32)
        return torch.tensor(arr, device=device)
