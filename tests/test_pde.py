import numpy as np
import torch
from q1pde.pde import PeriodicNonlinearPoisson
from q1pde.model import build_fno


def test_linear_solver_shape_1d():
    pde = PeriodicNonlinearPoisson(dimension=1, grid_size=32, domain_length=2*np.pi, kappa=1.0)
    f = np.sin(np.linspace(0, 2*np.pi, 32, endpoint=False))
    u = pde.linear_solve(f)
    assert u.shape == f.shape


def test_linear_solver_shape_2d():
    pde = PeriodicNonlinearPoisson(dimension=2, grid_size=16, domain_length=2*np.pi, kappa=1.0)
    f = np.random.randn(16, 16)
    u = pde.linear_solve(f)
    assert u.shape == f.shape


def test_fno_shape_2d():
    model = build_fno(dimension=2, width=8, depth=2, modes_x=4, modes_y=4)
    x = torch.randn(3, 1, 16, 16)
    y = model(x)
    assert y.shape == x.shape
