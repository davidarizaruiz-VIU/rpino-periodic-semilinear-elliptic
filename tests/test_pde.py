"""Unit tests for PDE solvers, models, and core components."""
import numpy as np
import torch
import pytest
from q1pde.pde import PeriodicNonlinearPoisson
from q1pde.pde_burgers import PeriodicBurgers2D
from q1pde.model import ResidualNet, RPINO, build_fno
from q1pde.model_deeponet import build_deeponet
from q1pde.torch_ops import TorchCoarseBackbone
from q1pde.torch_ops_ext import TorchCoarseBackboneBurgers
from q1pde.metrics import relative_l2_error


# ── PDE solvers ──

class TestPoissonSolver:
    def test_linear_solve_shape_1d(self):
        pde = PeriodicNonlinearPoisson(dimension=1, grid_size=32, domain_length=2*np.pi, kappa=1.0)
        f = np.sin(np.linspace(0, 2*np.pi, 32, endpoint=False))
        u = pde.linear_solve(f)
        assert u.shape == f.shape

    def test_linear_solve_shape_2d(self):
        pde = PeriodicNonlinearPoisson(dimension=2, grid_size=16, domain_length=2*np.pi, kappa=1.0)
        f = np.random.randn(16, 16)
        u = pde.linear_solve(f)
        assert u.shape == f.shape

    def test_coarse_linear_solve_2d(self):
        pde = PeriodicNonlinearPoisson(dimension=2, grid_size=16, domain_length=2*np.pi, kappa=1.0)
        f = np.random.randn(16, 16)
        u = pde.coarse_linear_solve(f, coarse_modes=4)
        assert u.shape == f.shape

    def test_solve_converges_2d(self):
        """Full nonlinear solve should converge for smooth forcing."""
        pde = PeriodicNonlinearPoisson(dimension=2, grid_size=16, domain_length=2*np.pi, kappa=2.0)
        f = 0.3 * np.random.randn(16, 16)
        u, converged = pde.solve(f, max_iter=500, tol=1e-8)
        assert converged, "Poisson solver did not converge"
        assert u.shape == f.shape


class TestBurgersSolver:
    def test_linear_solve_shape(self):
        pde = PeriodicBurgers2D(grid_size=16, domain_length=2*np.pi, nu=0.1, kappa=2.0)
        f = np.random.randn(16, 16)
        u = pde.linear_solve(f)
        assert u.shape == f.shape

    def test_nonlinear_term_shape(self):
        pde = PeriodicBurgers2D(grid_size=16, domain_length=2*np.pi, nu=0.1, kappa=2.0)
        u = np.random.randn(16, 16)
        nl = pde.nonlinear_term(u)
        assert nl.shape == u.shape

    def test_coarse_linear_solve(self):
        pde = PeriodicBurgers2D(grid_size=16, domain_length=2*np.pi, nu=0.1, kappa=2.0)
        f = np.random.randn(16, 16)
        u = pde.coarse_linear_solve(f, coarse_modes=4)
        assert u.shape == f.shape

    def test_solve_converges(self):
        pde = PeriodicBurgers2D(grid_size=16, domain_length=2*np.pi, nu=0.1, kappa=2.0)
        f = 0.3 * np.random.randn(16, 16)
        u, converged = pde.solve(f, max_iter=800, tol=1e-8, damping=0.75)
        assert converged, "Burgers solver did not converge"


# ── Models ──

class TestFNO:
    def test_fno_forward_2d(self):
        model = build_fno(dimension=2, width=8, depth=2, modes_x=4, modes_y=4)
        x = torch.randn(3, 1, 16, 16)
        y = model(x)
        assert y.shape == x.shape

    def test_fno_small_param_count(self):
        """FNO-Small should have roughly 214K parameters."""
        model = build_fno(dimension=2, width=23, depth=4, modes_x=10, modes_y=10)
        n = sum(p.numel() for p in model.parameters())
        assert 200_000 < n < 230_000, f"FNO-Small has {n} params, expected ~214K"


class TestDeepONet:
    def test_forward_shape(self):
        model = build_deeponet(grid_size=16, domain_length=2*np.pi, width=32, trunk_hidden=64)
        x = torch.randn(4, 1, 16, 16)
        y = model(x)
        assert y.shape == x.shape

    def test_param_count_at_grid48(self):
        """DeepONet at grid_size=48 should have roughly 209K parameters."""
        model = build_deeponet(grid_size=48, domain_length=2*np.pi, width=64, trunk_hidden=128)
        n = sum(p.numel() for p in model.parameters())
        assert 195_000 < n < 225_000, f"DeepONet has {n} params, expected ~209K"


class TestRPINO:
    def _build_rpino_poisson(self, grid_size=16):
        pde = PeriodicNonlinearPoisson(dimension=2, grid_size=grid_size,
                                        domain_length=2*np.pi, kappa=2.0)
        backbone = TorchCoarseBackbone(pde, coarse_modes=2, nonlinear_weight=1.0)
        residual = ResidualNet(dimension=2, hidden_channels=8, depth=2,
                               rho=0.45, kernel_size=3, controlled=True)
        return RPINO(backbone, residual, state_iters_train=3, state_iters_eval=5)

    def _build_rpino_burgers(self, grid_size=16):
        pde = PeriodicBurgers2D(grid_size=grid_size, domain_length=2*np.pi,
                                 nu=0.1, kappa=2.0)
        backbone = TorchCoarseBackboneBurgers(pde, coarse_modes=2, nonlinear_weight=0.65)
        residual = ResidualNet(dimension=2, hidden_channels=8, depth=2,
                               rho=0.45, kernel_size=3, controlled=True)
        return RPINO(backbone, residual, state_iters_train=3, state_iters_eval=5)

    def test_rpino_poisson_forward(self):
        model = self._build_rpino_poisson()
        x = torch.randn(2, 1, 16, 16)
        y = model(x)
        assert y.shape == x.shape

    def test_rpino_burgers_forward(self):
        model = self._build_rpino_burgers()
        x = torch.randn(2, 1, 16, 16)
        y = model(x)
        assert y.shape == x.shape

    def test_rpino_solve_vs_forward(self):
        """model(f) and model.solve(f, training=True) should agree."""
        model = self._build_rpino_poisson()
        f = torch.randn(2, 1, 16, 16)
        y1 = model(f)
        y2 = model.solve(f, training=True)
        assert torch.allclose(y1, y2, atol=1e-6)

    def test_step_is_contractive(self):
        """Two different inputs to step should produce outputs closer together."""
        model = self._build_rpino_poisson()
        f = torch.randn(1, 1, 16, 16)
        u1 = 0.1 * torch.randn(1, 1, 16, 16)
        u2 = 0.1 * torch.randn(1, 1, 16, 16)
        with torch.no_grad():
            v1 = model.step(u1, f)
            v2 = model.step(u2, f)
        d_in = torch.norm(u1 - u2).item()
        d_out = torch.norm(v1 - v2).item()
        # Contractivity: d_out < d_in (with some tolerance for numerical noise)
        assert d_out < d_in * 1.05, f"Not contractive: {d_out:.4f} >= {d_in:.4f}"


# ── Metrics ──

class TestMetrics:
    def test_relative_l2_error(self):
        pred = np.ones((5, 16, 16))
        true = np.ones((5, 16, 16))
        err = relative_l2_error(pred, true)
        assert np.allclose(err, 0.0)

    def test_relative_l2_error_nonzero(self):
        pred = np.zeros((5, 16, 16))
        true = np.ones((5, 16, 16))
        err = relative_l2_error(pred, true)
        assert np.allclose(err, 1.0)
