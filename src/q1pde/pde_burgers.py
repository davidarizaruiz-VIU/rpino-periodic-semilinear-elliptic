"""Steady viscous Burgers equation on periodic domain.

Solves:  -nu * Delta u  +  u * du/dx  +  kappa * u  =  f   on  T^d

with periodic boundary conditions, using a spectral (FFT) fixed-point
iteration.  The nonlinearity u * du/dx involves a first-order derivative,
making this problem structurally different from the cubic Poisson benchmark:
the operator is non-self-adjoint and the nonlinear term couples values
and derivatives.
"""
from __future__ import annotations
import math
import numpy as np


class PeriodicBurgers2D:
    """2-D steady viscous Burgers on [0, L)^2 with periodic BCs."""

    def __init__(self, grid_size: int, domain_length: float,
                 nu: float, kappa: float):
        self.dimension = 2
        self.n = grid_size
        self.L = domain_length
        self.nu = float(nu)
        self.kappa = float(kappa)
        self.h = domain_length / grid_size
        self._build_wavenumbers()

    # ------------------------------------------------------------------
    def _build_wavenumbers(self) -> None:
        freq = 2.0 * math.pi * np.fft.fftfreq(self.n, d=self.h)
        kx, ky = np.meshgrid(freq, freq, indexing="ij")
        self.kx = kx                          # for spectral derivative
        self.k2 = kx ** 2 + ky ** 2
        self.linear_symbol = self.nu * self.k2 + self.kappa
        self.linear_symbol[0, 0] = self.kappa  # k = 0 mode

    # ------------------------------------------------------------------
    # Spectral operations
    # ------------------------------------------------------------------
    def spectral_dx(self, u: np.ndarray) -> np.ndarray:
        """Compute du/dx via FFT."""
        return np.fft.ifftn(1j * self.kx * np.fft.fftn(u)).real

    def nonlinear_term(self, u: np.ndarray) -> np.ndarray:
        """Compute u * du/dx  (Burgers nonlinearity)."""
        return u * self.spectral_dx(u)

    # ------------------------------------------------------------------
    # Linear solvers
    # ------------------------------------------------------------------
    def linear_solve(self, rhs: np.ndarray) -> np.ndarray:
        return np.fft.ifftn(np.fft.fftn(rhs) / self.linear_symbol).real

    def coarse_linear_solve(self, rhs: np.ndarray,
                            coarse_modes: int) -> np.ndarray:
        rhs_hat = np.fft.fftn(rhs)
        mask = self.low_mode_mask(coarse_modes)
        u_hat = np.where(mask, rhs_hat / self.linear_symbol, 0.0)
        return np.fft.ifftn(u_hat).real

    def low_mode_mask(self, coarse_modes: int) -> np.ndarray:
        idx = np.fft.fftfreq(self.n) * self.n
        kx, ky = np.meshgrid(idx, idx, indexing="ij")
        return (np.abs(kx) <= coarse_modes) & (np.abs(ky) <= coarse_modes)

    # ------------------------------------------------------------------
    # Fixed-point maps
    # ------------------------------------------------------------------
    def fixed_point_map(self, u: np.ndarray, f: np.ndarray) -> np.ndarray:
        return self.linear_solve(f - self.nonlinear_term(u))

    def coarse_backbone_map(self, u: np.ndarray, f: np.ndarray,
                            coarse_modes: int) -> np.ndarray:
        return self.coarse_linear_solve(f - self.nonlinear_term(u),
                                        coarse_modes=coarse_modes)

    # ------------------------------------------------------------------
    # Reference solver
    # ------------------------------------------------------------------
    def solve_reference(self, f: np.ndarray, max_iter: int = 900,
                        tol: float = 1e-10, damping: float = 0.85):
        u = np.zeros_like(f)
        history: list[dict] = []
        for it in range(max_iter):
            t = self.fixed_point_map(u, f)
            new_u = (1.0 - damping) * u + damping * t
            inc = np.linalg.norm((new_u - u).ravel()) / math.sqrt(new_u.size)
            res = np.linalg.norm(
                (new_u - self.fixed_point_map(new_u, f)).ravel()
            ) / math.sqrt(new_u.size)
            history.append({"iter": it + 1, "increment": inc, "residual": res})
            u = new_u
            if inc < tol:
                break
        return u, history

    # ------------------------------------------------------------------
    # Forcing (same spectral structure as Poisson benchmark)
    # ------------------------------------------------------------------
    def forcing_sample(self, rng: np.random.Generator,
                       amplitude: float, modes: int) -> np.ndarray:
        x = np.linspace(0.0, self.L, self.n, endpoint=False)
        X, Y = np.meshgrid(x, x, indexing="ij")
        f = np.zeros((self.n, self.n), dtype=float)
        for k1 in range(0, modes + 1):
            for k2 in range(0, modes + 1):
                if k1 == 0 and k2 == 0:
                    continue
                scale = amplitude / ((1 + k1 + k2) ** 2)
                coeffs = rng.normal(scale=scale, size=4)
                f += coeffs[0] * np.cos(k1 * X) * np.cos(k2 * Y)
                f += coeffs[1] * np.cos(k1 * X) * np.sin(k2 * Y)
                f += coeffs[2] * np.sin(k1 * X) * np.cos(k2 * Y)
                f += coeffs[3] * np.sin(k1 * X) * np.sin(k2 * Y)
        return f

    def l2_error(self, u: np.ndarray, v: np.ndarray) -> float:
        return float(np.sqrt(np.mean((u - v) ** 2)))
