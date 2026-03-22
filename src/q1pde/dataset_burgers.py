"""Dataset generation for the steady viscous Burgers equation."""
from __future__ import annotations
import numpy as np
from q1pde.dataset import DataBundle
from q1pde.pde_burgers import PeriodicBurgers2D


def make_burgers_dataset(
    config: dict,
    split: str,
    *,
    amplitude: float | None = None,
    modes: int | None = None,
    grid_size: int | None = None,
) -> DataBundle:
    n_samples = {
        'train': config['training']['n_train'],
        'val': config['training']['n_val'],
        'test': config['training']['n_test'],
    }[split]

    gs = grid_size or config['grid_size']
    pde = PeriodicBurgers2D(
        grid_size=gs,
        domain_length=config['domain_length'],
        nu=config['nu'],
        kappa=config['kappa'],
    )
    rng = np.random.default_rng(
        config['seed'] + {'train': 0, 'val': 10_000, 'test': 20_000}[split] + gs
    )
    amp = float(amplitude if amplitude is not None else config['forcing_amplitude'])
    md = int(modes if modes is not None else config['forcing_modes'])
    f_list, u_list = [], []
    for i in range(n_samples):
        f = pde.forcing_sample(rng, amplitude=amp, modes=md)
        u, hist = pde.solve_reference(
            f,
            max_iter=config['reference_solver']['max_iter'],
            tol=config['reference_solver']['tol'],
            damping=config['reference_solver']['damping'],
        )
        f_list.append(f.astype(np.float32))
        u_list.append(u.astype(np.float32))
        if (i + 1) % 50 == 0:
            print(f'  [{split}] {i+1}/{n_samples} samples  (last converged in {len(hist)} iters)')
    return DataBundle(f=np.stack(f_list), u=np.stack(u_list))
