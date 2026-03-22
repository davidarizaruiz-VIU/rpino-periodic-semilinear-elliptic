from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from q1pde.pde import PeriodicNonlinearPoisson


@dataclass
class DataBundle:
    f: np.ndarray
    u: np.ndarray


def _counts(config: dict, split: str) -> int:
    return {
        'train': config['training']['n_train'],
        'val': config['training']['n_val'],
        'test': config['training']['n_test'],
    }[split]


def make_dataset(config: dict, split: str, *, amplitude: float | None = None, modes: int | None = None, grid_size: int | None = None) -> DataBundle:
    n_samples = _counts(config, split)
    pde = PeriodicNonlinearPoisson(
        dimension=config['dimension'],
        grid_size=grid_size or config['grid_size'],
        domain_length=config['domain_length'],
        kappa=config['kappa'],
    )
    rng = np.random.default_rng(config['seed'] + {'train': 0, 'val': 10_000, 'test': 20_000}[split] + (grid_size or config['grid_size']))
    f_list = []
    u_list = []
    amp = float(amplitude if amplitude is not None else config['forcing_amplitude'])
    md = int(modes if modes is not None else config['forcing_modes'])
    for _ in range(n_samples):
        f = pde.forcing_sample(rng, amplitude=amp, modes=md)
        u, _ = pde.solve_reference(
            f,
            max_iter=config['reference_solver']['max_iter'],
            tol=config['reference_solver']['tol'],
            damping=config['reference_solver']['damping'],
        )
        f_list.append(f.astype(np.float32))
        u_list.append(u.astype(np.float32))
    return DataBundle(f=np.stack(f_list), u=np.stack(u_list))
