from __future__ import annotations
import numpy as np
from q1pde.dataset import make_dataset
from q1pde.paths import dataset_dir, experiment_root
from q1pde.utils import set_seed, write_json


def run_generate_dataset(config: dict) -> None:
    set_seed(config['seed'])
    root = experiment_root(config)
    ddir = dataset_dir(config)
    summary = {}
    for split in ('train', 'val', 'test'):
        data = make_dataset(config, split)
        np.savez_compressed(ddir / f'{split}.npz', f=data.f, u=data.u)
        summary[split] = {'n_samples': int(data.f.shape[0]), 'grid_shape': list(data.f.shape[1:])}

    if 'shift_test' in config:
        shift = config['shift_test']
        data = make_dataset(
            config,
            'test',
            amplitude=shift.get('amplitude', config['forcing_amplitude']),
            modes=shift.get('modes', config['forcing_modes']),
        )
        np.savez_compressed(ddir / 'test_shift.npz', f=data.f, u=data.u)
        summary['test_shift'] = {'n_samples': int(data.f.shape[0]), 'grid_shape': list(data.f.shape[1:])}

    if 'cross_resolution' in config:
        cr = config['cross_resolution']
        high_n = int(cr['grid_size'])
        data = make_dataset(config, 'test', grid_size=high_n)
        np.savez_compressed(ddir / 'test_crossres.npz', f=data.f, u=data.u)
        summary['test_crossres'] = {'n_samples': int(data.f.shape[0]), 'grid_shape': list(data.f.shape[1:])}

    write_json(summary, root / 'dataset_summary.json')
    print(f'Dataset generated in {ddir}')
