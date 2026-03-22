#!/usr/bin/env python3
"""
Ablation study for RP-INO on Burgers 2-D.
==========================================

Trains the *free* (unconstrained) residual variant on Burgers,
then compares backbone-only, free, and controlled RP-INO.

This complements the Poisson ablation (script 06) so that Table 5
in the manuscript covers both problems.

Output:
    results/burgers_2d_v1/ablations/ablation_summary.csv

Usage (from pde_project/):
    python3 scripts/13_ablation_burgers.py

Prerequisites:
    - Burgers dataset generated  (script 10, phase burgers_data)
    - Controlled RP-INO trained  (script 10, phase burgers_train)

CPU-only, ~15–25 min on iMac M1 (training the free variant).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import pandas as pd
import torch

from q1pde.config import load_config
from q1pde.utils import set_seed, ensure_dir
from q1pde.paths import dataset_dir, training_dir
from q1pde.metrics import relative_l2_error
from q1pde.torch_data import make_loader
from q1pde.pde_burgers import PeriodicBurgers2D
from q1pde.torch_ops_ext import TorchCoarseBackboneBurgers
from q1pde.model import ResidualNet, RPINO

# Reuse the generic training loop from script 10
from importlib.util import spec_from_file_location, module_from_spec
_spec = spec_from_file_location("ext", ROOT / 'scripts' / '10_run_extended_experiments.py')
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)
train_model = _mod.train_model
parameter_count = _mod.parameter_count


def build_rpino_burgers(config, *, variant='controlled'):
    pde = PeriodicBurgers2D(
        grid_size=config['grid_size'],
        domain_length=config['domain_length'],
        nu=config['nu'],
        kappa=config['kappa'],
    )
    backbone = TorchCoarseBackboneBurgers(
        pde,
        coarse_modes=config['backbone']['coarse_modes'],
        nonlinear_weight=config['backbone'].get('nonlinear_weight', 1.0),
    )
    controlled = variant != 'free'
    rho = config['training']['rho'] if controlled else config['training'].get('free_rho', 1.0)
    residual = ResidualNet(
        dimension=config['dimension'],
        hidden_channels=config['training']['hidden_channels'],
        depth=config['training']['depth'],
        rho=rho,
        kernel_size=config['training'].get('kernel_size', 5),
        controlled=controlled,
    )
    return RPINO(
        backbone_operator=backbone,
        residual_net=residual,
        state_iters_train=config['training']['state_iters_train'],
        state_iters_eval=config['training']['state_iters_eval'],
    )


def tensor_with_channel(x, dim):
    if dim == 1:
        return torch.tensor(x[:, None, :], dtype=torch.float32)
    return torch.tensor(x[:, None, :, :], dtype=torch.float32)


def main():
    print('=' * 60)
    print('Ablation: RP-INO (free vs controlled) on Burgers 2-D')
    print('=' * 60)

    config = load_config(ROOT / 'configs' / 'burgers_2d.yaml')
    set_seed(config['seed'])

    ddir = dataset_dir(config)
    dim = config['dimension']
    bs = config['training']['batch_size']

    # ── 1. Train free variant (if not already trained) ──
    free_dir = training_dir(config, 'rpino_free')
    if (free_dir / 'best_model.pt').exists():
        print(f'\nFree variant already trained at {free_dir}, skipping.')
    else:
        print('\n--- Training free (unconstrained) RP-INO on Burgers ---')
        train_data = np.load(ddir / 'train.npz')
        val_data = np.load(ddir / 'val.npz')
        train_loader = make_loader(train_data['f'], train_data['u'], dim, bs, True)
        val_loader = make_loader(val_data['f'], val_data['u'], dim, bs, False)

        free_model = build_rpino_burgers(config, variant='free')
        print(f'  Free RP-INO: {parameter_count(free_model):,} params')
        train_model(free_model, train_loader, val_loader, config,
                    model_name='rpino_free', save_dir=free_dir)

    # ── 2. Evaluate all three variants on test set ──
    print('\n--- Evaluating ablation variants ---')
    data = np.load(ddir / 'test.npz')
    f, u_true = data['f'], data['u']
    f_t = tensor_with_channel(f, dim)
    rows = []

    # (a) Backbone only (from controlled model)
    ctrl_model = build_rpino_burgers(config, variant='controlled')
    ctrl_weights = training_dir(config, 'rpino') / 'best_model.pt'
    if not ctrl_weights.exists():
        print(f'ERROR: Controlled weights not found at {ctrl_weights}')
        sys.exit(1)
    ctrl_model.load_state_dict(torch.load(ctrl_weights, map_location='cpu'))
    ctrl_model.eval()

    with torch.no_grad():
        back = ctrl_model.backbone_only(f_t, eval_mode=True).cpu().numpy()[:, 0]
    rel_back = relative_l2_error(back, u_true)
    rows.append({
        'variant': 'Backbone',
        'mean_rel_l2': float(rel_back.mean()),
        'median_rel_l2': float(np.median(rel_back)),
    })
    print(f'  Backbone:          mean={rel_back.mean():.4f}  median={np.median(rel_back):.4f}')

    # (b) Free variant
    free_model = build_rpino_burgers(config, variant='free')
    free_model.load_state_dict(torch.load(free_dir / 'best_model.pt', map_location='cpu'))
    free_model.eval()
    with torch.no_grad():
        pred_free = free_model.solve(f_t, training=False).cpu().numpy()[:, 0]
    rel_free = relative_l2_error(pred_free, u_true)
    rows.append({
        'variant': 'Free residual',
        'mean_rel_l2': float(rel_free.mean()),
        'median_rel_l2': float(np.median(rel_free)),
    })
    print(f'  Free residual:     mean={rel_free.mean():.4f}  median={np.median(rel_free):.4f}')

    # (c) Controlled variant
    with torch.no_grad():
        pred_ctrl = ctrl_model.solve(f_t, training=False).cpu().numpy()[:, 0]
    rel_ctrl = relative_l2_error(pred_ctrl, u_true)
    rows.append({
        'variant': 'Controlled RP-INO',
        'mean_rel_l2': float(rel_ctrl.mean()),
        'median_rel_l2': float(np.median(rel_ctrl)),
    })
    print(f'  Controlled RP-INO: mean={rel_ctrl.mean():.4f}  median={np.median(rel_ctrl):.4f}')

    # ── 3. Save ──
    out_dir = ensure_dir(Path('results') / config['experiment_name'] / 'ablations')
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'ablation_summary.csv', index=False)
    print(f'\nSaved: {out_dir / "ablation_summary.csv"}')
    print('\n' + df.to_string(index=False))


if __name__ == '__main__':
    main()
