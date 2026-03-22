#!/usr/bin/env python3
"""
Diagnostic evaluation for RP-INO on Burgers 2-D.
=================================================

Generates the per-sample diagnostic data needed for the unified A+B
figures in the manuscript:

  1. Per-sample relative L2 errors   → sample_metrics.csv
  2. Contraction traces              → contraction_metrics_test.csv
  3. Stability perturbation ratios   → stability_metrics_test.csv
  4. Iteration-sweep (K=1,2,3,5,8)  → iteration_sweep.csv

These complement the existing Poisson diagnostics in
  results/nonlinear_poisson_2d_v3/evaluation_rpino/

Output directory:
  results/burgers_2d_v1/evaluation_rpino/

Usage (from pde_project/):
    python3 scripts/11_evaluate_burgers_diagnostics.py

Requires: trained RP-INO weights at
    results/burgers_2d_v1/training_rpino/best_model.pt

CPU-only, ~5–10 minutes on iMac M1.
"""
from __future__ import annotations

import sys
from pathlib import Path

# ---- path setup ----
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import pandas as pd
import torch

from q1pde.config import load_config
from q1pde.utils import set_seed, write_json, ensure_dir
from q1pde.paths import dataset_dir, training_dir, evaluation_dir
from q1pde.metrics import relative_l2_error
from q1pde.pde_burgers import PeriodicBurgers2D
from q1pde.torch_ops_ext import TorchCoarseBackboneBurgers
from q1pde.model import ResidualNet, RPINO


# ======================================================================
#  Build RP-INO for Burgers (mirrors build_rpino_burgers in script 10)
# ======================================================================

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


def tensor_with_channel(x: np.ndarray, dimension: int) -> torch.Tensor:
    if dimension == 1:
        return torch.tensor(x[:, None, :], dtype=torch.float32)
    return torch.tensor(x[:, None, :, :], dtype=torch.float32)


# ======================================================================
#  1. Per-sample evaluation
# ======================================================================

def evaluate_per_sample(model, config, out_dir):
    """Compute per-sample relative L2 errors on the test set."""
    print('\n--- Per-sample evaluation on Burgers test set ---')
    ddir = dataset_dir(config)
    data = np.load(ddir / 'test.npz')
    f, u_true = data['f'], data['u']
    f_t = tensor_with_channel(f, config['dimension'])

    with torch.no_grad():
        u_pred = model.solve(f_t, training=False).cpu().numpy()[:, 0]
        u_back = model.backbone_only(f_t, eval_mode=True).cpu().numpy()[:, 0]

    rel_model = relative_l2_error(u_pred, u_true)
    rel_back = relative_l2_error(u_back, u_true)

    df = pd.DataFrame({
        'sample': np.arange(len(f)),
        'rel_l2_rpino': rel_model,
        'rel_l2_backbone': rel_back,
        'improvement_factor_vs_backbone': rel_back / (rel_model + 1e-12),
    })
    df.to_csv(out_dir / 'sample_metrics.csv', index=False)

    summary = {
        'mean_rel_l2_rpino': float(rel_model.mean()),
        'median_rel_l2_rpino': float(np.median(rel_model)),
        'p90_rel_l2_rpino': float(np.quantile(rel_model, 0.9)),
        'mean_rel_l2_backbone': float(rel_back.mean()),
        'fraction_model_better_than_backbone': float(np.mean(rel_model < rel_back)),
        'mean_improvement_factor': float(np.mean(rel_back / (rel_model + 1e-12))),
    }
    write_json(summary, out_dir / 'eval_summary_diagnostics.json')
    print(f'  Mean rel L2 (RP-INO):    {summary["mean_rel_l2_rpino"]:.4f}')
    print(f'  Mean rel L2 (backbone):  {summary["mean_rel_l2_backbone"]:.4f}')
    print(f'  Improvement factor:      {summary["mean_improvement_factor"]:.2f}x')
    return f, u_true


# ======================================================================
#  2. Contraction traces
# ======================================================================

def evaluate_contraction(model, config, f, out_dir, dataset_label='test'):
    """Record ||u^{k+1}-u^k|| per iteration for each test sample."""
    print(f'\n--- Contraction traces ({dataset_label}) ---')
    rows = []
    n_trace = min(8, len(f))  # 8 representative samples
    n_iters = config['training']['state_iters_eval']

    with torch.no_grad():
        for i in range(n_trace):
            ff = tensor_with_channel(f[i:i + 1], config['dimension'])
            u = torch.zeros_like(ff)
            prev_inc = None
            for k in range(n_iters):
                u_next = model.step(u, ff)
                inc = float(torch.sqrt(torch.mean((u_next - u) ** 2)).item())
                ratio = (inc / prev_inc) if prev_inc not in (None, 0.0) else float('nan')
                rows.append({
                    'sample': i,
                    'iter': k + 1,
                    'increment': inc,
                    'increment_ratio': ratio,
                })
                u = u_next
                prev_inc = inc
            print(f'  sample {i}: final increment = {inc:.6e}')

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f'contraction_metrics_{dataset_label}.csv', index=False)
    print(f'  Saved contraction_metrics_{dataset_label}.csv  ({len(rows)} rows)')


# ======================================================================
#  3. Stability perturbation test
# ======================================================================

def evaluate_stability(model, config, f, out_dir, dataset_label='test'):
    """Perturb f → f+δf, measure ||u(f+δf)-u(f)|| / ||δf||."""
    print(f'\n--- Stability perturbation test ({dataset_label}) ---')
    rng = np.random.default_rng(config['seed'] + 999)
    rows = []
    n_pairs = min(config['stability']['n_pairs'], len(f))
    eps = config['stability']['perturbation_scale']

    for i in range(n_pairs):
        base_f = f[i]
        delta = eps * rng.standard_normal(size=base_f.shape)
        pert_f = base_f + delta

        with torch.no_grad():
            u1 = model.solve(
                tensor_with_channel(base_f[None, ...], config['dimension']),
                training=False
            ).cpu().numpy()[0, 0]
            u2 = model.solve(
                tensor_with_channel(pert_f[None, ...], config['dimension']),
                training=False
            ).cpu().numpy()[0, 0]

        inp_norm = np.sqrt(np.mean((pert_f - base_f) ** 2))
        out_norm = np.sqrt(np.mean((u2 - u1) ** 2))
        ratio = out_norm / (inp_norm + 1e-12)
        rows.append({
            'pair': i,
            'dataset': dataset_label,
            'input_perturbation': inp_norm,
            'output_change': out_norm,
            'ratio': ratio,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f'stability_metrics_{dataset_label}.csv', index=False)
    mean_ratio = float(np.mean([r['ratio'] for r in rows]))
    print(f'  Mean stability ratio: {mean_ratio:.4f}')
    print(f'  Saved stability_metrics_{dataset_label}.csv  ({len(rows)} rows)')
    return mean_ratio


# ======================================================================
#  4. Iteration sweep: error vs K
# ======================================================================

def evaluate_iteration_sweep(model, config, out_dir):
    """Evaluate RP-INO test error for K=1,2,3,5,8 fixed-point steps."""
    print('\n--- Iteration sweep (K = 1, 2, 3, 5, 8) ---')
    ddir = dataset_dir(config)
    data = np.load(ddir / 'test.npz')
    f, u_true = data['f'], data['u']
    f_t = tensor_with_channel(f, config['dimension'])

    K_values = [1, 2, 3, 5, 8]
    rows = []

    for K in K_values:
        with torch.no_grad():
            u_pred = model.solve(f_t, training=False, n_iters=K).cpu().numpy()[:, 0]
        rel = relative_l2_error(u_pred, u_true)
        row = {
            'K': K,
            'mean_rel_l2': float(rel.mean()),
            'median_rel_l2': float(np.median(rel)),
            'p90_rel_l2': float(np.quantile(rel, 0.9)),
        }
        rows.append(row)
        print(f'  K={K}: mean={row["mean_rel_l2"]:.4f}  median={row["median_rel_l2"]:.4f}  p90={row["p90_rel_l2"]:.4f}')

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'iteration_sweep.csv', index=False)
    print(f'  Saved iteration_sweep.csv')


# ======================================================================
#  5. (Bonus) Also generate iteration_sweep.csv for Poisson if missing
# ======================================================================

def maybe_generate_poisson_iteration_sweep():
    """If Poisson iteration_sweep.csv is missing, generate it too."""
    cfg_path = ROOT / 'configs' / 'nonlinear_poisson_2d.yaml'
    if not cfg_path.exists():
        return
    config = load_config(cfg_path)
    out_dir = evaluation_dir(config, 'rpino')
    sweep_path = out_dir / 'iteration_sweep.csv'
    if sweep_path.exists():
        print(f'\n  Poisson iteration_sweep.csv already exists, skipping.')
        return

    print('\n--- Also generating Poisson iteration sweep ---')
    from q1pde.pde import PeriodicNonlinearPoisson
    from q1pde.torch_ops import TorchCoarseBackbone

    pde = PeriodicNonlinearPoisson(
        dimension=config['dimension'],
        grid_size=config['grid_size'],
        domain_length=config['domain_length'],
        kappa=config['kappa'],
    )
    backbone = TorchCoarseBackbone(
        pde,
        coarse_modes=config['backbone']['coarse_modes'],
        nonlinear_weight=config['backbone'].get('nonlinear_weight', 1.0),
    )
    residual = ResidualNet(
        dimension=config['dimension'],
        hidden_channels=config['training']['hidden_channels'],
        depth=config['training']['depth'],
        rho=config['training']['rho'],
        kernel_size=config['training'].get('kernel_size', 5),
        controlled=True,
    )
    model = RPINO(backbone, residual,
                  state_iters_train=config['training']['state_iters_train'],
                  state_iters_eval=config['training']['state_iters_eval'])

    weights_path = training_dir(config, 'rpino') / 'best_model.pt'
    if not weights_path.exists():
        print('  Poisson RP-INO weights not found, skipping.')
        return

    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()

    ddir = dataset_dir(config)
    data = np.load(ddir / 'test.npz')
    f_t = tensor_with_channel(data['f'], config['dimension'])
    u_true = data['u']

    K_values = [1, 2, 3, 5, 8]
    rows = []
    for K in K_values:
        with torch.no_grad():
            u_pred = model.solve(f_t, training=False, n_iters=K).cpu().numpy()[:, 0]
        rel = relative_l2_error(u_pred, u_true)
        rows.append({
            'K': K,
            'mean_rel_l2': float(rel.mean()),
            'median_rel_l2': float(np.median(rel)),
            'p90_rel_l2': float(np.quantile(rel, 0.9)),
        })
        print(f'  K={K}: mean={rows[-1]["mean_rel_l2"]:.4f}')

    pd.DataFrame(rows).to_csv(sweep_path, index=False)
    print(f'  Saved Poisson iteration_sweep.csv')


# ======================================================================
#  Main
# ======================================================================

def main():
    print('=' * 60)
    print('Diagnostic evaluation: RP-INO on Burgers 2-D')
    print('=' * 60)

    config = load_config(ROOT / 'configs' / 'burgers_2d.yaml')
    set_seed(config['seed'])

    # Build model
    model = build_rpino_burgers(config, variant='controlled')
    weights_path = training_dir(config, 'rpino') / 'best_model.pt'
    if not weights_path.exists():
        print(f'\nERROR: Trained weights not found at {weights_path}')
        print('Run script 10 first (phase burgers_train).')
        sys.exit(1)

    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Loaded RP-INO Burgers ({n_params:,} params)')

    out_dir = evaluation_dir(config, 'rpino')
    print(f'  Output directory: {out_dir}')

    # Run all diagnostics
    f, u_true = evaluate_per_sample(model, config, out_dir)
    evaluate_contraction(model, config, f, out_dir, dataset_label='test')
    evaluate_stability(model, config, f, out_dir, dataset_label='test')
    evaluate_iteration_sweep(model, config, out_dir)

    # Also make sure Poisson has iteration_sweep.csv
    maybe_generate_poisson_iteration_sweep()

    print('\n' + '=' * 60)
    print('All Burgers diagnostics complete.')
    print(f'Results in: {out_dir}')
    print('=' * 60)
    print('\nGenerated files:')
    print('  - sample_metrics.csv')
    print('  - contraction_metrics_test.csv')
    print('  - stability_metrics_test.csv')
    print('  - iteration_sweep.csv')
    print('  - eval_summary_diagnostics.json')


if __name__ == '__main__':
    main()
