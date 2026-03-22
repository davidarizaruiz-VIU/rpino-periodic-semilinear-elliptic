#!/usr/bin/env python3
"""
Extended experiments for the RP-INO paper.
==========================================

Adds:
  1. Steady viscous Burgers 2-D benchmark.
  2. FNO-Small  (~214 K params, fair comparison with RP-INO ~207 K).
  3. DeepONet baseline (~209 K params).
  4. Learning-curve analysis  (25 %, 50 %, 75 %, 100 % of training data).
  5. Summary tables printed to stdout and saved as CSV.

Usage (from pde_project/):
    python scripts/10_run_extended_experiments.py --phase all

Phases can be run individually:
    --phase burgers_data
    --phase burgers_train
    --phase poisson_extra
    --phase learning_curves
    --phase summary
    --phase all              (default)

CPU-only, compatible with iMac M1.
"""
from __future__ import annotations

import sys, argparse, time, math
from pathlib import Path

# ---- path setup ----
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import pandas as pd
import torch
from torch import nn

from q1pde.config import load_config
from q1pde.utils import set_seed, write_json, ensure_dir
from q1pde.paths import dataset_dir, training_dir, evaluation_dir, experiment_root
from q1pde.metrics import relative_l2_error
from q1pde.torch_data import make_loader

# ---------- PDE solvers ----------
from q1pde.pde import PeriodicNonlinearPoisson
from q1pde.pde_burgers import PeriodicBurgers2D
from q1pde.dataset import DataBundle, make_dataset
from q1pde.dataset_burgers import make_burgers_dataset

# ---------- Models ----------
from q1pde.model import ResidualNet, RPINO, build_fno
from q1pde.model_deeponet import build_deeponet
from q1pde.torch_ops import TorchCoarseBackbone
from q1pde.torch_ops_ext import TorchCoarseBackboneBurgers

# ======================================================================
#  Helpers
# ======================================================================

def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def train_model(model, train_loader, val_loader, config, *, model_name, save_dir):
    """Generic training loop (Adam + cosine annealing + grad clip)."""
    device = torch.device('cpu')
    model = model.to(device)
    tcfg = config['training']
    opt = torch.optim.Adam(model.parameters(), lr=tcfg['lr'],
                           weight_decay=tcfg.get('weight_decay', 0.0))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg['epochs'])
    loss_fn = nn.MSELoss()
    best_val = float('inf')
    history = []

    for epoch in range(1, tcfg['epochs'] + 1):
        model.train()
        train_loss = 0.0
        for fb, ub in train_loader:
            fb, ub = fb.to(device), ub.to(device)
            pred = model(fb)
            loss = loss_fn(pred, ub)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            train_loss += loss.item() * fb.size(0)
        scheduler.step()
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for fb, ub in val_loader:
                fb, ub = fb.to(device), ub.to(device)
                val_loss += loss_fn(model(fb), ub).item() * fb.size(0)
        val_loss /= len(val_loader.dataset)

        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
        if epoch % 10 == 0 or epoch == 1:
            print(f'  [{model_name}] epoch {epoch:3d}  train={train_loss:.4e}  val={val_loss:.4e}')
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_dir / 'best_model.pt')

    pd.DataFrame(history).to_csv(save_dir / 'history.csv', index=False)
    write_json({'best_val_loss': best_val, 'params': parameter_count(model)}, save_dir / 'training_summary.json')
    return best_val


def evaluate_model(model, test_npz, config, *, model_name, save_dir):
    """Evaluate a model on a test set and return summary dict."""
    data = np.load(test_npz)
    dim = config['dimension']
    loader = make_loader(data['f'], data['u'], dim, config['training']['batch_size'], False)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for fb, ub in loader:
            preds.append(model(fb).cpu().numpy())
            trues.append(ub.cpu().numpy())
    pred = np.concatenate(preds)[:, 0]
    true = np.concatenate(trues)[:, 0]
    rel = relative_l2_error(pred, true)
    summary = {
        f'mean_rel_l2': float(rel.mean()),
        f'median_rel_l2': float(np.median(rel)),
        f'p90_rel_l2': float(np.quantile(rel, 0.9)),
        'params': parameter_count(model),
        'model': model_name,
    }
    write_json(summary, save_dir / 'eval_summary.json')
    return summary


# ======================================================================
#  Builder functions
# ======================================================================

def build_rpino_poisson(config, *, variant='controlled'):
    pde = PeriodicNonlinearPoisson(
        dimension=config['dimension'],
        grid_size=config['grid_size'],
        domain_length=config['domain_length'],
        kappa=config['kappa'],
    )
    backbone = TorchCoarseBackbone(pde,
                                   coarse_modes=config['backbone']['coarse_modes'],
                                   nonlinear_weight=config['backbone'].get('nonlinear_weight', 1.0))
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
    return RPINO(backbone, residual,
                 state_iters_train=config['training']['state_iters_train'],
                 state_iters_eval=config['training']['state_iters_eval'])


def build_rpino_burgers(config, *, variant='controlled'):
    pde = PeriodicBurgers2D(
        grid_size=config['grid_size'],
        domain_length=config['domain_length'],
        nu=config['nu'],
        kappa=config['kappa'],
    )
    backbone = TorchCoarseBackboneBurgers(pde,
                                          coarse_modes=config['backbone']['coarse_modes'],
                                          nonlinear_weight=config['backbone'].get('nonlinear_weight', 1.0))
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
    return RPINO(backbone, residual,
                 state_iters_train=config['training']['state_iters_train'],
                 state_iters_eval=config['training']['state_iters_eval'])


def build_fno_small(config):
    fcfg = config['fno_small']
    return build_fno(dimension=config['dimension'],
                     width=fcfg['width'], depth=fcfg['depth'],
                     modes_x=fcfg['modes_x'], modes_y=fcfg.get('modes_y'))


def build_deeponet_model(config):
    dcfg = config['deeponet']
    return build_deeponet(grid_size=config['grid_size'],
                          domain_length=config['domain_length'],
                          width=dcfg['width'],
                          trunk_hidden=dcfg['trunk_hidden'])


# ======================================================================
#  Phase 1: Burgers dataset
# ======================================================================

def phase_burgers_data(cfg_burgers):
    print('\n' + '='*60)
    print('PHASE 1: Generate Burgers 2-D dataset')
    print('='*60)
    set_seed(cfg_burgers['seed'])
    ddir = dataset_dir(cfg_burgers)
    for split in ('train', 'val', 'test'):
        print(f'  Generating {split} split ...')
        data = make_burgers_dataset(cfg_burgers, split)
        np.savez_compressed(ddir / f'{split}.npz', f=data.f, u=data.u)
        print(f'    -> {data.f.shape[0]} samples, grid {data.f.shape[1:]}')
    # shift test
    if 'shift_test' in cfg_burgers:
        s = cfg_burgers['shift_test']
        data = make_burgers_dataset(cfg_burgers, 'test', amplitude=s['amplitude'], modes=s['modes'])
        np.savez_compressed(ddir / 'test_shift.npz', f=data.f, u=data.u)
    print('  Burgers dataset done.\n')


# ======================================================================
#  Phase 2: Train all models on Burgers
# ======================================================================

def phase_burgers_train(cfg_burgers):
    print('\n' + '='*60)
    print('PHASE 2: Train models on Burgers 2-D')
    print('='*60)
    set_seed(cfg_burgers['seed'])
    ddir = dataset_dir(cfg_burgers)
    dim = cfg_burgers['dimension']
    bs = cfg_burgers['training']['batch_size']
    train_data = np.load(ddir / 'train.npz')
    val_data = np.load(ddir / 'val.npz')
    train_loader = make_loader(train_data['f'], train_data['u'], dim, bs, True)
    val_loader = make_loader(val_data['f'], val_data['u'], dim, bs, False)

    models = {
        'rpino': build_rpino_burgers(cfg_burgers, variant='controlled'),
        'fno': build_fno(dimension=dim, width=cfg_burgers['fno']['width'],
                         depth=cfg_burgers['fno']['depth'],
                         modes_x=cfg_burgers['fno']['modes_x'],
                         modes_y=cfg_burgers['fno'].get('modes_y')),
        'fno_small': build_fno_small(cfg_burgers),
        'deeponet': build_deeponet_model(cfg_burgers),
    }

    for name, model in models.items():
        print(f'\n--- Training {name} ({parameter_count(model):,} params) ---')
        sdir = training_dir(cfg_burgers, name)
        train_model(model, train_loader, val_loader, cfg_burgers, model_name=name, save_dir=sdir)

    # Evaluate all on test
    print('\n--- Evaluating on Burgers test set ---')
    results = []
    for name, model in models.items():
        sdir = training_dir(cfg_burgers, name)
        model.load_state_dict(torch.load(sdir / 'best_model.pt', weights_only=True))
        edir = evaluation_dir(cfg_burgers, name)
        summary = evaluate_model(model, ddir / 'test.npz', cfg_burgers, model_name=name, save_dir=edir)
        results.append(summary)
        print(f'  {name:15s}: mean_rel_l2 = {summary["mean_rel_l2"]:.4f}  ({summary["params"]:,} params)')

    # Shift test
    if (ddir / 'test_shift.npz').exists():
        print('\n--- Shift-test on Burgers ---')
        for name, model in models.items():
            sdir = training_dir(cfg_burgers, name)
            model.load_state_dict(torch.load(sdir / 'best_model.pt', weights_only=True))
            edir = evaluation_dir(cfg_burgers, name)
            summary = evaluate_model(model, ddir / 'test_shift.npz', cfg_burgers,
                                     model_name=name, save_dir=ensure_dir(edir / 'shift'))
            print(f'  {name:15s}: mean_rel_l2 (shift) = {summary["mean_rel_l2"]:.4f}')


# ======================================================================
#  Phase 3: Extra baselines on Poisson
# ======================================================================

def phase_poisson_extra(cfg_poisson):
    print('\n' + '='*60)
    print('PHASE 3: FNO-Small + DeepONet on Poisson 2-D')
    print('='*60)
    set_seed(cfg_poisson['seed'])
    ddir = dataset_dir(cfg_poisson)
    dim = cfg_poisson['dimension']
    bs = cfg_poisson['training']['batch_size']
    train_data = np.load(ddir / 'train.npz')
    val_data = np.load(ddir / 'val.npz')
    train_loader = make_loader(train_data['f'], train_data['u'], dim, bs, True)
    val_loader = make_loader(val_data['f'], val_data['u'], dim, bs, False)

    models = {
        'fno_small': build_fno_small(cfg_poisson),
        'deeponet': build_deeponet_model(cfg_poisson),
    }

    for name, model in models.items():
        print(f'\n--- Training {name} ({parameter_count(model):,} params) on Poisson ---')
        sdir = training_dir(cfg_poisson, name)
        train_model(model, train_loader, val_loader, cfg_poisson, model_name=name, save_dir=sdir)

    # Evaluate
    print('\n--- Evaluating on Poisson test set ---')
    for name, model in models.items():
        sdir = training_dir(cfg_poisson, name)
        model.load_state_dict(torch.load(sdir / 'best_model.pt', weights_only=True))
        edir = evaluation_dir(cfg_poisson, name)
        summary = evaluate_model(model, ddir / 'test.npz', cfg_poisson, model_name=name, save_dir=edir)
        print(f'  {name:15s}: mean_rel_l2 = {summary["mean_rel_l2"]:.4f}  ({summary["params"]:,} params)')

    # Shift test
    print('\n--- Shift-test on Poisson ---')
    for name, model in models.items():
        sdir = training_dir(cfg_poisson, name)
        model.load_state_dict(torch.load(sdir / 'best_model.pt', weights_only=True))
        edir = evaluation_dir(cfg_poisson, name)
        if (ddir / 'test_shift.npz').exists():
            summary = evaluate_model(model, ddir / 'test_shift.npz', cfg_poisson,
                                     model_name=name, save_dir=ensure_dir(edir / 'shift'))
            print(f'  {name:15s}: mean_rel_l2 (shift) = {summary["mean_rel_l2"]:.4f}')


# ======================================================================
#  Phase 4: Learning curves
# ======================================================================

def phase_learning_curves(cfg_poisson, cfg_burgers):
    print('\n' + '='*60)
    print('PHASE 4: Learning curves (error vs training set size)')
    print('='*60)

    fractions = [0.25, 0.50, 0.75, 1.00]

    for label, cfg, builder in [
        ('Poisson', cfg_poisson, lambda c: build_rpino_poisson(c)),
        ('Burgers', cfg_burgers, lambda c: build_rpino_burgers(c)),
    ]:
        print(f'\n--- Learning curves for {label} ---')
        ddir = dataset_dir(cfg)
        train_data = np.load(ddir / 'train.npz')
        val_data = np.load(ddir / 'val.npz')
        dim = cfg['dimension']
        bs = cfg['training']['batch_size']
        val_loader = make_loader(val_data['f'], val_data['u'], dim, bs, False)
        n_full = train_data['f'].shape[0]

        lc_results = []
        for frac in fractions:
            n_sub = max(int(n_full * frac), bs)
            print(f'  fraction={frac:.0%}  ({n_sub} samples)')

            sub_f = train_data['f'][:n_sub]
            sub_u = train_data['u'][:n_sub]
            sub_loader = make_loader(sub_f, sub_u, dim, bs, True)

            # RP-INO
            set_seed(cfg['seed'])
            rpino = builder(cfg)
            sdir = ensure_dir(experiment_root(cfg) / f'learning_curve/rpino_frac{frac:.2f}')
            bv = train_model(rpino, sub_loader, val_loader, cfg, model_name=f'rpino_{frac:.0%}', save_dir=sdir)
            rpino.load_state_dict(torch.load(sdir / 'best_model.pt', weights_only=True))
            ev = evaluate_model(rpino, ddir / 'test.npz', cfg, model_name=f'rpino_{frac:.0%}', save_dir=sdir)
            lc_results.append({'pde': label, 'model': 'RP-INO', 'frac': frac, 'n_train': n_sub,
                               'mean_rel_l2': ev['mean_rel_l2']})

            # FNO-Small
            set_seed(cfg['seed'])
            fno_s = build_fno_small(cfg)
            sdir = ensure_dir(experiment_root(cfg) / f'learning_curve/fno_small_frac{frac:.2f}')
            train_model(fno_s, sub_loader, val_loader, cfg, model_name=f'fno_small_{frac:.0%}', save_dir=sdir)
            fno_s.load_state_dict(torch.load(sdir / 'best_model.pt', weights_only=True))
            ev = evaluate_model(fno_s, ddir / 'test.npz', cfg, model_name=f'fno_small_{frac:.0%}', save_dir=sdir)
            lc_results.append({'pde': label, 'model': 'FNO-Small', 'frac': frac, 'n_train': n_sub,
                               'mean_rel_l2': ev['mean_rel_l2']})

            # DeepONet
            set_seed(cfg['seed'])
            don = build_deeponet_model(cfg)
            sdir = ensure_dir(experiment_root(cfg) / f'learning_curve/deeponet_frac{frac:.2f}')
            train_model(don, sub_loader, val_loader, cfg, model_name=f'deeponet_{frac:.0%}', save_dir=sdir)
            don.load_state_dict(torch.load(sdir / 'best_model.pt', weights_only=True))
            ev = evaluate_model(don, ddir / 'test.npz', cfg, model_name=f'deeponet_{frac:.0%}', save_dir=sdir)
            lc_results.append({'pde': label, 'model': 'DeepONet', 'frac': frac, 'n_train': n_sub,
                               'mean_rel_l2': ev['mean_rel_l2']})

        df = pd.DataFrame(lc_results)
        lc_path = experiment_root(cfg) / 'learning_curve' / 'learning_curves.csv'
        df.to_csv(lc_path, index=False)
        print(f'  Saved learning curves -> {lc_path}')
        print(df.to_string(index=False))


# ======================================================================
#  Phase 5: Grand summary table
# ======================================================================

def phase_summary(cfg_poisson, cfg_burgers):
    print('\n' + '='*60)
    print('PHASE 5: Summary tables')
    print('='*60)

    rows = []
    for pde_label, cfg in [('Poisson', cfg_poisson), ('Burgers', cfg_burgers)]:
        root = experiment_root(cfg)
        for model_name in ['rpino', 'rpino_free', 'fno', 'fno_small', 'deeponet']:
            edir = root / f'evaluation_{model_name}'
            jpath = edir / 'eval_summary.json'
            if jpath.exists():
                import json
                with open(jpath) as f:
                    s = json.load(f)
                rows.append({
                    'PDE': pde_label,
                    'Model': s.get('model', model_name),
                    'Params': s.get('params', ''),
                    'Mean Rel L2': f'{s["mean_rel_l2"]:.4f}',
                    'Median Rel L2': f'{s.get("median_rel_l2", float("nan")):.4f}',
                    'P90 Rel L2': f'{s.get("p90_rel_l2", float("nan")):.4f}',
                })

    if rows:
        df = pd.DataFrame(rows)
        print('\n' + df.to_string(index=False))
        summary_path = ROOT / 'results' / 'extended_summary.csv'
        df.to_csv(summary_path, index=False)
        print(f'\nSaved -> {summary_path}')
    else:
        print('No evaluation results found yet. Run the training phases first.')


# ======================================================================
#  Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description='Extended RP-INO experiments')
    parser.add_argument('--phase', default='all',
                        choices=['burgers_data', 'burgers_train', 'poisson_extra',
                                 'learning_curves', 'summary', 'all'])
    args = parser.parse_args()

    cfg_poisson = load_config(ROOT / 'configs' / 'nonlinear_poisson_2d.yaml')
    cfg_burgers = load_config(ROOT / 'configs' / 'burgers_2d.yaml')

    t0 = time.time()

    if args.phase in ('all', 'burgers_data'):
        phase_burgers_data(cfg_burgers)

    if args.phase in ('all', 'burgers_train'):
        phase_burgers_train(cfg_burgers)

    if args.phase in ('all', 'poisson_extra'):
        phase_poisson_extra(cfg_poisson)

    if args.phase in ('all', 'learning_curves'):
        phase_learning_curves(cfg_poisson, cfg_burgers)

    if args.phase in ('all', 'summary'):
        phase_summary(cfg_poisson, cfg_burgers)

    elapsed = time.time() - t0
    print(f'\nTotal elapsed time: {elapsed/60:.1f} minutes')


if __name__ == '__main__':
    main()
