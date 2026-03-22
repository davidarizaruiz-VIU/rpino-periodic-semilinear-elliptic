from __future__ import annotations
import json
import numpy as np
import pandas as pd
import torch
from q1pde.experiments.common import build_rpino_model, tensor_with_channel
from q1pde.paths import aux_dir, dataset_dir, evaluation_dir, training_dir
from q1pde.metrics import relative_l2_error


def run_ablation(config: dict) -> None:
    data = np.load(dataset_dir(config) / 'test.npz')
    f = data['f']
    u_true = data['u']
    f_t = tensor_with_channel(f, config['dimension'])
    rows = []

    # Backbone only from trained controlled model structure
    model = build_rpino_model(config, variant='controlled')
    model.load_state_dict(torch.load(training_dir(config, 'rpino') / 'best_model.pt', map_location='cpu'))
    model.eval()
    with torch.no_grad():
        back = model.backbone_only(f_t, eval_mode=True).cpu().numpy()[:, 0]
    rel_back = relative_l2_error(back, u_true)
    rows.append({'variant': 'backbone', 'mean_rel_l2': float(rel_back.mean()), 'median_rel_l2': float(np.median(rel_back)), 'fraction_better_than_backbone': np.nan})

    for variant in ('free', 'controlled'):
        name = 'rpino' if variant == 'controlled' else f'rpino_{variant}'
        model = build_rpino_model(config, variant=variant)
        model.load_state_dict(torch.load(training_dir(config, name) / 'best_model.pt', map_location='cpu'))
        model.eval()
        with torch.no_grad():
            pred = model.solve(f_t, training=False).cpu().numpy()[:, 0]
        rel = relative_l2_error(pred, u_true)
        rows.append({
            'variant': name,
            'mean_rel_l2': float(rel.mean()),
            'median_rel_l2': float(np.median(rel)),
            'fraction_better_than_backbone': float(np.mean(rel < rel_back)),
        })

    out = aux_dir(config, 'ablations')
    pd.DataFrame(rows).to_csv(out / 'ablation_summary.csv', index=False)
    print(pd.DataFrame(rows))


def run_iteration_sweep(config: dict, eval_iters_list: list[int]) -> None:
    data = np.load(dataset_dir(config) / 'test.npz')
    f = data['f']
    u_true = data['u']
    f_t = tensor_with_channel(f, config['dimension'])
    model = build_rpino_model(config, variant='controlled')
    model.load_state_dict(torch.load(training_dir(config, 'rpino') / 'best_model.pt', map_location='cpu'))
    model.eval()
    rows = []
    with torch.no_grad():
        for k in eval_iters_list:
            pred = model.solve(f_t, training=False, n_iters=k).cpu().numpy()[:, 0]
            rel = relative_l2_error(pred, u_true)
            rows.append({'eval_iters': k, 'mean_rel_l2': float(rel.mean()), 'median_rel_l2': float(np.median(rel)), 'p90_rel_l2': float(np.quantile(rel, 0.9))})
    out = aux_dir(config, 'ablations')
    pd.DataFrame(rows).to_csv(out / 'iteration_sweep.csv', index=False)
    print(pd.DataFrame(rows))
