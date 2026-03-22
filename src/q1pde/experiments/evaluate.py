from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from q1pde.metrics import relative_l2_error
from q1pde.model import ResidualNet, RPINO
from q1pde.pde import PeriodicNonlinearPoisson
from q1pde.paths import dataset_dir, evaluation_dir, training_dir
from q1pde.torch_ops import TorchCoarseBackbone
from q1pde.utils import set_seed, write_json


def _tensor_with_channel(x: np.ndarray, dimension: int) -> torch.Tensor:
    if dimension == 1:
        return torch.tensor(x[:, None, :], dtype=torch.float32)
    return torch.tensor(x[:, None, :, :], dtype=torch.float32)


def run_evaluation(config: dict) -> None:
    set_seed(config['seed'])
    device = torch.device('cpu')
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
    )
    model = RPINO(
        backbone_operator=backbone,
        residual_net=residual,
        state_iters_train=config['training']['state_iters_train'],
        state_iters_eval=config['training']['state_iters_eval'],
    ).to(device)
    model.load_state_dict(torch.load(training_dir(config) / 'best_model.pt', map_location=device))
    model.eval()

    test = np.load(dataset_dir(config) / 'test.npz')
    f = test['f']
    u_true = test['u']
    f_t = _tensor_with_channel(f, config['dimension']).to(device)

    with torch.no_grad():
        u_pred = model(f_t).cpu().numpy()[:, 0]
        u_backbone = model.backbone_only(f_t, eval_mode=True).cpu().numpy()[:, 0]

    rel_err_model = relative_l2_error(u_pred, u_true)
    rel_err_backbone = relative_l2_error(u_backbone, u_true)

    ed = evaluation_dir(config)
    pd.DataFrame({
        'sample': np.arange(len(f)),
        'rel_l2_backbone': rel_err_backbone,
        'rel_l2_rpino': rel_err_model,
        'improvement_factor': rel_err_backbone / (rel_err_model + 1e-12),
    }).to_csv(ed / 'sample_metrics.csv', index=False)

    rng = np.random.default_rng(config['seed'] + 999)
    rows = []
    n_pairs = min(config['stability']['n_pairs'], len(f))
    for i in range(n_pairs):
        base_f = f[i]
        delta = config['stability']['perturbation_scale'] * rng.standard_normal(size=base_f.shape)
        pert_f = base_f + delta
        with torch.no_grad():
            u1 = model(_tensor_with_channel(base_f[None, ...], config['dimension']).to(device)).cpu().numpy()[0, 0]
            u2 = model(_tensor_with_channel(pert_f[None, ...], config['dimension']).to(device)).cpu().numpy()[0, 0]
        inp = np.sqrt(np.mean((pert_f - base_f) ** 2))
        out = np.sqrt(np.mean((u2 - u1) ** 2))
        rows.append({'pair': i, 'input_perturbation': inp, 'output_change': out, 'ratio': out / (inp + 1e-12)})
    pd.DataFrame(rows).to_csv(ed / 'stability_metrics.csv', index=False)

    contraction_rows = []
    n_trace = min(6, len(f))
    for i in range(n_trace):
        ff = f[i]
        uu = np.zeros_like(ff)
        prev_inc = None
        for k in range(config['training']['state_iters_eval']):
            uu_next = pde.coarse_linear_solve(ff - config['backbone'].get('nonlinear_weight', 1.0) * uu**3, config['backbone']['coarse_modes'])
            inc = np.sqrt(np.mean((uu_next - uu) ** 2))
            contraction_rows.append({
                'sample': i,
                'iter': k + 1,
                'increment': inc,
                'increment_ratio': (inc / prev_inc) if prev_inc not in (None, 0.0) else np.nan,
            })
            uu = uu_next
            prev_inc = inc
    pd.DataFrame(contraction_rows).to_csv(ed / 'contraction_metrics.csv', index=False)

    summary = {
        'mean_rel_l2_backbone': float(rel_err_backbone.mean()),
        'mean_rel_l2_rpino': float(rel_err_model.mean()),
        'median_rel_l2_backbone': float(np.median(rel_err_backbone)),
        'median_rel_l2_rpino': float(np.median(rel_err_model)),
        'mean_improvement_factor': float(np.mean(rel_err_backbone / (rel_err_model + 1e-12))),
        'fraction_rpino_better': float(np.mean(rel_err_model < rel_err_backbone)),
        'mean_stability_ratio': float(np.mean([r['ratio'] for r in rows])),
    }
    write_json(summary, ed / 'summary.json')
    print(summary)
