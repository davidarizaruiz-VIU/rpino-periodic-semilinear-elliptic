from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from q1pde.experiments.common import build_rpino_model, evaluate_predictions, measure_inference_time, parameter_count, save_eval_summary, tensor_with_channel
from q1pde.paths import dataset_dir, evaluation_dir, training_dir
from q1pde.utils import set_seed


def _load_model(config: dict, variant: str = 'controlled', grid_size: int | None = None):
    device = torch.device('cpu')
    model = build_rpino_model(config, variant=variant, grid_size=grid_size).to(device)
    name = 'rpino' if variant == 'controlled' else f'rpino_{variant}'
    model.load_state_dict(torch.load(training_dir(config, name) / 'best_model.pt', map_location=device))
    model.eval()
    return model, name


def _evaluate_dataset(config: dict, dataset_name: str, *, variant: str = 'controlled', eval_iters: int | None = None, grid_size: int | None = None, compare_backbone: bool = True):
    model, name = _load_model(config, variant=variant, grid_size=grid_size)
    data = np.load(dataset_dir(config) / f'{dataset_name}.npz')
    f = data['f']
    u_true = data['u']
    f_t = tensor_with_channel(f, config['dimension'])
    with torch.no_grad():
        u_pred = model.solve(f_t, training=False, n_iters=eval_iters).cpu().numpy()[:, 0]
        u_back = model.backbone_only(f_t, eval_mode=True, n_iters=eval_iters).cpu().numpy()[:, 0] if compare_backbone else None
    summary, rel_model = evaluate_predictions(u_pred, u_true, model_label=name)
    rows = {'sample': np.arange(len(f)), f'rel_l2_{name}': rel_model}
    if u_back is not None:
        _, rel_back = evaluate_predictions(u_back, u_true, model_label='backbone')
        rows['rel_l2_backbone'] = rel_back
        rows['improvement_factor_vs_backbone'] = rel_back / (rel_model + 1e-12)
        summary.update({
            'mean_rel_l2_backbone': float(rel_back.mean()),
            'median_rel_l2_backbone': float(np.median(rel_back)),
            'fraction_model_better_than_backbone': float(np.mean(rel_model < rel_back)),
            'mean_improvement_factor_vs_backbone': float(np.mean(rel_back / (rel_model + 1e-12))),
        })
    summary['n_parameters'] = parameter_count(model)
    summary['mean_inference_time_seconds'] = measure_inference_time(lambda x: model.solve(x, training=False, n_iters=eval_iters), f_t)
    return summary, pd.DataFrame(rows), model, f, u_true


def _stability_and_contraction(config: dict, model, f: np.ndarray, dataset_label: str, out_dir):
    rng = np.random.default_rng(config['seed'] + 999)
    st_rows = []
    n_pairs = min(config['stability']['n_pairs'], len(f))
    for i in range(n_pairs):
        base_f = f[i]
        delta = config['stability']['perturbation_scale'] * rng.standard_normal(size=base_f.shape)
        pert_f = base_f + delta
        with torch.no_grad():
            u1 = model.solve(tensor_with_channel(base_f[None, ...], config['dimension']), training=False).cpu().numpy()[0, 0]
            u2 = model.solve(tensor_with_channel(pert_f[None, ...], config['dimension']), training=False).cpu().numpy()[0, 0]
        inp = np.sqrt(np.mean((pert_f - base_f) ** 2))
        out = np.sqrt(np.mean((u2 - u1) ** 2))
        st_rows.append({'pair': i, 'dataset': dataset_label, 'input_perturbation': inp, 'output_change': out, 'ratio': out / (inp + 1e-12)})
    pd.DataFrame(st_rows).to_csv(out_dir / f'stability_metrics_{dataset_label}.csv', index=False)

    contraction_rows = []
    n_trace = min(6, len(f))
    with torch.no_grad():
        for i in range(n_trace):
            ff = tensor_with_channel(f[i:i+1], config['dimension'])
            u = torch.zeros_like(ff)
            prev_inc = None
            for k in range(config['training']['state_iters_eval']):
                u_next = model.step(u, ff)
                inc = float(torch.sqrt(torch.mean((u_next - u) ** 2)).item())
                contraction_rows.append({'sample': i, 'iter': k + 1, 'increment': inc, 'increment_ratio': (inc / prev_inc) if prev_inc not in (None, 0.0) else np.nan})
                u = u_next
                prev_inc = inc
    pd.DataFrame(contraction_rows).to_csv(out_dir / f'contraction_metrics_{dataset_label}.csv', index=False)
    return float(np.mean([r['ratio'] for r in st_rows]))


def run_evaluation(config: dict, *, variant: str = 'controlled') -> None:
    set_seed(config['seed'])
    name = 'rpino' if variant == 'controlled' else f'rpino_{variant}'
    out_dir = evaluation_dir(config, name)
    summary, sample_df, model, f, _ = _evaluate_dataset(config, 'test', variant=variant)
    mean_stab = _stability_and_contraction(config, model, f, 'test', out_dir)
    summary['mean_stability_ratio'] = mean_stab
    sample_df.to_csv(out_dir / 'sample_metrics.csv', index=False)
    save_eval_summary(config, name, summary)
    print(summary)


def run_shift_evaluation(config: dict, *, variant: str = 'controlled') -> None:
    name = 'rpino' if variant == 'controlled' else f'rpino_{variant}'
    out_dir = evaluation_dir(config, name)
    summary, sample_df, model, f, _ = _evaluate_dataset(config, 'test_shift', variant=variant)
    mean_stab = _stability_and_contraction(config, model, f, 'test_shift', out_dir)
    summary['mean_stability_ratio'] = mean_stab
    sample_df.to_csv(out_dir / 'sample_metrics_shift.csv', index=False)
    save_eval_summary(config, name + '_shift', summary)
    print(summary)


def run_crossres_evaluation(config: dict, *, variant: str = 'controlled') -> None:
    high_n = int(config['cross_resolution']['grid_size'])
    name = 'rpino' if variant == 'controlled' else f'rpino_{variant}'
    out_dir = evaluation_dir(config, name)
    summary, sample_df, model, f, _ = _evaluate_dataset(config, 'test_crossres', variant=variant, grid_size=high_n)
    mean_stab = _stability_and_contraction(config, model, f, 'test_crossres', out_dir)
    summary['mean_stability_ratio'] = mean_stab
    sample_df.to_csv(out_dir / 'sample_metrics_crossres.csv', index=False)
    save_eval_summary(config, name + '_crossres', summary)
    print(summary)
