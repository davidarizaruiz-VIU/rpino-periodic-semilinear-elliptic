from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from q1pde.experiments.common import build_fno_model, evaluate_predictions, measure_inference_time, parameter_count, save_eval_summary, tensor_with_channel
from q1pde.paths import dataset_dir, evaluation_dir, training_dir
from q1pde.utils import set_seed


def _load_fno(config: dict):
    device = torch.device('cpu')
    model = build_fno_model(config).to(device)
    model.load_state_dict(torch.load(training_dir(config, 'fno') / 'best_model.pt', map_location=device))
    model.eval()
    return model


def _stability(config: dict, model, f: np.ndarray):
    rng = np.random.default_rng(config['seed'] + 1234)
    rows = []
    n_pairs = min(config['stability']['n_pairs'], len(f))
    for i in range(n_pairs):
        base_f = f[i]
        delta = config['stability']['perturbation_scale'] * rng.standard_normal(size=base_f.shape)
        pert_f = base_f + delta
        with torch.no_grad():
            u1 = model(tensor_with_channel(base_f[None, ...], config['dimension'])).cpu().numpy()[0, 0]
            u2 = model(tensor_with_channel(pert_f[None, ...], config['dimension'])).cpu().numpy()[0, 0]
        inp = np.sqrt(np.mean((pert_f - base_f) ** 2))
        out = np.sqrt(np.mean((u2 - u1) ** 2))
        rows.append({'pair': i, 'input_perturbation': inp, 'output_change': out, 'ratio': out / (inp + 1e-12)})
    return rows


def run_evaluation_fno(config: dict, dataset_name: str = 'test') -> None:
    set_seed(config['seed'])
    model = _load_fno(config)
    data = np.load(dataset_dir(config) / f'{dataset_name}.npz')
    f = data['f']
    u_true = data['u']
    f_t = tensor_with_channel(f, config['dimension'])
    with torch.no_grad():
        pred = model(f_t).cpu().numpy()[:, 0]
    summary, rel = evaluate_predictions(pred, u_true, model_label='fno')
    summary['n_parameters'] = parameter_count(model)
    summary['mean_inference_time_seconds'] = measure_inference_time(model, f_t)
    st_rows = _stability(config, model, f)
    summary['mean_stability_ratio'] = float(np.mean([r['ratio'] for r in st_rows]))
    out_dir = evaluation_dir(config, 'fno')
    pd.DataFrame({'sample': np.arange(len(f)), 'rel_l2_fno': rel}).to_csv(out_dir / (f'sample_metrics_{dataset_name}.csv' if dataset_name != 'test' else 'sample_metrics.csv'), index=False)
    pd.DataFrame(st_rows).to_csv(out_dir / (f'stability_metrics_{dataset_name}.csv' if dataset_name != 'test' else 'stability_metrics.csv'), index=False)
    save_eval_summary(config, 'fno' if dataset_name == 'test' else f'fno_{dataset_name}', summary)
    print(summary)
