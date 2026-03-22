from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from q1pde.metrics import relative_l2_error
from q1pde.model import ResidualNet, RPINO, build_fno
from q1pde.paths import training_dir, evaluation_dir
from q1pde.pde import PeriodicNonlinearPoisson
from q1pde.torch_data import make_loader
from q1pde.torch_ops import TorchCoarseBackbone
from q1pde.utils import write_json


def tensor_with_channel(x: np.ndarray, dimension: int) -> torch.Tensor:
    if dimension == 1:
        return torch.tensor(x[:, None, :], dtype=torch.float32)
    return torch.tensor(x[:, None, :, :], dtype=torch.float32)


def build_rpino_model(config: dict, *, grid_size: int | None = None, variant: str = 'controlled') -> RPINO:
    pde = PeriodicNonlinearPoisson(
        dimension=config['dimension'],
        grid_size=grid_size or config['grid_size'],
        domain_length=config['domain_length'],
        kappa=config['kappa'],
    )
    backbone = TorchCoarseBackbone(
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


def build_fno_model(config: dict):
    fcfg = config['fno']
    return build_fno(
        dimension=config['dimension'],
        width=fcfg['width'],
        depth=fcfg['depth'],
        modes_x=fcfg['modes_x'],
        modes_y=fcfg.get('modes_y'),
    )


def train_supervised(model, train_npz: Path, val_npz: Path, config: dict, *, model_name: str, baseline_predictor=None):
    device = torch.device('cpu')
    model = model.to(device)
    train = np.load(train_npz)
    val = np.load(val_npz)
    train_loader = make_loader(train['f'], train['u'], config['dimension'], config['training']['batch_size'], True)
    val_loader = make_loader(val['f'], val['u'], config['dimension'], config['training']['batch_size'], False)
    opt = torch.optim.Adam(model.parameters(), lr=config['training']['lr'], weight_decay=config['training'].get('weight_decay', 0.0))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config['training']['epochs'])
    loss_fn = nn.MSELoss()
    history = []
    best_val = float('inf')
    tdir = training_dir(config, model_name)

    for epoch in range(1, config['training']['epochs'] + 1):
        model.train()
        train_loss = 0.0
        for f_batch, u_batch in train_loader:
            f_batch = f_batch.to(device)
            u_batch = u_batch.to(device)
            pred = model(f_batch)
            loss = loss_fn(pred, u_batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            train_loss += loss.item() * f_batch.size(0)
        scheduler.step()
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        baseline_loss = 0.0
        with torch.no_grad():
            for f_batch, u_batch in val_loader:
                f_batch = f_batch.to(device)
                u_batch = u_batch.to(device)
                pred = model(f_batch)
                loss = loss_fn(pred, u_batch)
                val_loss += loss.item() * f_batch.size(0)
                if baseline_predictor is not None:
                    base = baseline_predictor(f_batch)
                    baseline_loss += loss_fn(base, u_batch).item() * f_batch.size(0)
        val_loss /= len(val_loader.dataset)
        baseline_loss = baseline_loss / len(val_loader.dataset) if baseline_predictor is not None else np.nan
        history.append({
            'epoch': epoch,
            'lr': scheduler.get_last_lr()[0],
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_baseline_loss': baseline_loss,
        })
        if np.isfinite(baseline_loss):
            print(f"epoch={epoch} train_loss={train_loss:.6e} val_loss={val_loss:.6e} val_baseline_loss={baseline_loss:.6e}")
        else:
            print(f"epoch={epoch} train_loss={train_loss:.6e} val_loss={val_loss:.6e}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), tdir / 'best_model.pt')

    pd.DataFrame(history).to_csv(tdir / 'history.csv', index=False)
    write_json({'best_val_loss': best_val}, tdir / 'training_summary.json')
    return best_val


def evaluate_predictions(pred: np.ndarray, true: np.ndarray, *, model_label: str):
    rel = relative_l2_error(pred, true)
    return {
        f'mean_rel_l2_{model_label}': float(rel.mean()),
        f'median_rel_l2_{model_label}': float(np.median(rel)),
        f'p90_rel_l2_{model_label}': float(np.quantile(rel, 0.9)),
    }, rel


def measure_inference_time(predict_fn, f_t: torch.Tensor, repeats: int = 5) -> float:
    times = []
    with torch.no_grad():
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = predict_fn(f_t)
            times.append(time.perf_counter() - t0)
    return float(np.mean(times) / f_t.shape[0])


def parameter_count(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def save_eval_summary(config: dict, model_name: str, summary: dict):
    ed = evaluation_dir(config, model_name)
    write_json(summary, ed / 'summary.json')
    return ed
