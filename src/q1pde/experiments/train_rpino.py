from __future__ import annotations
import numpy as np
from q1pde.experiments.common import build_rpino_model, train_supervised
from q1pde.paths import dataset_dir
from q1pde.utils import set_seed


def run_train(config: dict, *, variant: str = 'controlled') -> None:
    set_seed(config['seed'])
    ddir = dataset_dir(config)
    model = build_rpino_model(config, variant=variant)
    baseline_predictor = None
    if variant in ('controlled', 'free'):
        baseline_predictor = lambda f_batch: model.backbone_only(f_batch, eval_mode=True)
    best_val = train_supervised(
        model,
        ddir / 'train.npz',
        ddir / 'val.npz',
        config,
        model_name='rpino' if variant == 'controlled' else f'rpino_{variant}',
        baseline_predictor=baseline_predictor,
    )
    label = 'RP-INO' if variant == 'controlled' else f'RP-INO ({variant})'
    print(f'{label} training finished. Best val loss = {best_val:.6e}')
