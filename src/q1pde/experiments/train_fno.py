from __future__ import annotations
from q1pde.experiments.common import build_fno_model, build_rpino_model, train_supervised
from q1pde.paths import dataset_dir
from q1pde.utils import set_seed


def run_train_fno(config: dict) -> None:
    set_seed(config['seed'])
    ddir = dataset_dir(config)
    model = build_fno_model(config)
    rpino = build_rpino_model(config, variant='controlled')
    baseline_predictor = lambda f_batch: rpino.backbone_only(f_batch, eval_mode=True)
    best_val = train_supervised(model, ddir / 'train.npz', ddir / 'val.npz', config, model_name='fno', baseline_predictor=baseline_predictor)
    print(f'FNO training finished. Best val loss = {best_val:.6e}')
