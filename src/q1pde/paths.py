from __future__ import annotations
from pathlib import Path
from q1pde.utils import ensure_dir


def experiment_root(config: dict) -> Path:
    return ensure_dir(Path('results') / config['experiment_name'])


def dataset_dir(config: dict) -> Path:
    return ensure_dir(experiment_root(config) / 'dataset')


def training_dir(config: dict, model_name: str = 'rpino') -> Path:
    return ensure_dir(experiment_root(config) / f'training_{model_name}')


def evaluation_dir(config: dict, model_name: str = 'rpino') -> Path:
    return ensure_dir(experiment_root(config) / f'evaluation_{model_name}')


def figures_dir(config: dict) -> Path:
    return ensure_dir(experiment_root(config) / 'figures')


def tables_dir(config: dict) -> Path:
    return ensure_dir(experiment_root(config) / 'tables')


def aux_dir(config: dict, name: str) -> Path:
    return ensure_dir(experiment_root(config) / name)
