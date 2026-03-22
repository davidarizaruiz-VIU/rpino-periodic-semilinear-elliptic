# RP-INO for periodic semilinear elliptic problems

Code accompanying the paper:

“Certified residual-preconditioned implicit neural operators in periodic Sobolev spaces”

## Overview

This repository contains the implementation used for the numerical experiments in the paper, including:

- contractive backbone solvers,
- controlled RP-INO models,
- free-residual ablations,
- FNO baseline experiments,
- iteration sweep experiments,
- scripts to generate figures and tables for the manuscript.

## PDE benchmark

The main benchmark studied in the paper is the periodic semilinear elliptic problem

\[
-\Delta u + \kappa u + u^3 = f
\qquad \text{on } \mathbb{T}^d.
\]

We report experiments in one and two spatial dimensions, with the main results focused on the two-dimensional setting.

## Repository structure

- `src/`: source code
- `scripts/`: executable experiment scripts
- `configs/`: experiment configurations
- `results/`: generated outputs
- `figures/`: figures used in the manuscript

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Reproducing the main 2D experiments

```bash
python scripts/01_generate_dataset.py --config configs/nonlinear_poisson_2d.yaml
python scripts/02_train_rpino.py --config configs/nonlinear_poisson_2d.yaml --variant controlled
python scripts/03_evaluate_rpino.py --config configs/nonlinear_poisson_2d.yaml --variant controlled --dataset test
python scripts/04_train_fno.py --config configs/nonlinear_poisson_2d.yaml
python scripts/05_evaluate_fno.py --config configs/nonlinear_poisson_2d.yaml --dataset test
python scripts/06_run_ablation.py --config configs/nonlinear_poisson_2d.yaml
python scripts/07_run_iteration_sweep.py --config configs/nonlinear_poisson_2d.yaml --iters 1 2 3 5 8
python scripts/08_make_paper_figures.py --config configs/nonlinear_poisson_2d.yaml
python scripts/09_make_paper_tables.py --config configs/nonlinear_poisson_2d.yaml
```

## Repository

GitHub repository:
`https://github.com/davidarizaruiz-VIU/rpino-periodic-semilinear-elliptic`

## Reproducibility note

For a citable frozen version of the implementation, create a tagged release on GitHub and archive it with Zenodo. The Zenodo DOI should be cited in the paper alongside the repository URL.

## Citation

If you use this code, please cite the associated paper and the archived software release.
