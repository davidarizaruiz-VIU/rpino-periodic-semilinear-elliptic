# Certified Implicit Neural Operators on Periodic Sobolev Spaces

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19172623.svg)](https://doi.org/10.5281/zenodo.19172623)

Code accompanying the paper:

> **Certified Implicit Neural Operators on Periodic Sobolev Spaces**
> 
> David Ariza-Ruiz
>
> *Submitted to Journal of Scientific Computing*, 2026.

Archived software release on Zenodo: [10.5281/zenodo.19172623](https://doi.org/10.5281/zenodo.19172623)

## What this code does

This repository reproduces **all** numerical experiments in the paper. It implements:

- **RP-INO** (Residual-Preconditioned Implicit Neural Operator): an implicit neural operator with certified contractivity, Lipschitz stability, and convergence guarantees enforced at design time through operator-norm budgets in periodic Sobolev spaces.
- **Three baselines**: FNO (595K params), parameter-matched FNO-Small (214K params), and DeepONet (209K params).
- **Two PDE benchmarks** on the 2-D periodic torus:
  - **Problem A** — Semilinear Poisson: $-\Delta u + \kappa u + u^3 = f$
  - **Problem B** — Steady viscous Burgers: $-\nu\Delta u + u\,\partial_{x_1}u + \kappa u = f$
- **Full evaluation suite**: in-distribution accuracy, distributional shift, learning curves, ablation (controlled vs. free residual), iteration sweep, contraction traces, and empirical stability.

## Repository structure

```
├── configs/                 # YAML experiment configurations
│   ├── nonlinear_poisson_2d.yaml
│   └── burgers_2d.yaml
├── src/q1pde/               # Python package
│   ├── pde.py               # FFT Poisson solver
│   ├── pde_burgers.py        # FFT Burgers solver
│   ├── model.py              # ResidualNet + RPINO + FNO
│   ├── model_deeponet.py     # DeepONet (CNN branch + MLP trunk)
│   ├── torch_ops.py          # Backbone wrapper (Poisson)
│   ├── torch_ops_ext.py      # Backbone wrapper (Burgers)
│   ├── dataset.py            # Poisson dataset generator
│   ├── dataset_burgers.py    # Burgers dataset generator
│   └── experiments/          # Evaluation and training modules
├── scripts/                  # Executable experiment scripts
│   ├── 01–09                 # Core Poisson experiments
│   ├── 10_run_extended_experiments.py   # Burgers + baselines + learning curves
│   ├── 11_evaluate_burgers_diagnostics.py  # Per-sample diagnostics (Burgers)
│   └── 12_make_final_figures.py            # All 8 manuscript figures
├── tests/                    # Unit tests
├── pyproject.toml
├── requirements.txt
├── LICENSE                   # MIT
└── CITATION.cff
```

## Requirements

- Python >= 3.10
- PyTorch (CPU is sufficient; MPS/CUDA optional)
- NumPy, SciPy, Matplotlib, Pandas, PyYAML

Tested on macOS (Apple Silicon M1) and Ubuntu 22.04.

## Installation

```bash
git clone https://github.com/davidarizaruiz-VIU/rpino-periodic-semilinear-elliptic.git
cd rpino-periodic-semilinear-elliptic

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Reproducing the experiments

All commands are run from the repository root.

### Step 1 — Poisson baseline (RP-INO + FNO)

```bash
python3 scripts/01_generate_dataset.py --config configs/nonlinear_poisson_2d.yaml
python3 scripts/02_train_rpino.py      --config configs/nonlinear_poisson_2d.yaml --variant controlled
python3 scripts/04_train_fno.py        --config configs/nonlinear_poisson_2d.yaml
python3 scripts/03_evaluate_rpino.py   --config configs/nonlinear_poisson_2d.yaml --variant controlled --dataset test
python3 scripts/05_evaluate_fno.py     --config configs/nonlinear_poisson_2d.yaml --dataset test
python3 scripts/06_run_ablation.py     --config configs/nonlinear_poisson_2d.yaml
python3 scripts/07_run_iteration_sweep.py --config configs/nonlinear_poisson_2d.yaml --iters 1 2 3 5 8
```

### Step 2 — Extended experiments (Burgers + FNO-S + DeepONet + learning curves)

```bash
# All at once (~2–4 h on CPU):
python3 scripts/10_run_extended_experiments.py --phase all

# Or phase by phase:
python3 scripts/10_run_extended_experiments.py --phase burgers_data
python3 scripts/10_run_extended_experiments.py --phase burgers_train
python3 scripts/10_run_extended_experiments.py --phase poisson_extra
python3 scripts/10_run_extended_experiments.py --phase learning_curves
python3 scripts/10_run_extended_experiments.py --phase summary
```

### Step 3 — Per-sample diagnostics for Burgers

```bash
python3 scripts/11_evaluate_burgers_diagnostics.py
```

Generates contraction traces, stability ratios, and iteration sweep data for Problem B.

### Step 4 — Figures and tables

```bash
python3 scripts/12_make_final_figures.py
python3 scripts/08_make_paper_figures.py --config configs/nonlinear_poisson_2d.yaml
python3 scripts/09_make_paper_tables.py  --config configs/nonlinear_poisson_2d.yaml
```

All figures are saved to `results/` and `figures/`.

## Results overview

| Model | Params | Poisson (rel. L²) | Burgers (rel. L²) | Shift degrad. (A / B) |
|-------|--------|--------------------|--------------------|-----------------------|
| **RP-INO** | 207K | **0.031** | 0.113 | **+4% / +3%** |
| FNO | 595K | 0.106 | **0.064** | +11% / +55% |
| FNO-S | 214K | 0.107 | 0.065 | +10% / +55% |
| DeepONet | 209K | 0.286 | 0.422 | +2% / +7% |

RP-INO achieves the best accuracy on Poisson (where the backbone captures the linear PDE structure) and the best shift robustness on both problems.

## Tests

```bash
pytest tests/ -v
```

## Citation

If you use this code, please cite:

### Manuscript

```bibtex
@article{arizaruiz2026certified,
  title   = {Certified Implicit Neural Operators on Periodic Sobolev Spaces},
  author  = {Ariza-Ruiz, David},
  journal = {Submitted to Journal of Scientific Computing},
  year    = {2026}
}
```

### Software release

```bibtex
@software{ariza_ruiz_2026_19172623,
  author  = {Ariza-Ruiz, David},
  title   = {rpino-periodic-semilinear-elliptic},
  year    = {2026},
  doi     = {10.5281/zenodo.19172623},
  url     = {https://doi.org/10.5281/zenodo.19172623}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
