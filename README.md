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

$$-\Delta u + \kappa\ u + u^3 = f
\qquad \text{on } \mathbb{T}^d.$$

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
