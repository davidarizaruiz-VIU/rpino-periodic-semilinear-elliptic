import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))
import argparse
from q1pde.config import load_config
from q1pde.experiments.ablation import run_iteration_sweep

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
parser.add_argument('--iters', nargs='+', type=int, default=[1, 2, 3, 5, 8])
args = parser.parse_args()
run_iteration_sweep(load_config(args.config), eval_iters_list=args.iters)
