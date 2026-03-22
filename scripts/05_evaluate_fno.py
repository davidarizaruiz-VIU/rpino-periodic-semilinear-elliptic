import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))
import argparse
from q1pde.config import load_config
from q1pde.experiments.evaluate_fno import run_evaluation_fno

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
parser.add_argument('--dataset', choices=['test', 'test_shift', 'test_crossres'], default='test')
args = parser.parse_args()
run_evaluation_fno(load_config(args.config), dataset_name=args.dataset)
