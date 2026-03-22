import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))
import argparse
from q1pde.config import load_config
from q1pde.experiments.evaluate_rpino import run_evaluation, run_shift_evaluation, run_crossres_evaluation

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
parser.add_argument('--variant', choices=['controlled', 'free'], default='controlled')
parser.add_argument('--dataset', choices=['test', 'shift', 'crossres'], default='test')
args = parser.parse_args()
config = load_config(args.config)
if args.dataset == 'test':
    run_evaluation(config, variant=args.variant)
elif args.dataset == 'shift':
    run_shift_evaluation(config, variant=args.variant)
else:
    run_crossres_evaluation(config, variant=args.variant)
