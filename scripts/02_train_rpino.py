import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))
import argparse
from q1pde.config import load_config
from q1pde.experiments.train_rpino import run_train

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
parser.add_argument('--variant', choices=['controlled', 'free'], default='controlled')
args = parser.parse_args()
run_train(load_config(args.config), variant=args.variant)
