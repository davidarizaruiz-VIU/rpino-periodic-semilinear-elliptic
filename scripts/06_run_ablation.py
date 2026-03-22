import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))
import argparse
from q1pde.config import load_config
from q1pde.experiments.ablation import run_ablation

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
args = parser.parse_args()
run_ablation(load_config(args.config))
