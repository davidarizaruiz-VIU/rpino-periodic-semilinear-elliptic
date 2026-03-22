import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))
import argparse
from q1pde.config import load_config
from q1pde.paper.tables import make_tables

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
args = parser.parse_args()
make_tables(load_config(args.config))
