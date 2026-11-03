import json
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.reporting.neurips_figure1_manuscript import export
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--copy-to-manuscript', action='store_true')
    args = parser.parse_args()
    print(export(ROOT / 'experiments/results/neurips_figure1_matched', args.copy_to_manuscript))
