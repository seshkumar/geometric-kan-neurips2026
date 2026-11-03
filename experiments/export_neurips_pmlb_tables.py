from pathlib import Path
import shutil
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.reporting.neurips_pmlb_tables import export
if __name__ == '__main__':
    out = export(ROOT, ROOT / 'experiments/results/neurips_pmlb15/pmlb15_pbs_v1')
    print(out)
