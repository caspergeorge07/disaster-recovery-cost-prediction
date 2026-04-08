from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.processing.validate_data import run_all_validations

if __name__ == "__main__":
    run_all_validations()
