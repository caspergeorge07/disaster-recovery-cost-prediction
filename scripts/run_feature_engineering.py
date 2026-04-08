from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.processing.feature_engineering import build_processed_dataset, inspect_processed_dataset

if __name__ == "__main__":
    df = build_processed_dataset()
    inspect_processed_dataset(df)