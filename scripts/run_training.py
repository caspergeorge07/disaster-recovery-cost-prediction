from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.train import run_training_pipeline

if __name__ == "__main__":
    results_df, best_model_name = run_training_pipeline()
    print(results_df)
    print(f"Best model: {best_model_name}")