from pathlib import Path
import sys

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.ingestion.fema_api import run_full_ingestion

if __name__ == "__main__":
    run_full_ingestion(force_refresh=False)