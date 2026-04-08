from src.config import RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR
from src.utils import create_directories

def main():
    create_directories([
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR / "trained",
        MODELS_DIR / "artifacts",
        REPORTS_DIR / "figures",
        REPORTS_DIR / "weekly_updates"
    ])
    print("Project folders created successfully.")

if __name__ == "__main__":
    main()