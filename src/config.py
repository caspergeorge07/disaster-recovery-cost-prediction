from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

FEMA_BASE_URL = "https://www.fema.gov/api/open"

DECLARATIONS_ENDPOINT = f"{FEMA_BASE_URL}/v2/DisasterDeclarationsSummaries"
PUBLIC_ASSISTANCE_ENDPOINT = f"{FEMA_BASE_URL}/v2/PublicAssistanceFundedProjectsDetails"
DISASTER_SUMMARIES_ENDPOINT = f"{FEMA_BASE_URL}/v1/FemaWebDisasterSummaries"