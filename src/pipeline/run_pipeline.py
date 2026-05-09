from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.ingestion.fema_api import run_full_ingestion
from src.processing.validate_data import run_all_validations
from src.processing.feature_engineering import build_processed_dataset
from src.models.train import run_training_pipeline


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
BEST_METADATA_PATH = MODELS_DIR / "best_model_metadata.json"

BACKUP_MODEL_PATH = MODELS_DIR / "best_model_backup.pkl"
BACKUP_METADATA_PATH = MODELS_DIR / "best_model_metadata_backup.json"

PIPELINE_LOG_PATH = PROJECT_ROOT / "pipeline_run_summary.json"


def load_current_best_score() -> float | None:
    """
    Load the current best model score from metadata if available.
    """
    if not BEST_METADATA_PATH.exists():
        logger.warning("No existing metadata found. New model will be accepted if training succeeds.")
        return None

    with open(BEST_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Handles different metadata formats from earlier project stages
    if "cv_results" in metadata:
        results = metadata["cv_results"]
        if results:
            return max(row.get("mean_cv_r2", float("-inf")) for row in results)

    if "test_results" in metadata:
        results = metadata["test_results"]
        if results:
            return max(row.get("test_r2", float("-inf")) for row in results)

    if "best_cv_r2" in metadata:
        return metadata["best_cv_r2"]

    return None


def backup_current_model() -> None:
    """
    Backup existing model and metadata before retraining.
    """
    if BEST_MODEL_PATH.exists():
        shutil.copy2(BEST_MODEL_PATH, BACKUP_MODEL_PATH)
        logger.info("Backed up current model to %s", BACKUP_MODEL_PATH)

    if BEST_METADATA_PATH.exists():
        shutil.copy2(BEST_METADATA_PATH, BACKUP_METADATA_PATH)
        logger.info("Backed up current metadata to %s", BACKUP_METADATA_PATH)


def restore_backup_model() -> None:
    """
    Restore previous model if new model does not improve.
    """
    if BACKUP_MODEL_PATH.exists():
        shutil.copy2(BACKUP_MODEL_PATH, BEST_MODEL_PATH)
        logger.info("Restored previous best model.")

    if BACKUP_METADATA_PATH.exists():
        shutil.copy2(BACKUP_METADATA_PATH, BEST_METADATA_PATH)
        logger.info("Restored previous best metadata.")


def get_new_best_score(results_df: pd.DataFrame) -> float:
    """
    Extract best mean CV R2 from latest training results.
    """
    return float(results_df["mean_cv_r2"].max())


def write_pipeline_summary(summary: dict) -> None:
    """
    Save pipeline run summary for auditability.
    """
    with open(PIPELINE_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved pipeline summary to %s", PIPELINE_LOG_PATH)


def run_full_pipeline() -> dict:
    """
    Run full data-to-model pipeline:
    1. Fetch FEMA data
    2. Validate raw data
    3. Build processed features
    4. Retrain models
    5. Compare new model with existing best model
    """
    start_time = datetime.now(timezone.utc)

    summary = {
        "pipeline_start_utc": start_time.isoformat(),
        "status": "started",
        "steps": {},
    }

    logger.info("Starting full FEMA disaster recovery ML pipeline...")

    try:
        # ---------------------------------------------------------------------
        # Step 1: Ingestion
        # ---------------------------------------------------------------------
        logger.info("Step 1/5: Fetching FEMA data...")
        run_full_ingestion()
        summary["steps"]["ingestion"] = "completed"

        # ---------------------------------------------------------------------
        # Step 2: Validation
        # ---------------------------------------------------------------------
        logger.info("Step 2/5: Validating raw data...")
        validation_results = run_all_validations()
        summary["steps"]["validation"] = "completed"

        validation_failures = {}
        for dataset_name, payload in validation_results.items():
            report = payload["report"]
            failed_count = int((~report["passed"]).sum())
            validation_failures[dataset_name] = failed_count

        summary["validation_failures"] = validation_failures

        # ---------------------------------------------------------------------
        # Step 3: Feature engineering
        # ---------------------------------------------------------------------
        logger.info("Step 3/5: Running feature engineering...")
        processed_df = build_processed_dataset()
        summary["steps"]["feature_engineering"] = "completed"
        summary["processed_shape"] = list(processed_df.shape)

        # ---------------------------------------------------------------------
        # Step 4: Model training
        # ---------------------------------------------------------------------
        logger.info("Step 4/5: Backing up current model and retraining...")
        old_score = load_current_best_score()
        backup_current_model()

        results_df, best_model_name = run_training_pipeline()
        new_score = get_new_best_score(results_df)

        summary["steps"]["training"] = "completed"
        summary["old_best_score"] = old_score
        summary["new_best_score"] = new_score
        summary["new_best_model_name"] = best_model_name

        # ---------------------------------------------------------------------
        # Step 5: Compare and accept/reject model
        # ---------------------------------------------------------------------
        logger.info("Step 5/5: Comparing new model against current best...")

        if old_score is None or new_score >= old_score:
            logger.info(
                "New model accepted. old_score=%s | new_score=%s",
                old_score,
                new_score,
            )
            summary["model_update_decision"] = "accepted"
        else:
            logger.warning(
                "New model rejected. old_score=%s | new_score=%s",
                old_score,
                new_score,
            )
            restore_backup_model()
            summary["model_update_decision"] = "rejected_previous_model_restored"

        end_time = datetime.now(timezone.utc)
        summary["pipeline_end_utc"] = end_time.isoformat()
        summary["duration_seconds"] = (end_time - start_time).total_seconds()
        summary["status"] = "completed"

        write_pipeline_summary(summary)
        logger.info("Pipeline completed successfully.")

        return summary

    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)

        summary["status"] = "failed"
        summary["error"] = str(exc)
        summary["pipeline_end_utc"] = datetime.now(timezone.utc).isoformat()

        write_pipeline_summary(summary)

        raise


if __name__ == "__main__":
    run_full_pipeline()