from contextlib import asynccontextmanager
from pathlib import Path
import json
import logging
import pickle
import time

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, HTTPException

from api.schemas import (
    PredictRequest,
    PredictResponse,
    HealthResponse,
    ModelInfoResponse,
)

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
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "best_model_metadata.json"

# -----------------------------------------------------------------------------
# Global model state
# -----------------------------------------------------------------------------
model_artifacts = {
    "model": None,
    "metadata": None,
    "loaded": False,
}


def map_season(month: int) -> str:
    if month in [12, 1, 2]:
        return "Winter"
    if month in [3, 4, 5]:
        return "Spring"
    if month in [6, 7, 8]:
        return "Summer"
    if month in [9, 10, 11]:
        return "Autumn"
    return "Unknown"


def map_census_region(state: str) -> str:
    census_region_map = {
        "CT": "Northeast", "ME": "Northeast", "MA": "Northeast", "NH": "Northeast",
        "RI": "Northeast", "VT": "Northeast", "NJ": "Northeast", "NY": "Northeast",
        "PA": "Northeast",

        "IL": "Midwest", "IN": "Midwest", "MI": "Midwest", "OH": "Midwest",
        "WI": "Midwest", "IA": "Midwest", "KS": "Midwest", "MN": "Midwest",
        "MO": "Midwest", "NE": "Midwest", "ND": "Midwest", "SD": "Midwest",

        "DE": "South", "FL": "South", "GA": "South", "MD": "South",
        "NC": "South", "SC": "South", "VA": "South", "DC": "South",
        "WV": "South", "AL": "South", "KY": "South", "MS": "South",
        "TN": "South", "AR": "South", "LA": "South", "OK": "South",
        "TX": "South",

        "AZ": "West", "CO": "West", "ID": "West", "MT": "West",
        "NV": "West", "NM": "West", "UT": "West", "WY": "West",
        "AK": "West", "CA": "West", "HI": "West", "OR": "West", "WA": "West",

        "PR": "Territory", "VI": "Territory", "GU": "Territory",
        "AS": "Territory", "MP": "Territory",
    }
    return census_region_map.get(state.upper(), "Unknown")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model once when the API starts.
    """
    try:
        with open(MODEL_PATH, "rb") as f:
            model_artifacts["model"] = pickle.load(f)

        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            model_artifacts["metadata"] = json.load(f)

        model_artifacts["loaded"] = True
        logger.info("Model loaded successfully from %s", MODEL_PATH)

    except Exception as exc:
        model_artifacts["loaded"] = False
        logger.exception("Failed to load model: %s", exc)

    yield

    logger.info("API shutting down.")


app = FastAPI(
    title="Disaster Recovery Cost Prediction API",
    description="API for predicting disaster recovery cost using FEMA disaster features.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    logger.info(
        "%s %s | status=%s | duration=%.4fs",
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )

    return response


@app.get("/health", response_model=HealthResponse)
def health_check():
    metadata = model_artifacts.get("metadata") or {}
    return {
        "status": "ok" if model_artifacts["loaded"] else "error",
        "model_loaded": model_artifacts["loaded"],
        "model_name": metadata.get("best_model_name", "unknown"),
    }


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    if not model_artifacts["loaded"]:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    metadata = model_artifacts["metadata"]

    return {
        "model_name": metadata.get("best_model_name", "unknown"),
        "model_version": metadata.get("model_version", "1.0.0"),
        "training_date": metadata.get("training_date", "not_available"),
        "target_column": metadata.get("target_column", "log_total_obligated_amount"),
        "metrics": metadata.get("metrics", metadata.get("test_results", {})),
        "numeric_features": metadata.get("numeric_features", []),
        "categorical_features": metadata.get("categorical_features", []),
    }


@app.post("/predict-cost", response_model=PredictResponse)
def predict_cost(request: PredictRequest):
    if not model_artifacts["loaded"]:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    model = model_artifacts["model"]
    metadata = model_artifacts["metadata"]

    payload = request.model_dump()

    # Auto-fill engineered fields if missing
    if payload.get("season") is None:
        payload["season"] = map_season(payload["declaration_month"])

    if payload.get("census_region") is None:
        payload["census_region"] = map_census_region(payload["state"])

    feature_cols = metadata.get("numeric_features", []) + metadata.get("categorical_features", [])

    input_df = pd.DataFrame([payload])

    # Ensure correct model feature order
    missing_cols = [col for col in feature_cols if col not in input_df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required model input columns: {missing_cols}"
        )

    input_df = input_df[feature_cols]

    predicted_log_cost = float(model.predict(input_df)[0])
    predicted_cost_usd = float(np.expm1(predicted_log_cost))

    return {
        "predicted_log_cost": predicted_log_cost,
        "predicted_cost_usd": max(predicted_cost_usd, 0.0),
        "model_name": metadata.get("best_model_name", "unknown"),
        "model_version": metadata.get("model_version", "1.0.0"),
        "target": metadata.get("target_column", "log_total_obligated_amount"),
        "message": "Prediction generated successfully.",
    }