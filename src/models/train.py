from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBRegressor

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_disasters.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUTPUT_PATH = MODELS_DIR / "best_model.pkl"
METADATA_OUTPUT_PATH = MODELS_DIR / "best_model_metadata.json"

MLFLOW_TRACKING_URI = (PROJECT_ROOT / "mlruns").resolve().as_uri()


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def load_processed_data() -> pd.DataFrame:
    """
    Load the processed disaster-level dataset.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    logger.info("Loaded processed dataset | shape=%s", df.shape)
    return df


# -----------------------------------------------------------------------------
# Feature definitions
# -----------------------------------------------------------------------------
def define_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str], str]:
    """
    Define target, numeric features, and categorical features.

    We exclude target leakage columns and identifier-style columns.
    """
    target_col = "log_total_obligated_amount"

    exclude_cols = {
        # target/leakage
        "log_total_obligated_amount",
        "total_obligated_amount",
        "totalObligatedAmountPa",
        "totalObligatedAmountCatAb",
        "totalObligatedAmountCatC2g",
        "totalObligatedAmountHmgp",
        "project_count",
        "avg_project_amount",

        # identifiers
        "disasterNumber",
        "id",
        "hash",
        "femaDeclarationString",
        "declarationRequestNumber",
        "incidentId",
        "lastRefresh",
        
        # PROGRAM FLAGS (CRITICAL — NEW FIX)
        "ihProgramDeclared",
        "iaProgramDeclared",
        "paProgramDeclared",
        "hmProgramDeclared",

        # raw dates
        "declarationDate",
        "incidentBeginDate",
        "incidentEndDate",
        "disasterCloseoutDate",
        "lastIAFilingDate",

        # high-cardinality / text-heavy columns
        "declarationTitle",
        "designatedArea",
        "designatedIncidentTypes",
    }

    candidate_features = [col for col in df.columns if col not in exclude_cols]

    numeric_features = []
    categorical_features = []

    for col in candidate_features:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_features.append(col)
        else:
            categorical_features.append(col)

    return numeric_features, categorical_features, target_col


# -----------------------------------------------------------------------------
# Preprocessor
# -----------------------------------------------------------------------------
def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """
    Build a preprocessing transformer for numeric and categorical columns.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


# -----------------------------------------------------------------------------
# Model definitions
# -----------------------------------------------------------------------------
def build_model_pipelines(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    """
    Build sklearn pipelines for each model.
    """
    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror",
            eval_metric="rmse",
        ),
    }

    pipelines = {
        model_name: Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        for model_name, model in models.items()
    }

    return pipelines


# -----------------------------------------------------------------------------
# Cross-validation
# -----------------------------------------------------------------------------
def evaluate_models(
    pipelines: dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate all pipelines using 5-fold cross-validation.
    """
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "r2": "r2",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
    }

    results = []

    for model_name, pipeline in pipelines.items():
        logger.info("Evaluating model: %s", model_name)

        cv_results = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        mean_r2 = np.mean(cv_results["test_r2"])
        std_r2 = np.std(cv_results["test_r2"])

        mean_rmse = -np.mean(cv_results["test_rmse"])
        std_rmse = np.std(-cv_results["test_rmse"])

        mean_mae = -np.mean(cv_results["test_mae"])
        std_mae = np.std(-cv_results["test_mae"])

        results.append({
            "model_name": model_name,
            "mean_cv_r2": mean_r2,
            "std_cv_r2": std_r2,
            "mean_cv_rmse": mean_rmse,
            "std_cv_rmse": std_rmse,
            "mean_cv_mae": mean_mae,
            "std_cv_mae": std_mae,
        })

    return pd.DataFrame(results).sort_values("mean_cv_r2", ascending=False).reset_index(drop=True)


# -----------------------------------------------------------------------------
# MLflow logging
# -----------------------------------------------------------------------------
def log_results_to_mlflow(
    results_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> None:
    """
    Log CV results for each model to MLflow.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("disaster_recovery_cost_prediction")

    for _, row in results_df.iterrows():
        with mlflow.start_run(run_name=row["model_name"]):
            mlflow.log_param("model_name", row["model_name"])
            mlflow.log_param("n_numeric_features", len(numeric_features))
            mlflow.log_param("n_categorical_features", len(categorical_features))
            mlflow.log_param("numeric_features", ",".join(numeric_features))
            mlflow.log_param("categorical_features", ",".join(categorical_features))

            mlflow.log_metric("mean_cv_r2", float(row["mean_cv_r2"]))
            mlflow.log_metric("std_cv_r2", float(row["std_cv_r2"]))
            mlflow.log_metric("mean_cv_rmse", float(row["mean_cv_rmse"]))
            mlflow.log_metric("std_cv_rmse", float(row["std_cv_rmse"]))
            mlflow.log_metric("mean_cv_mae", float(row["mean_cv_mae"]))
            mlflow.log_metric("std_cv_mae", float(row["std_cv_mae"]))


# -----------------------------------------------------------------------------
# Train and save best model
# -----------------------------------------------------------------------------
def train_and_save_best_model(
    best_model_name: str,
    pipelines: dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    results_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> None:
    """
    Fit the best model on the full dataset and save it with metadata.
    """
    best_pipeline = pipelines[best_model_name]
    best_pipeline.fit(X, y)

    with open(MODEL_OUTPUT_PATH, "wb") as f:
        pickle.dump(best_pipeline, f)

    metadata = {
        "best_model_name": best_model_name,
        "target_column": "log_total_obligated_amount",
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "n_rows": len(X),
        "n_columns": X.shape[1],
        "model_output_path": str(MODEL_OUTPUT_PATH),
        "cv_results": results_df.to_dict(orient="records"),
    }

    with open(METADATA_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved best model to %s", MODEL_OUTPUT_PATH)
    logger.info("Saved metadata to %s", METADATA_OUTPUT_PATH)


# -----------------------------------------------------------------------------
# Main training pipeline
# -----------------------------------------------------------------------------
def run_training_pipeline() -> tuple[pd.DataFrame, str]:
    """
    End-to-end training workflow.
    """
    df = load_processed_data()

    numeric_features, categorical_features, target_col = define_feature_lists(df)

    logger.info("Numeric features (%s): %s", len(numeric_features), numeric_features)
    logger.info("Categorical features (%s): %s", len(categorical_features), categorical_features)

    X = df[numeric_features + categorical_features].copy()
    y = df[target_col].copy()

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    pipelines = build_model_pipelines(preprocessor)

    results_df = evaluate_models(pipelines, X, y)
    logger.info("Cross-validation results:\n%s", results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["model_name"]
    logger.info("Best model selected: %s", best_model_name)

    log_results_to_mlflow(results_df, numeric_features, categorical_features)

    train_and_save_best_model(
        best_model_name=best_model_name,
        pipelines=pipelines,
        X=X,
        y=y,
        results_df=results_df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    return results_df, best_model_name


if __name__ == "__main__":
    results_df, best_model_name = run_training_pipeline()

    print("\nTraining complete.")
    print("\nCross-validation results:")
    print(results_df.to_string(index=False))
    print(f"\nBest model: {best_model_name}")
    print(f"Model saved to: {MODEL_OUTPUT_PATH}")