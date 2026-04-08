from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

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
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# -----------------------------------------------------------------------------
# Dataset files
# -----------------------------------------------------------------------------
DATASET_FILES = {
    "disaster_declarations": RAW_DATA_DIR / "disaster_declarations_summaries.csv",
    "public_assistance": RAW_DATA_DIR / "public_assistance_funded_projects_details.csv",
    "fema_web_disaster_summaries": RAW_DATA_DIR / "fema_web_disaster_summaries.csv",
}

# -----------------------------------------------------------------------------
# Validation configs
# -----------------------------------------------------------------------------
VALIDATION_RULES = {
    "disaster_declarations": {
        "required_columns": [
            "disasterNumber",
            "state",
            "declarationType",
            "declarationDate",
            "incidentType",
            "incidentBeginDate",
            "incidentEndDate",
        ],
        "expected_types": {
            "disasterNumber": "numeric",
            "state": "string",
            "declarationType": "string",
            "declarationDate": "datetime",
            "incidentType": "string",
            "incidentBeginDate": "datetime",
            "incidentEndDate": "datetime",
        },
        "null_thresholds": {
            "disasterNumber": 0.00,
            "state": 0.00,
            "declarationType": 0.00,
            "declarationDate": 0.00,
            "incidentType": 0.00,
            "incidentBeginDate": 0.00,
            "incidentEndDate": 0.02,  # allow small null rate
        },
        "range_checks": {}
    },

    "public_assistance": {
        "required_columns": [
            "disasterNumber",
            "declarationDate",
            "incidentType",
            "stateAbbreviation",
            "projectAmount",
            "federalShareObligated",
            "totalObligated",
        ],
        "expected_types": {
            "disasterNumber": "numeric",
            "declarationDate": "datetime",
            "incidentType": "string",
            "stateAbbreviation": "string",
            "projectAmount": "numeric",
            "federalShareObligated": "numeric",
            "totalObligated": "numeric",
        },
        "null_thresholds": {
            "disasterNumber": 0.00,
            "declarationDate": 0.00,
            "incidentType": 0.00,
            "stateAbbreviation": 0.00,
            "projectAmount": 0.00,
            "federalShareObligated": 0.00,
            "totalObligated": 0.00,
        },
        "range_checks": {
            "projectAmount": {"min": 0},
            "federalShareObligated": {"min": 0},
            "totalObligated": {"min": 0},
        }
    },

    "fema_web_disaster_summaries": {
        "required_columns": [
            "disasterNumber",
            "hash",
            "lastRefresh",
        ],
        "expected_types": {
            "disasterNumber": "numeric",
            "hash": "string",
            "lastRefresh": "string",
        },
        "null_thresholds": {
            "disasterNumber": 0.00,
            "hash": 0.00,
            "lastRefresh": 0.00,
        },
        "range_checks": {
            "totalObligatedAmountPa": {"min": 0},
            "totalObligatedAmountCatAb": {"min": 0},
            "totalObligatedAmountCatC2g": {"min": 0},
            "totalObligatedAmountHmgp": {"min": 0},
        }
    }
}


@dataclass
class ValidationResult:
    dataset: str
    check_type: str
    column: str
    passed: bool
    details: str


def load_dataset(dataset_name: str) -> pd.DataFrame:
    """
    Load one raw dataset by name.
    """
    file_path = DATASET_FILES[dataset_name]
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    df = pd.read_csv(file_path)
    logger.info("Loaded %s | shape=%s", dataset_name, df.shape)
    return df


def coerce_known_types(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Safely coerce expected datetime/numeric columns in memory before validation.
    """
    rules = VALIDATION_RULES[dataset_name]
    expected_types = rules.get("expected_types", {})

    df = df.copy()

    for col, expected_type in expected_types.items():
        if col not in df.columns:
            continue

        if expected_type == "datetime":
            df[col] = pd.to_datetime(df[col], errors="coerce")

        elif expected_type == "numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce")

        elif expected_type == "string":
            # keep missing values as NaN, convert non-missing to string
            df[col] = df[col].where(df[col].isna(), df[col].astype(str))

    return df


def check_required_columns(df: pd.DataFrame, dataset_name: str) -> list[ValidationResult]:
    results = []
    required_columns = VALIDATION_RULES[dataset_name]["required_columns"]

    for col in required_columns:
        exists = col in df.columns
        results.append(
            ValidationResult(
                dataset=dataset_name,
                check_type="required_column",
                column=col,
                passed=exists,
                details="Column exists" if exists else "Missing required column",
            )
        )
    return results


def check_expected_types(df: pd.DataFrame, dataset_name: str) -> list[ValidationResult]:
    results = []
    expected_types = VALIDATION_RULES[dataset_name]["expected_types"]

    for col, expected_type in expected_types.items():
        if col not in df.columns:
            results.append(
                ValidationResult(
                    dataset=dataset_name,
                    check_type="dtype_check",
                    column=col,
                    passed=False,
                    details="Column missing, cannot validate dtype",
                )
            )
            continue

        series = df[col]

        if expected_type == "numeric":
            passed = pd.api.types.is_numeric_dtype(series)

        elif expected_type == "datetime":
            passed = pd.api.types.is_datetime64_any_dtype(series)

        elif expected_type == "string":
            passed = series.dtype == "object" or pd.api.types.is_string_dtype(series)

        else:
            passed = False

        results.append(
            ValidationResult(
                dataset=dataset_name,
                check_type="dtype_check",
                column=col,
                passed=passed,
                details=f"Expected {expected_type}, got {series.dtype}",
            )
        )

    return results


def check_null_thresholds(df: pd.DataFrame, dataset_name: str) -> list[ValidationResult]:
    results = []
    null_thresholds = VALIDATION_RULES[dataset_name]["null_thresholds"]

    for col, threshold in null_thresholds.items():
        if col not in df.columns:
            results.append(
                ValidationResult(
                    dataset=dataset_name,
                    check_type="null_check",
                    column=col,
                    passed=False,
                    details="Column missing, cannot validate null rate",
                )
            )
            continue

        null_rate = df[col].isna().mean()
        passed = null_rate <= threshold

        results.append(
            ValidationResult(
                dataset=dataset_name,
                check_type="null_check",
                column=col,
                passed=passed,
                details=f"Null rate={null_rate:.2%}, threshold={threshold:.2%}",
            )
        )

    return results


def check_value_ranges(df: pd.DataFrame, dataset_name: str) -> list[ValidationResult]:
    results = []
    range_checks = VALIDATION_RULES[dataset_name].get("range_checks", {})

    for col, rule in range_checks.items():
        if col not in df.columns:
            results.append(
                ValidationResult(
                    dataset=dataset_name,
                    check_type="range_check",
                    column=col,
                    passed=False,
                    details="Column missing, cannot validate range",
                )
            )
            continue

        series = pd.to_numeric(df[col], errors="coerce")

        passed = True
        details = []

        if "min" in rule:
            min_value = series.min(skipna=True)
            min_passed = pd.isna(min_value) or min_value >= rule["min"]
            passed = passed and min_passed
            details.append(f"min={min_value}, expected >= {rule['min']}")

        if "max" in rule:
            max_value = series.max(skipna=True)
            max_passed = pd.isna(max_value) or max_value <= rule["max"]
            passed = passed and max_passed
            details.append(f"max={max_value}, expected <= {rule['max']}")

        results.append(
            ValidationResult(
                dataset=dataset_name,
                check_type="range_check",
                column=col,
                passed=passed,
                details=" | ".join(details),
            )
        )

    return results


def validate_dataset(dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate one dataset and return:
    1) cleaned/coerced dataframe (in memory)
    2) validation report dataframe
    """
    df = load_dataset(dataset_name)
    df = coerce_known_types(df, dataset_name)

    if dataset_name == "public_assistance":
        df = clip_negative_values(df, [
            "projectAmount",
            "federalShareObligated",
            "totalObligated"
        ])

    elif dataset_name == "fema_web_disaster_summaries":
        df = clip_negative_values(df, [
            "totalObligatedAmountPa",
            "totalObligatedAmountCatAb",
            "totalObligatedAmountCatC2g",
            "totalObligatedAmountHmgp"
        ])

    results = []
    results.extend(check_required_columns(df, dataset_name))
    results.extend(check_expected_types(df, dataset_name))
    results.extend(check_null_thresholds(df, dataset_name))
    results.extend(check_value_ranges(df, dataset_name))

    report_df = pd.DataFrame([vars(r) for r in results])
    return df, report_df


def print_validation_summary(report_df: pd.DataFrame, dataset_name: str) -> None:
    total_checks = len(report_df)
    passed_checks = int(report_df["passed"].sum())
    failed_checks = total_checks - passed_checks

    print(f"\n{'=' * 70}")
    print(f"VALIDATION REPORT: {dataset_name}")
    print(f"{'=' * 70}")
    print(f"Total checks : {total_checks}")
    print(f"Passed       : {passed_checks}")
    print(f"Failed       : {failed_checks}")

    if failed_checks == 0:
        print("Overall      : PASS")
    else:
        print("Overall      : FAIL")
        print("\nFailed checks:")
        display_cols = ["check_type", "column", "details"]
        print(report_df.loc[~report_df["passed"], display_cols].to_string(index=False))


def run_all_validations() -> dict[str, dict[str, pd.DataFrame]]:
    """
    Run validation for all datasets and return results.
    """
    all_results = {}

    for dataset_name in DATASET_FILES.keys():
        df, report_df = validate_dataset(dataset_name)
        print_validation_summary(report_df, dataset_name)

        all_results[dataset_name] = {
            "dataframe": df,
            "report": report_df,
        }

    return all_results


def clip_negative_values(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Replace negative values with 0 for specified numeric columns.
    """
    df = df.copy()

    for col in columns:
        if col in df.columns:
            negative_count = (df[col] < 0).sum()

            if negative_count > 0:
                print(f"[FIX] {col}: {negative_count} negative values → set to 0")

                df[col] = df[col].clip(lower=0)

    return df


if __name__ == "__main__":
    run_all_validations()