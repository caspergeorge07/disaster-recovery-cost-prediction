from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

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
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = PROCESSED_DATA_DIR / "processed_disasters.csv"


# -----------------------------------------------------------------------------
# Helper mappings
# -----------------------------------------------------------------------------
CENSUS_REGION_MAP = {
    # Northeast
    "CT": "Northeast", "ME": "Northeast", "MA": "Northeast", "NH": "Northeast",
    "RI": "Northeast", "VT": "Northeast", "NJ": "Northeast", "NY": "Northeast",
    "PA": "Northeast",

    # Midwest
    "IL": "Midwest", "IN": "Midwest", "MI": "Midwest", "OH": "Midwest",
    "WI": "Midwest", "IA": "Midwest", "KS": "Midwest", "MN": "Midwest",
    "MO": "Midwest", "NE": "Midwest", "ND": "Midwest", "SD": "Midwest",

    # South
    "DE": "South", "FL": "South", "GA": "South", "MD": "South",
    "NC": "South", "SC": "South", "VA": "South", "DC": "South",
    "WV": "South", "AL": "South", "KY": "South", "MS": "South",
    "TN": "South", "AR": "South", "LA": "South", "OK": "South",
    "TX": "South",

    # West
    "AZ": "West", "CO": "West", "ID": "West", "MT": "West",
    "NV": "West", "NM": "West", "UT": "West", "WY": "West",
    "AK": "West", "CA": "West", "HI": "West", "OR": "West",
    "WA": "West",

    # Territories
    "PR": "Territory", "VI": "Territory", "GU": "Territory",
    "AS": "Territory", "MP": "Territory"
}

HIGH_COST_INCIDENT_TYPES = {"Hurricane", "Flood", "Tornado", "Severe Storm"}


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------
def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three raw FEMA datasets.
    """
    df_decl = pd.read_csv(RAW_DATA_DIR / "disaster_declarations_summaries.csv")
    df_pa = pd.read_csv(RAW_DATA_DIR / "public_assistance_funded_projects_details.csv")
    df_sum = pd.read_csv(RAW_DATA_DIR / "fema_web_disaster_summaries.csv")

    logger.info("Loaded declarations | shape=%s", df_decl.shape)
    logger.info("Loaded public assistance | shape=%s", df_pa.shape)
    logger.info("Loaded web summaries | shape=%s", df_sum.shape)

    return df_decl, df_pa, df_sum


# -----------------------------------------------------------------------------
# Cleaning helpers
# -----------------------------------------------------------------------------
def clip_negative_values(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Clip negative numeric values to zero for selected columns.
    This is appropriate here because negative FEMA obligations can represent
    accounting adjustments, but the modelling target should represent
    non-negative recovery expenditure.
    """
    df = df.copy()

    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            negative_count = (df[col] < 0).sum()

            if negative_count > 0:
                logger.info("Clipping %s negative values in column '%s' to zero", negative_count, col)
                df[col] = df[col].clip(lower=0)

    return df


def preprocess_declarations(df_decl: pd.DataFrame) -> pd.DataFrame:
    """
    Parse declaration-level fields needed for feature engineering.
    """
    df = df_decl.copy()

    date_cols = ["declarationDate", "incidentBeginDate", "incidentEndDate"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def preprocess_public_assistance(df_pa: pd.DataFrame) -> pd.DataFrame:
    """
    Parse and clean project-level PA data before aggregation.
    """
    df = df_pa.copy()

    if "declarationDate" in df.columns:
        df["declarationDate"] = pd.to_datetime(df["declarationDate"], errors="coerce")

    numeric_cols = ["projectAmount", "federalShareObligated", "totalObligated"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = clip_negative_values(df, numeric_cols)

    return df


def preprocess_web_summaries(df_sum: pd.DataFrame) -> pd.DataFrame:
    """
    Parse and lightly clean web summary fields.
    """
    df = df_sum.copy()

    if "lastRefresh" in df.columns:
        df["lastRefresh"] = pd.to_datetime(df["lastRefresh"], errors="coerce")

    numeric_cols = [
        "totalObligatedAmountPa",
        "totalObligatedAmountCatAb",
        "totalObligatedAmountCatC2g",
        "totalObligatedAmountHmgp",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = clip_negative_values(df, numeric_cols)

    return df


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------
def aggregate_public_assistance(df_pa: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate project-level PA records to disaster-level features.
    """
    agg_df = (
        df_pa.groupby("disasterNumber", as_index=False)
        .agg(
            total_obligated_amount=("totalObligated", "sum"),
            project_count=("gmProjectId", "count"),
            avg_project_amount=("projectAmount", "mean"),
        )
    )

    return agg_df


# -----------------------------------------------------------------------------
# Feature engineering helpers
# -----------------------------------------------------------------------------
def map_season(month: float) -> str:
    if pd.isna(month):
        return "Unknown"
    month = int(month)

    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Autumn"
    return "Unknown"


def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["declaration_year"] = df["declarationDate"].dt.year
    df["declaration_month"] = df["declarationDate"].dt.month

    df["incident_duration_days"] = (
        df["incidentEndDate"] - df["incidentBeginDate"]
    ).dt.days

    # Handle impossible negative durations if they exist
    df.loc[df["incident_duration_days"] < 0, "incident_duration_days"] = np.nan

    return df


def engineer_seasonality_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["season"] = df["declaration_month"].apply(map_season)
    return df


def engineer_geographic_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["census_region"] = df["state"].map(CENSUS_REGION_MAP).fillna("Unknown")
    return df


def engineer_risk_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["high_cost_incident"] = df["incidentType"].isin(HIGH_COST_INCIDENT_TYPES)
    return df


def engineer_historical_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a 5-year rolling disaster count per state, based on declaration year.

    For each row, count how many disasters occurred in the same state during:
    [current_year - 4, current_year]
    """
    df = df.copy()

    counts = (
        df.groupby(["state", "declaration_year"])
        .size()
        .reset_index(name="yearly_state_disaster_count")
        .sort_values(["state", "declaration_year"])
    )

    rolling_frames = []

    for state, group in counts.groupby("state"):
        group = group.sort_values("declaration_year").copy()

        rolling_values = []
        years = group["declaration_year"].tolist()
        yearly_counts = group["yearly_state_disaster_count"].tolist()

        for current_year in years:
            total_count = 0
            for year, count in zip(years, yearly_counts):
                if current_year - 4 <= year <= current_year:
                    total_count += count
            rolling_values.append(total_count)

        group["state_5yr_disaster_count"] = rolling_values
        rolling_frames.append(group)

    rolling_df = pd.concat(rolling_frames, ignore_index=True)

    df = df.merge(
        rolling_df[["state", "declaration_year", "state_5yr_disaster_count"]],
        on=["state", "declaration_year"],
        how="left"
    )

    return df


def add_log_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_obligated_amount"] = df["total_obligated_amount"].fillna(0)
    df["log_total_obligated_amount"] = np.log1p(df["total_obligated_amount"])
    return df


# -----------------------------------------------------------------------------
# Merge pipeline
# -----------------------------------------------------------------------------
def build_processed_dataset() -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    """
    df_decl, df_pa, df_sum = load_raw_data()

    df_decl = preprocess_declarations(df_decl)
    df_pa = preprocess_public_assistance(df_pa)
    df_sum = preprocess_web_summaries(df_sum)

    # Aggregate PA to disaster level
    pa_agg = aggregate_public_assistance(df_pa)
    logger.info("Aggregated PA to disaster level | shape=%s", pa_agg.shape)

    # Select useful columns from web summaries
    web_cols = [
        "disasterNumber",
        "totalObligatedAmountPa",
        "totalObligatedAmountCatAb",
        "totalObligatedAmountCatC2g",
        "totalObligatedAmountHmgp",
    ]
    web_cols = [col for col in web_cols if col in df_sum.columns]
    df_sum_selected = df_sum[web_cols].copy()

    # Merge all tables on disasterNumber
    df = df_decl.merge(pa_agg, on="disasterNumber", how="left")
    df = df.merge(df_sum_selected, on="disasterNumber", how="left")

    logger.info("Merged dataset shape after joins | shape=%s", df.shape)

    # Engineer features
    df = engineer_temporal_features(df)
    df = engineer_seasonality_feature(df)
    df = engineer_geographic_feature(df)
    df = engineer_historical_frequency(df)
    df = engineer_risk_flag(df)
    df = add_log_target(df)

    # Fill project aggregates if missing
    df["project_count"] = df["project_count"].fillna(0)
    df["avg_project_amount"] = df["avg_project_amount"].fillna(0)

    # Optional: keep a stable column order for key engineered fields first
    preferred_cols = [
        "disasterNumber",
        "state",
        "incidentType",
        "declarationType",
        "declarationDate",
        "incidentBeginDate",
        "incidentEndDate",
        "declaration_year",
        "declaration_month",
        "incident_duration_days",
        "season",
        "census_region",
        "state_5yr_disaster_count",
        "high_cost_incident",
        "project_count",
        "avg_project_amount",
        "total_obligated_amount",
        "log_total_obligated_amount",
    ]

    remaining_cols = [col for col in df.columns if col not in preferred_cols]
    df = df[[col for col in preferred_cols if col in df.columns] + remaining_cols]

    # Save output
    df.to_csv(OUTPUT_PATH, index=False)
    logger.info("Saved processed dataset to %s | shape=%s", OUTPUT_PATH, df.shape)

    return df


def inspect_processed_dataset(df: pd.DataFrame) -> None:
    """
    Print a quick inspection of the processed output.
    """
    print("\nProcessed dataset shape:", df.shape)
    print("\nProcessed dataset columns:")
    for col in df.columns:
        print("-", col)

    print("\nSample rows:")
    print(df.head(5).to_string())


if __name__ == "__main__":
    processed_df = build_processed_dataset()
    inspect_processed_dataset(processed_df)