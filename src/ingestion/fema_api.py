from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

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
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# API configuration
# -----------------------------------------------------------------------------
BASE_URL = "https://www.fema.gov/api/open"

ENDPOINTS = {
    "disaster_declarations_summaries": f"{BASE_URL}/v2/DisasterDeclarationsSummaries",
    "public_assistance_funded_projects_details": f"{BASE_URL}/v2/PublicAssistanceFundedProjectsDetails",
    "fema_web_disaster_summaries": f"{BASE_URL}/v1/FemaWebDisasterSummaries",
}

OUTPUT_FILES = {
    "disaster_declarations_summaries": RAW_DATA_DIR / "disaster_declarations_summaries.csv",
    "public_assistance_funded_projects_details": RAW_DATA_DIR / "public_assistance_funded_projects_details.csv",
    "fema_web_disaster_summaries": RAW_DATA_DIR / "fema_web_disaster_summaries.csv",
}

PAGE_SIZE = 1000
REQUEST_DELAY_SECONDS = 0.3
MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 1.0
FRESHNESS_DAYS = 7


def _is_fresh(file_path: Path, max_age_days: int = FRESHNESS_DAYS) -> bool:
    """
    Return True if file exists and is newer than max_age_days.
    """
    if not file_path.exists():
        return False

    age_seconds = time.time() - file_path.stat().st_mtime
    age_days = age_seconds / (60 * 60 * 24)
    return age_days < max_age_days


def _extract_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """
    OpenFEMA responses include metadata and one list-valued key containing records.
    This helper extracts that list robustly.
    """
    for value in payload.values():
        if isinstance(value, list):
            return value
    raise ValueError("No list of records found in FEMA API response.")


def _request_with_retry(
    url: str,
    params: dict[str, Any],
    session: requests.Session,
    max_retries: int = MAX_RETRIES,
    backoff_base: float = BACKOFF_BASE_SECONDS,
) -> dict[str, Any]:
    """
    Perform a GET request with exponential backoff retry logic.
    Retries on HTTP errors, connection errors, and timeouts.
    """
    for attempt in range(max_retries):
        try:
            response = session.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response.json()

        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as exc:
            is_last_attempt = attempt == max_retries - 1
            wait_time = backoff_base * (2 ** attempt)

            logger.warning(
                "Request failed for %s with params=%s | attempt=%s/%s | error=%s",
                url,
                params,
                attempt + 1,
                max_retries,
                exc,
            )

            if is_last_attempt:
                logger.error("Max retries reached. Failing request for %s", url)
                raise

            logger.info("Sleeping %.1f seconds before retry...", wait_time)
            time.sleep(wait_time)

    raise RuntimeError("Unexpected retry loop exit.")


def _fetch_paginated(
    url: str,
    page_size: int = PAGE_SIZE,
    request_delay_seconds: float = REQUEST_DELAY_SECONDS,
) -> pd.DataFrame:
    """
    Fetch all records from a paginated OpenFEMA endpoint using $top and $skip.
    """
    all_records: list[dict[str, Any]] = []
    skip = 0
    page_number = 1

    with requests.Session() as session:
        while True:
            params = {
                "$top": page_size,
                "$skip": skip,
                "$format": "json",
            }

            payload = _request_with_retry(url=url, params=params, session=session)
            records = _extract_records(payload)
            batch_size = len(records)

            logger.info(
                "Fetched page %s from %s | skip=%s | batch_size=%s",
                page_number,
                url,
                skip,
                batch_size,
            )

            if batch_size == 0:
                break

            all_records.extend(records)

            if batch_size < page_size:
                # Last page reached
                break

            skip += page_size
            page_number += 1
            time.sleep(request_delay_seconds)

    df = pd.DataFrame(all_records)
    logger.info("Completed fetch from %s | total_records=%s", url, len(df))
    return df


def _fetch_and_save(
    dataset_name: str,
    endpoint_url: str,
    output_path: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Generic dataset fetcher with freshness check and CSV persistence.
    """
    if not force_refresh and _is_fresh(output_path):
        logger.info(
            "Skipping fetch for %s because %s is newer than %s days.",
            dataset_name,
            output_path.name,
            FRESHNESS_DAYS,
        )
        df_existing = pd.read_csv(output_path)
        logger.info(
            "Loaded existing %s from disk | record_count=%s",
            dataset_name,
            len(df_existing),
        )
        return df_existing

    logger.info("Starting fetch for dataset: %s", dataset_name)
    df = _fetch_paginated(endpoint_url)
    df.to_csv(output_path, index=False)
    logger.info(
        "Saved %s to %s | record_count=%s",
        dataset_name,
        output_path,
        len(df),
    )
    return df


def fetch_disaster_declarations(force_refresh: bool = False) -> pd.DataFrame:
    return _fetch_and_save(
        dataset_name="disaster_declarations_summaries",
        endpoint_url=ENDPOINTS["disaster_declarations_summaries"],
        output_path=OUTPUT_FILES["disaster_declarations_summaries"],
        force_refresh=force_refresh,
    )


def fetch_public_assistance_projects(force_refresh: bool = False) -> pd.DataFrame:
    return _fetch_and_save(
        dataset_name="public_assistance_funded_projects_details",
        endpoint_url=ENDPOINTS["public_assistance_funded_projects_details"],
        output_path=OUTPUT_FILES["public_assistance_funded_projects_details"],
        force_refresh=force_refresh,
    )


def fetch_fema_web_disaster_summaries(force_refresh: bool = False) -> pd.DataFrame:
    return _fetch_and_save(
        dataset_name="fema_web_disaster_summaries",
        endpoint_url=ENDPOINTS["fema_web_disaster_summaries"],
        output_path=OUTPUT_FILES["fema_web_disaster_summaries"],
        force_refresh=force_refresh,
    )


def run_full_ingestion(force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    """
    Fetch all three required FEMA datasets and return them in a dictionary.
    """
    results = {
        "disaster_declarations_summaries": fetch_disaster_declarations(force_refresh=force_refresh),
        "public_assistance_funded_projects_details": fetch_public_assistance_projects(force_refresh=force_refresh),
        "fema_web_disaster_summaries": fetch_fema_web_disaster_summaries(force_refresh=force_refresh),
    }

    logger.info(
        "Full ingestion complete | declarations=%s | public_assistance=%s | web_summaries=%s",
        len(results["disaster_declarations_summaries"]),
        len(results["public_assistance_funded_projects_details"]),
        len(results["fema_web_disaster_summaries"]),
    )
    return results


if __name__ == "__main__":
    run_full_ingestion(force_refresh=False)