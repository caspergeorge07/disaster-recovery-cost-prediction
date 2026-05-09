from pathlib import Path
from typing import Dict

import os
import requests
import streamlit as st
import plotly.graph_objects as go


# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Disaster Recovery Cost Predictor",
    page_icon="🌪️",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict-cost"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"
MODEL_INFO_ENDPOINT = f"{API_BASE_URL}/model-info"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHAP_IMAGE_PATH = PROJECT_ROOT / "models" / "shap_summary.png"


# -----------------------------------------------------------------------------
# State and region mappings
# -----------------------------------------------------------------------------
US_STATES: Dict[str, str] = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
    "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska",
    "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}

FIPS_STATE_CODES = {
    "AL": 1, "AK": 2, "AZ": 4, "AR": 5, "CA": 6, "CO": 8, "CT": 9, "DE": 10,
    "DC": 11, "FL": 12, "GA": 13, "HI": 15, "ID": 16, "IL": 17, "IN": 18,
    "IA": 19, "KS": 20, "KY": 21, "LA": 22, "ME": 23, "MD": 24, "MA": 25,
    "MI": 26, "MN": 27, "MS": 28, "MO": 29, "MT": 30, "NE": 31, "NV": 32,
    "NH": 33, "NJ": 34, "NM": 35, "NY": 36, "NC": 37, "ND": 38, "OH": 39,
    "OK": 40, "OR": 41, "PA": 42, "RI": 44, "SC": 45, "SD": 46, "TN": 47,
    "TX": 48, "UT": 49, "VT": 50, "VA": 51, "WA": 53, "WV": 54, "WI": 55,
    "WY": 56,
}

HIGH_COST_INCIDENTS = {"Hurricane", "Flood", "Tornado", "Severe Storm"}

INCIDENT_TYPES = [
    "Fire", "Severe Storm", "Flood", "Hurricane", "Snowstorm", "Tornado",
    "Biological", "Typhoon", "Severe Ice Storm", "Drought", "Earthquake",
    "Coastal Storm", "Freezing", "Other"
]

DECLARATION_TYPES = ["DR", "FM", "EM"]


def get_season(month: int) -> str:
    """Map month to season."""
    if month in [12, 1, 2]:
        return "Winter"
    if month in [3, 4, 5]:
        return "Spring"
    if month in [6, 7, 8]:
        return "Summer"
    if month in [9, 10, 11]:
        return "Autumn"
    return "Unknown"


def get_census_region(state: str) -> str:
    """Map US state code to Census region."""
    northeast = {"CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"}
    midwest = {"IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"}
    south = {"DE", "FL", "GA", "MD", "NC", "SC", "VA", "DC", "WV", "AL", "KY", "MS", "TN", "AR", "LA", "OK", "TX"}
    west = {"AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"}

    if state in northeast:
        return "Northeast"
    if state in midwest:
        return "Midwest"
    if state in south:
        return "South"
    if state in west:
        return "West"
    return "Unknown"


def check_api_health() -> dict | None:
    """Check if FastAPI backend is available."""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None


def call_prediction_api(payload: dict) -> dict:
    """Send prediction payload to FastAPI."""
    response = requests.post(PREDICT_ENDPOINT, json=payload, timeout=15)
    response.raise_for_status()
    return response.json()


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title("🌪️ Disaster Recovery Cost Prediction Dashboard")
st.markdown(
    """
    This dashboard estimates disaster recovery cost using the trained FEMA disaster recovery model.
    Adjust the inputs in the sidebar, submit the scenario, and compare the predicted cost against an allocated budget.
    """
)


# -----------------------------------------------------------------------------
# API health status
# -----------------------------------------------------------------------------
health = check_api_health()

if health is None:
    st.error(
        "API connection failed. Please start the FastAPI backend first using: "
        "`uvicorn api.main:app --reload`"
    )
    st.stop()

if not health.get("model_loaded", False):
    st.error("API is running, but the model is not loaded.")
    st.stop()

st.success(f"API connected. Model loaded: `{health.get('model_name', 'unknown')}`")


# -----------------------------------------------------------------------------
# Sidebar inputs
# -----------------------------------------------------------------------------
st.sidebar.header("Scenario Inputs")

incident_type = st.sidebar.selectbox("Incident Type", INCIDENT_TYPES)
declaration_type = st.sidebar.selectbox("Declaration Type", DECLARATION_TYPES)

state_label = st.sidebar.selectbox(
    "State",
    options=[f"{code} - {name}" for code, name in US_STATES.items()],
    index=list(US_STATES.keys()).index("TX"),
)
state = state_label.split(" - ")[0]

declaration_year = st.sidebar.number_input(
    "Declaration Year",
    min_value=1950,
    max_value=2100,
    value=2024,
    step=1,
)

declaration_month = st.sidebar.slider(
    "Declaration Month",
    min_value=1,
    max_value=12,
    value=9,
)

incident_duration_days = st.sidebar.slider(
    "Incident Duration (days)",
    min_value=1,
    max_value=180,
    value=14,
)

state_5yr_disaster_count = st.sidebar.slider(
    "5-Year State Disaster Frequency",
    min_value=0,
    max_value=20,
    value=5,
)

# These are collected for scenario context.
# The latest leakage-safe model does not use project_count or avg_project_amount,
# but showing them helps users think about budget planning.
project_count = st.sidebar.number_input(
    "Project Count (scenario context)",
    min_value=0,
    value=0,
    step=1,
)

avg_project_amount = st.sidebar.number_input(
    "Average Project Amount (scenario context)",
    min_value=0.0,
    value=0.0,
    step=10000.0,
)

budget_allocated = st.sidebar.number_input(
    "Allocated Budget (USD)",
    min_value=0.0,
    value=10_000_000.0,
    step=100_000.0,
)

with st.sidebar.expander("Advanced API Inputs"):
    fy_declared = st.number_input(
        "FY Declared",
        min_value=1950,
        max_value=2100,
        value=int(declaration_year),
        step=1,
    )

    tribal_request = st.checkbox("Tribal Request", value=False)

    fips_county_code = st.number_input(
        "FIPS County Code",
        min_value=0,
        value=0,
        step=1,
    )

    place_code = st.number_input(
        "Place Code",
        min_value=0,
        value=0,
        step=1,
    )

    region = st.number_input(
        "FEMA Region Code",
        min_value=0,
        value=0,
        step=1,
    )


# -----------------------------------------------------------------------------
# Derived features
# -----------------------------------------------------------------------------
season = get_season(declaration_month)
census_region = get_census_region(state)
high_cost_incident = incident_type in HIGH_COST_INCIDENTS
fips_state_code = FIPS_STATE_CODES.get(state, 0)


# -----------------------------------------------------------------------------
# Display scenario summary
# -----------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Incident Type", incident_type)
col2.metric("State", state)
col3.metric("Season", season)
col4.metric("High-Cost Flag", str(high_cost_incident))


# -----------------------------------------------------------------------------
# Prediction payload
# -----------------------------------------------------------------------------
payload = {
    "declaration_year": int(declaration_year),
    "declaration_month": int(declaration_month),
    "incident_duration_days": float(incident_duration_days),
    "state_5yr_disaster_count": float(state_5yr_disaster_count),
    "high_cost_incident": bool(high_cost_incident),
    "fyDeclared": int(fy_declared),
    "tribalRequest": bool(tribal_request),
    "fipsStateCode": int(fips_state_code),
    "fipsCountyCode": int(fips_county_code),
    "placeCode": int(place_code),
    "region": int(region),
    "state": state,
    "incidentType": incident_type,
    "declarationType": declaration_type,
    "season": season,
    "census_region": census_region,
}


# -----------------------------------------------------------------------------
# Prediction section
# -----------------------------------------------------------------------------
st.subheader("Prediction Result")

if st.button("Predict Recovery Cost", type="primary"):
    try:
        result = call_prediction_api(payload)

        predicted_cost = result["predicted_cost_usd"]
        predicted_log_cost = result["predicted_log_cost"]

        st.session_state["prediction_result"] = result
        st.session_state["predicted_cost"] = predicted_cost
        st.session_state["predicted_log_cost"] = predicted_log_cost

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the prediction API. Make sure FastAPI is running.")
    except requests.exceptions.Timeout:
        st.error("The prediction API took too long to respond. Please try again.")
    except requests.exceptions.HTTPError as exc:
        st.error(f"API returned an error: {exc.response.text}")
    except Exception as exc:
        st.error(f"Unexpected error while generating prediction: {exc}")


if "predicted_cost" in st.session_state:
    predicted_cost = st.session_state["predicted_cost"]
    predicted_log_cost = st.session_state["predicted_log_cost"]

    budget_gap = budget_allocated - predicted_cost
    budget_gap_pct = (budget_gap / predicted_cost * 100) if predicted_cost > 0 else 0

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    metric_col1.metric(
        "Predicted Recovery Cost",
        f"${predicted_cost:,.0f}",
        help=f"Log prediction: {predicted_log_cost:.4f}",
    )

    metric_col2.metric(
        "Allocated Budget",
        f"${budget_allocated:,.0f}",
    )

    metric_col3.metric(
        "Budget Gap",
        f"${budget_gap:,.0f}",
        delta=f"{budget_gap_pct:.1f}%",
        help="Positive means allocated budget is above predicted cost. Negative means a shortfall.",
    )

    fig = go.Figure(
        data=[
            go.Bar(name="Predicted Cost", x=["Predicted"], y=[predicted_cost]),
            go.Bar(name="Allocated Budget", x=["Allocated"], y=[budget_allocated]),
        ]
    )

    fig.update_layout(
        title="Predicted vs Allocated Budget",
        yaxis_title="USD",
        barmode="group",
        height=450,
    )

    st.plotly_chart(fig, use_container_width=True)

    if budget_gap < 0:
        st.warning(
            f"The allocated budget is below the predicted recovery cost by "
            f"${abs(budget_gap):,.0f}."
        )
    else:
        st.info(
            f"The allocated budget exceeds the predicted recovery cost by "
            f"${budget_gap:,.0f}."
        )


# -----------------------------------------------------------------------------
# Scenario context
# -----------------------------------------------------------------------------
st.subheader("Scenario Context")

context_df = {
    "Field": [
        "Incident Type",
        "Declaration Type",
        "State",
        "Census Region",
        "Season",
        "High-Cost Incident",
        "Incident Duration",
        "5-Year Disaster Frequency",
        "Project Count",
        "Average Project Amount",
    ],
    "Value": [
        incident_type,
        declaration_type,
        state,
        census_region,
        season,
        high_cost_incident,
        incident_duration_days,
        state_5yr_disaster_count,
        project_count,
        f"${avg_project_amount:,.0f}",
    ],
}

st.dataframe(context_df, use_container_width=True)


# -----------------------------------------------------------------------------
# Feature importance / SHAP section
# -----------------------------------------------------------------------------
st.subheader("Model Explainability")

st.markdown(
    """
    The SHAP summary plot shows which features had the strongest influence on model predictions during evaluation.
    Features near the top have higher global impact on model output.
    """
)

if SHAP_IMAGE_PATH.exists():
    st.image(str(SHAP_IMAGE_PATH), caption="SHAP Summary Plot", use_container_width=True)
else:
    st.warning(
        "SHAP summary image not found. Expected file: "
        f"`{SHAP_IMAGE_PATH}`"
    )


# -----------------------------------------------------------------------------
# Model information
# -----------------------------------------------------------------------------
with st.expander("Model Information"):
    try:
        model_info_response = requests.get(MODEL_INFO_ENDPOINT, timeout=5)
        model_info_response.raise_for_status()
        model_info = model_info_response.json()
        st.json(model_info)
    except Exception:
        st.warning("Could not retrieve model information from the API.")