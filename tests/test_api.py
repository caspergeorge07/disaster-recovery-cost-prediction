import pytest
from fastapi.testclient import TestClient

from api.main import app, model_artifacts


class MockModel:
    def predict(self, X):
        return [16.0]


@pytest.fixture(autouse=True)
def mock_model_artifact():
    model_artifacts["model"] = MockModel()
    model_artifacts["metadata"] = {
        "best_model_name": "mock_model",
        "model_version": "test",
        "training_date": "2026-01-01",
        "target_column": "log_total_obligated_amount",
        "metrics": {"r2": 0.85},
        "numeric_features": [
            "declaration_year",
            "declaration_month",
            "incident_duration_days",
            "state_5yr_disaster_count",
            "high_cost_incident",
            "fyDeclared",
            "tribalRequest",
            "fipsStateCode",
            "fipsCountyCode",
            "placeCode",
            "region",
        ],
        "categorical_features": [
            "state",
            "incidentType",
            "declarationType",
            "season",
            "census_region",
        ],
    }
    model_artifacts["loaded"] = True
    yield


client = TestClient(app)


def valid_payload():
    return {
        "declaration_year": 2024,
        "declaration_month": 9,
        "incident_duration_days": 14,
        "state_5yr_disaster_count": 120,
        "high_cost_incident": True,
        "fyDeclared": 2024,
        "tribalRequest": False,
        "fipsStateCode": 48,
        "fipsCountyCode": 201,
        "placeCode": 12345,
        "region": 6,
        "state": "TX",
        "incidentType": "Hurricane",
        "declarationType": "DR",
    }


def test_health_status_code_and_body():
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_name"] == "mock_model"


def test_predict_cost_valid_request_returns_expected_structure():
    response = client.post("/predict-cost", json=valid_payload())

    assert response.status_code == 200
    body = response.json()

    assert "predicted_log_cost" in body
    assert "predicted_cost_usd" in body
    assert "model_name" in body
    assert "model_version" in body
    assert "target" in body
    assert body["model_name"] == "mock_model"
    assert body["predicted_log_cost"] == 16.0
    assert body["predicted_cost_usd"] > 0


def test_predict_cost_missing_required_fields_returns_422():
    bad_payload = valid_payload()
    bad_payload.pop("declaration_year")

    response = client.post("/predict-cost", json=bad_payload)

    assert response.status_code == 422


def test_model_info_returns_expected_fields():
    response = client.get("/model-info")

    assert response.status_code == 200
    body = response.json()

    assert "model_name" in body
    assert "model_version" in body
    assert "training_date" in body
    assert "target_column" in body
    assert "metrics" in body
    assert "numeric_features" in body
    assert "categorical_features" in body