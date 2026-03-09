"""
test_api.py — Tests for the FastAPI endpoints.

Key design for CI compatibility:
  - The 'client' fixture ALWAYS overrides get_model with a mock.
  - No real MLflow server or registered model is ever needed.
  - Tests run identically locally and in GitHub Actions CI.

Root cause of the CI failures:
  - The old file had TWO fixtures both named 'client'.
  - Python silently uses the LAST definition, which had no mock.
  - So every test called the real get_model() → MLflow not found → 503.
  - FastAPI's Depends() runs BEFORE body validation, so 503 appeared
    instead of the expected 422, even on completely empty requests.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from api.main import app
from api.routers.predict import get_model


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_model():
    """
    A MagicMock that behaves like a fitted sklearn pipeline.

    predict() returns 30,000.0 (realistic MW value) for any input.
    Using MagicMock instead of a real sklearn model keeps the fixture
    fast and avoids any dependency on feature column names or counts.
    """
    model = MagicMock()
    model.predict = lambda X: np.full(len(X), 30_000.0)
    return model


@pytest.fixture
def client(mock_model):
    """
    TestClient with the MLflow model dependency replaced by mock_model.

    This is the ONLY 'client' fixture in this file.
    dependency_overrides tells FastAPI: when any endpoint calls
    Depends(get_model), return mock_model instantly instead.
    No network call, no MLflow, no 503.
    """
    app.dependency_overrides[get_model] = lambda: mock_model
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def one_data_point():
    """One complete, valid hourly data point."""
    return {
        "time":                             "2023-01-01T00:00:00Z",
        "generation_fossil_gas":            2500.0,
        "generation_fossil_hard_coal":      800.0,
        "generation_nuclear":               6000.0,
        "generation_wind_onshore":          1200.0,
        "generation_solar":                 0.0,
        "generation_hydro_water_reservoir": 800.0,
        "generation_biomass":               400.0,
        "forecast_solar_day_ahead":         0.0,
        "forecast_wind_onshore_day_ahead":  1200.0,
        "total_load_forecast":              25000.0,
        "total_load_actual":                24800.0,
        "temp":                             285.0,
        "temp_min":                         282.0,
        "temp_max":                         288.0,
        "pressure":                         1013.0,
        "humidity":                         65.0,
        "wind_speed":                       3.5,
        "wind_deg":                         180.0,
        "rain_1h":                          0.0,
        "rain_3h":                          0.0,
        "snow_3h":                          0.0,
        "clouds_all":                       20.0,
    }


@pytest.fixture
def enough_data(one_data_point):
    """
    340 hourly data points — above the 336-row minimum so the router's
    data-size check passes and we reach actual prediction logic.
    """
    timestamps = pd.date_range("2023-01-01", periods=340, freq="h", tz="UTC")
    data = []
    for ts in timestamps:
        point = dict(one_data_point)
        point["time"] = ts.isoformat()
        data.append(point)
    return data


# ── Health endpoint ────────────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_returns_200(self, client):
        """Health endpoint must always return 200."""
        assert client.get("/health").status_code == 200

    def test_has_all_required_fields(self, client):
        """All documented fields must be present in the health response."""
        data = client.get("/health").json()
        for field in ("status", "version", "model_loaded", "uptime_seconds", "timestamp"):
            assert field in data, f"Missing field: {field}"

    def test_version_is_string(self, client):
        assert isinstance(client.get("/health").json()["version"], str)

    def test_model_loaded_is_bool(self, client):
        assert isinstance(client.get("/health").json()["model_loaded"], bool)

    def test_uptime_is_non_negative(self, client):
        assert client.get("/health").json()["uptime_seconds"] >= 0


# ── Root endpoint ──────────────────────────────────────────────────────────────

class TestRootEndpoint:

    def test_returns_200(self, client):
        assert client.get("/").status_code == 200

    def test_has_required_fields(self, client):
        data = client.get("/").json()
        for field in ("message", "version", "docs", "health"):
            assert field in data, f"Missing field in / response: {field}"


# ── Predict endpoint: validation (these were the failing tests) ───────────────

class TestPredictValidation:

    def test_empty_body_returns_422(self, client):
        """
        {} has no 'data' field → Pydantic must return 422.

        This was the failing test. It returned 503 because:
          1. The old file had two 'client' fixtures, second overwrote first.
          2. The mock was never applied.
          3. get_model() ran for real → MLflow not found → 503.
          4. FastAPI's Depends() runs BEFORE body validation, so 503 came
             first and 422 was never reached.

        With the mock applied correctly, get_model() returns instantly,
        FastAPI validates the body, and 422 is returned as expected.
        """
        response = client.post("/api/v1/predict", json={})
        assert response.status_code == 422

    def test_missing_data_field_returns_422(self, client):
        """Request body with no 'data' key must fail Pydantic validation."""
        response = client.post("/api/v1/predict", json={"hours_ahead": 24})
        assert response.status_code == 422

    def test_data_wrong_type_returns_422(self, client):
        """'data' must be a list — passing a string must fail validation."""
        response = client.post("/api/v1/predict", json={"data": "not-a-list"})
        assert response.status_code == 422

    def test_too_few_rows_returns_400(self, client, one_data_point):
        """
        One row is far below the 336-hour minimum.
        After body validation passes (422 check), the router checks
        row count and returns 400 with a clear message.
        """
        response = client.post("/api/v1/predict", json={"data": [one_data_point]})
        assert response.status_code == 400
        assert "Insufficient historical data" in response.json()["detail"]


# ── Predict endpoint: successful request ──────────────────────────────────────

class TestPredictSuccess:

    def test_valid_request_returns_200(self, client, enough_data):
        """340 rows + mocked model → must return 200."""
        response = client.post(
            "/api/v1/predict",
            json={"data": enough_data, "hours_ahead": 24}
        )
        assert response.status_code == 200

    def test_response_contains_predictions(self, client, enough_data):
        data = client.post("/api/v1/predict", json={"data": enough_data}).json()
        assert "predictions" in data
        assert len(data["predictions"]) > 0

    def test_response_contains_model_info(self, client, enough_data):
        data = client.post("/api/v1/predict", json={"data": enough_data}).json()
        assert "model_info" in data

    def test_response_contains_prediction_range(self, client, enough_data):
        data = client.post("/api/v1/predict", json={"data": enough_data}).json()
        rng = data["prediction_range"]
        for key in ("min_mw", "max_mw", "mean_mw"):
            assert key in rng, f"Missing key in prediction_range: {key}"

    def test_each_prediction_has_correct_keys(self, client, enough_data):
        data = client.post("/api/v1/predict", json={"data": enough_data}).json()
        for point in data["predictions"]:
            assert "timestamp"           in point
            assert "predicted_demand_mw" in point
            assert isinstance(point["predicted_demand_mw"], float)

    def test_prediction_range_values_are_consistent(self, client, enough_data):
        """min_mw <= mean_mw <= max_mw must always hold."""
        rng = client.post(
            "/api/v1/predict", json={"data": enough_data}
        ).json()["prediction_range"]
        assert rng["min_mw"] <= rng["mean_mw"] <= rng["max_mw"]