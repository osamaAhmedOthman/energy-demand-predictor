"""
test_api.py — Tests for the FastAPI endpoints.

Key design for CI compatibility:
  - The 'client' fixture always overrides get_model with a mock.
  - This means tests never need a real MLflow server or registered model.
  - Tests run identically locally and in GitHub Actions CI.
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from api.main import app
from api.routers.predict import get_model


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_pipeline():
    """
    A minimal trained sklearn Pipeline that the API can call .predict() on.

    We use Ridge (fast to fit) wrapped in a Pipeline to match the real model's
    interface. It is fitted on random data — the actual predictions don't matter
    for API contract tests.
    """
    # For testing we don't care about realistic predictions at all — we just
    # need something with a .predict() method that returns non-negative numbers
    # of the correct length. A DummyRegressor set to a constant value is ideal
    # because its output is independent of the input features and it cannot
    # explode to absurd values when the feature distribution changes.
    from sklearn.dummy import DummyRegressor

    pipe = Pipeline([
        # scaler is unnecessary because DummyRegressor ignores X, but we include
        # it to maintain the same interface as the real pipeline (which starts
        # with a transformer). Keeping the pipeline shape makes the test code
        # less sensitive to refactoring in production code.
        ('scaler', StandardScaler()),
        ('model', DummyRegressor(strategy='constant', constant=30000.0)),
    ])
    # Fit on dummy data so sklearn is happy; the regressor ignores X anyway.
    rng = np.random.default_rng(42)
    from src.config import ALL_FEATURES
    n_feats = len(ALL_FEATURES)
    X = rng.random((10, n_feats))   # small random matrix
    y = np.full(10, 30000.0)
    pipe.fit(X, y)
    return pipe


@pytest.fixture
def client(mock_pipeline):
    """
    TestClient with get_model dependency overridden.

    CRITICAL: This is the ONLY 'client' fixture in the file.
    The original file had two fixtures named 'client' — the second one
    (plain, no mock) silently overwrote the first (with mock), so the mock
    was never applied and every request hit the real MLflow connection.

    With dependency_overrides, FastAPI replaces get_model() with a lambda
    returning our mock pipeline. No MLflow connection is ever attempted.
    """
    app.dependency_overrides[get_model] = lambda: mock_pipeline
    yield TestClient(app)
    app.dependency_overrides.clear()   # clean up after every test


@pytest.fixture
def minimal_data_point():
    """One valid hourly data point. Used to build request payloads."""
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
def sufficient_data(minimal_data_point):
    """
    340 hourly data points (above the 336-row minimum for lag features).
    Each point gets a unique timestamp one hour apart.
    """
    import pandas as pd
    timestamps = pd.date_range("2023-01-01", periods=340, freq="h", tz="UTC")
    data = []
    for ts in timestamps:
        point = dict(minimal_data_point)
        point["time"] = ts.isoformat()
        data.append(point)
    return data


# ── Health endpoint ────────────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_returns_200(self, client):
        """Health endpoint must always return 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_response_has_required_fields(self, client):
        """Health response must contain all documented fields."""
        data = client.get("/health").json()
        for field in ("status", "timestamp", "version", "model_loaded", "uptime_seconds"):
            assert field in data, f"Missing field in /health response: {field}"

    def test_version_is_string(self, client):
        data = client.get("/health").json()
        assert isinstance(data["version"], str)

    def test_model_loaded_is_bool(self, client):
        data = client.get("/health").json()
        assert isinstance(data["model_loaded"], bool)

    def test_uptime_is_non_negative(self, client):
        data = client.get("/health").json()
        assert data["uptime_seconds"] >= 0


# ── Root endpoint ──────────────────────────────────────────────────────────────

class TestRootEndpoint:

    def test_returns_200(self, client):
        assert client.get("/").status_code == 200

    def test_response_has_required_fields(self, client):
        data = client.get("/").json()
        for field in ("message", "version", "docs", "health"):
            assert field in data, f"Missing field in / response: {field}"


# ── Predict endpoint: request validation ──────────────────────────────────────

class TestPredictValidation:

    def test_empty_body_returns_422(self, client):
        """
        FastAPI must reject a completely empty request body with 422.

        WHY THIS TEST FAILED BEFORE:
        The file had two fixtures named 'client'. The second (plain, no mock)
        overwrote the first (with mock). So get_model() tried to connect to
        MLflow, failed, and returned 503 before FastAPI even read the request
        body. With the mock in place, FastAPI reads the body first and
        correctly returns 422 for missing required fields.
        """
        response = client.post("/api/v1/predict", json={})
        assert response.status_code == 422

    def test_missing_data_field_returns_422(self, client):
        """Request without the 'data' field must return 422."""
        response = client.post("/api/v1/predict", json={"hours_ahead": 24})
        assert response.status_code == 422

    def test_data_as_wrong_type_returns_422(self, client):
        """'data' must be a list — a string should fail validation."""
        response = client.post("/api/v1/predict", json={"data": "not-a-list"})
        assert response.status_code == 422

    def test_insufficient_data_returns_400(self, client, minimal_data_point):
        """
        A single data point is far below the 336-hour minimum.
        The API must return 400 with a clear error message.
        """
        response = client.post(
            "/api/v1/predict",
            json={"data": [minimal_data_point]}
        )
        assert response.status_code == 400
        assert "Insufficient historical data" in response.json()["detail"]


# ── Predict endpoint: successful request ──────────────────────────────────────

class TestPredictSuccess:

    def test_valid_request_returns_200(self, client, sufficient_data):
        """A well-formed request with enough data must return 200."""
        response = client.post(
            "/api/v1/predict",
            json={"data": sufficient_data, "hours_ahead": 24}
        )
        assert response.status_code == 200

    def test_response_has_predictions_key(self, client, sufficient_data):
        data = client.post("/api/v1/predict", json={"data": sufficient_data}).json()
        assert "predictions" in data

    def test_response_has_model_info(self, client, sufficient_data):
        data = client.post("/api/v1/predict", json={"data": sufficient_data}).json()
        assert "model_info" in data

    def test_response_has_prediction_range(self, client, sufficient_data):
        data = client.post("/api/v1/predict", json={"data": sufficient_data}).json()
        assert "prediction_range" in data
        for key in ("min_mw", "max_mw", "mean_mw"):
            assert key in data["prediction_range"]

    def test_predictions_list_is_non_empty(self, client, sufficient_data):
        data = client.post("/api/v1/predict", json={"data": sufficient_data}).json()
        assert len(data["predictions"]) > 0

    def test_each_prediction_has_timestamp_and_value(self, client, sufficient_data):
        data = client.post("/api/v1/predict", json={"data": sufficient_data}).json()
        for point in data["predictions"]:
            assert "timestamp" in point
            assert "predicted_demand_mw" in point
            assert isinstance(point["predicted_demand_mw"], float)