"""
test_api.py — Tests for the FastAPI endpoints.

Tests the health check and prediction endpoints.
"""

import pytest
from fastapi.testclient import TestClient
import pandas as pd
from datetime import datetime, timedelta

from api.main import app
from tests.conftest import sample_timestamps


@pytest.fixture
def client():
    """Test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_energy_data(sample_timestamps):
    """Sample energy data for testing."""
    n = len(sample_timestamps)
    rng = pd.np.random.default_rng(42)

    # Create sample data similar to conftest.py
    data = []
    for i, ts in enumerate(sample_timestamps):
        hour_effect = 3000 * pd.np.sin(2 * pd.np.pi * (i % 24) / 24)
        demand = pd.np.clip(28000 + hour_effect + rng.normal(0, 300), 18000, 42000)

        data.append({
            "time": ts.isoformat(),
            "generation_fossil_gas": float(rng.uniform(2000, 8000)),
            "generation_fossil_hard_coal": float(rng.uniform(500, 3000)),
            "generation_nuclear": float(rng.uniform(5000, 8000)),
            "generation_wind_onshore": float(rng.uniform(500, 5000)),
            "generation_solar": float(rng.uniform(0, 2000)),
            "generation_hydro_water_reservoir": float(rng.uniform(500, 2000)),
            "generation_biomass": float(rng.uniform(200, 600)),
            "forecast_solar_day_ahead": float(rng.uniform(0, 2000)),
            "forecast_wind_onshore_day_ahead": float(rng.uniform(500, 5000)),
            "total_load_forecast": float(demand + rng.normal(0, 100)),
            "total_load_actual": float(demand),
            "temp": float(rng.uniform(275, 310)),
            "temp_min": float(rng.uniform(270, 305)),
            "temp_max": float(rng.uniform(280, 315)),
            "pressure": float(rng.uniform(990, 1030)),
            "humidity": float(rng.uniform(20, 100)),
            "wind_speed": float(rng.uniform(0, 15)),
            "wind_deg": float(rng.uniform(0, 360)),
            "rain_1h": float(rng.uniform(0, 5)),
            "rain_3h": float(rng.uniform(0, 10)),
            "snow_3h": float(rng.uniform(0, 2)),
            "clouds_all": float(rng.uniform(0, 100)),
        })

    return data


class TestHealthEndpoint:

    def test_health_endpoint_returns_200(self, client):
        """Health endpoint should return 200 and health status."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "model_loaded" in data
        assert "uptime_seconds" in data

    def test_health_endpoint_has_correct_structure(self, client):
        """Health response should have all required fields."""
        response = client.get("/health")
        data = response.json()

        assert data["version"] == "1.0.0"
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0


class TestPredictEndpoint:

    def test_predict_endpoint_requires_data(self, client):
        """Predict endpoint should reject requests without data."""
        response = client.post("/api/v1/predict", json={})
        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_validates_minimum_data(self, client):
        """Predict endpoint should require at least 336 data points."""
        insufficient_data = [
            {
                "time": "2023-01-01T00:00:00Z",
                "generation_fossil_gas": 2500.0,
                "generation_fossil_hard_coal": 800.0,
                "generation_nuclear": 6000.0,
                "generation_wind_onshore": 1200.0,
                "generation_solar": 0.0,
                "generation_hydro_water_reservoir": 800.0,
                "generation_biomass": 400.0,
                "forecast_solar_day_ahead": 0.0,
                "forecast_wind_onshore_day_ahead": 1200.0,
                "total_load_forecast": 25000.0,
                "temp": 285.0,
                "temp_min": 282.0,
                "temp_max": 288.0,
                "pressure": 1013.0,
                "humidity": 65.0,
                "wind_speed": 3.5,
                "wind_deg": 180.0,
                "rain_1h": 0.0,
                "rain_3h": 0.0,
                "snow_3h": 0.0,
                "clouds_all": 20.0
            }
        ]

        response = client.post("/api/v1/predict", json={"data": insufficient_data})
        assert response.status_code == 400
        assert "Insufficient historical data" in response.json()["detail"]

    def test_predict_endpoint_accepts_valid_request(self, client, sample_energy_data):
        """Predict endpoint should accept valid requests (may fail if model not loaded)."""
        request_data = {
            "data": sample_energy_data,
            "predict_start": None,
            "hours_ahead": 24
        }

        response = client.post("/api/v1/predict", json=request_data)

        # Either succeeds (200) if model is loaded, or fails (503) if not
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "model_info" in data
            assert "prediction_range" in data
            assert "request_metadata" in data
            assert len(data["predictions"]) > 0
        else:
            assert "ML model not available" in response.json()["detail"]


class TestRootEndpoint:

    def test_root_endpoint_returns_info(self, client):
        """Root endpoint should return API information."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data
        assert "health" in data