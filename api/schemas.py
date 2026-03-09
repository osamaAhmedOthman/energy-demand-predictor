"""
schemas.py — Pydantic models for API request/response validation.

Defines the structure of data sent to and received from the API endpoints.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime


class EnergyDataPoint(BaseModel):
    """Single row of energy and weather data."""
    time: datetime = Field(..., description="Timestamp in ISO format")
    generation_fossil_gas: float = Field(..., ge=0, description="Fossil gas generation (MW)")
    generation_fossil_hard_coal: float = Field(..., ge=0, description="Hard coal generation (MW)")
    generation_nuclear: float = Field(..., ge=0, description="Nuclear generation (MW)")
    generation_wind_onshore: float = Field(..., ge=0, description="Onshore wind generation (MW)")
    generation_solar: float = Field(..., ge=0, description="Solar generation (MW)")
    generation_hydro_water_reservoir: float = Field(..., ge=0, description="Hydro reservoir generation (MW)")
    generation_biomass: float = Field(..., ge=0, description="Biomass generation (MW)")
    forecast_solar_day_ahead: float = Field(..., ge=0, description="Solar forecast (MW)")
    forecast_wind_onshore_day_ahead: float = Field(..., ge=0, description="Wind forecast (MW)")
    total_load_forecast: float = Field(..., ge=0, description="Load forecast (MW)")
    total_load_actual: Optional[float] = Field(None, ge=0, description="Actual load (MW) - optional for predictions")
    temp: float = Field(..., description="Temperature (K)")
    temp_min: float = Field(..., description="Min temperature (K)")
    temp_max: float = Field(..., description="Max temperature (K)")
    pressure: float = Field(..., ge=0, description="Pressure (hPa)")
    humidity: float = Field(..., ge=0, le=100, description="Humidity (%)")
    wind_speed: float = Field(..., ge=0, description="Wind speed (m/s)")
    wind_deg: float = Field(..., ge=0, le=360, description="Wind direction (degrees)")
    rain_1h: float = Field(..., ge=0, description="Rain in last hour (mm)")
    rain_3h: float = Field(..., ge=0, description="Rain in last 3 hours (mm)")
    snow_3h: float = Field(..., ge=0, description="Snow in last 3 hours (mm)")
    clouds_all: float = Field(..., ge=0, le=100, description="Cloud cover (%)")


class PredictRequest(BaseModel):
    """Request model for energy demand prediction."""
    data: List[EnergyDataPoint] = Field(..., description="Historical energy and weather data")
    predict_start: Optional[str] = Field(None, description="Start date for predictions (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)")
    hours_ahead: Optional[int] = Field(24, ge=1, le=168, description="Number of hours to predict ahead")

    @validator('predict_start')
    def validate_predict_start(cls, v):
        if v is not None:
            try:
                datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError('predict_start must be in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)')
        return v


class PredictionPoint(BaseModel):
    """Single prediction point with timestamp and predicted demand."""
    timestamp: datetime = Field(..., description="Prediction timestamp")
    predicted_demand_mw: float = Field(..., ge=0, description="Predicted energy demand in MW")


class PredictResponse(BaseModel):
    """Response model for prediction results."""
    predictions: List[PredictionPoint] = Field(..., description="List of predictions")
    model_info: Dict[str, Any] = Field(..., description="Information about the model used")
    prediction_range: Dict[str, float] = Field(..., description="Min/max predicted values")
    request_metadata: Dict[str, Any] = Field(..., description="Metadata about the request")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class ErrorResponse(BaseModel):
    """Response model for error cases."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
