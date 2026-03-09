"""
routers/predict.py — FastAPI router for energy demand prediction endpoint.

Handles the /predict endpoint with input validation, model loading, and prediction.
"""

import logging
import sys
from pathlib import Path
from typing import List
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
import pandas as pd
from datetime import datetime, timedelta

from ..schemas import PredictRequest, PredictResponse, PredictionPoint, ErrorResponse

# Add src directory to path for imports
SRC_DIR = Path(__file__).resolve().parent.parent.parent / 'src'
sys.path.insert(0, str(SRC_DIR))

from src.predict import load_model_from_registry, predict, validate_predictions
from src.config import TIME_COL, TARGET, LAG_CONTEXT_ROWS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


def get_model():
    """Dependency to load the ML model."""
    try:
        model = load_model_from_registry()
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(
            status_code=503,
            detail="ML model not available. Please ensure the model is registered in MLflow."
        )


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse}
    }
)
async def predict_energy_demand(
    request: PredictRequest,
    model = Depends(get_model)
) -> PredictResponse:
    """
    Predict energy demand for the next hours.

    Requires at least 336 hours of historical data for lag feature computation.
    Returns predictions starting from predict_start (or the end of historical data).
    """
    try:
        # Convert request data to DataFrame
        data_dicts = [point.dict() for point in request.data]
        df = pd.DataFrame(data_dicts)
        
        # Map API column names to original CSV format
        # The Pydantic schema uses underscores, but the backend expects the original column names
        column_mapping = {
            'generation_fossil_gas': 'generation fossil gas',
            'generation_fossil_hard_coal': 'generation fossil hard coal',
            'generation_nuclear': 'generation nuclear',
            'generation_wind_onshore': 'generation wind onshore',
            'generation_solar': 'generation solar',
            'generation_hydro_water_reservoir': 'generation hydro water reservoir',
            'generation_biomass': 'generation biomass',
            'forecast_solar_day_ahead': 'forecast solar day ahead',
            'forecast_wind_onshore_day_ahead': 'forecast wind onshore day ahead',
            'total_load_forecast': 'total load forecast',
            'total_load_actual': 'total load actual',
            'temp_min': 'temp_min',
            'temp_max': 'temp_max',
            'wind_speed': 'wind_speed',
            'wind_deg': 'wind_deg',
            'rain_1h': 'rain_1h',
            'rain_3h': 'rain_3h',
            'snow_3h': 'snow_3h',
            'clouds_all': 'clouds_all',
        }
        df = df.rename(columns=column_mapping)

        # Ensure time column is datetime
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
        df = df.sort_values(TIME_COL).reset_index(drop=True)

        # Validate we have enough historical data
        if len(df) < LAG_CONTEXT_ROWS:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient historical data. Need at least {LAG_CONTEXT_ROWS} hours, got {len(df)}."
            )

        # Make predictions
        predictions_series = predict(
            pipeline=model,
            df_with_context=df,
            predict_start=request.predict_start
        )

        # Validate predictions
        is_valid = validate_predictions(predictions_series)
        if not is_valid:
            logger.warning("Predictions failed validation checks")

        # Convert to response format
        predictions = [
            PredictionPoint(
                timestamp=ts.to_pydatetime(),
                predicted_demand_mw=float(value)
            )
            for ts, value in predictions_series.items()
        ]

        # Prepare response metadata
        model_info = {
            "type": "sklearn_pipeline",
            "features_used": len(predictions_series),
            "predictions_valid": is_valid
        }

        prediction_range = {
            "min_mw": float(predictions_series.min()),
            "max_mw": float(predictions_series.max()),
            "mean_mw": float(predictions_series.mean())
        }

        request_metadata = {
            "historical_data_points": len(df),
            "prediction_points": len(predictions),
            "predict_start": request.predict_start,
            "hours_requested": request.hours_ahead
        }

        return PredictResponse(
            predictions=predictions,
            model_info=model_info,
            prediction_range=prediction_range,
            request_metadata=request_metadata
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
