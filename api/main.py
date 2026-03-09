"""
main.py — FastAPI application for Energy Demand Prediction API.

Provides endpoints for:
- Health check (/health)
- Energy demand prediction (/predict)

The API serves a machine learning model that predicts Spanish electricity demand
based on historical energy generation data and weather features.
"""

import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routers.predict import router as predict_router
from .schemas import HealthResponse, ErrorResponse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for app state
app_start_time = time.time()
model_loaded = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    global model_loaded

    # Startup
    logger.info("Starting Energy Demand Prediction API")
    try:
        # Try to load the model to check if it's available
        from src.predict import load_model_from_registry
        load_model_from_registry()
        model_loaded = True
        logger.info("ML model loaded successfully")
    except Exception as e:
        logger.warning(f"ML model not available at startup: {e}")
        model_loaded = False

    yield

    # Shutdown
    logger.info("Shutting down Energy Demand Prediction API")


# Create FastAPI app
app = FastAPI(
    title="Energy Demand Prediction API",
    description="""
    Machine Learning API for predicting Spanish electricity demand.

    This API provides real-time predictions of energy demand based on:
    - Historical energy generation data (nuclear, solar, wind, etc.)
    - Weather conditions (temperature, humidity, wind, etc.)
    - Temporal features (hour, day of week, season, etc.)

    **Features:**
    - Predict demand for the next 1-168 hours
    - Requires 336+ hours of historical context data
    - Validates predictions against physical constraints
    - Returns predictions with confidence metadata
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the API and ML model are healthy and ready to serve predictions."
)
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - app_start_time

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        version="1.0.0",
        model_loaded=model_loaded,
        uptime_seconds=uptime
    )


# Include routers
app.include_router(
    predict_router,
    prefix="/api/v1",
    tags=["predictions"]
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Energy Demand Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
