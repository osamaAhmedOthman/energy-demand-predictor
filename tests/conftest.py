"""
conftest.py — Shared pytest fixtures for all test files.

pytest finds this file automatically and makes every fixture here
available to every test in the tests/ directory — no imports needed.

Design rule: ALL fixtures use synthetic data built in memory.
Tests must NEVER read from data/raw/ or data/processed/.
This means tests run instantly, anywhere, without the real CSVs.
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# ── Make src/ importable from tests/ ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Shared constants ───────────────────────────────────────────────────────────
N_ROWS   = 600      # must be > max lag (336) + rolling window (168) + buffer
TARGET   = 'total load actual'
TIME_COL = 'time'


# ── Timestamp fixture ──────────────────────────────────────────────────────────

@pytest.fixture
def sample_timestamps():
    """600 consecutive hourly UTC timestamps starting 2016-01-01."""
    return pd.date_range(
        start='2016-01-01',
        periods=N_ROWS,
        freq='h',
        tz='UTC',
    )


# ── Raw data fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def raw_energy_df(sample_timestamps):
    """
    Synthetic raw energy DataFrame — mirrors structure of energy_dataset.csv.

    Includes the useless columns (null / zero) so data_loader tests
    can verify they get dropped correctly.
    """
    rng = np.random.default_rng(42)
    n   = len(sample_timestamps)

    hour_idx    = np.arange(n) % 24
    hour_effect = 3_000 * np.sin(2 * np.pi * hour_idx / 24)
    demand      = np.clip(28_000 + hour_effect + rng.normal(0, 300, n), 18_000, 42_000)

    return pd.DataFrame({
        TIME_COL:                                      sample_timestamps,
        'generation fossil gas':                       rng.uniform(2000, 8000, n),
        'generation fossil hard coal':                 rng.uniform(500, 3000, n),
        'generation nuclear':                          rng.uniform(5000, 8000, n),
        'generation wind onshore':                     rng.uniform(500, 5000, n),
        'generation solar':                            rng.uniform(0, 2000, n),
        'generation hydro water reservoir':            rng.uniform(500, 2000, n),
        'generation biomass':                          rng.uniform(200, 600, n),
        'forecast solar day ahead':                    rng.uniform(0, 2000, n),
        'forecast wind onshore day ahead':             rng.uniform(500, 5000, n),
        'total load forecast':                         demand + rng.normal(0, 100, n),
        TARGET:                                        demand,
        # Must be DROPPED — 100% null
        'generation hydro pumped storage aggregated':  np.full(n, np.nan),
        'forecast wind offshore eday ahead':           np.full(n, np.nan),
        # Must be DROPPED — 99.9% zero
        'generation fossil coal-derived gas':          np.zeros(n),
        'generation fossil oil shale':                 np.zeros(n),
        'generation fossil peat':                      np.zeros(n),
        'generation geothermal':                       np.zeros(n),
        'generation marine':                           np.zeros(n),
        'generation wind offshore':                    np.zeros(n),
    })


@pytest.fixture
def raw_weather_df(sample_timestamps):
    """
    Synthetic raw weather DataFrame — mirrors structure of weather_features.csv.
    5 cities x N_ROWS timestamps.
    """
    rng    = np.random.default_rng(42)
    cities = ['Madrid', 'Barcelona', 'Seville', 'Valencia', 'Bilbao']
    rows   = []

    for city in cities:
        base_temp = 285 + rng.normal(0, 1)
        for ts in sample_timestamps:
            rows.append({
                'dt_iso':     ts,
                'city_name':  city,
                'temp':       base_temp + rng.normal(0, 3),
                'temp_min':   base_temp - 2,
                'temp_max':   base_temp + 2,
                'pressure':   rng.uniform(1010, 1020),
                'humidity':   rng.uniform(40, 80),
                'wind_speed': rng.uniform(1, 10),
                'wind_deg':   rng.uniform(0, 360),
                'rain_1h':    rng.exponential(0.5),
                'rain_3h':    rng.exponential(1.0),
                'snow_3h':    0.0,
                'clouds_all': rng.uniform(0, 100),
            })

    return pd.DataFrame(rows)


# ── Processed data fixtures ────────────────────────────────────────────────────

@pytest.fixture
def merged_df(raw_energy_df, raw_weather_df):
    """
    Merged + cleaned DataFrame — what train/val/test CSVs look like.
    Built from synthetic fixtures, not from real files.
    """
    weather_numeric = [
        'temp', 'temp_min', 'temp_max', 'pressure', 'humidity',
        'wind_speed', 'wind_deg', 'rain_1h', 'rain_3h', 'snow_3h', 'clouds_all',
    ]
    weather_agg = (
        raw_weather_df
        .groupby('dt_iso')[weather_numeric]
        .mean()
        .reset_index()
    )

    drop_cols = [
        'generation hydro pumped storage aggregated',
        'forecast wind offshore eday ahead',
        'generation fossil coal-derived gas',
        'generation fossil oil shale',
        'generation fossil peat',
        'generation geothermal',
        'generation marine',
        'generation wind offshore',
    ]
    energy_clean = raw_energy_df.drop(columns=drop_cols)

    merged = energy_clean.merge(
        weather_agg, left_on=TIME_COL, right_on='dt_iso', how='inner'
    ).drop(columns=['dt_iso'])

    return merged.sort_values(TIME_COL).reset_index(drop=True)


@pytest.fixture
def transformed_df(merged_df):
    """
    Feature-engineered DataFrame — output of EnergyFeatureTransformer.
    Used by model and prediction tests.
    """
    from src.transformers import EnergyFeatureTransformer
    fe = EnergyFeatureTransformer()
    return fe.fit_transform(merged_df)


@pytest.fixture
def fitted_ridge_pipeline(transformed_df):
    """Ridge pipeline fitted on synthetic data. Used by prediction tests."""
    from src.config import ALL_FEATURES
    from src.pipelines import build_pipeline

    X = transformed_df[ALL_FEATURES]
    y = transformed_df[TARGET]

    pipeline = build_pipeline('Ridge')
    pipeline.fit(X, y)
    return pipeline
