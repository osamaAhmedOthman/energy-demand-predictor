"""
config.py — Single source of truth for all project constants.

Every other script imports from here.
Changing a value here changes it everywhere.
"""

import os
from pathlib import Path

# ── Project root ───────────────────────────────────────────────────────────────
# Works regardless of where you call the script from
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DATA_RAW       = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR     = PROJECT_ROOT / 'models'
REPORTS_DIR    = PROJECT_ROOT / 'reports'

# ── Raw file names ─────────────────────────────────────────────────────────────
ENERGY_FILE  = DATA_RAW / 'energy_dataset.csv'
WEATHER_FILE = DATA_RAW / 'weather_features.csv'

# ── Processed file names ───────────────────────────────────────────────────────
TRAIN_FILE          = DATA_PROCESSED / 'train.csv'
VAL_FILE            = DATA_PROCESSED / 'val.csv'
TEST_FILE           = DATA_PROCESSED / 'test.csv'
META_FILE           = DATA_PROCESSED / 'meta.json'
FEATURE_CONFIG_FILE = DATA_PROCESSED / 'feature_config.json'
BEST_MODEL_FILE     = DATA_PROCESSED / 'best_model.txt'
BEST_PARAMS_FILE    = DATA_PROCESSED / 'best_params.json'

# ── Target ─────────────────────────────────────────────────────────────────────
TARGET   = 'total load actual'
TIME_COL = 'time'

# ── Chronological split ratios ─────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ── Columns to drop from raw energy data ──────────────────────────────────────
# Reason: either 100% null or 99.9% zero — confirmed in notebook 01
DROP_COLS = [
    'generation hydro pumped storage aggregated',  # 100% null
    'forecast wind offshore eday ahead',           # 100% null
    'generation fossil coal-derived gas',          # 99.9% zero
    'generation fossil oil shale',                 # 99.9% zero
    'generation fossil peat',                      # 99.9% zero
    'generation geothermal',                       # 99.9% zero
    'generation marine',                           # 99.9% zero
    'generation wind offshore',                    # 99.9% zero
]

# ── Weather columns to aggregate across 5 cities ──────────────────────────────
WEATHER_NUMERIC_COLS = [
    'temp', 'temp_min', 'temp_max', 'pressure',
    'humidity', 'wind_speed', 'wind_deg',
    'rain_1h', 'rain_3h', 'snow_3h', 'clouds_all',
]

# ── Feature groups ─────────────────────────────────────────────────────────────
CALENDAR_FEATURES = [
    'hour', 'dow', 'month', 'quarter',
    'is_weekend', 'season', 'is_holiday',
]

CYCLICAL_FEATURES = [
    'hour_sin', 'hour_cos',
    'dow_sin',  'dow_cos',
    'month_sin','month_cos',
    'doy_sin',  'doy_cos',
]

LAG_HOURS     = [1, 24, 48, 168, 336]
LAG_FEATURES  = [f'lag_{h}h' for h in LAG_HOURS]

ROLLING_FEATURES = [
    'rolling_mean_24h', 'rolling_mean_168h',
    'rolling_std_24h',  'rolling_max_24h', 'rolling_min_24h',
]

WEATHER_ORIG = [
    'temp', 'humidity', 'wind_speed', 'wind_deg',
    'clouds_all', 'pressure', 'rain_1h', 'snow_3h',
]

WEATHER_ENG = [
    'temp_squared', 'is_cold', 'is_hot',
    'temp_humidity', 'is_raining', 'wind_chill',
]

LOAD_FORECAST = ['total load forecast']

ALL_FEATURES = (
    CALENDAR_FEATURES
    + CYCLICAL_FEATURES
    + LAG_FEATURES
    + ROLLING_FEATURES
    + WEATHER_ORIG
    + WEATHER_ENG
    + LOAD_FORECAST
)

# ── Cross-validation ───────────────────────────────────────────────────────────
CV_N_SPLITS = 5
CV_GAP      = 24   # hours between train/val in each fold — prevents leakage

# ── Temperature thresholds (Kelvin) ───────────────────────────────────────────
TEMP_COLD_K = 280   # below 7°C  → heating demand
TEMP_HOT_K  = 298   # above 25°C → cooling demand

# ── Lag context rows needed for test set predictions ──────────────────────────
# Must be >= max(LAG_HOURS)
LAG_CONTEXT_ROWS = max(LAG_HOURS)  # 336

# ── MLflow ─────────────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT        = 'energy-demand-prediction'
MLFLOW_TRACKING_URI      = os.environ.get('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000')
REGISTERED_MODEL_NAME    = 'energy_demand_model'

# ── Optuna ─────────────────────────────────────────────────────────────────────
OPTUNA_N_TRIALS    = 30
OPTUNA_RANDOM_SEED = 42

# ── General ───────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
