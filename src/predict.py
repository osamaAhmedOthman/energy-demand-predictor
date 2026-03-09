"""
predict.py — Load the registered MLflow model and predict on new data.

This is the inference script used in production / after evaluate.py.

Usage:
    # From another script or API:
    from predict import load_model, predict

    model = load_model()
    df    = pd.read_csv('new_data.csv')
    preds = predict(model, df)

    # Or run standalone:
    python src/predict.py

Input format:
    A DataFrame with the same raw columns as data/processed/train.csv
    (energy + weather features, with a 'time' column).
    The caller must also provide at least LAG_CONTEXT_ROWS (336) rows
    of HISTORICAL data BEFORE the rows to be predicted, so that
    lag features can be computed correctly.

Output:
    pd.Series of predicted energy demand in MW, indexed by timestamp.
"""

import json
import logging
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from config import (
    FEATURE_CONFIG_FILE, BEST_MODEL_FILE, BEST_PARAMS_FILE,
    TARGET, TIME_COL,
    MLFLOW_EXPERIMENT,
    REGISTERED_MODEL_NAME,
    LAG_CONTEXT_ROWS,
)
from transformers import EnergyFeatureTransformer
from pipelines import build_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)s  %(message)s')
log = logging.getLogger(__name__)


def load_model_from_registry(model_name: str = None, stage: str = 'Production'):
    """
    Load the registered model from MLflow Model Registry.

    Parameters
    ----------
    model_name : str, optional
        Registered model name. Defaults to REGISTERED_MODEL_NAME from config.
    stage : str
        Model stage to load: 'Production', 'Staging', 'None'.

    Returns
    -------
    Loaded sklearn pipeline (mlflow.pyfunc.PythonModelContext or sklearn model)
    """
    model_name = model_name or REGISTERED_MODEL_NAME
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    model_uri = f'models:/{model_name}/{stage}'
    log.info(f'Loading model from registry: {model_uri}')
    model = mlflow.sklearn.load_model(model_uri)
    log.info('Model loaded ✅')
    return model


def load_model_from_params() -> object:
    """
    Rebuild and reload the model using saved best_params.json.
    Fallback when the MLflow registry is not available.

    This rebuilds the exact pipeline that evaluate.py trained.
    The model weights are NOT preserved — use this only to retrain
    from scratch, not to load a previously trained model.
    """
    log.info('Loading model from best_params.json (rebuild from scratch)')

    with open(BEST_MODEL_FILE) as f:
        model_name = f.read().strip()

    with open(BEST_PARAMS_FILE) as f:
        best_params = json.load(f)

    pipeline = build_pipeline(model_name, best_params)
    log.info(f'Pipeline built: {model_name}')
    return pipeline


def predict(
    pipeline,
    df_with_context: pd.DataFrame,
    predict_start: str = None,
) -> pd.Series:
    """
    Generate predictions for new data.

    Parameters
    ----------
    pipeline : sklearn Pipeline
        The trained pipeline (from load_model_from_registry or fit()).

    df_with_context : pd.DataFrame
        DataFrame that includes:
        - The rows to predict (the actual forecast period)
        - At least LAG_CONTEXT_ROWS (336) rows of HISTORICAL data
          BEFORE the forecast period so lag features can be computed.

        Must include all raw columns from train.csv (energy + weather).

    predict_start : str, optional
        If provided, filter predictions to only rows >= this timestamp.
        Format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'.

    Returns
    -------
    pd.Series
        Predicted energy demand in MW, indexed by 'time'.

    Example
    -------
    # Load last 336 hours of history + next 24 hours to forecast
    history  = pd.read_csv('last_336_hours.csv')
    future   = pd.read_csv('next_24_hours.csv')
    combined = pd.concat([history, future]).sort_values('time')

    model    = load_model_from_registry()
    preds    = predict(model, combined, predict_start='2019-01-01')
    """
    with open(FEATURE_CONFIG_FILE) as f:
        config = json.load(f)

    fe = EnergyFeatureTransformer()
    df_transformed = fe.transform(df_with_context)

    # Filter to only the requested prediction period
    if predict_start is not None:
        start_ts = pd.Timestamp(predict_start)
        # Ensure both timestamps have the same timezone awareness
        if df_transformed[TIME_COL].dt.tz is not None:
            # Data is tz-aware, make start_ts tz-aware too
            if start_ts.tz is None:
                start_ts = start_ts.tz_localize('UTC')
        else:
            # Data is tz-naive, make start_ts tz-naive too
            if start_ts.tz is not None:
                start_ts = start_ts.tz_convert('UTC').tz_localize(None)
        df_transformed = df_transformed[df_transformed[TIME_COL] >= start_ts]

    if df_transformed.empty:
        log.warning('No rows to predict after filtering. Check predict_start and input data.')
        return pd.Series(dtype=float)

    X = df_transformed[config['all_features']]
    y_pred = pipeline.predict(X)

    predictions = pd.Series(
        y_pred,
        index=df_transformed[TIME_COL].values,
        name='predicted_demand_mw',
    )

    log.info(f'Predictions: {len(predictions):,} rows  |  '
             f'mean: {predictions.mean():,.0f} MW  |  '
             f'range: [{predictions.min():,.0f}, {predictions.max():,.0f}]')

    return predictions


def validate_predictions(predictions: pd.Series) -> bool:
    """
    Sanity check predictions against known reasonable bounds
    for Spanish electricity demand.

    Spanish hourly demand historically ranges ~18,000–42,000 MW.
    Returns True if all predictions are within bounds.
    """
    MIN_REASONABLE_MW = 10_000
    MAX_REASONABLE_MW = 55_000

    n_below = (predictions < MIN_REASONABLE_MW).sum()
    n_above = (predictions > MAX_REASONABLE_MW).sum()

    if n_below > 0:
        log.warning(f'{n_below} predictions below {MIN_REASONABLE_MW:,} MW — check input data')
    if n_above > 0:
        log.warning(f'{n_above} predictions above {MAX_REASONABLE_MW:,} MW — check input data')

    is_valid = (n_below == 0) and (n_above == 0)
    if is_valid:
        log.info('✅ All predictions within reasonable range')
    return bool(is_valid)


# ── Standalone demo: predict on test set ──────────────────────────────────────

def run():
    """
    Demo: load model, predict on test set, print summary.
    Mirrors what evaluate.py does but without re-running the full eval.
    """
    log.info('PREDICT — demo run on test set')

    from config import TRAIN_FILE, VAL_FILE, TEST_FILE
    import pandas as pd

    train_raw = pd.read_csv(TRAIN_FILE)
    val_raw   = pd.read_csv(VAL_FILE)
    test_raw  = pd.read_csv(TEST_FILE)

    for df in [train_raw, val_raw, test_raw]:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True)

    # Combine train+val for context, then append test
    trainval = pd.concat([train_raw, val_raw]).sort_values(TIME_COL).reset_index(drop=True)
    context  = trainval.tail(LAG_CONTEXT_ROWS)
    combined = pd.concat([context, test_raw]).sort_values(TIME_COL).reset_index(drop=True)

    # Try loading from registry; fall back to rebuilding + fitting
    try:
        pipeline = load_model_from_registry()
    except Exception as e:
        log.warning(f'Registry load failed ({e}). Rebuilding and refitting model...')
        pipeline = load_model_from_params()

        # Retrain (since we don't have saved weights outside MLflow)
        with open(FEATURE_CONFIG_FILE) as f:
            config = json.load(f)
        fe          = EnergyFeatureTransformer()
        trainval_fe = fe.fit_transform(trainval)
        pipeline.fit(trainval_fe[config['all_features']], trainval_fe[TARGET])

    # Predict
    test_start  = str(test_raw[TIME_COL].min())
    predictions = predict(pipeline, combined, predict_start=test_start)
    validate_predictions(predictions)

    print('\nFirst 5 predictions:')
    print(predictions.head().to_string())
    log.info('PREDICT — done ✅')


if __name__ == '__main__':
    run()
