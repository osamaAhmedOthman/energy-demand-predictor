"""
train.py — Train and compare all candidate models. Log each to MLflow.

Extracted from notebooks 04_baseline_model and 05_model_comparison.

Usage:
    python src/train.py

What it does:
    1. Loads train.csv + feature_config.json from data/processed/
    2. Applies EnergyFeatureTransformer to build X_train, y_train
    3. For each model in the registry:
        a. Runs TimeSeriesSplit CV (n=5, gap=24h)
        b. Logs CV metrics + trained pipeline to MLflow
    4. Picks winner by lowest CV RMSE
    5. Saves winner name to data/processed/best_model.txt

After this script, run: python src/tune.py
"""

import json
import time
import logging
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

from config import (
    TRAIN_FILE, FEATURE_CONFIG_FILE, BEST_MODEL_FILE,
    TARGET, TIME_COL,
    MLFLOW_EXPERIMENT,
    CV_N_SPLITS, CV_GAP,
    RANDOM_STATE,
)
from transformers import EnergyFeatureTransformer
from pipelines import get_default_models, build_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)s  %(message)s')
log = logging.getLogger(__name__)


def load_train_data():
    """Load raw train CSV and feature config."""
    train_raw = pd.read_csv(TRAIN_FILE)
    train_raw[TIME_COL] = pd.to_datetime(train_raw[TIME_COL], utc=True)

    with open(FEATURE_CONFIG_FILE) as f:
        config = json.load(f)

    return train_raw, config


def build_features(train_raw: pd.DataFrame, config: dict):
    """Apply feature engineering to get X_train, y_train."""
    fe        = EnergyFeatureTransformer()
    train_fe  = fe.fit_transform(train_raw)
    X_train   = train_fe[config['all_features']]
    y_train   = train_fe[TARGET]
    log.info(f'X_train: {X_train.shape}  |  y_train mean: {y_train.mean():,.0f} MW')
    return X_train, y_train


def run_naive_baseline(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Log the naive baseline: predict last week's same-hour demand.
    This is the floor — any ML model must beat this RMSE.
    """
    log.info('Running naive baseline (lag_168h)...')
    y_pred = X_train['lag_168h']
    mask   = y_pred.notna()

    rmse = np.sqrt(mean_squared_error(y_train[mask], y_pred[mask]))
    r2   = r2_score(y_train[mask], y_pred[mask])

    with mlflow.start_run(run_name='naive_baseline_lag168h'):
        mlflow.log_param('model_type', 'Naive')
        mlflow.log_param('strategy',   'lag_168h — last week same hour')
        mlflow.log_metric('train_rmse', rmse)
        mlflow.log_metric('train_r2',   r2)

    log.info(f'  Naive RMSE: {rmse:,.2f} MW  |  R²: {r2:.4f}  ← floor to beat')
    return rmse


def train_one_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tscv: TimeSeriesSplit,
) -> dict:
    """
    Train one model, run TimeSeriesSplit CV, log everything to MLflow.

    Returns a metrics dict for the comparison table.
    """
    log.info(f'Training {model_name}...')
    t0       = time.time()
    pipeline = build_pipeline(model_name)

    # ── Cross-validation ───────────────────────────────────────────────────────
    # TimeSeriesSplit with gap=24 prevents adjacent-hour leakage between
    # the training window and the validation window in each fold.
    # NEVER use KFold on time-series data.
    cv_rmse = -cross_val_score(pipeline, X_train, y_train, cv=tscv,
                                scoring='neg_root_mean_squared_error')
    cv_mae  = -cross_val_score(pipeline, X_train, y_train, cv=tscv,
                                scoring='neg_mean_absolute_error')
    cv_r2   =  cross_val_score(pipeline, X_train, y_train, cv=tscv,
                                scoring='r2')

    # ── Train metrics (for overfit gap) ───────────────────────────────────────
    pipeline.fit(X_train, y_train)
    y_pred_train = pipeline.predict(X_train)
    train_rmse   = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_r2     = r2_score(y_train, y_pred_train)
    overfit_gap  = train_r2 - cv_r2.mean()
    elapsed      = time.time() - t0

    # ── Log to MLflow ──────────────────────────────────────────────────────────
    with mlflow.start_run(run_name=f'compare_{model_name}'):
        mlflow.log_param('model_type',       model_name)
        mlflow.log_param('default_params',   True)
        mlflow.log_param('n_features',       X_train.shape[1])
        mlflow.log_param('cv_strategy',      f'TimeSeriesSplit n={CV_N_SPLITS} gap={CV_GAP}')
        mlflow.log_metric('cv_rmse_mean',    cv_rmse.mean())
        mlflow.log_metric('cv_rmse_std',     cv_rmse.std())
        mlflow.log_metric('cv_mae_mean',     cv_mae.mean())
        mlflow.log_metric('cv_r2_mean',      cv_r2.mean())
        mlflow.log_metric('cv_r2_std',       cv_r2.std())
        mlflow.log_metric('train_rmse',      train_rmse)
        mlflow.log_metric('train_r2',        train_r2)
        mlflow.log_metric('overfit_gap_r2',  overfit_gap)
        mlflow.log_metric('training_time_s', elapsed)
        mlflow.sklearn.log_model(pipeline, artifact_path='model')

    log.info(
        f'  CV RMSE: {cv_rmse.mean():>8,.2f} ± {cv_rmse.std():.2f} MW  |'
        f'  CV R²: {cv_r2.mean():.4f}  |  Gap: {overfit_gap:.4f}  |  {elapsed:.1f}s  ✅'
    )

    return {
        'Model':       model_name,
        'CV RMSE':     round(cv_rmse.mean(), 2),
        'CV RMSE std': round(cv_rmse.std(), 2),
        'CV R²':       round(cv_r2.mean(), 4),
        'Train R²':    round(train_r2, 4),
        'Overfit Gap': round(overfit_gap, 4),
        'Time (s)':    round(elapsed, 1),
    }


def pick_winner(results: list[dict]) -> str:
    """
    Pick the best model by lowest CV RMSE.
    Ties are broken by overfit gap (lower is better).
    """
    results_df = (
        pd.DataFrame(results)
        .sort_values(['CV RMSE', 'Overfit Gap'])
        .reset_index(drop=True)
    )

    log.info('\nMODEL COMPARISON (sorted by CV RMSE):')
    log.info(results_df.to_string(index=False))

    winner = results_df.iloc[0]['Model']
    log.info(f'\n🏆 Winner: {winner}')
    log.info(f'   CV RMSE: {results_df.iloc[0]["CV RMSE"]:,.2f} MW  |  CV R²: {results_df.iloc[0]["CV R²"]:.4f}')
    return winner


def save_best_model_name(model_name: str):
    with open(BEST_MODEL_FILE, 'w') as f:
        f.write(model_name)
    log.info(f'✅ Saved best model name → {BEST_MODEL_FILE}')


def run():
    log.info('=' * 60)
    log.info('TRAIN — model comparison starting')
    log.info('=' * 60)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    tscv = TimeSeriesSplit(n_splits=CV_N_SPLITS, gap=CV_GAP)

    train_raw, config = load_train_data()
    X_train, y_train  = build_features(train_raw, config)

    # Naive baseline first — establishes the performance floor
    naive_rmse = run_naive_baseline(X_train, y_train)
    log.info(f'Naive baseline RMSE: {naive_rmse:,.2f} MW  ← models must beat this\n')

    # Train all models
    results = []
    for model_name in get_default_models():
        result = train_one_model(model_name, X_train, y_train, tscv)
        results.append(result)

    # Pick and save winner
    winner = pick_winner(results)
    save_best_model_name(winner)

    log.info('\nTRAIN — done ✅')
    log.info(f'Next step: python src/tune.py')


if __name__ == '__main__':
    run()
