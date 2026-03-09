"""
tune.py — Hyperparameter tuning with Optuna + nested MLflow runs.

Extracted from notebook 06_hyperparameter_tuning.ipynb.

Usage:
    python src/tune.py

What it does:
    1. Reads best_model.txt to know which model family to tune
    2. Defines a model-specific Optuna objective function
    3. Opens a parent MLflow run for the full study
    4. Each Optuna trial → nested child MLflow run (nested=True is REQUIRED)
    5. Objective returns CV RMSE → Optuna minimizes it
    6. Saves best params to data/processed/best_params.json

After this script, run: python src/evaluate.py

MLflow structure created:
    optuna_study_LightGBM          ← parent run
    ├── trial_000                  ← nested child run
    ├── trial_001
    ├── ...
    └── trial_059
"""

import json
import logging
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import optuna

from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from config import (
    TRAIN_FILE, FEATURE_CONFIG_FILE, BEST_MODEL_FILE, BEST_PARAMS_FILE,
    TARGET, TIME_COL,
    MLFLOW_EXPERIMENT,
    CV_N_SPLITS, CV_GAP,
    OPTUNA_N_TRIALS, OPTUNA_RANDOM_SEED,
    RANDOM_STATE,
)
from transformers import EnergyFeatureTransformer
from pipelines import build_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)s  %(message)s')
log = logging.getLogger(__name__)

# Suppress Optuna's per-trial console spam — we log to MLflow instead
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Objective functions ────────────────────────────────────────────────────────

def make_lgbm_objective(X_train, y_train, tscv):
    """
    Optuna objective for LightGBM.

    Called once per trial. Optuna's TPE sampler suggests hyperparameters,
    we CV them, log to a NESTED MLflow run, and return the CV RMSE.

    Optuna minimizes the return value.
    nested=True is REQUIRED — we are inside a parent run.
    """
    def objective(trial):
        params = {
            'n_estimators':       trial.suggest_int('n_estimators', 200, 1000),
            'max_depth':          trial.suggest_int('max_depth', 3, 10),
            'learning_rate':      trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'num_leaves':         trial.suggest_int('num_leaves', 20, 150),
            'min_child_samples':  trial.suggest_int('min_child_samples', 5, 50),
            'subsample':          trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree':   trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha':          trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda':         trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'random_state':       RANDOM_STATE,
            'verbose':            -1,
            'n_jobs':             -1,
        }
        pipeline = build_pipeline('LightGBM', params)
        cv_rmse  = _cv_rmse(pipeline, X_train, y_train, tscv)
        cv_r2    = _cv_r2(pipeline, X_train, y_train, tscv)

        # Each trial is a nested child run inside the parent study run
        with mlflow.start_run(run_name=f'trial_{trial.number:03d}', nested=True):
            mlflow.log_params(params)
            mlflow.log_metric('cv_rmse',     cv_rmse)
            mlflow.log_metric('cv_r2',       cv_r2)
            mlflow.log_metric('trial_number', trial.number)

        return cv_rmse  # Optuna minimizes this value

    return objective


def make_gbm_objective(X_train, y_train, tscv):
    """Optuna objective for sklearn GradientBoostingRegressor."""
    def objective(trial):
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 100, 500),
            'max_depth':         trial.suggest_int('max_depth', 2, 8),
            'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features':      trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state':      RANDOM_STATE,
        }
        pipeline = build_pipeline('GradientBoosting', params)
        cv_rmse  = _cv_rmse(pipeline, X_train, y_train, tscv)
        cv_r2    = _cv_r2(pipeline, X_train, y_train, tscv)

        with mlflow.start_run(run_name=f'trial_{trial.number:03d}', nested=True):
            mlflow.log_params(params)
            mlflow.log_metric('cv_rmse',      cv_rmse)
            mlflow.log_metric('cv_r2',        cv_r2)
            mlflow.log_metric('trial_number', trial.number)

        return cv_rmse

    return objective


def make_xgb_objective(X_train, y_train, tscv):
    """Optuna objective for XGBoost."""
    def objective(trial):
        params = {
            'n_estimators':    trial.suggest_int('n_estimators', 200, 1000),
            'max_depth':       trial.suggest_int('max_depth', 3, 10),
            'learning_rate':   trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'subsample':       trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree':trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha':       trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda':      trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'min_child_weight':trial.suggest_int('min_child_weight', 1, 10),
            'gamma':           trial.suggest_float('gamma', 0.0, 1.0),
            'random_state':    RANDOM_STATE,
            'verbosity':       0,
            'n_jobs':          -1,
        }
        pipeline = build_pipeline('XGBoost', params)
        cv_rmse  = _cv_rmse(pipeline, X_train, y_train, tscv)
        cv_r2    = _cv_r2(pipeline, X_train, y_train, tscv)

        with mlflow.start_run(run_name=f'trial_{trial.number:03d}', nested=True):
            mlflow.log_params(params)
            mlflow.log_metric('cv_rmse',      cv_rmse)
            mlflow.log_metric('cv_r2',        cv_r2)
            mlflow.log_metric('trial_number', trial.number)

        return cv_rmse

    return objective


def make_rf_objective(X_train, y_train, tscv):
    """Optuna objective for Random Forest."""
    def objective(trial):
        params = {
            'n_estimators':  trial.suggest_int('n_estimators', 100, 500),
            'max_depth':     trial.suggest_int('max_depth', 5, 30),
            'max_features':  trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5]),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 10),
            'random_state':  RANDOM_STATE,
            'n_jobs':        -1,
        }
        pipeline = build_pipeline('RandomForest', params)
        cv_rmse  = _cv_rmse(pipeline, X_train, y_train, tscv)
        cv_r2    = _cv_r2(pipeline, X_train, y_train, tscv)

        with mlflow.start_run(run_name=f'trial_{trial.number:03d}', nested=True):
            mlflow.log_params(params)
            mlflow.log_metric('cv_rmse',      cv_rmse)
            mlflow.log_metric('cv_r2',        cv_r2)
            mlflow.log_metric('trial_number', trial.number)

        return cv_rmse

    return objective


# ── Helpers ────────────────────────────────────────────────────────────────────

def _cv_rmse(pipeline, X, y, tscv) -> float:
    scores = cross_val_score(pipeline, X, y, cv=tscv,
                              scoring='neg_root_mean_squared_error')
    return float(-scores.mean())


def _cv_r2(pipeline, X, y, tscv) -> float:
    scores = cross_val_score(pipeline, X, y, cv=tscv, scoring='r2')
    return float(scores.mean())


OBJECTIVE_MAP = {
    'LightGBM':        make_lgbm_objective,
    'GradientBoosting':make_gbm_objective,
    'XGBoost':         make_xgb_objective,
    'RandomForest':    make_rf_objective,
}


def get_objective(model_name: str, X_train, y_train, tscv):
    if model_name not in OBJECTIVE_MAP:
        raise ValueError(
            f'No objective defined for {model_name}. '
            f'Available: {list(OBJECTIVE_MAP.keys())}'
        )
    return OBJECTIVE_MAP[model_name](X_train, y_train, tscv)


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    log.info('=' * 60)
    log.info('TUNE — Optuna hyperparameter search starting')
    log.info('=' * 60)

    # Load data
    train_raw = pd.read_csv(TRAIN_FILE)
    train_raw[TIME_COL] = pd.to_datetime(train_raw[TIME_COL], utc=True)

    with open(FEATURE_CONFIG_FILE) as f:
        config = json.load(f)

    with open(BEST_MODEL_FILE) as f:
        model_name = f.read().strip()

    # Apply feature engineering
    fe       = EnergyFeatureTransformer()
    train_fe = fe.fit_transform(train_raw)
    X_train  = train_fe[config['all_features']]
    y_train  = train_fe[TARGET]

    log.info(f'Tuning model: {model_name}')
    log.info(f'X_train: {X_train.shape}  |  Trials: {OPTUNA_N_TRIALS}')

    tscv      = TimeSeriesSplit(n_splits=CV_N_SPLITS, gap=CV_GAP)
    objective = get_objective(model_name, X_train, y_train, tscv)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # ── Parent run wraps the entire study ─────────────────────────────────────
    # Each trial opens a NESTED child run with nested=True
    with mlflow.start_run(run_name=f'optuna_study_{model_name}'):

        mlflow.log_param('model_type',  model_name)
        mlflow.log_param('n_trials',    OPTUNA_N_TRIALS)
        mlflow.log_param('cv_strategy', f'TimeSeriesSplit n={CV_N_SPLITS} gap={CV_GAP}')
        mlflow.log_param('sampler',     'TPE')
        mlflow.log_param('direction',   'minimize cv_rmse')
        mlflow.log_param('n_features',  X_train.shape[1])

        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=OPTUNA_RANDOM_SEED),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        )

        log.info(f'Starting study: {OPTUNA_N_TRIALS} trials...')
        study.optimize(objective, n_trials=OPTUNA_N_TRIALS, show_progress_bar=True)

        best_params = study.best_params
        best_rmse   = study.best_value

        mlflow.log_metric('best_cv_rmse',   best_rmse)
        mlflow.log_metric('best_trial_num', study.best_trial.number)
        mlflow.log_params({f'best_{k}': v for k, v in best_params.items()})

    log.info(f'\n✅ Study complete!')
    log.info(f'   Best trial:   #{study.best_trial.number}')
    log.info(f'   Best CV RMSE: {best_rmse:,.2f} MW')
    log.info(f'   Best params:')
    for k, v in best_params.items():
        log.info(f'     {k}: {v}')

    # ── Save best params ───────────────────────────────────────────────────────
    # Add fixed params that Optuna doesn't tune
    best_params_full = {**best_params, 'random_state': RANDOM_STATE}
    if model_name == 'LightGBM':
        best_params_full.update({'verbose': -1, 'n_jobs': -1})
    elif model_name in ('XGBoost', 'RandomForest'):
        best_params_full.update({'n_jobs': -1})

    with open(BEST_PARAMS_FILE, 'w') as f:
        json.dump(best_params_full, f, indent=2)

    log.info(f'\n✅ Best params saved → {BEST_PARAMS_FILE}')
    log.info('TUNE — done ✅')
    log.info('Next step: python src/evaluate.py')


if __name__ == '__main__':
    run()
