"""
evaluate.py — Final one-time evaluation on the locked test set.

Extracted from notebook 07_final_evaluation.ipynb.

Usage:
    python src/evaluate.py

⚠️  Run this script ONCE and ONLY ONCE.
    Looking at test results and then re-tuning is test set leakage.
    These are your real, honest numbers.

What it does:
    1. Loads train + val + test splits
    2. Retrains best model (best_model.txt + best_params.json) on train+val combined
       More training data → stronger final model for deployment
    3. Prepends lag context rows to test set so lag features have valid values
    4. Predicts on test set
    5. Computes full metrics: RMSE, MAE, R², MAPE, peak-hour RMSE, weekday/weekend split
    6. Saves diagnostic plots to reports/
    7. Logs everything to MLflow: params, metrics, plots, model artifact
    8. Runs mlflow.evaluate() for automatic residual plots + extra metrics
"""

import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mlflow
import mlflow.sklearn

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import (
    TRAIN_FILE, VAL_FILE, TEST_FILE,
    FEATURE_CONFIG_FILE, BEST_MODEL_FILE, BEST_PARAMS_FILE,
    REPORTS_DIR,
    TARGET, TIME_COL,
    MLFLOW_EXPERIMENT,
    REGISTERED_MODEL_NAME,
    LAG_CONTEXT_ROWS,
)
from transformers import EnergyFeatureTransformer
from pipelines import build_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)s  %(message)s')
log = logging.getLogger(__name__)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_all_splits():
    """Load train, val, test and all config files."""
    train_raw = pd.read_csv(TRAIN_FILE)
    val_raw   = pd.read_csv(VAL_FILE)
    test_raw  = pd.read_csv(TEST_FILE)

    for df in [train_raw, val_raw, test_raw]:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True)

    with open(FEATURE_CONFIG_FILE) as f:
        config = json.load(f)

    with open(BEST_MODEL_FILE) as f:
        model_name = f.read().strip()

    with open(BEST_PARAMS_FILE) as f:
        best_params = json.load(f)

    log.info(f'Train: {len(train_raw):,}  Val: {len(val_raw):,}  Test: {len(test_raw):,}')
    log.info(f'Model: {model_name}')
    return train_raw, val_raw, test_raw, config, model_name, best_params


# ── Training ───────────────────────────────────────────────────────────────────

def retrain_on_trainval(
    train_raw: pd.DataFrame,
    val_raw: pd.DataFrame,
    config: dict,
    model_name: str,
    best_params: dict,
):
    """
    Retrain the best model on train + val combined.

    Why combine train and val?
    During model selection (notebook 05) we used only train.
    During tuning (notebook 06) we used only train.
    Now that all decisions are frozen, we can use more data.
    More training data → better model for deployment.

    The test set is still untouched at this point.
    """
    trainval_raw = (
        pd.concat([train_raw, val_raw])
        .sort_values(TIME_COL)
        .reset_index(drop=True)
    )

    fe          = EnergyFeatureTransformer()
    trainval_fe = fe.fit_transform(trainval_raw)
    X_trainval  = trainval_fe[config['all_features']]
    y_trainval  = trainval_fe[TARGET]

    pipeline = build_pipeline(model_name, best_params)
    pipeline.fit(X_trainval, y_trainval)

    log.info(f'Retrained on {len(X_trainval):,} rows (train + val)')
    return pipeline, fe


# ── Test set inference ─────────────────────────────────────────────────────────

def predict_test(
    pipeline,
    fe: EnergyFeatureTransformer,
    train_raw: pd.DataFrame,
    val_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    config: dict,
):
    """
    Generate predictions for the test set.

    The lag context trick:
        lag_336h at the first test row needs demand from 336 hours before.
        Those 336 hours are at the END of the train+val period.
        We prepend them so the transformer can compute valid lag values.
        Then we filter back to only the actual test rows.

    Without this, the first 336 test predictions would use NaN lags,
    giving incorrect (or dropped) rows.
    """
    trainval_raw = (
        pd.concat([train_raw, val_raw])
        .sort_values(TIME_COL)
        .reset_index(drop=True)
    )

    # Prepend lag context rows from end of train+val
    context   = trainval_raw.tail(LAG_CONTEXT_ROWS)
    test_with_context = (
        pd.concat([context, test_raw])
        .sort_values(TIME_COL)
        .reset_index(drop=True)
    )

    test_fe    = fe.transform(test_with_context)

    # Filter back to only actual test period
    test_start = test_raw[TIME_COL].min()
    test_fe    = test_fe[test_fe[TIME_COL] >= test_start].reset_index(drop=True)

    X_test = test_fe[config['all_features']]
    y_test = test_fe[TARGET]
    y_pred = pipeline.predict(X_test)

    log.info(f'Test predictions: {len(y_pred):,} rows')
    return test_fe, X_test, y_test, y_pred


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_all_metrics(y_test: pd.Series, y_pred: np.ndarray, test_fe: pd.DataFrame) -> dict:
    """Compute comprehensive metrics including segment-level breakdowns."""
    rmse      = np.sqrt(mean_squared_error(y_test, y_pred))
    mae       = mean_absolute_error(y_test, y_pred)
    r2        = r2_score(y_test, y_pred)
    mape      = np.mean(np.abs((y_test.values - y_pred) / y_test.values)) * 100
    rmse_pct  = rmse / y_test.mean() * 100

    # Peak hours: morning (8-11) and evening (19-21) — grid operator cares most
    peak_mask = test_fe['hour'].isin([8, 9, 10, 11, 19, 20, 21])
    peak_rmse = np.sqrt(mean_squared_error(y_test[peak_mask], y_pred[peak_mask]))

    # Weekday vs weekend
    wd_mask  = test_fe['is_weekend'] == 0
    we_mask  = test_fe['is_weekend'] == 1
    wd_rmse  = np.sqrt(mean_squared_error(y_test[wd_mask], y_pred[wd_mask]))
    we_rmse  = np.sqrt(mean_squared_error(y_test[we_mask], y_pred[we_mask]))

    metrics = {
        'test_rmse':     rmse,
        'test_rmse_pct': rmse_pct,
        'test_mae':      mae,
        'test_r2':       r2,
        'test_mape':     mape,
        'peak_rmse':     peak_rmse,
        'weekday_rmse':  wd_rmse,
        'weekend_rmse':  we_rmse,
        'target_mean':   y_test.mean(),
        'target_std':    y_test.std(),
    }

    log.info('\nFINAL TEST SET METRICS:')
    log.info('=' * 50)
    log.info(f'  RMSE:             {rmse:>10,.2f} MW')
    log.info(f'  RMSE (% of mean): {rmse_pct:>10.2f} %')
    log.info(f'  MAE:              {mae:>10,.2f} MW')
    log.info(f'  R²:               {r2:>10.4f}')
    log.info(f'  MAPE:             {mape:>10.2f} %')
    log.info(f'  Peak hour RMSE:   {peak_rmse:>10,.2f} MW')
    log.info(f'  Weekday RMSE:     {wd_rmse:>10,.2f} MW')
    log.info(f'  Weekend RMSE:     {we_rmse:>10,.2f} MW')

    return metrics


# ── Plots ──────────────────────────────────────────────────────────────────────

def save_evaluation_plots(
    y_test: pd.Series,
    y_pred: np.ndarray,
    test_fe: pd.DataFrame,
    metrics: dict,
    model_name: str,
) -> dict:
    """
    Create and save 4 diagnostic plots:
    1. Full test period actual vs predicted
    2. Scatter actual vs predicted (should be tight around diagonal)
    3. Residuals over time (should be random, no pattern)
    4. Mean residual by hour of day (should be ~0 everywhere)
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    residuals = y_test.values - y_pred
    rmse      = metrics['test_rmse']
    r2        = metrics['test_r2']

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # ── 1. Actual vs predicted time series ──
    axes[0, 0].plot(test_fe[TIME_COL], y_test.values,
                    color='steelblue', lw=0.6, alpha=0.9, label='Actual')
    axes[0, 0].plot(test_fe[TIME_COL], y_pred,
                    color='tomato',    lw=0.6, alpha=0.7, label='Predicted')
    axes[0, 0].set_title(f'Actual vs Predicted — Full Test Period  |  RMSE: {rmse:,.0f} MW')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('MW')
    axes[0, 0].legend()
    axes[0, 0].xaxis.set_major_locator(mdates.MonthLocator())
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=20)

    # ── 2. Scatter ──
    axes[0, 1].scatter(y_test.values, y_pred, alpha=0.3, s=5, color='steelblue')
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    axes[0, 1].plot(lims, lims, 'r--', lw=2, label='Perfect prediction')
    axes[0, 1].set_title(f'Scatter: Actual vs Predicted  |  R² = {r2:.4f}')
    axes[0, 1].set_xlabel('Actual MW')
    axes[0, 1].set_ylabel('Predicted MW')
    axes[0, 1].legend()

    # ── 3. Residuals over time ──
    axes[1, 0].plot(test_fe[TIME_COL], residuals, color='tomato', lw=0.5, alpha=0.7)
    axes[1, 0].axhline(y=0,     color='black',  ls='--', lw=1)
    axes[1, 0].axhline(y=rmse,  color='orange', ls=':',  lw=1, label=f'+RMSE ({rmse:,.0f})')
    axes[1, 0].axhline(y=-rmse, color='orange', ls=':',  lw=1, label=f'-RMSE')
    axes[1, 0].set_title('Residuals Over Time  |  Random scatter = good model')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Residual (MW)')
    axes[1, 0].legend()
    axes[1, 0].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=20)

    # ── 4. Mean residual by hour ──
    test_fe_copy = test_fe.copy()
    test_fe_copy['residual'] = residuals
    hourly = test_fe_copy.groupby('hour')['residual'].agg(['mean', 'std'])
    axes[1, 1].bar(hourly.index, hourly['mean'], color='steelblue', edgecolor='white')
    axes[1, 1].errorbar(hourly.index, hourly['mean'], yerr=hourly['std'],
                        fmt='none', color='black', capsize=3)
    axes[1, 1].axhline(y=0, color='red', ls='--', lw=1)
    axes[1, 1].set_title('Mean Residual by Hour  |  Should be ~0 everywhere')
    axes[1, 1].set_xlabel('Hour of Day')
    axes[1, 1].set_ylabel('Mean Residual (MW)')

    plt.suptitle(
        f'Final Evaluation — {model_name} (Tuned)  |  RMSE={rmse:,.0f} MW  R²={r2:.4f}',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()

    eval_plot_path = REPORTS_DIR / 'final_eval_plots.png'
    plt.savefig(eval_plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    log.info(f'Saved: {eval_plot_path}')

    # ── Feature importance plot ──
    fi_plot_path = None
    model_step   = pipeline_ref.named_steps['model']
    if hasattr(model_step, 'feature_importances_'):
        importances = pd.Series(
            model_step.feature_importances_,
            index=all_features_ref,
        ).sort_values(ascending=True).tail(20)

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = [
            'seagreen'   if any(x in f for x in ['lag', 'rolling']) else
            'steelblue'  if any(x in f for x in ['temp', 'humid', 'wind', 'rain', 'cloud']) else
            'gold'
            for f in importances.index
        ]
        importances.plot(kind='barh', ax=ax, color=colors)
        ax.set_title(
            f'Top 20 Feature Importances — {model_name}\n'
            '🟩 lag/rolling  🟦 weather  🟨 calendar'
        )
        ax.set_xlabel('Importance')
        plt.tight_layout()

        fi_plot_path = REPORTS_DIR / 'feature_importances.png'
        plt.savefig(fi_plot_path, bbox_inches='tight', dpi=150)
        plt.close()
        log.info(f'Saved: {fi_plot_path}')

    return {
        'eval_plot':  str(eval_plot_path),
        'fi_plot':    str(fi_plot_path) if fi_plot_path else None,
    }


# ── MLflow logging ─────────────────────────────────────────────────────────────

def log_to_mlflow(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metrics: dict,
    plot_paths: dict,
    model_name: str,
    best_params: dict,
):
    """Log the final model, metrics, and plots to MLflow. Run mlflow.evaluate()."""
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name='final_model_evaluation'):

        # Params
        mlflow.log_param('model_type',  model_name)
        mlflow.log_param('tuned',       True)
        mlflow.log_param('train_on',    'train + val combined')
        mlflow.log_param('n_features',  X_test.shape[1])
        mlflow.log_params({f'hp_{k}': v for k, v in best_params.items()})

        # Metrics
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        # Plots
        if plot_paths.get('eval_plot'):
            mlflow.log_artifact(plot_paths['eval_plot'])
        if plot_paths.get('fi_plot'):
            mlflow.log_artifact(plot_paths['fi_plot'])

        # Log and register the pipeline
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path='final_pipeline',
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=X_test.head(5),
        )
        log.info(f'Model logged: {model_info.model_uri}')

        # mlflow.evaluate() — auto-generates residual plots, actual-vs-predicted,
        # and all standard regression metrics in the MLflow UI
        eval_data           = X_test.copy()
        eval_data['target'] = y_test.values

        eval_results = mlflow.evaluate(
            model=model_info.model_uri,
            data=eval_data,
            targets='target',
            model_type='regressor',
            evaluators='default',
        )

        log.info('\nmlflow.evaluate() auto-logged metrics:')
        for k, v in eval_results.metrics.items():
            log.info(f'  {k}: {v:.4f}')


# ── Globals for save_evaluation_plots (accessed by inner func) ─────────────────
pipeline_ref      = None
all_features_ref  = None


def run():
    global pipeline_ref, all_features_ref

    log.info('=' * 60)
    log.info('EVALUATE — final test set evaluation starting')
    log.info('=' * 60)
    log.info('⚠️  This is a ONE-TIME operation. Do not re-run after seeing results.')

    train_raw, val_raw, test_raw, config, model_name, best_params = load_all_splits()

    pipeline, fe = retrain_on_trainval(train_raw, val_raw, config, model_name, best_params)
    pipeline_ref     = pipeline
    all_features_ref = config['all_features']

    test_fe, X_test, y_test, y_pred = predict_test(
        pipeline, fe, train_raw, val_raw, test_raw, config
    )

    metrics    = compute_all_metrics(y_test, y_pred, test_fe)
    plot_paths = save_evaluation_plots(y_test, y_pred, test_fe, metrics, model_name)
    log_to_mlflow(pipeline, X_test, y_test, metrics, plot_paths, model_name, best_params)

    log.info('\n' + '=' * 60)
    log.info('PROJECT COMPLETE')
    log.info('=' * 60)
    log.info(f'  Model:   {model_name} (tuned)')
    log.info(f'  RMSE:    {metrics["test_rmse"]:,.2f} MW  ({metrics["test_rmse_pct"]:.1f}% of mean demand)')
    log.info(f'  MAE:     {metrics["test_mae"]:,.2f} MW')
    log.info(f'  R²:      {metrics["test_r2"]:.4f}')
    log.info(f'  MAPE:    {metrics["test_mape"]:.2f}%')
    log.info('EVALUATE — done ✅')


if __name__ == '__main__':
    run()
