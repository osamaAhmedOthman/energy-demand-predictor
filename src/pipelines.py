"""
pipelines.py — Build sklearn Pipeline objects for each model type.

Extracted from notebooks 04 and 05.

Why a separate pipelines.py?
  - train.py, tune.py, and evaluate.py all need to build the same pipeline.
  - One factory function = no duplication = no inconsistency.
  - The pipeline is the contract between feature engineering and the model.

Pipeline design decision:
  - Tree models (RF, GBM, XGBoost, LightGBM): no scaler needed.
    They split on thresholds — scale doesn't matter.
  - Ridge (linear): StandardScaler is required.
    Otherwise features with large ranges (temp²) dominate the loss.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from config import RANDOM_STATE

# Optional imports — gracefully handle missing libraries
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


# ── Model registry ─────────────────────────────────────────────────────────────

def get_default_models() -> dict:
    """
    Return a dict of {model_name: unfitted_model_instance} with default
    hyperparameters for initial comparison in notebook/script 05.

    Only models whose libraries are installed are included.
    """
    models = {
        'Ridge': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
        ),
    }

    if HAS_XGB:
        models['XGBoost'] = XGBRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=-1,
        )

    if HAS_LGB:
        models['LightGBM'] = LGBMRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=-1,
        )

    return models


def get_model_from_name(model_name: str, params: dict = None):
    """
    Instantiate a model by name with optional hyperparameters.

    Used by tune.py and evaluate.py to rebuild the model from
    the best_model.txt + best_params.json saved in data/processed/.

    Parameters
    ----------
    model_name : str
        One of: 'Ridge', 'RandomForest', 'GradientBoosting',
                'XGBoost', 'LightGBM'
    params : dict, optional
        Hyperparameters to pass to the model constructor.
        If None, model is created with default params.

    Returns
    -------
    sklearn estimator
    """
    params = params or {}

    if model_name == 'Ridge':
        return Ridge(**params)

    elif model_name == 'RandomForest':
        params.setdefault('random_state', RANDOM_STATE)
        params.setdefault('n_jobs', -1)
        return RandomForestRegressor(**params)

    elif model_name == 'GradientBoosting':
        params.setdefault('random_state', RANDOM_STATE)
        return GradientBoostingRegressor(**params)

    elif model_name == 'XGBoost':
        if not HAS_XGB:
            raise ImportError('XGBoost not installed: pip install xgboost')
        params.setdefault('random_state', RANDOM_STATE)
        params.setdefault('verbosity', 0)
        params.setdefault('n_jobs', -1)
        return XGBRegressor(**params)

    elif model_name == 'LightGBM':
        if not HAS_LGB:
            raise ImportError('LightGBM not installed: pip install lightgbm')
        params.setdefault('random_state', RANDOM_STATE)
        params.setdefault('verbose', -1)
        params.setdefault('n_jobs', -1)
        return LGBMRegressor(**params)

    else:
        raise ValueError(
            f'Unknown model: {model_name}. '
            f'Valid options: Ridge, RandomForest, GradientBoosting, XGBoost, LightGBM'
        )


# ── Pipeline factory ───────────────────────────────────────────────────────────

def build_pipeline(model_name: str, params: dict = None) -> Pipeline:
    """
    Build a complete sklearn Pipeline for the given model.

    For Ridge: Pipeline([scaler → model])
    For tree models: Pipeline([model])  — no scaler needed

    Parameters
    ----------
    model_name : str
        Model identifier. See get_model_from_name() for valid names.
    params : dict, optional
        Hyperparameters passed to the model constructor.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline ready for .fit() or cross_val_score().

    Example
    -------
    >>> pipeline = build_pipeline('LightGBM', {'n_estimators': 500, 'learning_rate': 0.05})
    >>> pipeline.fit(X_train, y_train)
    >>> y_pred = pipeline.predict(X_test)
    """
    model = get_model_from_name(model_name, params)

    if model_name == 'Ridge':
        # Linear model needs scaling — features with larger ranges would
        # otherwise dominate the regularization penalty
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model',  model),
        ])
    else:
        # Tree models are scale-invariant — they split on thresholds
        return Pipeline([
            ('model', model),
        ])
