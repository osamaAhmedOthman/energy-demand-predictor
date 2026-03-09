"""
test_pipelines.py — Tests for pipelines.py

Tests cover:
  1. build_pipeline returns a valid sklearn Pipeline
  2. Ridge pipeline includes a StandardScaler step
  3. Tree pipelines do NOT include a scaler (not needed, wastes time)
  4. All model names produce a fittable and predictable pipeline
  5. Unknown model name raises a clear ValueError
  6. get_default_models returns at least the 3 always-available models
  7. get_model_from_name with params passes params to the model
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / 'src'
sys.path.insert(0, str(SRC_DIR))

from src.pipelines import build_pipeline, get_default_models, get_model_from_name
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.config import ALL_FEATURES, TARGET


# ── build_pipeline returns correct type ───────────────────────────────────────

class TestBuildPipelineType:

    def test_returns_sklearn_pipeline(self):
        """build_pipeline must return a sklearn Pipeline object."""
        pipeline = build_pipeline('Ridge')
        assert isinstance(pipeline, Pipeline)

    def test_ridge_pipeline_has_scaler(self):
        """Ridge pipeline must include StandardScaler as first step."""
        pipeline = build_pipeline('Ridge')
        step_names = [name for name, _ in pipeline.steps]
        assert 'scaler' in step_names

        scaler_step = pipeline.named_steps['scaler']
        assert isinstance(scaler_step, StandardScaler)

    def test_random_forest_has_no_scaler(self):
        """RandomForest pipeline must NOT include a scaler — trees are scale-invariant."""
        pipeline   = build_pipeline('RandomForest')
        step_names = [name for name, _ in pipeline.steps]
        assert 'scaler' not in step_names

    def test_gradient_boosting_has_no_scaler(self):
        """GradientBoosting pipeline must NOT include a scaler."""
        pipeline   = build_pipeline('GradientBoosting')
        step_names = [name for name, _ in pipeline.steps]
        assert 'scaler' not in step_names

    def test_all_pipelines_have_model_step(self):
        """Every pipeline must have a 'model' step."""
        for model_name in ['Ridge', 'RandomForest', 'GradientBoosting']:
            pipeline   = build_pipeline(model_name)
            step_names = [name for name, _ in pipeline.steps]
            assert 'model' in step_names, f'{model_name} pipeline missing model step'

    def test_unknown_model_raises_value_error(self):
        """build_pipeline with an unknown model name must raise ValueError."""
        with pytest.raises(ValueError, match='Unknown model'):
            build_pipeline('NonExistentModel')


# ── Pipelines are fittable and predictable ─────────────────────────────────────

class TestPipelineFitPredict:

    def test_ridge_fits_and_predicts(self, transformed_df):
        """Ridge pipeline must fit on training data and produce predictions."""
        X = transformed_df[ALL_FEATURES]
        y = transformed_df[TARGET]

        pipeline = build_pipeline('Ridge')
        pipeline.fit(X, y)
        preds = pipeline.predict(X)

        assert len(preds) == len(X)
        assert not np.any(np.isnan(preds)), 'Ridge produced NaN predictions'

    def test_random_forest_fits_and_predicts(self, transformed_df):
        """RandomForest pipeline must fit and produce valid predictions."""
        X = transformed_df[ALL_FEATURES]
        y = transformed_df[TARGET]

        pipeline = build_pipeline('RandomForest', {'n_estimators': 10})
        pipeline.fit(X, y)
        preds = pipeline.predict(X)

        assert len(preds) == len(X)
        assert not np.any(np.isnan(preds))

    def test_gradient_boosting_fits_and_predicts(self, transformed_df):
        """GradientBoosting pipeline must fit and produce valid predictions."""
        X = transformed_df[ALL_FEATURES]
        y = transformed_df[TARGET]

        pipeline = build_pipeline('GradientBoosting', {'n_estimators': 10})
        pipeline.fit(X, y)
        preds = pipeline.predict(X)

        assert len(preds) == len(X)
        assert not np.any(np.isnan(preds))

    def test_predictions_are_finite(self, transformed_df):
        """Predictions must be finite real numbers — no inf, no NaN."""
        X = transformed_df[ALL_FEATURES]
        y = transformed_df[TARGET]

        params_by_model = {
            'Ridge': {'alpha': 1.0},
            'RandomForest': {'n_estimators': 5}
        }

        for model_name, params in params_by_model.items():
            pipeline = build_pipeline(model_name, params)
            pipeline.fit(X, y)
            preds = pipeline.predict(X)
            assert np.all(np.isfinite(preds)), \
                f'{model_name} produced non-finite predictions'


# ── Hyperparameters are passed through ────────────────────────────────────────

class TestHyperparameterPassthrough:

    def test_ridge_alpha_is_applied(self):
        """build_pipeline('Ridge', {'alpha': 10.0}) must create Ridge with alpha=10."""
        pipeline = build_pipeline('Ridge', {'alpha': 10.0})
        model    = pipeline.named_steps['model']
        assert model.alpha == 10.0

    def test_random_forest_n_estimators_is_applied(self):
        """n_estimators must be passed to RandomForest correctly."""
        pipeline = build_pipeline('RandomForest', {'n_estimators': 77})
        model    = pipeline.named_steps['model']
        assert model.n_estimators == 77

    def test_gradient_boosting_learning_rate_is_applied(self):
        """learning_rate must be passed to GradientBoosting correctly."""
        pipeline = build_pipeline('GradientBoosting', {'learning_rate': 0.05})
        model    = pipeline.named_steps['model']
        assert model.learning_rate == 0.05

    def test_params_none_uses_defaults(self):
        """Passing params=None must not raise and must use model defaults."""
        pipeline = build_pipeline('Ridge', params=None)
        assert pipeline is not None


# ── get_default_models ─────────────────────────────────────────────────────────

class TestGetDefaultModels:

    def test_returns_dict(self):
        """get_default_models must return a dictionary."""
        models = get_default_models()
        assert isinstance(models, dict)

    def test_always_contains_core_models(self):
        """Ridge, RandomForest, GradientBoosting must always be present."""
        models = get_default_models()
        for name in ['Ridge', 'RandomForest', 'GradientBoosting']:
            assert name in models, f'Core model {name} missing from default models'

    def test_model_values_are_estimators(self):
        """Every value in the dict must have fit() and predict() methods."""
        models = get_default_models()
        for name, model in models.items():
            assert hasattr(model, 'fit'),     f'{name}: missing fit() method'
            assert hasattr(model, 'predict'), f'{name}: missing predict() method'

    def test_model_names_are_strings(self):
        """All keys must be strings."""
        models = get_default_models()
        for name in models:
            assert isinstance(name, str)


# ── get_model_from_name ────────────────────────────────────────────────────────

class TestGetModelFromName:

    def test_ridge_by_name(self):
        from sklearn.linear_model import Ridge
        model = get_model_from_name('Ridge')
        assert isinstance(model, Ridge)

    def test_random_forest_by_name(self):
        from sklearn.ensemble import RandomForestRegressor
        model = get_model_from_name('RandomForest')
        assert isinstance(model, RandomForestRegressor)

    def test_gradient_boosting_by_name(self):
        from sklearn.ensemble import GradientBoostingRegressor
        model = get_model_from_name('GradientBoosting')
        assert isinstance(model, GradientBoostingRegressor)

    def test_params_are_passed_to_model(self):
        """Params dict must be forwarded to the model constructor."""
        model = get_model_from_name('Ridge', {'alpha': 99.0})
        assert model.alpha == 99.0

    def test_unknown_name_raises_value_error(self):
        with pytest.raises(ValueError):
            get_model_from_name('MadeUpModel')
