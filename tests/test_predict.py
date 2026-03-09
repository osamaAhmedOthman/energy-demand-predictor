"""
test_predict.py — Tests for the predict() function in predict.py

Tests cover:
  1. Predictions are within the physically reasonable MW range for Spain
  2. Prediction count matches the number of input rows (after filtering)
  3. predict_start filter returns only rows >= the given timestamp
  4. Empty input after filtering returns an empty Series (no crash)
  5. validate_predictions correctly flags out-of-range values
  6. Output is a pandas Series with a DatetimeIndex
  7. All predicted values are finite (no NaN, no inf)
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / 'src'
sys.path.insert(0, str(SRC_DIR))

from src.predict import predict, validate_predictions
from src.config import ALL_FEATURES, TARGET, TIME_COL, LAG_CONTEXT_ROWS


# ── Reasonable output range ────────────────────────────────────────────────────

class TestPredictionRange:

    def test_predictions_within_physical_bounds(self, fitted_ridge_pipeline, transformed_df):
        """
        Spanish electricity demand is physically bounded.
        Historical range: ~18,000–42,000 MW.
        We use generous bounds (10k–55k) to account for model error.

        If predictions fall outside this range, something is seriously wrong
        with feature engineering or the model — this catches bugs like
        accidentally predicting in kW instead of MW.
        """
        MIN_MW = 10_000
        MAX_MW = 55_000

        preds = fitted_ridge_pipeline.predict(transformed_df[ALL_FEATURES])

        assert preds.min() >= MIN_MW, \
            f'Predictions below {MIN_MW:,} MW: min={preds.min():,.0f}'
        assert preds.max() <= MAX_MW, \
            f'Predictions above {MAX_MW:,} MW: max={preds.max():,.0f}'

    def test_predictions_are_finite(self, fitted_ridge_pipeline, transformed_df):
        """No NaN or infinite values in predictions."""
        preds = fitted_ridge_pipeline.predict(transformed_df[ALL_FEATURES])
        assert np.all(np.isfinite(preds)), 'Predictions contain NaN or inf'

    def test_predictions_are_positive(self, fitted_ridge_pipeline, transformed_df):
        """Energy demand is always positive — negative predictions are a model bug."""
        preds = fitted_ridge_pipeline.predict(transformed_df[ALL_FEATURES])
        assert np.all(preds > 0), 'Predictions contain non-positive values'


# ── predict() function ─────────────────────────────────────────────────────────

class TestPredictFunction:

    def test_output_is_series(self, fitted_ridge_pipeline, merged_df):
        """predict() must return a pandas Series."""
        result = predict(fitted_ridge_pipeline, merged_df)
        assert isinstance(result, pd.Series)

    def test_output_length_matches_valid_rows(self, fitted_ridge_pipeline, merged_df):
        """
        predict() calls EnergyFeatureTransformer internally.
        The transformer drops the first max(lag_hours) rows.
        Output length must equal input rows minus those dropped rows.
        """
        result   = predict(fitted_ridge_pipeline, merged_df)
        expected = len(merged_df) - max([1, 24, 48, 168, 336])
        assert len(result) == expected

    def test_predict_start_filters_correctly(self, fitted_ridge_pipeline, merged_df):
        """
        predict_start='YYYY-MM-DD' must return only predictions for rows
        with timestamp >= that date.
        """
        predict_start = '2016-01-20'
        result        = predict(fitted_ridge_pipeline, merged_df,
                                predict_start=predict_start)

        cutoff = pd.Timestamp(predict_start)
        for ts in result.index:
            assert pd.Timestamp(ts) >= cutoff, \
                f'Prediction at {ts} is before predict_start {cutoff}'

    def test_predict_start_reduces_output_size(self, fitted_ridge_pipeline, merged_df):
        """predict_start filter must produce fewer rows than predicting on full data."""
        full   = predict(fitted_ridge_pipeline, merged_df)
        subset = predict(fitted_ridge_pipeline, merged_df, predict_start='2016-01-20')

        assert len(subset) < len(full), \
            'predict_start filter did not reduce the number of predictions'

    def test_empty_result_on_future_predict_start(self, fitted_ridge_pipeline, merged_df):
        """
        predict_start far in the future (beyond all data) must return
        an empty Series, not crash.
        """
        result = predict(fitted_ridge_pipeline, merged_df,
                         predict_start='2099-01-01')
        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_output_values_are_finite(self, fitted_ridge_pipeline, merged_df):
        """All predicted MW values must be finite real numbers."""
        result = predict(fitted_ridge_pipeline, merged_df)
        assert np.all(np.isfinite(result.values)), \
            'predict() returned NaN or inf values'

    def test_output_index_is_datetime(self, fitted_ridge_pipeline, merged_df):
        """Output Series index must contain timestamps, not integers."""
        result = predict(fitted_ridge_pipeline, merged_df)
        # Index should be datetime-like (or object with timestamps)
        first_ts = result.index[0]
        # Should be convertible to Timestamp without error
        pd.Timestamp(first_ts)


# ── validate_predictions ───────────────────────────────────────────────────────

class TestValidatePredictions:

    def test_valid_predictions_return_true(self):
        """All values in [10_000, 55_000] → must return True."""
        preds = pd.Series([20_000, 28_000, 35_000, 41_000])
        assert validate_predictions(preds) is True

    def test_too_low_returns_false(self):
        """A value below 10_000 MW must return False."""
        preds = pd.Series([28_000, 500])   # 500 is unrealistically low
        assert validate_predictions(preds) is False

    def test_too_high_returns_false(self):
        """A value above 55_000 MW must return False."""
        preds = pd.Series([28_000, 100_000])   # 100k MW is impossibly high
        assert validate_predictions(preds) is False

    def test_boundary_values_are_valid(self):
        """Exact boundary values (10_000 and 55_000) must be accepted."""
        preds = pd.Series([10_000, 55_000])
        assert validate_predictions(preds) is True

    def test_single_valid_value(self):
        """A single valid prediction must return True."""
        preds = pd.Series([28_700.0])
        assert validate_predictions(preds) is True

    def test_all_invalid_returns_false(self):
        """All values out of range must return False."""
        preds = pd.Series([100, 200, 300])
        assert validate_predictions(preds) is False


# ── Prediction consistency ─────────────────────────────────────────────────────

class TestPredictionConsistency:

    def test_same_input_same_output(self, fitted_ridge_pipeline, merged_df):
        """Calling predict twice on the same data must give identical results."""
        result1 = predict(fitted_ridge_pipeline, merged_df)
        result2 = predict(fitted_ridge_pipeline, merged_df)
        pd.testing.assert_series_equal(result1, result2)

    def test_predict_does_not_modify_input(self, fitted_ridge_pipeline, merged_df):
        """predict() must not modify the caller's DataFrame."""
        original_shape = merged_df.shape
        original_cols  = merged_df.columns.tolist()

        _ = predict(fitted_ridge_pipeline, merged_df)

        assert merged_df.shape == original_shape
        assert merged_df.columns.tolist() == original_cols
