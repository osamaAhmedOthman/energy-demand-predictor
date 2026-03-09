"""
test_transformers.py — Tests for EnergyFeatureTransformer in transformers.py

The most critical test file in the project.
Bad feature engineering = poisoned model = wrong production predictions.

Tests cover:
  1. sklearn contract: fit() returns self, fit_transform() == fit().transform()
  2. No input mutation: transform() never modifies the original DataFrame
  3. Output shape: correct columns exist, row count is reduced by NaN rows only
  4. Lag correctness: lag_24h[i] == target[i-24] — the no-leakage guarantee
  5. Rolling correctness: rolling window uses shift(1) — no current row leakage
  6. Calendar features: hour, dow, month values are within valid ranges
  7. Cyclical features: sin/cos values stay in [-1, 1] and have correct period
  8. Weather features: temp_squared == temp^2, binary flags are 0 or 1
  9. Custom lag_hours parameter: only requested lags are created
  10. No NaN in output: dropna() removes all incomplete rows
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / 'src'
sys.path.insert(0, str(SRC_DIR))

from src.transformers import EnergyFeatureTransformer
from src.config import (
    TARGET, TIME_COL,
    LAG_HOURS, LAG_FEATURES,
    CALENDAR_FEATURES, CYCLICAL_FEATURES,
    ROLLING_FEATURES, WEATHER_ENG,
    ALL_FEATURES,
    TEMP_COLD_K, TEMP_HOT_K,
)


# ── sklearn contract ───────────────────────────────────────────────────────────

class TestSklearnContract:

    def test_fit_returns_self(self, merged_df):
        """fit() must return the transformer itself — required by sklearn."""
        fe     = EnergyFeatureTransformer()
        result = fe.fit(merged_df)
        assert result is fe

    def test_fit_with_y_returns_self(self, merged_df):
        """fit(X, y) must also return self — sklearn pipelines pass y."""
        fe     = EnergyFeatureTransformer()
        y_fake = np.zeros(len(merged_df))
        result = fe.fit(merged_df, y_fake)
        assert result is fe

    def test_fit_transform_equals_fit_then_transform(self, merged_df):
        """fit_transform(X) must give identical output to fit(X).transform(X)."""
        fe1 = EnergyFeatureTransformer()
        fe2 = EnergyFeatureTransformer()

        result_combined  = fe1.fit_transform(merged_df)
        result_separate  = fe2.fit(merged_df).transform(merged_df)

        pd.testing.assert_frame_equal(result_combined, result_separate)

    def test_transform_called_without_fit_works(self, merged_df):
        """
        Because fit() is stateless (returns self, learns nothing),
        transform() should work even without calling fit() first.
        """
        fe     = EnergyFeatureTransformer()
        result = fe.transform(merged_df)
        assert len(result) > 0


# ── Input mutation ─────────────────────────────────────────────────────────────

class TestNoInputMutation:

    def test_transform_does_not_modify_input(self, merged_df):
        """
        transform() must never modify the caller's DataFrame.
        It must copy internally with df = X.copy().
        """
        original_cols   = merged_df.columns.tolist()
        original_shape  = merged_df.shape
        original_target = merged_df[TARGET].copy()

        fe = EnergyFeatureTransformer()
        _  = fe.transform(merged_df)

        # Columns must be unchanged
        assert merged_df.columns.tolist() == original_cols
        # Shape must be unchanged
        assert merged_df.shape == original_shape
        # Target values must be unchanged
        pd.testing.assert_series_equal(merged_df[TARGET], original_target)

    def test_calling_transform_twice_gives_same_result(self, merged_df):
        """Idempotency: calling transform on the same input twice gives identical output."""
        fe      = EnergyFeatureTransformer()
        result1 = fe.transform(merged_df)
        result2 = fe.transform(merged_df)
        pd.testing.assert_frame_equal(result1, result2)


# ── Output shape ───────────────────────────────────────────────────────────────

class TestOutputShape:

    def test_all_expected_features_are_present(self, transformed_df):
        """Every feature in ALL_FEATURES must exist in the output."""
        for col in ALL_FEATURES:
            assert col in transformed_df.columns, f'Missing feature: {col}'

    def test_time_column_is_present(self, transformed_df):
        """time column must survive transformation."""
        assert TIME_COL in transformed_df.columns

    def test_target_column_is_present(self, transformed_df):
        """Target column must survive transformation."""
        assert TARGET in transformed_df.columns

    def test_output_has_fewer_rows_than_input(self, merged_df, transformed_df):
        """
        Lag features introduce NaN for the first max(lag_hours) rows.
        dropna() removes them, so output must have fewer rows than input.
        """
        max_lag = max(LAG_HOURS)   # 336
        assert len(transformed_df) == len(merged_df) - max_lag

    def test_no_nulls_in_output(self, transformed_df):
        """After dropna(), there must be zero NaN values in the output."""
        assert transformed_df.isnull().sum().sum() == 0


# ── Lag features: no-leakage guarantee ────────────────────────────────────────

class TestLagFeatures:

    def test_lag_values_equal_past_target(self, transformed_df):
        """
        THE most important test in the project.

        lag_Nh at row i must equal the target at (i - N rows).
        If this fails, the model is training on future data.

        We test this by reconstructing the original target series
        and verifying the shift is correct.
        """
        fe = EnergyFeatureTransformer()

        # Use a fresh, simple DataFrame with known target values
        # so we can calculate exactly what the lag should be
        n   = 400
        rng = np.random.default_rng(99)
        ts  = pd.date_range('2020-01-01', periods=n, freq='h', tz='UTC')

        # Simple synthetic merged_df-like structure
        df = pd.DataFrame({
            TIME_COL:              ts,
            TARGET:                np.arange(n, dtype=float),  # 0, 1, 2, 3...
            'total load forecast': np.arange(n, dtype=float),
            'temp':     285.0, 'temp_min': 283.0, 'temp_max': 287.0,
            'pressure': 1015.0, 'humidity': 60.0,
            'wind_speed': 5.0, 'wind_deg': 180.0,
            'rain_1h': 0.0, 'rain_3h': 0.0, 'snow_3h': 0.0, 'clouds_all': 50.0,
        })

        result = fe.transform(df)

        # After dropna, the first valid row has index = max_lag in original df
        # At that first row, lag_24h should be the value from 24 rows before
        first_row = result.iloc[0]
        for lag in LAG_HOURS:
            lag_col   = f'lag_{lag}h'
            actual    = first_row[lag_col]
            # The original value at position (max_lag - lag) in the original df
            # = max_lag - lag  (since target = index value 0,1,2,...)
            expected  = float(max(LAG_HOURS) - lag)
            assert abs(actual - expected) < 1e-9, (
                f'{lag_col}: expected {expected}, got {actual}. '
                f'Lag feature is using wrong row — possible leakage!'
            )

    def test_lag_column_names_match_config(self, transformed_df):
        """Lag column names must match what ALL_FEATURES expects."""
        for col in LAG_FEATURES:
            assert col in transformed_df.columns

    def test_lag_shifts_are_positive(self):
        """
        Verify that the transformer uses positive shift (past data).
        Negative shift would be future data = leakage.
        Inspect the source to confirm shift direction.
        """
        import inspect
        from src.transformers import EnergyFeatureTransformer
        source = inspect.getsource(EnergyFeatureTransformer._add_lags)
        # Must use shift(lag) where lag is from self.lag_hours (all positive)
        assert 'shift(' in source
        assert 'shift(-' not in source, \
            'Negative shift detected in _add_lags! This is future data leakage.'

    def test_custom_lag_hours_creates_correct_columns(self, merged_df):
        """Custom lag_hours parameter must create exactly those lag columns."""
        custom_lags = [1, 12, 24]
        fe          = EnergyFeatureTransformer(lag_hours=custom_lags)
        result      = fe.transform(merged_df)

        for lag in custom_lags:
            assert f'lag_{lag}h' in result.columns

        # Default lags that were NOT requested must be absent
        for lag in LAG_HOURS:
            if lag not in custom_lags:
                assert f'lag_{lag}h' not in result.columns

    def test_default_lag_hours_match_config(self, transformed_df):
        """Default transformer must create exactly the lags from config.LAG_HOURS."""
        for lag in LAG_HOURS:
            assert f'lag_{lag}h' in transformed_df.columns


# ── Rolling features: shift(1) before rolling ─────────────────────────────────

class TestRollingFeatures:

    def test_rolling_uses_shift_before_rolling(self):
        """
        Verify that _add_rolling calls shift(1) BEFORE .rolling().
        Without shift(1), the current row is included in the window = leakage.
        """
        import inspect
        source = inspect.getsource(EnergyFeatureTransformer._add_rolling)

        # shift(1) must appear before any .rolling() call
        shift_pos   = source.find('shift(1)')
        rolling_pos = source.find('.rolling(')
        assert shift_pos != -1,  '_add_rolling must call shift(1) to prevent leakage'
        assert rolling_pos != -1, '_add_rolling must use .rolling()'
        assert shift_pos < rolling_pos, \
            'shift(1) must appear BEFORE .rolling() — otherwise current row is included'

    def test_all_rolling_columns_are_present(self, transformed_df):
        """All 5 rolling features must exist in the output."""
        for col in ROLLING_FEATURES:
            assert col in transformed_df.columns

    def test_rolling_mean_24h_is_reasonable(self, transformed_df):
        """rolling_mean_24h should be close to lag values — sanity check."""
        # Rolling mean over past 24 hours of demand should be near mean demand
        mean_demand = transformed_df[TARGET].mean()
        rolling_col = transformed_df['rolling_mean_24h']
        # All values should be within 50% of mean demand (generous bounds)
        assert (rolling_col > mean_demand * 0.5).all()
        assert (rolling_col < mean_demand * 1.5).all()


# ── Calendar features ──────────────────────────────────────────────────────────

class TestCalendarFeatures:

    def test_hour_range(self, transformed_df):
        """hour must be in [0, 23]."""
        assert transformed_df['hour'].between(0, 23).all()

    def test_dow_range(self, transformed_df):
        """dow (day of week) must be in [0, 6]."""
        assert transformed_df['dow'].between(0, 6).all()

    def test_month_range(self, transformed_df):
        """month must be in [1, 12]."""
        assert transformed_df['month'].between(1, 12).all()

    def test_quarter_range(self, transformed_df):
        """quarter must be in [1, 4]."""
        assert transformed_df['quarter'].between(1, 4).all()

    def test_is_weekend_is_binary(self, transformed_df):
        """is_weekend must only contain 0 or 1."""
        assert set(transformed_df['is_weekend'].unique()).issubset({0, 1})

    def test_season_is_valid(self, transformed_df):
        """season must be in {0, 1, 2, 3} (Spring/Summer/Autumn/Winter)."""
        assert set(transformed_df['season'].unique()).issubset({0, 1, 2, 3})

    def test_is_holiday_is_binary(self, transformed_df):
        """is_holiday must only contain 0 or 1."""
        assert set(transformed_df['is_holiday'].unique()).issubset({0, 1})

    def test_all_calendar_features_are_present(self, transformed_df):
        """All features from CALENDAR_FEATURES config must exist."""
        for col in CALENDAR_FEATURES:
            assert col in transformed_df.columns


# ── Cyclical features ──────────────────────────────────────────────────────────

class TestCyclicalFeatures:

    def test_sin_cos_values_in_minus1_to_1(self, transformed_df):
        """All sin/cos features must be in [-1, 1] — mathematical property."""
        sin_cos_cols = [c for c in transformed_df.columns
                        if c.endswith('_sin') or c.endswith('_cos')]
        assert len(sin_cos_cols) > 0, 'No cyclical features found'
        for col in sin_cos_cols:
            assert transformed_df[col].between(-1, 1).all(), \
                f'{col} has values outside [-1, 1]'

    def test_hour_sin_cos_encodes_full_cycle(self, transformed_df):
        """
        sin^2 + cos^2 == 1 for all rows (Pythagorean identity).
        This verifies the encoding formula is correct.
        """
        s2_plus_c2 = (transformed_df['hour_sin'] ** 2
                      + transformed_df['hour_cos'] ** 2)
        np.testing.assert_allclose(s2_plus_c2, 1.0, atol=1e-9)

    def test_dow_sin_cos_encodes_full_cycle(self, transformed_df):
        """Same Pythagorean check for day-of-week encoding."""
        s2_plus_c2 = (transformed_df['dow_sin'] ** 2
                      + transformed_df['dow_cos'] ** 2)
        np.testing.assert_allclose(s2_plus_c2, 1.0, atol=1e-9)

    def test_hour_0_and_23_are_adjacent(self, transformed_df):
        """
        The whole point of cyclical encoding: hour 23 and hour 0 must
        be numerically close (angular distance ≈ 2π/24), not far apart.
        """
        h0  = transformed_df[transformed_df['hour'] == 0].iloc[0]
        h23 = transformed_df[transformed_df['hour'] == 23].iloc[0]

        # Euclidean distance in sin-cos space
        dist = np.sqrt((h0['hour_sin'] - h23['hour_sin']) ** 2
                       + (h0['hour_cos'] - h23['hour_cos']) ** 2)

        # Max angular step between adjacent hours: 2*sin(pi/24) ≈ 0.261
        max_adjacent_dist = 2 * np.sin(np.pi / 24) + 1e-9
        assert dist < max_adjacent_dist, (
            f'hour 0 and hour 23 are too far apart in cyclical space: {dist:.4f}. '
            f'Cyclical encoding may be wrong.'
        )

    def test_all_cyclical_features_are_present(self, transformed_df):
        """All features from CYCLICAL_FEATURES config must exist."""
        for col in CYCLICAL_FEATURES:
            assert col in transformed_df.columns


# ── Weather engineering ────────────────────────────────────────────────────────

class TestWeatherFeatures:

    def test_temp_squared_equals_temp_squared(self, transformed_df):
        """temp_squared must literally equal temp^2 — not an approximation."""
        expected = transformed_df['temp'] ** 2
        np.testing.assert_allclose(
            transformed_df['temp_squared'].values,
            expected.values,
            rtol=1e-9,
        )

    def test_is_cold_is_binary(self, transformed_df):
        """is_cold must only contain 0 or 1."""
        assert set(transformed_df['is_cold'].unique()).issubset({0, 1})

    def test_is_hot_is_binary(self, transformed_df):
        """is_hot must only contain 0 or 1."""
        assert set(transformed_df['is_hot'].unique()).issubset({0, 1})

    def test_is_raining_is_binary(self, transformed_df):
        """is_raining must only contain 0 or 1."""
        assert set(transformed_df['is_raining'].unique()).issubset({0, 1})

    def test_is_cold_correct_threshold(self, transformed_df):
        """is_cold must be 1 exactly where temp < TEMP_COLD_K."""
        expected = (transformed_df['temp'] < TEMP_COLD_K).astype(int)
        pd.testing.assert_series_equal(
            transformed_df['is_cold'].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_is_hot_correct_threshold(self, transformed_df):
        """is_hot must be 1 exactly where temp > TEMP_HOT_K."""
        expected = (transformed_df['temp'] > TEMP_HOT_K).astype(int)
        pd.testing.assert_series_equal(
            transformed_df['is_hot'].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_is_raining_correct_threshold(self, transformed_df):
        """is_raining must be 1 exactly where rain_1h > 0."""
        expected = (transformed_df['rain_1h'] > 0).astype(int)
        pd.testing.assert_series_equal(
            transformed_df['is_raining'].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_all_weather_eng_features_present(self, transformed_df):
        """All features from WEATHER_ENG config must exist."""
        for col in WEATHER_ENG:
            assert col in transformed_df.columns
