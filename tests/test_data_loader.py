"""
test_data_loader.py — Tests for data_loader.py

Tests cover:
  1. Weather aggregation: 5 cities → 1 row per hour
  2. Merge: inner join preserves shape, drops dt_iso
  3. Drop useless columns: null and zero columns are gone
  4. Fill missing values: zero nulls after fill
  5. Chronological split: no overlap, no rows lost, correct ratios
  6. Split validation: detects overlapping splits (negative test)
  7. Sorted order: all splits are monotonically increasing in time
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / 'src'
sys.path.insert(0, str(SRC_DIR))

from src.data_loader import (  
    aggregate_weather,
    merge_datasets,
    drop_useless_columns,
    fill_missing_values,
    split_chronological,
    validate_splits,
)
from src.config import (
    DROP_COLS,
    WEATHER_NUMERIC_COLS,
    TARGET, TIME_COL,
    TRAIN_RATIO, VAL_RATIO,
)


# ── Weather aggregation ────────────────────────────────────────────────────────

class TestAggregateWeather:

    def test_output_has_one_row_per_timestamp(self, raw_weather_df):
        """After aggregation: rows == unique timestamps (not cities × timestamps)."""
        result = aggregate_weather(raw_weather_df)
        n_unique_ts = raw_weather_df['dt_iso'].nunique()
        assert len(result) == n_unique_ts

    def test_output_has_no_city_column(self, raw_weather_df):
        """City column must be gone — we aggregated across cities."""
        result = aggregate_weather(raw_weather_df)
        assert 'city_name' not in result.columns

    def test_output_contains_all_numeric_cols(self, raw_weather_df):
        """All weather numeric columns survive aggregation."""
        result = aggregate_weather(raw_weather_df)
        for col in WEATHER_NUMERIC_COLS:
            assert col in result.columns, f'Missing column after aggregation: {col}'

    def test_aggregated_values_are_means(self, raw_weather_df):
        """Verify aggregated temp equals mean across cities for one timestamp."""
        result     = aggregate_weather(raw_weather_df)
        ts         = raw_weather_df['dt_iso'].iloc[0]
        expected   = raw_weather_df[raw_weather_df['dt_iso'] == ts]['temp'].mean()
        actual     = result[result['dt_iso'] == ts]['temp'].values[0]
        assert abs(actual - expected) < 1e-9

    def test_no_nulls_in_output(self, raw_weather_df):
        """Aggregated weather should have zero nulls (fixture has clean data)."""
        result = aggregate_weather(raw_weather_df)
        assert result.isnull().sum().sum() == 0


# ── Merge ──────────────────────────────────────────────────────────────────────

class TestMergeDatasets:

    def test_dt_iso_column_is_dropped(self, raw_energy_df, raw_weather_df):
        """dt_iso is a duplicate of time — must be dropped after merge."""
        weather_agg = aggregate_weather(raw_weather_df)
        result      = merge_datasets(raw_energy_df, weather_agg)
        assert 'dt_iso' not in result.columns

    def test_merged_has_time_column(self, raw_energy_df, raw_weather_df):
        """time column must survive the merge."""
        weather_agg = aggregate_weather(raw_weather_df)
        result      = merge_datasets(raw_energy_df, weather_agg)
        assert TIME_COL in result.columns

    def test_merged_is_sorted_by_time(self, raw_energy_df, raw_weather_df):
        """Merged DataFrame must be sorted chronologically."""
        weather_agg = aggregate_weather(raw_weather_df)
        result      = merge_datasets(raw_energy_df, weather_agg)
        assert result[TIME_COL].is_monotonic_increasing

    def test_inner_join_preserves_shared_timestamps(self, raw_energy_df, raw_weather_df):
        """Inner join: result rows <= min(energy rows, unique weather timestamps)."""
        weather_agg = aggregate_weather(raw_weather_df)
        result      = merge_datasets(raw_energy_df, weather_agg)
        max_possible = min(len(raw_energy_df), weather_agg['dt_iso'].nunique())
        assert len(result) <= max_possible
        assert len(result) > 0


# ── Drop useless columns ───────────────────────────────────────────────────────

class TestDropUselessColumns:

    def test_all_drop_cols_are_removed(self, raw_energy_df):
        """Every column in DROP_COLS must be absent after dropping."""
        result = drop_useless_columns(raw_energy_df)
        for col in DROP_COLS:
            assert col not in result.columns, f'Column still present after drop: {col}'

    def test_useful_columns_are_kept(self, raw_energy_df):
        """The target and useful columns must survive."""
        result = drop_useless_columns(raw_energy_df)
        assert TARGET in result.columns
        assert TIME_COL in result.columns
        assert 'total load forecast' in result.columns

    def test_row_count_unchanged(self, raw_energy_df):
        """Dropping columns must not drop rows."""
        result = drop_useless_columns(raw_energy_df)
        assert len(result) == len(raw_energy_df)

    def test_fewer_columns_after_drop(self, raw_energy_df):
        """Column count must decrease by exactly len(DROP_COLS)."""
        n_before = len(raw_energy_df.columns)
        result   = drop_useless_columns(raw_energy_df)
        assert len(result.columns) == n_before - len(DROP_COLS)


# ── Fill missing values ────────────────────────────────────────────────────────

class TestFillMissingValues:

    def test_no_nulls_after_fill(self, merged_df):
        """After fill, the DataFrame must have zero nulls."""
        # Inject some NaNs into target to simulate real data gaps
        dirty         = merged_df.copy()
        dirty.loc[10:15, TARGET] = np.nan

        dirty         = dirty.set_index(TIME_COL)
        dirty         = dirty.reset_index()  # keep time column
        result        = fill_missing_values(dirty)
        assert result.isnull().sum().sum() == 0

    def test_target_nulls_are_interpolated(self, merged_df):
        """Target NaNs must be filled by interpolation, not left as NaN."""
        dirty         = merged_df.copy()
        null_indices  = [5, 6, 7]
        dirty.loc[null_indices, TARGET] = np.nan

        result = fill_missing_values(dirty)
        for i in null_indices:
            assert not np.isnan(result[TARGET].iloc[i])

    def test_time_column_is_preserved(self, merged_df):
        """fill_missing_values uses set_index/reset_index — time column must survive."""
        result = fill_missing_values(merged_df)
        assert TIME_COL in result.columns

    def test_row_count_unchanged_after_fill(self, merged_df):
        """Filling values must not add or remove rows."""
        result = fill_missing_values(merged_df)
        assert len(result) == len(merged_df)


# ── Chronological split ────────────────────────────────────────────────────────

class TestSplitChronological:

    def test_no_rows_lost(self, merged_df):
        """train + val + test must equal the original row count."""
        train, val, test = split_chronological(merged_df)
        assert len(train) + len(val) + len(test) == len(merged_df)

    def test_train_ends_before_val_starts(self, merged_df):
        """Last train timestamp must be strictly before first val timestamp."""
        train, val, _ = split_chronological(merged_df)
        assert train[TIME_COL].max() < val[TIME_COL].min()

    def test_val_ends_before_test_starts(self, merged_df):
        """Last val timestamp must be strictly before first test timestamp."""
        _, val, test = split_chronological(merged_df)
        assert val[TIME_COL].max() < test[TIME_COL].min()

    def test_train_is_largest_split(self, merged_df):
        """Train must be 70% — the largest of the three splits."""
        train, val, test = split_chronological(merged_df)
        assert len(train) > len(val)
        assert len(train) > len(test)

    def test_approximate_ratios(self, merged_df):
        """Each split size must be within 1% of its target ratio."""
        train, val, test = split_chronological(merged_df)
        n = len(merged_df)
        assert abs(len(train) / n - TRAIN_RATIO) < 0.01
        assert abs(len(val)   / n - VAL_RATIO)   < 0.01

    def test_all_splits_are_sorted(self, merged_df):
        """Every split must be monotonically increasing in time."""
        train, val, test = split_chronological(merged_df)
        assert train[TIME_COL].is_monotonic_increasing
        assert val[TIME_COL].is_monotonic_increasing
        assert test[TIME_COL].is_monotonic_increasing

    def test_splits_cover_full_time_range(self, merged_df):
        """Together the splits must cover the entire original time range."""
        train, val, test = split_chronological(merged_df)
        assert train[TIME_COL].min() == merged_df[TIME_COL].min()
        assert test[TIME_COL].max()  == merged_df[TIME_COL].max()


# ── Validate splits (negative tests) ──────────────────────────────────────────

class TestValidateSplits:

    def test_passes_for_correct_splits(self, merged_df):
        """validate_splits must not raise for a correctly created split."""
        train, val, test = split_chronological(merged_df)
        # Should not raise
        validate_splits(train, val, test, len(merged_df))

    def test_raises_on_train_val_overlap(self, merged_df):
        """Must detect when train and val share timestamps."""
        train, val, test = split_chronological(merged_df)

        # Force overlap: prepend last row of train onto val
        val_bad = pd.concat([train.tail(1), val]).reset_index(drop=True)

        with pytest.raises(AssertionError, match='overlap|Train'):
            validate_splits(train, val_bad, test, len(merged_df) + 1)

    def test_raises_on_row_count_mismatch(self, merged_df):
        """Must detect when rows are lost during splitting."""
        train, val, test = split_chronological(merged_df)

        with pytest.raises(AssertionError):
            validate_splits(train, val, test, len(merged_df) + 1)
