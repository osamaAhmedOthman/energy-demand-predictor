"""
data_loader.py — Load, merge, clean, and split the raw energy + weather data.

Extracted from notebook 01_split_data.ipynb.

Usage:
    python src/data_loader.py

What it does:
    1. Loads energy_dataset.csv and weather_features.csv from data/raw/
    2. Aggregates weather (5 cities → 1 row per hour)
    3. Merges both datasets on timestamp
    4. Drops useless columns (100% null, 99.9% zero)
    5. Fills missing values (interpolation for target, ffill for rest)
    6. Creates chronological 70/15/15 train/val/test split
    7. Validates no overlap between splits
    8. Saves train.csv, val.csv, test.csv, meta.json to data/processed/
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path

from config import (
    ENERGY_FILE, WEATHER_FILE,
    TRAIN_FILE, VAL_FILE, TEST_FILE, META_FILE,
    DATA_PROCESSED,
    DROP_COLS, WEATHER_NUMERIC_COLS,
    TARGET, TIME_COL,
    TRAIN_RATIO, VAL_RATIO,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)s  %(message)s')
log = logging.getLogger(__name__)


# ── Step 1: Load ───────────────────────────────────────────────────────────────

def load_energy(path: Path) -> pd.DataFrame:
    """Load energy dataset and parse the timestamp column."""
    log.info(f'Loading energy data from {path}')
    df = pd.read_csv(path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True)
    log.info(f'  Energy shape: {df.shape}  |  {df[TIME_COL].min()} → {df[TIME_COL].max()}')
    assert df[TIME_COL].is_monotonic_increasing, 'Energy timestamps are not sorted!'
    return df


def load_weather(path: Path) -> pd.DataFrame:
    """Load weather dataset and parse the timestamp column."""
    log.info(f'Loading weather data from {path}')
    df = pd.read_csv(path)
    df['dt_iso'] = pd.to_datetime(df['dt_iso'], utc=True)
    cities = df['city_name'].unique().tolist()
    log.info(f'  Weather shape: {df.shape}  |  Cities: {cities}')
    return df


# ── Step 2: Aggregate weather ──────────────────────────────────────────────────

def aggregate_weather(weather: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate weather from 5 cities → 1 row per hour by taking the mean
    of all numeric columns across cities.

    Input:  178,396 rows (5 cities × 35,064 hours)
    Output: 35,064 rows (1 per hour)
    """
    log.info('Aggregating weather: 5 cities → 1 row per hour')
    weather_agg = (
        weather
        .groupby('dt_iso')[WEATHER_NUMERIC_COLS]
        .mean()
        .reset_index()
    )
    log.info(f'  Weather agg shape: {weather_agg.shape}')
    return weather_agg


# ── Step 3: Merge ──────────────────────────────────────────────────────────────

def merge_datasets(energy: pd.DataFrame, weather_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Inner join energy + weather on timestamp.
    Only keeps hours that exist in both datasets.
    """
    log.info('Merging energy + weather on timestamp')
    merged = energy.merge(
        weather_agg,
        left_on=TIME_COL,
        right_on='dt_iso',
        how='inner',
    )
    merged = merged.drop(columns=['dt_iso'])
    merged = merged.sort_values(TIME_COL).reset_index(drop=True)
    log.info(f'  Merged shape: {merged.shape}')
    assert merged[TIME_COL].is_monotonic_increasing, 'Merged timestamps not sorted!'
    return merged


# ── Step 4: Drop useless columns ───────────────────────────────────────────────

def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are 100% null or 99.9% zero — confirmed in EDA.
    These columns carry zero signal and just add noise.
    """
    log.info(f'Dropping {len(DROP_COLS)} useless columns')
    for col in DROP_COLS:
        null_pct = df[col].isnull().mean() * 100
        zero_pct = (df[col] == 0).mean() * 100
        log.info(f'  DROP: {col}  (null={null_pct:.1f}%  zero={zero_pct:.1f}%)')
    df = df.drop(columns=DROP_COLS)
    log.info(f'  Columns after drop: {df.shape[1]}')
    return df


# ── Step 5: Fill missing values ────────────────────────────────────────────────

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values using domain-appropriate strategies:

    - Target (total load actual): time-based interpolation.
      Gaps are small (≤36 consecutive NaNs). Interpolation preserves
      the temporal trend better than ffill or mean.

    - Generation & forecast columns: forward-fill.
      Values are stable hour-to-hour; last known reading is a good estimate.

    - Weather columns: no nulls after 5-city aggregation.
    """
    nulls_before = df.isnull().sum().sum()
    log.info(f'Filling missing values  (total nulls before: {nulls_before})')

    # Set time as index so pandas can do time-weighted interpolation
    df = df.set_index(TIME_COL)

    # Target: time-weighted interpolation
    df[TARGET] = df[TARGET].interpolate(method='time')

    # Other columns: forward-fill
    cols_to_ffill = [c for c in df.columns if df[c].isnull().sum() > 0]
    if cols_to_ffill:
        df[cols_to_ffill] = df[cols_to_ffill].ffill()

    df = df.reset_index()
    nulls_after = df.isnull().sum().sum()
    log.info(f'  Nulls after fill: {nulls_after}')
    assert nulls_after == 0, f'Still have {nulls_after} nulls after filling!'
    return df


# ── Step 6: Chronological split ────────────────────────────────────────────────

def split_chronological(df: pd.DataFrame):
    """
    Split into train / val / test using purely chronological ordering.

    WHY NOT train_test_split with shuffle?
    Because shuffling leaks future data into training. In production,
    when you predict hour T, you only have data up to T-1. Random split
    breaks this constraint and gives falsely optimistic results.

    Returns
    -------
    train, val, test : pd.DataFrame
    """
    n         = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    train = df.iloc[:train_end].copy()
    val   = df.iloc[train_end:val_end].copy()
    test  = df.iloc[val_end:].copy()

    log.info('Chronological split:')
    log.info(f'  Train : {len(train):>6,} rows  {train[TIME_COL].min().date()} → {train[TIME_COL].max().date()}')
    log.info(f'  Val   : {len(val):>6,} rows  {val[TIME_COL].min().date()} → {val[TIME_COL].max().date()}')
    log.info(f'  Test  : {len(test):>6,} rows  {test[TIME_COL].min().date()} → {test[TIME_COL].max().date()}')

    return train, val, test


# ── Step 7: Validate splits ────────────────────────────────────────────────────

def validate_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, n_total: int):
    """
    Assert that splits are clean:
    - No overlap between any two splits
    - No rows lost
    - All splits are sorted chronologically
    """
    assert train[TIME_COL].max() < val[TIME_COL].min(),  'Train/Val overlap!'
    assert val[TIME_COL].max()   < test[TIME_COL].min(), 'Val/Test overlap!'
    assert len(train) + len(val) + len(test) == n_total, 'Rows lost in split!'
    assert train[TIME_COL].is_monotonic_increasing, 'Train not sorted!'
    assert val[TIME_COL].is_monotonic_increasing,   'Val not sorted!'
    assert test[TIME_COL].is_monotonic_increasing,  'Test not sorted!'
    log.info('✅ All split validation checks passed')


# ── Step 8: Save ───────────────────────────────────────────────────────────────

def save_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, merged: pd.DataFrame):
    """Save splits and metadata to data/processed/."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    train.to_csv(TRAIN_FILE, index=False)
    val.to_csv(VAL_FILE,     index=False)
    test.to_csv(TEST_FILE,   index=False)

    meta = {
        'target_col'  : TARGET,
        'time_col'    : TIME_COL,
        'train_rows'  : len(train),
        'val_rows'    : len(val),
        'test_rows'   : len(test),
        'train_start' : str(train[TIME_COL].min()),
        'train_end'   : str(train[TIME_COL].max()),
        'val_start'   : str(val[TIME_COL].min()),
        'val_end'     : str(val[TIME_COL].max()),
        'test_start'  : str(test[TIME_COL].min()),
        'test_end'    : str(test[TIME_COL].max()),
        'all_columns' : merged.columns.tolist(),
        'dropped_cols': DROP_COLS,
        'weather_cols': WEATHER_NUMERIC_COLS,
    }
    with open(META_FILE, 'w') as f:
        json.dump(meta, f, indent=2)

    log.info(f'✅ Saved:')
    log.info(f'   {TRAIN_FILE}  ({len(train):,} rows)')
    log.info(f'   {VAL_FILE}    ({len(val):,} rows)')
    log.info(f'   {TEST_FILE}   ({len(test):,} rows)')
    log.info(f'   {META_FILE}')


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run():
    """Full data loading and splitting pipeline."""
    log.info('=' * 60)
    log.info('DATA LOADER — starting')
    log.info('=' * 60)

    energy      = load_energy(ENERGY_FILE)
    weather     = load_weather(WEATHER_FILE)
    weather_agg = aggregate_weather(weather)
    merged      = merge_datasets(energy, weather_agg)
    merged      = drop_useless_columns(merged)
    merged      = fill_missing_values(merged)

    n_total            = len(merged)
    train, val, test   = split_chronological(merged)
    validate_splits(train, val, test, n_total)
    save_splits(train, val, test, merged)

    log.info('DATA LOADER — done ✅')


if __name__ == '__main__':
    run()
