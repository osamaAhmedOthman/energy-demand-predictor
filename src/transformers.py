"""
transformers.py — Custom sklearn transformer for energy demand feature engineering.

Extracted from notebook 03_feature_engineering.ipynb.

The EnergyFeatureTransformer is a proper sklearn transformer:
  - Implements fit() / transform() contract
  - Can be placed inside a sklearn Pipeline
  - fit() is stateless (returns self) — all transforms are deterministic
  - transform() always copies input — never modifies the original DataFrame

Feature groups created:
  1. Calendar    — hour, dow, month, quarter, is_weekend, season, is_holiday
  2. Cyclical    — sin/cos encoding for hour, dow, month, day_of_year
  3. Lag         — demand 1h, 24h, 48h, 168h, 336h ago
  4. Rolling     — mean/std/max/min over past 24h and 168h windows
  5. Weather     — temp², is_cold, is_hot, temp_humidity, is_raining, wind_chill
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from config import (
    TARGET, TIME_COL,
    LAG_HOURS,
    TEMP_COLD_K, TEMP_HOT_K,
    ALL_FEATURES,
)


class EnergyFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Full feature engineering pipeline for energy demand prediction.

    Transforms the raw merged dataframe (energy + weather columns)
    into a model-ready feature matrix.

    Parameters
    ----------
    lag_hours : list of int, optional
        Which lag offsets to compute. Default: [1, 24, 48, 168, 336].
    add_cyclical : bool, optional
        Whether to add sin/cos cyclical features. Default: True.
        Set to False to reduce feature count for linear models.

    Notes
    -----
    - fit() is stateless — it returns self immediately.
      Nothing is learned from the training data.
      All transformations are deterministic from the input.

    - transform() calls dropna() at the end because lag features
      introduce NaN for the first max(lag_hours) rows.
      Callers must be aware that output has fewer rows than input.

    - For test set inference, prepend LAG_CONTEXT_ROWS rows from the
      end of the training set before calling transform(), then filter
      back to only the test period. See evaluate.py for the pattern.
    """

    def __init__(self, lag_hours: list = None, add_cyclical: bool = True):
        self.lag_hours    = lag_hours or LAG_HOURS
        self.add_cyclical = add_cyclical

    def fit(self, X, y=None):
        # Stateless transformer — nothing to fit
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()  # never modify the original

        # Ensure time column is parsed
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True)
        df = df.sort_values(TIME_COL).reset_index(drop=True)

        # Apply feature groups in order
        df = self._add_calendar(df)

        if self.add_cyclical:
            df = self._add_cyclical(df)

        df = self._add_lags(df)
        df = self._add_rolling(df)
        df = self._add_weather(df)

        # Drop NaN rows introduced by lag features
        # (first max(lag_hours) rows will have NaN lags)
        df = df.dropna().reset_index(drop=True)

        return df

    # ── Private helpers ────────────────────────────────────────────────────────

    def _add_calendar(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add calendar features derived purely from the timestamp.
        No target data is used → no leakage risk.
        """
        t = df[TIME_COL]

        df['hour']         = t.dt.hour                           # 0–23
        df['dow']          = t.dt.dayofweek                      # 0=Mon, 6=Sun
        df['month']        = t.dt.month                          # 1–12
        df['day_of_year']  = t.dt.dayofyear                      # 1–365
        df['week_of_year'] = t.dt.isocalendar().week.astype(int) # 1–53
        df['quarter']      = t.dt.quarter                        # 1–4
        df['is_weekend']   = (t.dt.dayofweek >= 5).astype(int)   # 0 or 1

        # Season: 0=Spring, 1=Summer, 2=Autumn, 3=Winter
        month_to_season = {
            12: 3, 1: 3, 2: 3,   # Winter
             3: 0, 4: 0, 5: 0,   # Spring
             6: 1, 7: 1, 8: 1,   # Summer
             9: 2,10: 2,11: 2,   # Autumn
        }
        df['season'] = df['month'].map(month_to_season)

        # Spanish public holidays
        # Uses the 'holidays' library if installed; falls back to zero otherwise.
        # Install: pip install holidays
        try:
            import holidays
            es_holidays  = holidays.Spain(years=range(2015, 2020))
            holiday_strs = {str(h) for h in es_holidays}
            df['is_holiday'] = t.dt.date.astype(str).isin(holiday_strs).astype(int)
        except ImportError:
            df['is_holiday'] = 0

        return df

    def _add_cyclical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode periodic features as (sin, cos) pairs.

        Why: hour 23 and hour 0 are adjacent in reality but far apart
        numerically (23 vs 0). Sin/cos encoding wraps the cycle so that
        the distance between any two adjacent hours is always the same.

        For a period P:
            sin_feature = sin(2π × value / P)
            cos_feature = cos(2π × value / P)
        """
        df['hour_sin']  = np.sin(2 * np.pi * df['hour']        / 24)
        df['hour_cos']  = np.cos(2 * np.pi * df['hour']        / 24)
        df['dow_sin']   = np.sin(2 * np.pi * df['dow']         /  7)
        df['dow_cos']   = np.cos(2 * np.pi * df['dow']         /  7)
        df['month_sin'] = np.sin(2 * np.pi * df['month']       / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']       / 12)
        df['doy_sin']   = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['doy_cos']   = np.cos(2 * np.pi * df['day_of_year'] / 365)
        return df

    def _add_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add lag features: demand N hours ago.

        CRITICAL RULE: shift(n) where n > 0 shifts values DOWN,
        meaning row i gets the value from row i-n (the past).

        df['lag_24h'].iloc[i] == df[TARGET].iloc[i - 24]  ← past data

        NEVER use negative shift — that would be future data (leakage).
        """
        for lag in self.lag_hours:
            df[f'lag_{lag}h'] = df[TARGET].shift(lag)
        return df

    def _add_rolling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling window statistics.

        CRITICAL RULE: call shift(1) BEFORE rolling.
        This excludes the current row from all rolling windows.

        Without shift(1):
            rolling_mean_24h at time T uses hours T, T-1, ..., T-23
            → includes time T itself → leakage!

        With shift(1):
            rolling_mean_24h at time T uses hours T-1, T-2, ..., T-24
            → only past data → no leakage ✅
        """
        shifted = df[TARGET].shift(1)  # excludes current row

        df['rolling_mean_24h']  = shifted.rolling(window=24,  min_periods=12).mean()
        df['rolling_mean_168h'] = shifted.rolling(window=168, min_periods=84).mean()
        df['rolling_std_24h']   = shifted.rolling(window=24,  min_periods=12).std()
        df['rolling_max_24h']   = shifted.rolling(window=24,  min_periods=12).max()
        df['rolling_min_24h']   = shifted.rolling(window=24,  min_periods=12).min()
        return df

    def _add_weather(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered weather features.

        Key insight from EDA: temperature has a U-shaped relationship
        with energy demand — both very cold (heating) and very hot
        (cooling) lead to high demand. temp² captures this non-linearity.

        Temperature in this dataset is in Kelvin:
            TEMP_COLD_K = 280K = 7°C
            TEMP_HOT_K  = 298K = 25°C
        """
        df['temp_squared']  = df['temp'] ** 2
        df['is_cold']       = (df['temp'] < TEMP_COLD_K).astype(int)
        df['is_hot']        = (df['temp'] > TEMP_HOT_K).astype(int)
        df['temp_humidity'] = df['temp'] * df['humidity'] / 100
        df['is_raining']    = (df['rain_1h'] > 0).astype(int)

        # Wind chill: high wind + cold temperature → more heating demand
        df['wind_chill']    = df['wind_speed'] * (1 - df['temp'] / 310)
        return df
