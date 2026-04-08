"""
features.py
-----------
Leak-free feature engineering for MNREGA district-level forecasting.

All features are computed from lagged/historical values only.
Leaky columns (works_completed, expenditure_lakhs, etc.) are excluded.

Features:
    lag1_pd, lag2_pd, lag3_pd
    roll2_mean, roll3_mean, roll3_std
    lag1_adj          (lag1 deflated by COVID multiplier when lag year = 2020)
    lag_yoy, lag2_yoy, momentum
    district_capacity, blended_capacity
    relative_to_state, state_lag1_norm
    lag1_vs_capacity, lag1_zscore, state_lag1_zscore
    lag1_extreme, lag1_is_covid
    history_length
    avg_wage_rate, wage_yoy
    is_covid, is_post_covid, is_2022_anomaly, year_trend
    state_enc, district_enc
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

COVID_MULTIPLIER = 1.447
TARGET = "person_days_lakhs"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[features] Building leak-free features...")

    df = df.sort_values(["state", "district", "financial_year"]).reset_index(drop=True)

    df = _lag_features(df)
    df = _rolling_features(df)
    df = _covid_features(df)
    df = _trend_features(df)
    df = _capacity_features(df)
    df = _anomaly_features(df)
    df = _state_features(df)
    df = _temporal_flags(df)
    df = _wage_features(df)
    df = _encode_categoricals(df)

    before = len(df)
    df = df.dropna(subset=["lag1_pd", "lag2_pd"]).reset_index(drop=True)
    print(f"[features] Dropped {before - len(df)} rows (insufficient history)")
    print(f"[features] Done. Final shape: {df.shape}")
    return df


def _lag_features(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["state", "district"])
    df["lag1_pd"] = grp[TARGET].shift(1)
    df["lag2_pd"] = grp[TARGET].shift(2)
    df["lag3_pd"] = grp[TARGET].shift(3)
    df["lag1_hh"] = grp["households_availed"].shift(1)
    return df


def _rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    lag1 = df.groupby(["state", "district"])["lag1_pd"]
    df["roll2_mean"] = lag1.transform(lambda s: s.rolling(2, min_periods=1).mean())
    df["roll3_mean"] = lag1.transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["roll3_std"]  = lag1.transform(
        lambda s: s.rolling(3, min_periods=1).std().fillna(0)
    )
    return df


def _covid_features(df: pd.DataFrame) -> pd.DataFrame:
    df["lag1_is_covid"] = (df["financial_year"] - 1 == 2020).astype(int)
    df["lag1_adj"] = np.where(
        df["lag1_is_covid"] == 1,
        df["lag1_pd"] / COVID_MULTIPLIER,
        df["lag1_pd"]
    )
    return df


def _trend_features(df: pd.DataFrame) -> pd.DataFrame:
    df["lag_yoy"] = (
        (df["lag1_pd"] - df["lag2_pd"]) / df["lag2_pd"].replace(0, np.nan)
    ).clip(-1, 3)
    df["lag2_yoy"] = (
        (df["lag2_pd"] - df["lag3_pd"]) / df["lag3_pd"].replace(0, np.nan)
    ).clip(-1, 3)
    df["momentum"] = df["lag_yoy"] - df["lag2_yoy"]
    return df


def _capacity_features(df: pd.DataFrame) -> pd.DataFrame:
    df["district_capacity"] = df.groupby(["state", "district"])["lag1_pd"].transform(
        lambda s: s.expanding().mean()
    )
    df["history_length"] = df.groupby(["state", "district"]).cumcount()

    state_mean = df.groupby(["state", "financial_year"])["lag1_pd"].transform("mean")
    df["blended_capacity"] = np.where(
        df["history_length"] < 3,
        0.5 * df["district_capacity"].fillna(state_mean) + 0.5 * state_mean,
        df["district_capacity"]
    )

    df["lag1_vs_capacity"] = (
        df["lag1_pd"] / df["blended_capacity"].replace(0, np.nan)
    ).clip(0, 5).fillna(1.0)

    df["lag1_hh_ratio"] = (
        df["lag1_hh"] / df["blended_capacity"].replace(0, np.nan)
    ).clip(0, 5).fillna(1.0)

    return df


def _rolling_zscore(s: pd.Series) -> pd.Series:
    exp_mean = s.shift(1).expanding().mean()
    exp_std  = s.shift(1).expanding().std().fillna(1).replace(0, 1)
    return ((s - exp_mean) / exp_std).clip(-4, 4)


def _anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    df["lag1_zscore"] = df.groupby(["state", "district"])[TARGET].transform(
        lambda s: _rolling_zscore(s).shift(1)
    ).fillna(0)
    df["lag1_extreme"] = (df["lag1_zscore"].abs() > 2.5).astype(int)
    return df


def _state_features(df: pd.DataFrame) -> pd.DataFrame:
    state_yr = (
        df.groupby(["state", "financial_year"])[TARGET]
        .sum().reset_index()
        .rename(columns={TARGET: "state_total"})
    )
    state_yr["state_total_lag1"] = state_yr.groupby("state")["state_total"].shift(1)
    state_yr["state_lag1_zscore"] = state_yr.groupby("state")["state_total"].transform(
        lambda s: _rolling_zscore(s)
    )

    state_hist_mean = state_yr.groupby("state")["state_total_lag1"].transform("mean")
    state_yr["state_lag1_norm"] = (
        state_yr["state_total_lag1"] / state_hist_mean.replace(0, np.nan)
    ).clip(0, 5).fillna(1.0)

    df = df.merge(
        state_yr[["state", "financial_year", "state_lag1_zscore", "state_lag1_norm"]],
        on=["state", "financial_year"],
        how="left"
    )

    state_yr_lag = df.groupby(["state", "financial_year"])["lag1_pd"].transform("mean")
    df["relative_to_state"] = (
        df["lag1_pd"] / state_yr_lag.replace(0, np.nan)
    ).clip(0, 10).fillna(1.0)

    return df


def _temporal_flags(df: pd.DataFrame) -> pd.DataFrame:
    fy_min = df["financial_year"].min()
    df["year_trend"]      = df["financial_year"] - fy_min
    df["is_covid"]        = (df["financial_year"] == 2020).astype(int)
    df["is_post_covid"]   = (df["financial_year"] >= 2021).astype(int)
    df["is_2022_anomaly"] = (df["financial_year"] == 2022).astype(int)
    return df


def _wage_features(df: pd.DataFrame) -> pd.DataFrame:
    if "avg_wage_rate" not in df.columns:
        return df
    df["wage_yoy"] = (
        df.groupby(["state", "district"])["avg_wage_rate"]
        .pct_change(fill_method=None)
        .fillna(0)
        .clip(-0.2, 0.5)
    )
    return df


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    le_state = LabelEncoder()
    le_dist  = LabelEncoder()
    df["state_enc"]    = le_state.fit_transform(df["state"].astype(str))
    df["district_enc"] = le_dist.fit_transform(
        (df["district"] + "|" + df["state"]).astype(str)
    )
    return df


FEATURE_COLS = [
    "lag1_pd",
    "roll2_mean",
    "roll3_mean",
    "lag1_adj",
    "lag2_pd",
    "lag3_pd",
    "roll3_std",
    "state_lag1_norm",
    "relative_to_state",
    "blended_capacity",
    "lag1_vs_capacity",
    "state_lag1_zscore",
    "state_enc",
    "is_covid",
    "lag1_is_covid",
    "wage_yoy",
    "avg_wage_rate",
]
