"""Unified feature engineering pipeline."""

import pandas as pd

from src.features.lags import add_lag_features
from src.features.rolling import add_rolling_features
from src.features.temporal import add_temporal_features
from src.features.trend import add_trend_features


def build_feature_matrix(
    df_gistemp: pd.DataFrame,
    df_co2: pd.DataFrame | None = None,
    df_nino: pd.DataFrame | None = None,
    target_col: str = "anomaly_c",
) -> pd.DataFrame:
    """Build standardized leak-free feature matrix from GISTEMP and covariates."""
    df_feat = df_gistemp.copy()
    df_feat = df_feat.sort_values("date").reset_index(drop=True)

    # 1. Temporal & Calendar
    df_feat = add_temporal_features(df_feat, date_col="date")

    # 2. Anomaly Lags
    df_feat = add_lag_features(df_feat, col=target_col, lags=[1, 3, 6, 12, 24, 60])

    # 3. Anomaly Rolling Statistics
    df_feat = add_rolling_features(
        df_feat, col=target_col, windows=[12, 24, 60], stats=["mean", "std", "min", "max"]
    )

    # 4. Anomaly Differences & Trend
    df_feat = add_trend_features(df_feat, col=target_col, diffs=[1, 12])

    # 5. Merge CO2 Covariate
    if df_co2 is not None and not df_co2.empty and "date" in df_co2.columns:
        co2_cols = ["date"]
        if "co2_ppm" in df_co2.columns:
            co2_cols.append("co2_ppm")

        df_co2_sub = df_co2[co2_cols].copy()
        df_feat = pd.merge_asof(df_feat, df_co2_sub, on="date", direction="backward")

        if "co2_ppm" in df_feat.columns:
            df_feat["co2_ppm_lag_1"] = df_feat["co2_ppm"].shift(1)
            df_feat["co2_ppm_diff_12"] = df_feat["co2_ppm"].shift(1).diff(12)

    # 6. Merge Nino Covariate
    if df_nino is not None and not df_nino.empty and "date" in df_nino.columns:
        nino_cols = ["date"]
        if "nino34_anom" in df_nino.columns:
            nino_cols.append("nino34_anom")

        df_nino_sub = df_nino[nino_cols].copy()
        df_feat = pd.merge_asof(df_feat, df_nino_sub, on="date", direction="backward")

        if "nino34_anom" in df_feat.columns:
            df_feat["nino34_anom_lag_1"] = df_feat["nino34_anom"].shift(1)
            df_feat["nino34_anom_lag_6"] = df_feat["nino34_anom"].shift(6)

    # Clean up unneeded raw covariate columns if merged
    cols_to_drop = [c for c in ["co2_ppm", "nino34_anom"] if c in df_feat.columns]
    if cols_to_drop:
        df_feat = df_feat.drop(columns=cols_to_drop)

    # 7. Target Head for 12-Month-Ahead Forecast (h=12)
    df_feat["anomaly_c_h12"] = df_feat[target_col].shift(-12)

    return df_feat.sort_values("date").reset_index(drop=True)
