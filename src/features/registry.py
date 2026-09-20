"""Feature registry defining metadata, formulas, and leakage notes for all features."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class FeatureInfo:
    name: str
    family: str
    formula: str
    window: str
    source_col: str
    leakage_note: str


FEATURE_REGISTRY: List[FeatureInfo] = [
    # Temporal & Calendar
    FeatureInfo("month", "Calendar", "month(date)", "1", "date", "Strictly deterministic calendar feature"),
    FeatureInfo("quarter", "Calendar", "quarter(date)", "1", "date", "Strictly deterministic calendar feature"),
    FeatureInfo("month_sin", "Calendar", "sin(2 * pi * month / 12)", "1", "date", "Cyclical transformation of month"),
    FeatureInfo("month_cos", "Calendar", "cos(2 * pi * month / 12)", "1", "date", "Cyclical transformation of month"),

    # Lags
    FeatureInfo("anomaly_c_lag_1", "Lags", "y(t-1)", "1", "anomaly_c", "Trailing lag-1 anomaly; no future leakage"),
    FeatureInfo("anomaly_c_lag_3", "Lags", "y(t-3)", "3", "anomaly_c", "Trailing lag-3 anomaly; no future leakage"),
    FeatureInfo("anomaly_c_lag_6", "Lags", "y(t-6)", "6", "anomaly_c", "Trailing lag-6 anomaly; no future leakage"),
    FeatureInfo("anomaly_c_lag_12", "Lags", "y(t-12)", "12", "anomaly_c", "Trailing lag-12 annual anomaly; no future leakage"),
    FeatureInfo("anomaly_c_lag_24", "Lags", "y(t-24)", "24", "anomaly_c", "Trailing lag-24 anomaly; no future leakage"),
    FeatureInfo("anomaly_c_lag_60", "Lags", "y(t-60)", "60", "anomaly_c", "Trailing lag-60 anomaly; no future leakage"),

    # Rolling Statistics (center=False strictly)
    FeatureInfo("anomaly_c_roll_mean_12", "Rolling", "mean(y(t-12:t-1))", "12", "anomaly_c", "12-month trailing mean (center=False); no future leakage"),
    FeatureInfo("anomaly_c_roll_std_12", "Rolling", "std(y(t-12:t-1))", "12", "anomaly_c", "12-month trailing std (center=False); no future leakage"),
    FeatureInfo("anomaly_c_roll_min_12", "Rolling", "min(y(t-12:t-1))", "12", "anomaly_c", "12-month trailing min (center=False); no future leakage"),
    FeatureInfo("anomaly_c_roll_max_12", "Rolling", "max(y(t-12:t-1))", "12", "anomaly_c", "12-month trailing max (center=False); no future leakage"),

    FeatureInfo("anomaly_c_roll_mean_24", "Rolling", "mean(y(t-24:t-1))", "24", "anomaly_c", "24-month trailing mean (center=False); no future leakage"),
    FeatureInfo("anomaly_c_roll_std_24", "Rolling", "std(y(t-24:t-1))", "24", "anomaly_c", "24-month trailing std (center=False); no future leakage"),

    FeatureInfo("anomaly_c_roll_mean_60", "Rolling", "mean(y(t-60:t-1))", "60", "anomaly_c", "60-month trailing mean (center=False); no future leakage"),
    FeatureInfo("anomaly_c_roll_std_60", "Rolling", "std(y(t-60:t-1))", "60", "anomaly_c", "60-month trailing std (center=False); no future leakage"),

    # Differences & Trend
    FeatureInfo("anomaly_c_diff_1", "Trend", "y(t-1) - y(t-2)", "2", "anomaly_c", "First difference of trailing anomaly"),
    FeatureInfo("anomaly_c_diff_12", "Trend", "y(t-1) - y(t-13)", "13", "anomaly_c", "12-month YoY change of trailing anomaly"),

    # Covariates
    FeatureInfo("co2_ppm_lag_1", "Covariate", "co2(t-1)", "1", "co2_ppm", "Trailing 1-month lagged global CO2 ppm"),
    FeatureInfo("co2_ppm_diff_12", "Covariate", "co2(t-1) - co2(t-13)", "13", "co2_ppm", "12-month trailing rate of change of CO2 ppm"),
    FeatureInfo("nino34_anom_lag_1", "Covariate", "nino34(t-1)", "1", "nino34_anom", "Trailing 1-month lagged NINO3.4 SST anomaly"),
    FeatureInfo("nino34_anom_lag_6", "Covariate", "nino34(t-6)", "6", "nino34_anom", "Trailing 6-month lagged NINO3.4 SST anomaly"),
]


def get_feature_registry_dict() -> Dict[str, FeatureInfo]:
    return {f.name: f for f in FEATURE_REGISTRY}
