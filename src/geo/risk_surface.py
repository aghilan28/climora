"""Risk Surface computation module for multi-station spatial panel."""

from typing import Dict, Optional

import pandas as pd

from src.geo.station_network import get_indian_station_network
from src.risk.methodology import compute_methodology_b_thresholds
from src.risk.scoring import compute_risk_score


def compute_station_risk_surface(
    global_anomaly: float,
    methodology: str = "A",
    thresholds: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Compute station-specific climate risk scores and bands based on dynamic anomaly regional scaling."""
    df_stations = get_indian_station_network().copy()

    if thresholds is None:
        if methodology.upper() == "B":
            thresholds = compute_methodology_b_thresholds()
        else:
            # Load processed amplification or train set default
            from dashboard.state import AppState
            thresholds = AppState.get_risk_thresholds("A")

    risk_scores = []
    risk_bands = []
    station_anomalies = []

    for _, row in df_stations.iterrows():
        factor = float(row.get("region_factor", 1.0))
        station_anom = float(global_anomaly) * factor
        score, band = compute_risk_score(station_anom, thresholds=thresholds, methodology=methodology)

        station_anomalies.append(round(station_anom, 2))
        risk_scores.append(score)
        risk_bands.append(band.value)

    df_stations["station_anomaly_c"] = station_anomalies
    df_stations["risk_score"] = risk_scores
    df_stations["band"] = risk_bands

    return df_stations
