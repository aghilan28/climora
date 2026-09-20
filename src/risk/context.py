"""Historical risk percentile and recurrence context helper."""

from typing import Any, Dict

import numpy as np
import pandas as pd


def compute_historical_context(
    anomaly: float, history_series: pd.Series
) -> Dict[str, Any]:
    """Compute historical percentile rank and exceedance recurrence for an anomaly."""
    clean = history_series.dropna()
    if clean.empty:
        return {"percentile": 50.0, "exceedance_pct": 50.0, "rank_in_history": 1, "total_history": 0}

    pct = float(stats_percentile_of_score(clean.values, anomaly))
    exceed_count = int((clean >= anomaly).sum())
    exceed_pct = float((exceed_count / len(clean)) * 100.0)

    return {
        "percentile": round(pct, 2),
        "exceedance_pct": round(exceed_pct, 2),
        "exceed_count": exceed_count,
        "total_history": len(clean),
    }


def stats_percentile_of_score(a: np.ndarray, score: float) -> float:
    """Compute percentile rank of score in array a."""
    if len(a) == 0:
        return 50.0
    count_below = np.sum(a < score)
    count_equal = np.sum(a == score)
    return float((count_below + 0.5 * count_equal) / len(a) * 100.0)
