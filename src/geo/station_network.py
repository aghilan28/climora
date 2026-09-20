"""Indian Climate Station Network module for spatial climate analysis."""

import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
from config.settings import settings
from src.utils.logging import logger

INDIAN_STATION_GEOGRAPHY: List[Dict[str, Any]] = [
    {"station": "Chennai", "lat": 13.0827, "lon": 80.2707, "elevation": 6.0, "state": "Tamil Nadu"},
    {"station": "Delhi", "lat": 28.6139, "lon": 77.2090, "elevation": 216.0, "state": "Delhi"},
    {"station": "Mumbai", "lat": 19.0760, "lon": 72.8777, "elevation": 14.0, "state": "Maharashtra"},
    {"station": "Kolkata", "lat": 22.5726, "lon": 88.3639, "elevation": 9.0, "state": "West Bengal"},
    {"station": "Bengaluru", "lat": 12.9716, "lon": 77.5946, "elevation": 920.0, "state": "Karnataka"},
    {"station": "Hyderabad", "lat": 17.3850, "lon": 78.4867, "elevation": 542.0, "state": "Telangana"},
    {"station": "Ahmedabad", "lat": 23.0225, "lon": 72.5714, "elevation": 53.0, "state": "Gujarat"},
    {"station": "Jaipur", "lat": 26.9124, "lon": 75.7873, "elevation": 431.0, "state": "Rajasthan"},
    {"station": "Pune", "lat": 18.5204, "lon": 73.8567, "elevation": 560.0, "state": "Maharashtra"},
    {"station": "Lucknow", "lat": 26.8467, "lon": 80.9462, "elevation": 123.0, "state": "Uttar Pradesh"},
    {"station": "Patna", "lat": 25.5941, "lon": 85.1376, "elevation": 53.0, "state": "Bihar"},
    {"station": "Guwahati", "lat": 26.1445, "lon": 91.7362, "elevation": 55.0, "state": "Assam"},
]


def load_station_amplification_factors() -> Dict[str, float]:
    """Load empirically computed station amplification factors from data/processed artifact."""
    factor_path = settings.base_dir / "data" / "processed" / "station_amplification.json"
    if not factor_path.exists():
        logger.warning("Station amplification artifact missing at %s", factor_path)
        raise FileNotFoundError(f"Empirical station amplification artifact missing: {factor_path}")

    data = json.loads(factor_path.read_text(encoding="utf-8"))
    return {k: float(v) for k, v in data.items()}


def get_indian_station_network() -> pd.DataFrame:
    """Return DataFrame of N=12 Indian climate stations with empirical amplification factors."""
    df = pd.DataFrame(INDIAN_STATION_GEOGRAPHY)
    factors = load_station_amplification_factors()

    df["region_factor"] = df["station"].map(factors).fillna(1.0)
    return df
