"""Indian Climate Station Network module for spatial climate analysis."""

from typing import Any, Dict, List

import pandas as pd

INDIAN_STATIONS: List[Dict[str, Any]] = [
    {"station": "Chennai", "lat": 13.0827, "lon": 80.2707, "elevation": 6.0, "state": "Tamil Nadu", "region_factor": 1.05},
    {"station": "Delhi", "lat": 28.6139, "lon": 77.2090, "elevation": 216.0, "state": "Delhi", "region_factor": 1.20},
    {"station": "Mumbai", "lat": 19.0760, "lon": 72.8777, "elevation": 14.0, "state": "Maharashtra", "region_factor": 1.08},
    {"station": "Kolkata", "lat": 22.5726, "lon": 88.3639, "elevation": 9.0, "state": "West Bengal", "region_factor": 1.10},
    {"station": "Bengaluru", "lat": 12.9716, "lon": 77.5946, "elevation": 920.0, "state": "Karnataka", "region_factor": 0.95},
    {"station": "Hyderabad", "lat": 17.3850, "lon": 78.4867, "elevation": 542.0, "state": "Telangana", "region_factor": 1.02},
    {"station": "Ahmedabad", "lat": 23.0225, "lon": 72.5714, "elevation": 53.0, "state": "Gujarat", "region_factor": 1.15},
    {"station": "Jaipur", "lat": 26.9124, "lon": 75.7873, "elevation": 431.0, "state": "Rajasthan", "region_factor": 1.18},
    {"station": "Pune", "lat": 18.5204, "lon": 73.8567, "elevation": 560.0, "state": "Maharashtra", "region_factor": 0.98},
    {"station": "Lucknow", "lat": 26.8467, "lon": 80.9462, "elevation": 123.0, "state": "Uttar Pradesh", "region_factor": 1.12},
    {"station": "Patna", "lat": 25.5941, "lon": 85.1376, "elevation": 53.0, "state": "Bihar", "region_factor": 1.14},
    {"station": "Guwahati", "lat": 26.1445, "lon": 91.7362, "elevation": 55.0, "state": "Assam", "region_factor": 1.01},
]


def get_indian_station_network() -> pd.DataFrame:
    """Return DataFrame of N=12 Indian climate stations with geographic metadata."""
    df = pd.DataFrame(INDIAN_STATIONS)
    return df
