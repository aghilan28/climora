"""Risk bands enumeration and color mapping."""

from enum import Enum


class RiskBand(str, Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    EXTREME = "Extreme"


RISK_COLORS = {
    RiskBand.LOW: "#10b981",       # Emerald Green
    RiskBand.MODERATE: "#f59e0b",  # Amber Yellow
    RiskBand.HIGH: "#f97316",      # Orange
    RiskBand.EXTREME: "#ef4444",   # Red
}
