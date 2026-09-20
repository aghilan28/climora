"""Model performance comparison table formatting and evaluation reporting."""

from typing import Dict

import pandas as pd


def build_comparison_table(metrics_summary: Dict[str, Dict[str, float]], best_criterion: str = "test_rmse") -> pd.DataFrame:
    """Build standardized comparison DataFrame across all evaluated models."""
    records = []
    for model_name, metrics in metrics_summary.items():
        records.append({
            "Model": model_name,
            "MAE": metrics.get("mae", 0.0),
            "RMSE": metrics.get("rmse", 0.0),
            "MAPE (%)": metrics.get("mape", 0.0),
            "R²": metrics.get("r2", 0.0),
            "Skill Score": metrics.get("skill_score", 0.0),
        })

    df_comp = pd.DataFrame(records)
    if not df_comp.empty and "RMSE" in df_comp.columns:
        df_comp = df_comp.sort_values("RMSE").reset_index(drop=True)

    return df_comp
