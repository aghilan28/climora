"""Seasonality analysis with monthly boxplots and seasonal amplitude computation."""

from typing import Dict, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def compute_seasonality_metrics(df: pd.DataFrame, val_col: str = "anomaly_c") -> Dict[str, float]:
    """Compute monthly seasonal means and peak-to-trough seasonal amplitude."""
    if "month" not in df.columns or val_col not in df.columns:
        raise ValueError(f"Required columns 'month' and '{val_col}' must exist")

    monthly_means = df.groupby("month")[val_col].mean()
    amplitude = float(monthly_means.max() - monthly_means.min())
    return {
        "seasonal_amplitude": amplitude,
        "max_month": int(monthly_means.idxmax()),
        "min_month": int(monthly_means.idxmin()),
    }


def plot_seasonality_boxplot(
    df: pd.DataFrame, val_col: str = "anomaly_c"
) -> Tuple[go.Figure, Dict[str, float]]:
    """Generate monthly seasonality boxplot and compute seasonal metrics."""
    if "month" not in df.columns or val_col not in df.columns:
        raise ValueError(f"Columns 'month' and '{val_col}' must be present in DataFrame")

    metrics = compute_seasonality_metrics(df, val_col=val_col)

    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }
    plot_df = df.copy()
    plot_df["month_name"] = plot_df["month"].map(month_names)

    fig = px.box(
        plot_df,
        x="month_name",
        y=val_col,
        category_orders={"month_name": list(month_names.values())},
        title=f"Monthly Seasonality Distribution (Amplitude: {metrics['seasonal_amplitude']:.2f} °C)",
        color_discrete_sequence=["#a855f7"],
    )
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Month",
        yaxis_title="Anomaly (°C)",
        height=400,
    )
    return fig, metrics
