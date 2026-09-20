"""Long-term linear and rolling trend line visualizations."""

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_trend_line(
    df: pd.DataFrame,
    date_col: str = "date",
    val_col: str = "anomaly_c",
    rolling_window: int = 12,
) -> Optional[go.Figure]:
    """Plot long-term temperature anomaly time-series with rolling mean and OLS trend."""
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' missing from DataFrame")
    if val_col not in df.columns:
        raise ValueError(f"Value column '{val_col}' missing from DataFrame")

    clean_df = df.dropna(subset=[date_col, val_col]).sort_values(date_col).copy()
    if clean_df.empty:
        return None

    clean_df["rolling_mean"] = clean_df[val_col].rolling(window=rolling_window, center=False).mean()

    # Compute OLS linear trend line
    x_num = (clean_df[date_col] - clean_df[date_col].min()).dt.days.values
    y_vals = clean_df[val_col].values
    valid_mask = ~np.isnan(y_vals)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=clean_df[date_col],
            y=clean_df[val_col],
            mode="lines",
            name="Monthly Anomaly",
            line=dict(color="#38bdf8", width=1.0),
            opacity=0.6,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=clean_df[date_col],
            y=clean_df["rolling_mean"],
            mode="lines",
            name=f"{rolling_window}-Month Rolling Mean",
            line=dict(color="#f59e0b", width=2.0),
        )
    )

    if np.sum(valid_mask) > 1:
        poly = np.polyfit(x_num[valid_mask], y_vals[valid_mask], 1)
        trend_vals = np.polyval(poly, x_num)
        fig.add_trace(
            go.Scatter(
                x=clean_df[date_col],
                y=trend_vals,
                mode="lines",
                name="Linear OLS Trend",
                line=dict(color="#ef4444", width=2.0, dash="dash"),
            )
        )

    fig.update_layout(
        title=f"Long-Term Climate Trend Analysis ({val_col})",
        xaxis_title="Date",
        yaxis_title="Anomaly (°C)",
        template="plotly_dark",
        height=450,
        hovermode="x unified",
    )
    return fig
