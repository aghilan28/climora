"""Distribution plot generators."""

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_distribution(
    df: pd.DataFrame, col: str = "anomaly_c", title: Optional[str] = None
) -> Optional[go.Figure]:
    """Generate a Plotly histogram/KDE distribution chart for a numeric column."""
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame columns: {list(df.columns)}")

    series = df[col].dropna()
    if series.empty:
        return None

    fig = px.histogram(
        df,
        x=col,
        nbins=40,
        marginal="box",
        title=title or f"Distribution of {col}",
        color_discrete_sequence=["#38bdf8"],
    )
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=40, r=40, t=50, b=40))
    return fig
