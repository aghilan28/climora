"""Correlation matrix analysis and heatmap generation."""

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_correlation_matrix(
    df: pd.DataFrame, method: str = "pearson"
) -> Optional[go.Figure]:
    """Generate correlation heatmap for numeric features."""
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty or numeric_df.shape[1] < 2:
        return None

    corr = numeric_df.corr(method=method)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title=f"Feature Correlation Matrix ({method.title()})",
    )
    fig.update_layout(template="plotly_dark", height=450)
    return fig
