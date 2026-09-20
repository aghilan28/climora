"""Climate EDA & Analysis page module for CLIMORA AI dashboard."""

import streamlit as st

from dashboard.state import AppState, StateStage, get_clean_gistemp, get_feature_df
from src.eda.correlation import plot_correlation_matrix
from src.eda.distributions import plot_distribution
from src.eda.seasonality import compute_seasonality_metrics, plot_seasonality_boxplot
from src.eda.trend import plot_trend_line


def render_climate_analysis_page() -> None:
    """Render Climate Analysis page with interactive EDA charts."""
    st.title("🌡️ Climate Trend & Seasonality Analysis")
    st.caption("Exploratory climate analysis over historical observation backbone")

    if not AppState.require_stage(StateStage.LOADED):
        return

    clean_df = get_clean_gistemp()
    feat_df = get_feature_df()

    if clean_df is None or len(clean_df) == 0:
        st.error("No dataset available for analysis.")
        return

    # Date Window Filter
    st.subheader("🗓️ Temporal Filter")
    if "date" in clean_df.columns:
        min_date = clean_df["date"].min().date()
        max_date = clean_df["date"].max().date()
        date_range = st.slider(
            "Select Analysis Window",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM",
        )
        mask = (clean_df["date"].dt.date >= date_range[0]) & (clean_df["date"].dt.date <= date_range[1])
        filtered_df = clean_df.loc[mask].copy()
    else:
        filtered_df = clean_df.copy()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Trend Line & OLS", "🍂 Seasonality & Amplitude", "📊 Distributions", "🔗 Feature Correlations"]
    )

    with tab1:
        st.subheader("Long-Term Temperature Trend & OLS Regression")
        fig_trend = plot_trend_line(filtered_df, date_col="date", val_col="anomaly_c", rolling_window=12)
        if fig_trend is not None:
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Insufficient data to render trend line.")

    with tab2:
        st.subheader("Monthly Seasonality & Seasonal Amplitude")
        if "month" in filtered_df.columns and "anomaly_c" in filtered_df.columns:
            metrics = compute_seasonality_metrics(filtered_df, val_col="anomaly_c")
            col1, col2, col3 = st.columns(3)
            col1.metric("Peak-to-Trough Amplitude", f"{metrics['seasonal_amplitude']:.3f} °C")
            col2.metric("Warmest Anomaly Month", f"Month {metrics['max_month']}")
            col3.metric("Coolest Anomaly Month", f"Month {metrics['min_month']}")

            fig_seas, _ = plot_seasonality_boxplot(filtered_df, val_col="anomaly_c")
            if fig_seas is not None:
                st.plotly_chart(fig_seas, use_container_width=True)
        else:
            st.info("Month column missing for seasonality analysis.")

    with tab3:
        st.subheader("Variable Distribution Plots")
        target_var = st.selectbox("Select Variable for Distribution Plot", options=["anomaly_c"])
        if target_var in filtered_df.columns:
            fig_dist = plot_distribution(filtered_df, col=target_var, title=f"Distribution of {target_var}")
            if fig_dist is not None:
                st.plotly_chart(fig_dist, use_container_width=True)

    with tab4:
        st.subheader("Feature Correlation Heatmap")
        target_df = feat_df if feat_df is not None else filtered_df
        method = st.radio("Correlation Metric", ["pearson", "spearman"], horizontal=True)
        fig_corr = plot_correlation_matrix(target_df, method=method)
        if fig_corr is not None:
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Insufficient numeric columns for correlation matrix.")
