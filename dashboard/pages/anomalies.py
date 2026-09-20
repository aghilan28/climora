"""Anomaly Detection page module for CLIMORA AI dashboard."""

import plotly.express as px
import streamlit as st

from dashboard.state import AppState, StateStage, get_feature_df
from src.anomaly.isolation_forest import ClimateAnomalyDetector


def render_anomalies_page() -> None:
    """Render Anomaly Detection page with Isolation Forest outputs, score distributions, and top-K inspection."""
    st.title("🚨 Station-Panel & Climate Extreme Anomaly Detection")
    st.caption("Unsupervised Isolation Forest anomaly detection over multivariate feature matrix")

    if not AppState.require_stage(StateStage.ANOMALY_READY):
        return

    anomaly_res = st.session_state.get("anomaly_result")
    feat_df = get_feature_df()

    if not anomaly_res or feat_df is None:
        st.error("Anomaly detection result not available.")
        return

    res_df = anomaly_res["df"]
    meta = anomaly_res["metadata"]
    detector: ClimateAnomalyDetector = anomaly_res["detector"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Observations", f"{meta['total_observations']:,}")
    col2.metric("Normal Records", f"{meta['normal_count']:,}")
    col3.metric("Anomalies Flagged", f"{meta['anomaly_count']:,}")
    col4.metric("Anomaly Rate", f"{meta['anomaly_pct']:.2f}%")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Anomaly Timeline", "📊 Score Distribution", "🏆 Top-K Anomalies", "🧪 Sensitivity Analysis"]
    )

    with tab1:
        st.subheader("Historical Timeline of Flagged Heat-Extreme Anomalies")
        if "date" in res_df.columns and "anomaly_c" in res_df.columns:
            fig = px.scatter(
                res_df,
                x="date",
                y="anomaly_c",
                color=res_df["is_anomaly"].map({0: "Normal", 1: "Anomaly"}),
                color_discrete_map={"Normal": "#64748B", "Anomaly": "#EF4444"},
                title="Anomaly Classification Timeline (Isolation Forest)",
                labels={"date": "Date", "anomaly_c": "Anomaly (°C)", "color": "Status"},
            )
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Normalized Anomaly Score Distribution (0–1)")
        if "anomaly_score" in res_df.columns:
            fig_hist = px.histogram(
                res_df,
                x="anomaly_score",
                nbins=50,
                color="is_anomaly",
                color_discrete_map={0: "#0D9488", 1: "#EF4444"},
                title="Normalized Isolation Forest Anomaly Score Distribution",
                labels={"anomaly_score": "Normalized Anomaly Score (1 = Extreme)", "is_anomaly": "Flagged"},
            )
            fig_hist.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_hist, use_container_width=True)

    with tab3:
        st.subheader("🏆 Top-K Most Anomalous Observations")
        k = st.slider("Select K (Top Anomalous Rows)", min_value=5, max_value=50, value=10)
        top_k = res_df.sort_values("anomaly_score", ascending=False).head(k)

        display_cols = [c for c in ["date", "year", "month", "anomaly_c", "anomaly_score", "anomaly_rank"] if c in top_k.columns]
        st.dataframe(top_k[display_cols], use_container_width=True)

        with st.expander("🔍 Inspect Single Record Drivers"):
            selected_idx = st.selectbox("Select Record Index for Deep Inspection", options=top_k.index.tolist())
            selected_row = top_k.loc[selected_idx]
            st.json(selected_row.to_dict())

    with tab4:
        st.subheader("🧪 Contamination Sensitivity Analysis")
        st.markdown("Evaluate sensitivity across contamination thresholds $\\in \\{0.01, 0.05, 0.10\\}$.")
        feature_cols = [c for c in feat_df.columns if c not in ["date", "year", "month", "anomaly_c"]]
        sens_df = detector.compute_sensitivity_table(feat_df, feature_cols=feature_cols)
        st.dataframe(sens_df, use_container_width=True)
