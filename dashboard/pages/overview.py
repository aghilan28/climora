"""Overview page module for CLIMORA AI dashboard."""

import plotly.express as px
import streamlit as st

from dashboard.state import AppState, StateStage, get_clean_gistemp, get_feature_df, get_models
from src.risk.scoring import compute_risk_score


def render_overview_page() -> None:
    """Render Overview page with live state KPIs and Climate Intelligence Summary."""
    st.title("🌐 Climate Intelligence Overview")
    st.caption("Real-time summary derived from NASA GISTEMP v4 & ERA5 reanalysis")

    if not AppState.require_stage(StateStage.LOADED):
        return

    df = get_clean_gistemp()
    feat_df = get_feature_df()
    models = get_models()

    if df is None or len(df) == 0:
        st.error("No dataset available.")
        return

    # Live KPIs
    total_records = len(df)
    variable_count = len(df.columns)
    min_date = df["date"].min().strftime("%Y-%m") if "date" in df.columns else "N/A"
    max_date = df["date"].max().strftime("%Y-%m") if "date" in df.columns else "N/A"
    latest_anomaly = float(df["anomaly_c"].iloc[-1]) if "anomaly_c" in df.columns else 0.0

    # Risk Calculation from latest row
    thresholds_a = AppState.get_risk_thresholds("A")
    risk_score, risk_band = compute_risk_score(latest_anomaly, thresholds=thresholds_a)

    # Anomaly Count from session state if ready
    anomaly_res = st.session_state.get("anomaly_result", {})
    if anomaly_res and "metadata" in anomaly_res:
        anom_count = anomaly_res["metadata"]["anomaly_count"]
        anom_pct = anomaly_res["metadata"]["anomaly_pct"]
    else:
        anom_count = 0
        anom_pct = 0.0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Records", f"{total_records:,}")
    col2.metric("Coverage", f"{min_date} → {max_date}")
    col3.metric("Latest Anomaly", f"{latest_anomaly:+.2f} °C")
    col4.metric("Risk Level", f"{risk_band.value} ({risk_score:.0f}/100)")
    col5.metric("Anomalies Flagged", f"{anom_count} ({anom_pct:.1f}%)")

    st.divider()

    # Climate Intelligence Summary Section
    st.subheader("💡 Climate Intelligence Summary")
    st.markdown(
        f"""
        - **What's Happening**: The latest global mean surface temperature anomaly is **{latest_anomaly:+.2f} °C** (relative to 1951–1980 baseline).
        - **Data Scope**: **{variable_count}** columns loaded across **{total_records:,}** monthly records.
        - **Model Status**: **{len(models)}** model architecture(s) loaded ({', '.join(models.keys()) if models else 'None trained'}).
        - **Current Risk Classification**: **{risk_band.value}** (Analytical Index Score: **{risk_score:.0f}/100**).
        - **Heat-Extreme Anomalies**: **{anom_count}** temporal observations flagged as statistically anomalous ({anom_pct:.1f}% of total).
        - **Data Provenance**: NASA GISTEMP v4 monthly global reanalysis backbone spanning 1880 to present.
        """
    )

    # Interactive Overview Chart
    st.subheader("📈 Historical Global Temperature Anomaly Timeline (1880–Present)")
    if "date" in df.columns and "anomaly_c" in df.columns:
        fig = px.line(
            df,
            x="date",
            y="anomaly_c",
            title="Global Mean Surface Temperature Anomaly (°C vs 1951–1980)",
            labels={"date": "Date", "anomaly_c": "Anomaly (°C)"},
            color_discrete_sequence=["#0D9488"],
        )
        fig.add_hline(y=0.0, line_dash="dash", line_color="#94A3B8")
        if feat_df is not None and "anomaly_c_roll12_mean" in feat_df.columns:
            fig.add_scatter(
                x=feat_df["date"],
                y=feat_df["anomaly_c_roll12_mean"],
                mode="lines",
                name="12-Month Trailing Mean",
                line=dict(color="#F59E0B", width=2),
            )
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Model status chips
    st.subheader("🤖 Model Registry Status")
    mcols = st.columns(3)
    for i, m_name in enumerate(["xgboost", "lightgbm", "lstm"]):
        with mcols[i]:
            if m_name in models:
                st.success(f"✅ **{m_name.upper()}**: Loaded & Ready")
                metrics = models[m_name].get("metrics", {})
                if metrics:
                    mae = metrics.get("test_mae", "N/A")
                    st.caption(f"Test MAE: {mae}")
            else:
                st.warning(f"⚠️ **{m_name.upper()}**: Not Trained")
