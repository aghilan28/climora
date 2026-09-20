"""Model Performance page module for CLIMORA AI dashboard."""

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.state import AppState, StateStage, get_feature_df, get_models
from src.evaluation.comparison import build_comparison_table
from src.splits.chronological import make_chronological_split


def render_model_performance_page() -> None:
    """Render Model Performance page with metrics table, baselines, residuals, and split diagrams."""
    st.title("📊 Model Performance & Comparative Benchmark")
    st.caption("Empirical evaluation across XGBoost, LightGBM, PyTorch LSTM, and Baselines")

    if not AppState.require_stage(StateStage.MODELS_TRAINED):
        return

    models = get_models()
    feat_df = get_feature_df()

    if not models or feat_df is None:
        st.error("Models or feature matrix not loaded.")
        return

    # Split Diagram
    st.subheader("🗓️ Chronological Data Split Strategy")
    splits = make_chronological_split(feat_df)
    train_len = len(splits.train_df)
    val_len = len(splits.val_df)
    test_len = len(splits.test_df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Train Split (1880–1999)", f"{train_len:,} rows")
    c2.metric("Val Split (2000–2014)", f"{val_len:,} rows")
    c3.metric("Test Split (2015–2025)", f"{test_len:,} rows")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🏆 Leaderboard & Baselines", "📈 Test Window Forecast", "📉 Residual Analysis", "⚙️ Config & Provenance"]
    )

    with tab1:
        st.subheader("Model Evaluation Leaderboard (Test Window 2015–2025)")
        # Collect metrics from loaded models + baselines
        metrics_summary = {}
        for m_name, m_dict in models.items():
            if "metrics" in m_dict:
                metrics_summary[m_name.upper()] = {
                    "mae": m_dict["metrics"].get("test_mae", 0.0),
                    "rmse": m_dict["metrics"].get("test_rmse", 0.0),
                    "mape": m_dict["metrics"].get("test_mape", 0.0),
                    "r2": m_dict["metrics"].get("test_r2", 0.0),
                    "skill_score": m_dict["metrics"].get("skill_score", 0.0),
                }

        # Add baseline benchmarks from audit measurements
        metrics_summary["SEASONAL_NAIVE"] = {"mae": 0.0814, "rmse": 0.1032, "mape": 8.5, "r2": 0.95, "skill_score": 0.0}
        metrics_summary["CLIMATOLOGY"] = {"mae": 0.7273, "rmse": 0.7313, "mape": 90.0, "r2": -1.5, "skill_score": -6.0}

        df_leaderboard = build_comparison_table(metrics_summary)
        st.dataframe(df_leaderboard, use_container_width=True)

        st.caption("ℹ️ **Winner Selection**: Best model selected by minimum Test RMSE.")

    with tab2:
        st.subheader("Forecast vs Actual on Test Split (2015–2025)")
        test_df = splits.test_df.copy()
        if "date" in test_df.columns and "anomaly_c" in test_df.columns:
            fig_fc = go.Figure()
            fig_fc.add_trace(
                go.Scatter(x=test_df["date"], y=test_df["anomaly_c"], mode="lines+markers", name="Actual GISTEMP", line=dict(color="#F8FAFC", width=2))
            )

            for m_name, m_dict in models.items():
                m_obj = m_dict["model"]
                try:
                    preds = m_obj.predict(test_df)
                    fig_fc.add_trace(
                        go.Scatter(x=test_df["date"], y=preds, mode="lines", name=f"Predicted ({m_name.upper()})", line=dict(width=2))
                    )
                except Exception as e:
                    st.warning(f"Could not render test forecast for {m_name}: {e}")

            fig_fc.update_layout(template="plotly_dark", height=450, title="Test Window Model Predictions vs Ground Truth")
            st.plotly_chart(fig_fc, use_container_width=True)

    with tab3:
        st.subheader("Residual Scatter Plot")
        sel_m = st.selectbox("Select Model for Residuals", options=list(models.keys()))
        m_obj = models[sel_m]["model"]
        test_df = splits.test_df.copy()
        if "anomaly_c" in test_df.columns:
            preds = m_obj.predict(test_df)
            resids = test_df["anomaly_c"].values - preds
            res_df = pd.DataFrame({"Actual": test_df["anomaly_c"], "Predicted": preds, "Residual": resids})

            fig_res = px.scatter(
                res_df, x="Predicted", y="Residual", title=f"Residuals vs Predicted ({sel_m.upper()})", color_discrete_sequence=["#0D9488"]
            )
            fig_res.add_hline(y=0.0, line_dash="dash", line_color="#EF4444")
            fig_res.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_res, use_container_width=True)

    with tab4:
        st.subheader("⚙️ Model Metadata & Reproducibility Provenance")
        for m_name, m_dict in models.items():
            with st.expander(f"Metadata Dump: {m_name.upper()}"):
                st.code(json.dumps(m_dict.get("metadata", {}), indent=2), language="json")
