"""Predictions page module for CLIMORA AI dashboard."""

import pandas as pd
import pydeck as pdk
import streamlit as st

from dashboard.state import AppState, StateStage, get_feature_df, get_models
from src.risk.scoring import compute_risk_score


def render_predictions_page() -> None:
    """Render Predictions page with real model inference, uncertainty intervals, risk, and pydeck map."""
    st.title("🎯 Model Prediction & Climate Forecasting")
    st.caption("Generate model-grounded temperature anomaly predictions with quantified uncertainty")

    if not AppState.require_stage(StateStage.MODELS_TRAINED):
        return

    models = get_models()
    feat_df = get_feature_df()

    if not models or feat_df is None:
        st.error("Models or feature matrix not loaded.")
        return

    col_select, col_mode = st.columns(2)
    with col_select:
        selected_model_name = st.selectbox("Select Model Architecture", options=list(models.keys()))
    with col_mode:
        input_mode = st.radio("Input Mode", ["Historical Record Selection", "Custom Parameter Inputs"], horizontal=True)

    model_entry = models[selected_model_name]
    model_obj = model_entry["model"]

    if input_mode == "Historical Record Selection":
        st.subheader("Select Historical Observation Row")
        dates = feat_df["date"].dt.strftime("%Y-%m").tolist()
        selected_date = st.selectbox("Select Date", options=dates, index=len(dates) - 1)
        row_idx = dates.index(selected_date)
        input_row = feat_df.iloc[[row_idx]].copy()
        actual_val = float(input_row["anomaly_c"].values[0]) if "anomaly_c" in input_row.columns else None
    else:
        st.subheader("Custom Input Parameters")
        latest_row = feat_df.iloc[-1].copy()
        lag1 = st.slider("Lag-1 Anomaly (°C)", min_value=-2.0, max_value=3.0, value=float(latest_row.get("anomaly_c_lag_1", 0.5)), step=0.05)
        lag12 = st.slider("Lag-12 Anomaly (°C)", min_value=-2.0, max_value=3.0, value=float(latest_row.get("anomaly_c_lag_12", 0.5)), step=0.05)
        co2_val = st.slider("CO₂ Concentration (ppm)", min_value=300.0, max_value=450.0, value=float(latest_row.get("co2_average", 420.0)), step=1.0)
        nino_val = st.slider("Niño3.4 SST Anomaly (°C)", min_value=-3.0, max_value=3.0, value=float(latest_row.get("nino34_anom", 0.0)), step=0.1)

        input_row = feat_df.iloc[[-1]].copy()
        if "anomaly_c_lag_1" in input_row.columns:
            input_row["anomaly_c_lag_1"] = lag1
        if "anomaly_c_lag_12" in input_row.columns:
            input_row["anomaly_c_lag_12"] = lag12
        if "co2_average" in input_row.columns:
            input_row["co2_average"] = co2_val
        if "nino34_anom" in input_row.columns:
            input_row["nino34_anom"] = nino_val
        actual_val = None

    if st.button("🔮 Run Model Prediction", type="primary", use_container_width=True):
        try:
            preds = model_obj.predict(input_row)
            pred_val = float(preds[0])

            # Real Uncertainty estimation (std dev estimate based on model residuals/variance)
            metrics = model_entry.get("metrics", {})
            rmse = metrics.get("test_rmse", 0.15)
            ci_lower = pred_val - 1.96 * rmse
            ci_upper = pred_val + 1.96 * rmse

            # Risk Score computation
            risk_score, risk_band = compute_risk_score(pred_val)

            st.divider()
            st.subheader("📌 Prediction Output & Risk Assessment")
            res1, res2, res3, res4 = st.columns(4)
            res1.metric("Predicted Anomaly", f"{pred_val:+.2f} °C")
            res2.metric("95% CI Uncertainty", f"[{ci_lower:+.2f}, {ci_upper:+.2f}] °C")
            res3.metric("Risk Score", f"{risk_score:.0f}/100")
            res4.metric("Risk Band", risk_band.value)

            if actual_val is not None:
                st.info(f"Actual Historical Anomaly for {selected_date}: **{actual_val:+.2f} °C** (Abs Error: {abs(pred_val - actual_val):.2f} °C)")

            st.divider()
            st.subheader("🗺️ Global Station Risk Map (pydeck Visualization)")
            # Pydeck layer with station points
            station_data = pd.DataFrame(
                {
                    "station": ["Chennai", "Delhi", "Mumbai", "Kolkata", "Bengaluru", "Hyderabad"],
                    "lat": [13.0827, 28.6139, 19.0760, 22.5726, 12.9716, 17.3850],
                    "lon": [80.2707, 77.2090, 72.8777, 88.3639, 77.5946, 78.4867],
                    "risk_score": [risk_score] * 6,
                }
            )

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=station_data,
                get_position=["lon", "lat"],
                get_color="[255, 100, 100, 200]",
                get_radius=100000,
                pickable=True,
            )

            view_state = pdk.ViewState(latitude=20.5937, longitude=78.9629, zoom=3, pitch=0)
            r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{station}: Risk Score {risk_score}"})
            st.pydeck_chart(r)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
