"""Risk Analysis page module for CLIMORA AI dashboard."""

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

from dashboard.state import AppState, StateStage, get_clean_gistemp
from src.risk.methodology import compute_methodology_a_thresholds, compute_methodology_b_thresholds
from src.risk.scoring import compute_risk_score


def render_risk_page() -> None:
    """Render Risk Analysis page with dual methodology selection, map, and mandatory disclaimers."""
    st.title("⚡ Climate Risk Classification & Indexing")
    st.caption("Dual-methodology analytical climate risk scoring system")

    if not AppState.require_stage(StateStage.LOADED):
        return

    clean_df = get_clean_gistemp()

    if clean_df is None or len(clean_df) == 0:
        st.error("No dataset available.")
        return

    # Mandatory Notice
    st.info(
        "ℹ️ **Analytical Notice**: This is an internally consistent analytical index derived from "
        "the model and dataset. It is not a certified or operational climate risk assessment."
    )

    col1, col2 = st.columns(2)
    with col1:
        methodology = st.radio(
            "Select Risk Methodology",
            ["Method A — Empirical Distribution Quantiles", "Method B — Literature Pre-Industrial Baseline Target (+1.5°C/+2.0°C)"],
            index=0,
        )
        meth_code = "A" if "Method A" in methodology else "B"
    with col2:
        latest_anomaly = float(clean_df["anomaly_c"].iloc[-1]) if "anomaly_c" in clean_df.columns else 0.0
        thresholds_selected = AppState.get_risk_thresholds(meth_code)
        risk_score, risk_band = compute_risk_score(latest_anomaly, thresholds=thresholds_selected, methodology=meth_code)
        st.metric("Current Anomaly", f"{latest_anomaly:+.2f} °C")
        st.metric("Risk Level", f"{risk_band.value} ({risk_score:.0f}/100)")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📊 Thresholds & Exceedance Trend", "🗺️ Regional Risk Map", "📚 Methodology Details"])

    with tab1:
        st.subheader("Threshold Breakdown")
        train_anom = clean_df.loc[clean_df["year"] <= 1999, "anomaly_c"].dropna() if "year" in clean_df.columns else clean_df["anomaly_c"].dropna()

        if meth_code == "A":
            t_dict = compute_methodology_a_thresholds(train_anom)
        else:
            t_dict = compute_methodology_b_thresholds()

        t_df = pd.DataFrame([{"Threshold Level": k, "Anomaly Cutoff (°C)": v} for k, v in t_dict.items()])
        st.dataframe(t_df, use_container_width=True)

        st.subheader("Historical Exceedance Timeline")
        if "date" in clean_df.columns and "anomaly_c" in clean_df.columns:
            fig = px.line(
                clean_df,
                x="date",
                y="anomaly_c",
                title="Historical Temperature Anomalies with Risk Cutoffs",
                labels={"date": "Date", "anomaly_c": "Anomaly (°C)"},
            )
            for k, v in t_dict.items():
                fig.add_hline(y=v, line_dash="dot", annotation_text=k)
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("🗺️ Regional Risk Map & Global Boundaries")
        from src.geo.choropleth import get_country_polygons
        from src.geo.frames import build_temporal_map_frames
        from src.geo.risk_surface import compute_station_risk_surface

        # 1. GeoJSON Choropleth Country Layer
        country_geojson = get_country_polygons()
        geojson_layer = pdk.Layer(
            "GeoJsonLayer",
            country_geojson,
            opacity=0.4,
            stroked=True,
            filled=True,
            extruded=False,
            wireframe=True,
            get_fill_color="properties.fill_color",
            get_line_color="[200, 200, 200, 120]",
            get_line_width=5000,
            pickable=True,
        )

        # 2. Historical Temporal Scrubber
        st.markdown("#### ⏱️ Historical Map Scrubber")
        map_frames = build_temporal_map_frames(clean_df, sample_interval=12)
        if map_frames:
            frame_dates = [f["date"] for f in map_frames]
            selected_date = st.select_slider("Select Historical Snapshot Date", options=frame_dates, value=frame_dates[-1])
            selected_frame = next(f for f in map_frames if f["date"] == selected_date)
            st.caption(f"Historical Snapshot: **{selected_date}** | Global Anomaly: **{selected_frame['global_anomaly_c']:+.2f} °C**")
            station_data = pd.DataFrame(selected_frame["stations"])
        else:
            station_data = compute_station_risk_surface(latest_anomaly, methodology=meth_code)

        station_layer = pdk.Layer(
            "ColumnLayer",
            data=station_data,
            get_position=["lon", "lat"],
            get_elevation="risk_score",
            elevation_scale=1000,
            radius=40000,
            get_fill_color="[239, 68, 68, 200]",
            pickable=True,
            auto_highlight=True,
        )

        view_state = pdk.ViewState(latitude=21.0, longitude=78.0, zoom=4, pitch=45)
        r = pdk.Deck(
            layers=[geojson_layer, station_layer],
            initial_view_state=view_state,
            tooltip={"text": "{station}: Risk Score {risk_score} ({band})"},
        )
        st.pydeck_chart(r)
        st.dataframe(station_data[["station", "state", "elevation", "station_anomaly_c", "risk_score", "band"]], use_container_width=True)

    with tab3:
        st.subheader("📚 Dual Methodology Specifications")
        st.markdown(
            """
            ### Method A — Empirical Distribution Quantiles
            - Derived from the training distribution (1880–1999) of warm anomalies.
            - Quantiles: 50th (Low), 75th (Moderate), 90th (High), 97th (Extreme).
            - Ensures threshold values are completely data-driven and leak-free.

            ### Method B — Literature Pre-Industrial Baseline
            - Anchored to IPCC framing relative to the 1850–1900 baseline.
            - Offset arithmetic: GISTEMP 1951–1980 baseline is shifted by $+0.25$ °C to align with pre-industrial levels.
            - Thresholds: $+1.5$ °C and $+2.0$ °C warming policy boundaries.
            """
        )
