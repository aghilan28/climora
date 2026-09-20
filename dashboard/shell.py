"""Main application shell and navigation dispatcher for CLIMORA AI."""

from typing import Callable, Dict

import streamlit as st

from dashboard.state import AppState, StateStage


def render_sidebar_header() -> None:
    """Render top branding and environment status in sidebar."""
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding: 10px 0 20px 0;">
            <h1 style="margin: 0; font-size: 26px; color: #0D9488; font-weight: 800; letter-spacing: -0.5px;">
                CLIMORA AI
            </h1>
            <p style="margin: 2px 0 0 0; font-size: 11px; color: #94A3B8; text-transform: uppercase; letter-spacing: 1px;">
                Climate Intelligence Platform
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stage = AppState.get_stage()
    stage_colors = {
        StateStage.NOT_LOADED: "🔴 NOT LOADED",
        StateStage.LOADED: "🟡 DATA LOADED",
        StateStage.VALIDATED: "🟡 VALIDATED",
        StateStage.FEATURES_READY: "🟡 FEATURES READY",
        StateStage.MODELS_NOT_TRAINED: "🟠 NO MODELS",
        StateStage.MODELS_TRAINED: "🟢 MODELS TRAINED",
        StateStage.PREDICTION_READY: "🟢 PREDICTION READY",
        StateStage.ANOMALY_READY: "🟢 ANOMALY READY",
        StateStage.FULL_ANALYSIS_READY: "🟢 SYSTEM READY",
    }
    status_str = stage_colors.get(stage, f"⚪ {stage.name}")
    st.sidebar.caption(f"System State: **{status_str}**")
    st.sidebar.divider()


def render_shell(page_map: Dict[str, Callable[[], None]]) -> None:
    """Render application sidebar and execute selected page module."""
    render_sidebar_header()

    selected_page = st.sidebar.radio(
        "Navigation",
        options=list(page_map.keys()),
        index=0,
    )

    st.sidebar.divider()
    st.sidebar.caption("CLIMORA AI v1.0.0 · Python 3.13")
    st.sidebar.caption("Baseline: GISTEMP v4 (1951–1980)")

    if st.sidebar.button("🔄 Force Reload State", use_container_width=True):
        AppState.bootstrap(force_reload=True)
        st.rerun()

    # Page Execution Boundary
    if selected_page in page_map:
        try:
            page_map[selected_page]()
        except Exception as e:
            st.error("🚨 An unexpected error occurred while rendering this page.")
            st.exception(e)
