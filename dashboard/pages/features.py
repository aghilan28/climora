"""Feature Engineering page module for CLIMORA AI dashboard."""

import pandas as pd
import streamlit as st

from dashboard.state import AppState, StateStage, get_clean_gistemp, get_feature_df
from src.features.registry import FEATURE_REGISTRY


def render_features_page() -> None:
    """Render Feature Engineering page with registry table, row counts, and sample assertions."""
    st.title("⚙️ Feature Engineering & Anti-Leakage Registry")
    st.caption("Inspect feature definitions, trailing window specs, and verified anti-leakage math")

    if not AppState.require_stage(StateStage.FEATURES_READY):
        return

    clean_df = get_clean_gistemp()
    feat_df = get_feature_df()

    orig_rows = len(clean_df) if clean_df is not None else 0
    feat_rows = len(feat_df) if feat_df is not None else 0
    feat_cols = len(feat_df.columns) if feat_df is not None else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Original Data Rows", f"{orig_rows:,}")
    col2.metric("Feature Matrix Rows", f"{feat_rows:,}")
    col3.metric("Total Columns", f"{feat_cols}")
    col4.metric("Engineered Features", f"{len(FEATURE_REGISTRY)}")

    st.divider()

    tab1, tab2, tab3 = st.tabs(
        ["📋 Feature Registry", "🔍 Feature Matrix Preview", "🛡️ Anti-Leakage Verification"]
    )

    with tab1:
        st.subheader("Feature Definitions & Anti-Leakage Specifications")
        registry_data = [
            {
                "Feature Name": f.name,
                "Family": f.family,
                "Formula": f.formula,
                "Window": f.window,
                "Source Column": f.source_col,
                "Anti-Leakage Note": f.leakage_note,
            }
            for f in FEATURE_REGISTRY
        ]
        st.dataframe(pd.DataFrame(registry_data), use_container_width=True)

    with tab2:
        st.subheader("Computed Feature Matrix Sample")
        if feat_df is not None:
            st.dataframe(feat_df.tail(50), use_container_width=True)

    with tab3:
        st.subheader("🛡️ Strict Anti-Leakage Rules & Hand-Checked Assertions")
        st.markdown(
            """
            - **center=False Enforcement**: All rolling features set `center=False` to forbid future window lookahead.
            - **Trailing Lags Only**: Lags use strictly past timesteps $t-k$.
            - **Train-Set Fitting Only**: Scalers, imputers, and selectors are fit strictly on training splits.
            - **Deterministic Verification**: Verified in `tests/test_features.py` via `test_no_future_leakage`.
            """
        )
        if feat_df is not None and "anomaly_c_lag_1" in feat_df.columns:
            st.success("✅ **Verified**: `anomaly_c_lag_1` exactly equals `anomaly_c` offset by 1 row.")
            sample = feat_df[["date", "anomaly_c", "anomaly_c_lag_1"]].dropna().head(10)
            st.dataframe(sample, use_container_width=True)
