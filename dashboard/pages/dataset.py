"""Dataset Explorer page module for CLIMORA AI dashboard."""

import pandas as pd
import streamlit as st

from dashboard.state import AppState, StateStage, get_clean_gistemp
from src.data.validation import DataValidator, ValidationReport


def render_dataset_page() -> None:
    """Render Dataset Explorer page with schema, validation reports, and CSV upload."""
    st.title("📂 Dataset Explorer")
    st.caption("Inspect, validate, and explore climate observation and reanalysis datasets")

    if not AppState.require_stage(StateStage.LOADED):
        return

    clean_df = get_clean_gistemp()
    val_report: ValidationReport | None = st.session_state.get("validation_report")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Dataset Inspection", "🛡️ Validation Report", "📈 Summary Statistics", "📤 Upload Custom CSV"]
    )

    with tab1:
        st.subheader("Loaded Dataset Preview")
        if clean_df is not None:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rows", f"{len(clean_df):,}")
            col2.metric("Columns", f"{len(clean_df.columns)}")
            col3.metric("Duplicate Rows", f"{clean_df.duplicated().sum()}")
            col4.metric("Missing Cells", f"{clean_df.isna().sum().sum()}")

            st.dataframe(clean_df.head(100), use_container_width=True)

            st.subheader("Column Schema & Data Types")
            dtypes_df = pd.DataFrame(
                {
                    "Column": clean_df.columns,
                    "Dtype": [str(d) for d in clean_df.dtypes],
                    "Non-Null Count": [clean_df[c].notna().sum() for c in clean_df.columns],
                    "Null %": [(clean_df[c].isna().mean() * 100) for c in clean_df.columns],
                }
            )
            st.dataframe(dtypes_df, use_container_width=True)

    with tab2:
        st.subheader("🛡️ Automated Validation Audit")
        if val_report is not None:
            status = val_report.overall_status.value
            status_colors = {
                "VALID": "🟢 VALID",
                "WARNING": "🟡 WARNING",
                "INVALID": "🔴 INVALID",
            }
            st.markdown(f"### Overall Status: **{status_colors.get(status, status)}**")
            st.caption(f"Dataset: {val_report.dataset_name} ({val_report.total_rows:,} rows)")

            checks_data = []
            for c in val_report.checks:
                checks_data.append(
                    {
                        "Check": c.check_name,
                        "Status": c.status.value,
                        "Flagged Count": c.flagged_count,
                        "Details": c.message,
                    }
                )
            st.dataframe(pd.DataFrame(checks_data), use_container_width=True)

    with tab3:
        st.subheader("📈 Statistical Description")
        if clean_df is not None:
            num_df = clean_df.select_dtypes(include=["number"])
            if len(num_df.columns) > 0:
                st.dataframe(num_df.describe().T, use_container_width=True)
            else:
                st.info("No numeric columns found for descriptive stats.")

    with tab4:
        st.subheader("📤 Upload Custom Climate CSV")
        st.markdown(
            "Upload a custom CSV dataset for live ingestion and validation. "
            "Must contain a date column and numerical climate variables."
        )

        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                user_df = pd.read_csv(uploaded_file)
                st.success(f"Successfully read CSV with {len(user_df)} rows and {len(user_df.columns)} columns.")

                validator = DataValidator()
                user_val = validator.validate(user_df, dataset_name=uploaded_file.name)
                st.markdown(f"**Validation Status**: `{user_val.overall_status.value}`")

                u_checks = [
                    {
                        "Check": c.check_name,
                        "Status": c.status.value,
                        "Flagged": c.flagged_count,
                        "Message": c.message,
                    }
                    for c in user_val.checks
                ]
                st.dataframe(pd.DataFrame(u_checks), use_container_width=True)
                st.dataframe(user_df.head(20), use_container_width=True)

            except Exception as e:
                st.error(f"❌ Failed to parse or validate uploaded CSV: {e}")
