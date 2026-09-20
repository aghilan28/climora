"""Streamlit application entry point for CLIMORA AI.

Wires page modules, manages global session state, and enforces error boundaries.
"""

import sys
from pathlib import Path

# Ensure repository root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st  # noqa: E402

from dashboard.pages.anomalies import render_anomalies_page  # noqa: E402
from dashboard.pages.climate_analysis import render_climate_analysis_page  # noqa: E402
from dashboard.pages.dataset import render_dataset_page  # noqa: E402
from dashboard.pages.explainability import render_explainability_page  # noqa: E402
from dashboard.pages.features import render_features_page  # noqa: E402
from dashboard.pages.methodology import render_methodology_page  # noqa: E402
from dashboard.pages.model_performance import render_model_performance_page  # noqa: E402
from dashboard.pages.overview import render_overview_page  # noqa: E402
from dashboard.pages.predictions import render_predictions_page  # noqa: E402
from dashboard.pages.risk import render_risk_page  # noqa: E402
from dashboard.shell import render_shell  # noqa: E402
from dashboard.state import AppState  # noqa: E402
from src.utils.logging import setup_logger  # noqa: E402

logger = setup_logger("climora.app")


def main() -> None:
    """Main application runner with page routing and error boundaries."""
    st.set_page_config(
        page_title="CLIMORA AI — Climate Intelligence Platform",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    try:
        # Bootstrap application state (loads data, runs validation & features, loads trained models)
        AppState.bootstrap()

        # Page Dispatcher Map
        page_map = {
            "🌐 Overview": render_overview_page,
            "📂 Dataset Explorer": render_dataset_page,
            "🌡️ Climate EDA": render_climate_analysis_page,
            "⚙️ Feature Engineering": render_features_page,
            "🎯 Model Predictions": render_predictions_page,
            "⚡ Risk Classification": render_risk_page,
            "🚨 Anomaly Detection": render_anomalies_page,
            "📊 Model Performance": render_model_performance_page,
            "🧠 Explainability": render_explainability_page,
            "📚 Methodology & About": render_methodology_page,
        }

        # Render Shell Navigation & Selected Page
        render_shell(page_map)

    except Exception as e:
        logger.error(f"Critical error in Streamlit application root: {e}", exc_info=True)
        st.error("🚨 **System Error**: An unexpected critical error occurred.")
        st.info("Please reload the application or check application logs.")
        st.exception(e)


if __name__ == "__main__":
    main()
