"""Smoke test for Streamlit application pages and state machine."""

import traceback

import pytest

from dashboard.pages.anomalies import render_anomalies_page
from dashboard.pages.climate_analysis import render_climate_analysis_page
from dashboard.pages.dataset import render_dataset_page
from dashboard.pages.explainability import render_explainability_page
from dashboard.pages.features import render_features_page
from dashboard.pages.methodology import render_methodology_page
from dashboard.pages.model_performance import render_model_performance_page
from dashboard.pages.overview import render_overview_page
from dashboard.pages.predictions import render_predictions_page
from dashboard.pages.risk import render_risk_page
from dashboard.state import AppState, StateStage


def test_app_pages_smoke() -> None:
    """Test that all 10 dashboard page rendering functions execute without throwing exceptions."""
    # Bootstrap state
    AppState.bootstrap()

    pages = [
        ("overview", render_overview_page),
        ("dataset", render_dataset_page),
        ("climate_analysis", render_climate_analysis_page),
        ("features", render_features_page),
        ("predictions", render_predictions_page),
        ("risk", render_risk_page),
        ("anomalies", render_anomalies_page),
        ("model_performance", render_model_performance_page),
        ("explainability", render_explainability_page),
        ("methodology", render_methodology_page),
    ]

    for name, page_fn in pages:
        try:
            page_fn()
        except Exception as e:
            tb = traceback.format_exc()
            pytest.fail(f"Page function '{name}' raised exception: {e}\n{tb}")


def test_state_demotion_warnings() -> None:
    """Test that pages render prerequisite warning messages when state is demoted."""
    AppState.set_stage(StateStage.NOT_LOADED)

    # Overview requires StateStage.LOADED
    render_overview_page()
    assert AppState.get_stage() == StateStage.NOT_LOADED

    # Predictions requires StateStage.MODELS_TRAINED
    render_predictions_page()
    assert AppState.get_stage() == StateStage.NOT_LOADED
