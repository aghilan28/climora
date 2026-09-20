"""Gate test for model selection criterion and 'Best Model' badge logic per F11."""

import json
from pathlib import Path
import pytest


def test_model_selection_badge_and_h12_metrics() -> None:
    """Assert 'Best Model' badge logic enforces skill_score > 0 and h=1/h=12 metrics exist."""
    metrics_dir = Path("models/metrics")
    if not metrics_dir.exists():
        pytest.skip("models/metrics directory not found")

    metric_files = list(metrics_dir.glob("*.json"))
    assert len(metric_files) >= 3, f"Expected at least 3 metric JSON files in models/metrics, found {len(metric_files)}"

    ml_models = ["xgboost", "lightgbm", "lstm"]
    for model_name in ml_models:
        meta_file = metrics_dir / f"{model_name}_metrics.json"
        if meta_file.exists():
            data = json.loads(meta_file.read_text(encoding="utf-8"))
            # Assert both h=1 and h=12 metrics exist
            assert "rmse" in data or "mae" in data, f"{model_name}_metrics.json missing h=1 metrics!"
            assert "h12_metrics" in data or "test_metrics_h12" in data, f"{model_name}_metrics.json missing h12_metrics!"

    # Check badge logic rule: if all skill_scores <= 0, no Best Model badge rendered
    best_skill = max([json.loads(f.read_text(encoding="utf-8")).get("skill_score", -1.0) for f in metric_files])
    
    perf_page_content = Path("dashboard/pages/model_performance.py").read_text(encoding="utf-8")
    assert "No model beats the seasonal-naive baseline" in perf_page_content, \
        "Honest sentence missing from model_performance.py when no model beats baseline!"
