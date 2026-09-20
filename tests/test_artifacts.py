"""Unit tests for artifact manager and metric consistency assertions."""

from src.models.artifacts import ArtifactManager


def test_artifact_manager_save_load(tmp_path) -> None:
    mgr = ArtifactManager(base_dir=tmp_path)

    metrics = {"mae": 0.1234, "rmse": 0.1567, "r2": 0.85}
    mgr.save_metrics("xgboost", metrics)

    loaded_metrics = mgr.load_metrics("xgboost")
    assert loaded_metrics["mae"] == 0.1234
    assert loaded_metrics["rmse"] == 0.1567
    assert loaded_metrics["r2"] == 0.85
