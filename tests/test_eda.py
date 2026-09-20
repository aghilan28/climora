"""Unit tests for EDA modules and figure generators."""

import pytest

from src.data.providers.nasa_giss import NasaGissProvider
from src.eda.correlation import plot_correlation_matrix
from src.eda.distributions import plot_distribution
from src.eda.outliers import detect_outliers
from src.eda.overview import compute_dataset_overview
from src.eda.seasonality import compute_seasonality_metrics, plot_seasonality_boxplot
from src.eda.trend import plot_trend_line


def test_overview_computation(gistemp_fixture_bytes: bytes, tmp_path) -> None:
    provider = NasaGissProvider(tmp_path)
    df = provider.parse(gistemp_fixture_bytes)

    overview = compute_dataset_overview(df, target_col="anomaly_c")
    assert overview["total_records"] == len(df)
    assert overview["start_date"] != "N/A"
    assert "mean" in overview


def test_plot_distribution(gistemp_fixture_bytes: bytes, tmp_path) -> None:
    provider = NasaGissProvider(tmp_path)
    df = provider.parse(gistemp_fixture_bytes)

    fig = plot_distribution(df, col="anomaly_c")
    assert fig is not None
    assert len(fig.data) > 0

    with pytest.raises(ValueError, match="Column 'non_existent' not found"):
        plot_distribution(df, col="non_existent")


def test_plot_trend_line(gistemp_fixture_bytes: bytes, tmp_path) -> None:
    provider = NasaGissProvider(tmp_path)
    df = provider.parse(gistemp_fixture_bytes)

    fig = plot_trend_line(df, date_col="date", val_col="anomaly_c")
    assert fig is not None
    assert len(fig.data) >= 2  # Raw line + rolling mean

    with pytest.raises(ValueError, match="missing from DataFrame"):
        plot_trend_line(df, val_col="missing_col")


def test_plot_seasonality(gistemp_fixture_bytes: bytes, tmp_path) -> None:
    provider = NasaGissProvider(tmp_path)
    df = provider.parse(gistemp_fixture_bytes)

    fig, metrics = plot_seasonality_boxplot(df, val_col="anomaly_c")
    assert fig is not None
    assert len(fig.data) > 0
    assert "seasonal_amplitude" in metrics
    assert metrics["seasonal_amplitude"] >= 0.0

    df_no_month = df.drop(columns=["month"])
    with pytest.raises(ValueError, match="Required columns"):
        compute_seasonality_metrics(df_no_month, val_col="anomaly_c")


def test_detect_outliers(gistemp_fixture_bytes: bytes, tmp_path) -> None:
    provider = NasaGissProvider(tmp_path)
    df = provider.parse(gistemp_fixture_bytes)

    outliers = detect_outliers(df, col="anomaly_c", z_threshold=2.0)
    assert isinstance(outliers, type(df))

    with pytest.raises(ValueError, match="missing from DataFrame"):
        detect_outliers(df, col="non_existent")


def test_plot_correlation_matrix(gistemp_fixture_bytes: bytes, tmp_path) -> None:
    provider = NasaGissProvider(tmp_path)
    df = provider.parse(gistemp_fixture_bytes)

    fig = plot_correlation_matrix(df)
    assert fig is not None
    assert len(fig.data) > 0
