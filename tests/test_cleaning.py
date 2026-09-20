"""Unit tests for dataset cleaning engine."""

import pandas as pd

from src.data.cleaning import DataCleaner
from src.data.providers.nasa_giss import NasaGissProvider
from src.data.validation import DataValidator, ValidationStatus


def test_cleaner_recovers_corrupted_dataset(gistemp_fixture_bytes: bytes, tmp_path) -> None:
    provider = NasaGissProvider(tmp_path)
    df_raw = provider.parse(gistemp_fixture_bytes)

    # Inject corruption: 1) impossible anomaly 9999, 2) duplicate row
    df_corrupted = df_raw.copy()
    df_corrupted.loc[0, "anomaly_c"] = 9999.0
    dup_row = df_corrupted.iloc[[1]].copy()
    df_corrupted = pd.concat([df_corrupted, dup_row], ignore_index=True)

    cleaner = DataCleaner()
    df_cleaned, report = cleaner.clean(df_corrupted)

    # Assert impossible value and duplicate were dropped
    assert len(df_cleaned) == len(df_raw) - 1  # 1 dropped for 9999, 1 duplicate removed

    # Validate cleaned dataframe is now VALID
    validator = DataValidator()
    val_report = validator.validate(df_cleaned, "Cleaned Dataset")
    assert val_report.overall_status in (ValidationStatus.VALID, ValidationStatus.WARNING)
    assert 9999.0 not in df_cleaned["anomaly_c"].values


def test_cleaner_preserves_valid_extremes(gistemp_fixture_bytes: bytes, tmp_path) -> None:
    provider = NasaGissProvider(tmp_path)
    df_raw = provider.parse(gistemp_fixture_bytes)

    cleaner = DataCleaner()
    df_cleaned, report = cleaner.clean(df_raw)

    # Valid extremes (e.g. +1.21 °C in 2025) must be preserved
    assert len(df_cleaned) == len(df_raw)
