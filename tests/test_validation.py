"""Unit tests for dataset validation engine."""

import pandas as pd

from src.data.providers.nasa_giss import NasaGissProvider
from src.data.validation import DataValidator, ValidationStatus


def test_validator_with_valid_fixture(gistemp_fixture_bytes: bytes, tmp_path) -> None:
    provider = NasaGissProvider(tmp_path)
    df = provider.parse(gistemp_fixture_bytes)

    validator = DataValidator()
    report = validator.validate(df, "GISTEMP Sample")

    assert report.total_rows > 0
    assert report.overall_status in (ValidationStatus.VALID, ValidationStatus.WARNING)


def test_validator_detects_corrupted_fixture(gistemp_fixture_bytes: bytes, tmp_path) -> None:
    provider = NasaGissProvider(tmp_path)
    df = provider.parse(gistemp_fixture_bytes)

    # Inject corruption: 1) physically impossible anomaly +9999, 2) duplicate date
    df_corrupted = df.copy()

    # Corrupt row 0 with 9999 anomaly
    df_corrupted.loc[0, "anomaly_c"] = 9999.0

    # Inject duplicate of row 1
    dup_row = df_corrupted.iloc[[1]].copy()
    df_corrupted = pd.concat([df_corrupted, dup_row], ignore_index=True)

    validator = DataValidator()
    report = validator.validate(df_corrupted, "Corrupted GISTEMP")

    # Assert overall status is INVALID
    assert report.overall_status == ValidationStatus.INVALID

    # Assert range check for anomaly_c flagged the 9999 anomaly
    range_check = next((c for c in report.checks if c.check_name == "range_anomaly_c"), None)
    assert range_check is not None
    assert range_check.status == ValidationStatus.INVALID
    assert range_check.flagged_count >= 1

    # Assert duplicate date check flagged the duplicate row
    dup_check = next((c for c in report.checks if c.check_name == "duplicate_dates"), None)
    assert dup_check is not None
    assert dup_check.status == ValidationStatus.WARNING
    assert dup_check.flagged_count == 1
