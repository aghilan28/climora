"""Validation engine for climate intelligence datasets."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

import pandas as pd

from src.data.schema import GISTEMP_SCHEMA, ColumnSchema


class ValidationStatus(str, Enum):
    VALID = "VALID"
    WARNING = "WARNING"
    INVALID = "INVALID"


@dataclass
class CheckResult:
    check_name: str
    status: ValidationStatus
    message: str
    flagged_count: int = 0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    dataset_name: str
    overall_status: ValidationStatus
    total_rows: int
    total_columns: int
    checks: List[CheckResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "overall_status": self.overall_status.value,
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "checks": [
                {
                    "check_name": c.check_name,
                    "status": c.status.value,
                    "message": c.message,
                    "flagged_count": c.flagged_count,
                    "details": c.details,
                }
                for c in self.checks
            ],
        }


class DataValidator:
    """Validator that checks schema compliance, physical plausibility, and integrity."""

    def __init__(self, schema: Dict[str, ColumnSchema] | None = None) -> None:
        self.schema = schema or GISTEMP_SCHEMA

    def validate(
        self, df: pd.DataFrame, dataset_name: str = "Dataset"
    ) -> ValidationReport:
        checks: List[CheckResult] = []
        has_invalid = False
        has_warning = False

        total_rows = len(df)
        total_cols = len(df.columns)

        if total_rows == 0:
            return ValidationReport(
                dataset_name=dataset_name,
                overall_status=ValidationStatus.INVALID,
                total_rows=0,
                total_columns=total_cols,
                checks=[
                    CheckResult(
                        "non_empty",
                        ValidationStatus.INVALID,
                        "DataFrame is completely empty (0 rows)",
                    )
                ],
            )

        # Check 1: Required Columns
        missing_cols = [c for c in self.schema.keys() if c not in df.columns]
        if missing_cols:
            checks.append(
                CheckResult(
                    "required_columns",
                    ValidationStatus.INVALID,
                    f"Missing required columns: {missing_cols}",
                    flagged_count=len(missing_cols),
                )
            )
            has_invalid = True
        else:
            checks.append(
                CheckResult(
                    "required_columns",
                    ValidationStatus.VALID,
                    "All required columns present",
                )
            )

        # Check 2: Duplicates
        if "date" in df.columns:
            dup_dates = df.duplicated(subset=["date"]).sum()
            if dup_dates > 0:
                checks.append(
                    CheckResult(
                        "duplicate_dates",
                        ValidationStatus.WARNING,
                        f"Found {dup_dates} duplicate date entries",
                        flagged_count=int(dup_dates),
                    )
                )
                has_warning = True
            else:
                checks.append(
                    CheckResult(
                        "duplicate_dates",
                        ValidationStatus.VALID,
                        "No duplicate dates found",
                    )
                )

        # Check 3: Monotonic Chronological Ordering
        if "date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["date"]):
            is_monotonic = df["date"].is_monotonic_increasing
            if not is_monotonic:
                checks.append(
                    CheckResult(
                        "chronological_ordering",
                        ValidationStatus.WARNING,
                        "Dates are not strictly monotonically increasing",
                        flagged_count=1,
                    )
                )
                has_warning = True
            else:
                checks.append(
                    CheckResult(
                        "chronological_ordering",
                        ValidationStatus.VALID,
                        "Dates are strictly monotonically increasing",
                    )
                )

        # Check 4: Physical Plausibility Ranges
        for col, col_schema in self.schema.items():
            if col not in df.columns or col == "date":
                continue

            numeric_series = pd.to_numeric(df[col], errors="coerce")
            out_of_bounds = numeric_series[
                (numeric_series < col_schema.min_val)
                | (numeric_series > col_schema.max_val)
            ]
            oob_count = len(out_of_bounds)

            if oob_count > 0:
                checks.append(
                    CheckResult(
                        f"range_{col}",
                        ValidationStatus.INVALID,
                        f"Column '{col}' has {oob_count} values outside physical limits [{col_schema.min_val}, {col_schema.max_val}] {col_schema.unit}",
                        flagged_count=oob_count,
                        details={
                            "min_val": float(numeric_series.min()),
                            "max_val": float(numeric_series.max()),
                        },
                    )
                )
                has_invalid = True
            else:
                checks.append(
                    CheckResult(
                        f"range_{col}",
                        ValidationStatus.VALID,
                        f"Column '{col}' values within physical range [{col_schema.min_val}, {col_schema.max_val}]",
                    )
                )

        # Check 5: Missing / Sentinel Values
        missing_count = int(df.isna().sum().sum())
        if missing_count > 0:
            checks.append(
                CheckResult(
                    "missing_values",
                    ValidationStatus.WARNING,
                    f"Total missing (NaN) values: {missing_count}",
                    flagged_count=missing_count,
                )
            )
            has_warning = True
        else:
            checks.append(
                CheckResult(
                    "missing_values",
                    ValidationStatus.VALID,
                    "Zero missing values found",
                )
            )

        if has_invalid:
            overall = ValidationStatus.INVALID
        elif has_warning:
            overall = ValidationStatus.WARNING
        else:
            overall = ValidationStatus.VALID

        return ValidationReport(
            dataset_name=dataset_name,
            overall_status=overall,
            total_rows=total_rows,
            total_columns=total_cols,
            checks=checks,
        )
