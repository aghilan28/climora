"""Data cleaning pipeline preserving extremes while dropping physically impossible values."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.data.schema import GISTEMP_SCHEMA, ColumnSchema
from src.utils.logging import logger


@dataclass
class CleaningAction:
    operation: str
    rows_affected: int
    details: str


@dataclass
class CleaningReport:
    original_rows: int
    cleaned_rows: int
    actions: List[CleaningAction] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_rows": self.original_rows,
            "cleaned_rows": self.cleaned_rows,
            "actions": [
                {
                    "operation": a.operation,
                    "rows_affected": a.rows_affected,
                    "details": a.details,
                }
                for a in self.actions
            ],
        }


class DataCleaner:
    """Cleaner for climate data with audit log of every operation."""

    def __init__(self, schema: Dict[str, ColumnSchema] | None = None) -> None:
        self.schema = schema or GISTEMP_SCHEMA

    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, CleaningReport]:
        df_clean = df.copy()
        orig_rows = len(df_clean)
        actions: List[CleaningAction] = []

        # 1. Normalise column names
        df_clean.columns = [c.strip().lower() for c in df_clean.columns]

        # 2. Convert and validate date column
        if "date" in df_clean.columns:
            df_clean["date"] = pd.to_datetime(df_clean["date"], errors="coerce")
            invalid_dates = df_clean["date"].isna().sum()
            if invalid_dates > 0:
                df_clean = df_clean.dropna(subset=["date"])
                actions.append(
                    CleaningAction(
                        "drop_invalid_dates",
                        int(invalid_dates),
                        f"Dropped {invalid_dates} rows with unparseable date strings",
                    )
                )

        # 3. Drop exact duplicates
        init_count = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=["date"] if "date" in df_clean.columns else None)
        dups_dropped = init_count - len(df_clean)
        if dups_dropped > 0:
            actions.append(
                CleaningAction(
                    "drop_duplicates",
                    dups_dropped,
                    f"Dropped {dups_dropped} duplicate rows",
                )
            )

        # 4. Filter physically impossible out-of-bounds values (keep plausible extremes)
        for col, col_schema in self.schema.items():
            if col not in df_clean.columns or col == "date":
                continue

            num_series = pd.to_numeric(df_clean[col], errors="coerce")

            # Physically impossible criteria: value < min_val OR value > max_val
            impossible_mask = (num_series < col_schema.min_val) | (num_series > col_schema.max_val)
            impossible_count = impossible_mask.sum()

            if impossible_count > 0:
                df_clean = df_clean[~impossible_mask]
                actions.append(
                    CleaningAction(
                        f"drop_physically_impossible_{col}",
                        int(impossible_count),
                        f"Dropped {impossible_count} impossible values outside range [{col_schema.min_val}, {col_schema.max_val}] for {col}",
                    )
                )

        # 5. Sort chronologically
        if "date" in df_clean.columns:
            df_clean = df_clean.sort_values("date").reset_index(drop=True)
            actions.append(
                CleaningAction(
                    "sort_chronological",
                    0,
                    "Sorted dataset in monotonic chronological order by date",
                )
            )

        report = CleaningReport(
            original_rows=orig_rows,
            cleaned_rows=len(df_clean),
            actions=actions,
        )
        logger.info(
            "Data cleaning complete: %d -> %d rows (%d actions logged)",
            orig_rows,
            len(df_clean),
            len(actions),
        )
        return df_clean, report
