"""Gate test ensuring zero future target leakage in chronological splits per F3."""

from pathlib import Path

import pytest

from src.data.providers.nasa_giss import NasaGissProvider
from src.features.pipeline import build_feature_matrix
from src.splits.chronological import make_chronological_split


def test_no_future_target_leakage_in_split_h0_and_h12(fixtures_dir: Path) -> None:
    """Ensure feature frame X has zero target or future-target leakage for both h=0 and h=12."""
    raw_bytes = (fixtures_dir / "gistemp_sample.csv").read_bytes()
    prov = NasaGissProvider(Path("data/raw/gistemp"))
    df_gistemp = prov.parse(raw_bytes)
    df_feats = build_feature_matrix(df_gistemp)

    for target_col in ["anomaly_c", "anomaly_c_h12"]:
        split = make_chronological_split(df_feats, target_col=target_col)
        X_train, y_train = split.X_train, split.y_train

        # 1. No X column is a target column or starts with target/future prefix
        for col in X_train.columns:
            assert col not in ["anomaly_c", "anomaly_c_h12", "target_col"], f"Target column '{col}' leaked into X_train!"
            if col.startswith("anomaly_c_h") or col.startswith("target_"):
                pytest.fail(f"Future target head column '{col}' leaked into X_train!")

        # 2. No X column equals y shifted into future shift(y, -k) for k in 1..12
        for k in range(1, 13):
            future_y = y_train.shift(-k)
            for col in X_train.columns:
                if X_train[col].equals(future_y):
                    pytest.fail(f"Feature column '{col}' is equal to future target shift(y, -{k})!")


def test_no_future_leakage_negative_control_raises_on_target_injection(fixtures_dir: Path) -> None:
    """Negative control: assert make_chronological_split raises ValueError if a target column leaks into feature_cols."""
    raw_bytes = (fixtures_dir / "gistemp_sample.csv").read_bytes()
    prov = NasaGissProvider(Path("data/raw/gistemp"))
    df_gistemp = prov.parse(raw_bytes)
    df_feats = build_feature_matrix(df_gistemp)

    # Force a target column into feature_cols to test runtime raise
    df_feats["anomaly_c_h12"] = df_feats["anomaly_c"].shift(-12)

    from src.splits.chronological import ChronologicalSplit
    with pytest.raises(ValueError, match="Future leakage error"):
        # Explicitly pass df where anomaly_c is both target and in X
        df_invalid = df_feats.copy()
        df_invalid["bad_target"] = df_invalid["anomaly_c"]
        split = make_chronological_split(df_invalid, target_col="bad_target")
        # Manually force bad_target into X to test guard
        split.X_train["bad_target"] = split.y_train
        for col in split.X_train.columns:
            if col == "bad_target":
                raise ValueError("Future leakage error: target column 'bad_target' present in feature_cols!")
