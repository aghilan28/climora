"""Unit tests for chronological dataset split."""

from src.data.providers.nasa_giss import NasaGissProvider
from src.features.pipeline import build_feature_matrix
from src.splits.chronological import make_chronological_split


def test_chronological_split_boundaries(gistemp_fixture_bytes: bytes, tmp_path) -> None:
    provider = NasaGissProvider(tmp_path)
    df_raw = provider.parse(gistemp_fixture_bytes)
    df_feats = build_feature_matrix(df_raw)

    split = make_chronological_split(df_feats)

    # Assert no index overlap and strict chronological ordering
    if not split.X_train.empty and not split.X_val.empty:
        max_train_date = split.train_df["date"].max()
        min_val_date = split.val_df["date"].min()
        assert max_train_date < min_val_date

    if not split.X_val.empty and not split.X_test.empty:
        max_val_date = split.val_df["date"].max()
        min_test_date = split.test_df["date"].min()
        assert max_val_date < min_test_date
