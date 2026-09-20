"""Model-grounded narrative builder derived strictly from numeric contributions."""

from typing import List, Tuple

import numpy as np


def build_explanation_narrative(
    feature_names: List[str],
    contributions: np.ndarray,
    top_k: int = 3,
    reason_if_empty: str = "No contribution values provided",
) -> str:
    """Build model-grounded narrative listing top positive and negative driver features."""
    if contributions is None or len(contributions) == 0 or len(feature_names) == 0:
        return f"EXPLANATION UNAVAILABLE: {reason_if_empty}"

    if len(contributions.shape) > 1:
        contrib_vector = contributions[0]
    else:
        contrib_vector = contributions

    if len(contrib_vector) != len(feature_names):
        return f"EXPLANATION UNAVAILABLE: Feature length mismatch ({len(feature_names)} vs {len(contrib_vector)})"

    # Pair features with contributions
    paired: List[Tuple[str, float]] = list(zip(feature_names, [float(c) for c in contrib_vector], strict=False))

    # Sort by absolute magnitude
    sorted_paired = sorted(paired, key=lambda x: abs(x[1]), reverse=True)
    top_drivers = sorted_paired[:top_k]

    pos_drivers = [f"{feat} (+{val:.3f})" for feat, val in top_drivers if val > 0]
    neg_drivers = [f"{feat} ({val:.3f})" for feat, val in top_drivers if val < 0]

    parts = []
    if pos_drivers:
        parts.append(f"Positive drivers: {', '.join(pos_drivers)}")
    if neg_drivers:
        parts.append(f"Negative drivers: {', '.join(neg_drivers)}")

    if not parts:
        return "Model prediction influenced equally across neutral features."

    return " | ".join(parts)
