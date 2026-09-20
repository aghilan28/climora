"""Verification script ensuring published README §12 metrics match on-disk JSON artifacts exactly to 4 decimal places."""
# ruff: noqa: E402

import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def verify_readme_metrics(readme_path: Path | None = None) -> bool:
    """Verify README §12 metrics table against models/metrics/*.json artifacts to 4 decimal places."""
    readme_file = readme_path if readme_path is not None else (ROOT_DIR / "README.md")
    metrics_dir = ROOT_DIR / "models" / "metrics"

    if not readme_file.exists():
        print(f"ERROR: {readme_file} not found.")
        return False

    content = readme_file.read_text(encoding="utf-8")

    # Map README model display names to metric JSON files
    model_map = {
        "Seasonal Naive Baseline": "seasonal_naive_metrics.json",
        "SeasonalNaive": "seasonal_naive_metrics.json",
        "Climatology Baseline": "climatology_metrics.json",
        "Climatology": "climatology_metrics.json",
        "PyTorch LSTM": "lstm_metrics.json",
        "LSTM": "lstm_metrics.json",
        "LightGBM": "lightgbm_metrics.json",
        "XGBoost": "xgboost_metrics.json",
    }

    # Extract markdown table rows: Model | MAE | RMSE | Skill h=1 | Skill h=12
    table_pattern = re.compile(
        r"\|\s*([^|]+)\s*\|\s*\*?\*?([\d.-]+)\*?\*?\s*\|\s*\*?\*?([\d.-]+)\*?\*?\s*\|\s*\*?\*?([\d.-]+)\*?\*?\s*\|\s*\*?\*?([\d.-]+)\*?\*?\s*\|"
    )
    matches = table_pattern.findall(content)

    if not matches:
        print(f"ERROR: Could not parse metrics table in {readme_file}.")
        return False

    errors = []
    checked_count = 0

    for model_name_raw, mae_str, rmse_str, skill_str, h12_skill_str in matches:
        model_name = model_name_raw.strip().replace("**", "")
        if model_name not in model_map:
            continue

        json_file = metrics_dir / model_map[model_name]
        if not json_file.exists():
            errors.append(f"Metrics JSON missing for {model_name}: {json_file}")
            continue

        data = json.loads(json_file.read_text(encoding="utf-8"))

        expected_mae = float(data.get("mae", 0.0))
        expected_rmse = float(data.get("rmse", 0.0))
        expected_skill = float(data.get("skill_score", 0.0))
        expected_h12_skill = float(data.get("h12_metrics", {}).get("skill_score", data.get("skill_score_h12", 0.0)))

        actual_mae = float(mae_str)
        actual_rmse = float(rmse_str)
        actual_skill = float(skill_str)
        actual_h12_skill = float(h12_skill_str)

        if abs(actual_mae - expected_mae) > 1e-4:
            errors.append(f"{model_name} MAE mismatch: README={actual_mae:.4f} vs JSON={expected_mae:.4f}")
        if abs(actual_rmse - expected_rmse) > 1e-4:
            errors.append(f"{model_name} RMSE mismatch: README={actual_rmse:.4f} vs JSON={expected_rmse:.4f}")
        if abs(actual_skill - expected_skill) > 1e-4:
            errors.append(f"{model_name} Skill Score mismatch: README={actual_skill:.4f} vs JSON={expected_skill:.4f}")
        if abs(actual_h12_skill - expected_h12_skill) > 1e-4:
            errors.append(f"{model_name} h=12 Skill Score mismatch: README={actual_h12_skill:.4f} vs JSON={expected_h12_skill:.4f}")

        checked_count += 1

    if errors:
        print("=======================================================")
        print("README METRICS VERIFICATION FAILED:")
        for err in errors:
            print(" -", err)
        print("=======================================================")
        return False

    print(f"[PASS] README §12 metrics table matches on-disk JSON artifacts to 4 decimal places ({checked_count} models checked).")
    return True


def main() -> None:
    if verify_readme_metrics():
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
