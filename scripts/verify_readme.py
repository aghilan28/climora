"""Verification script ensuring published README §12 metrics match on-disk JSON artifacts exactly to 4 decimal places."""
# ruff: noqa: E402

import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main() -> None:
    readme_file = ROOT_DIR / "README.md"
    metrics_dir = ROOT_DIR / "models" / "metrics"

    if not readme_file.exists():
        print("ERROR: README.md not found.")
        sys.exit(1)

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

    # Extract markdown table rows
    table_pattern = re.compile(r"\|\s*([^|]+)\s*\|\s*([\d.-]+)\s*\|\s*([\d.-]+)\s*\|\s*([\d.-]+)\s*\|\s*([\d.-]+)\s*\|")
    matches = table_pattern.findall(content)

    if not matches:
        print("ERROR: Could not parse metrics table in README.md §12.")
        sys.exit(1)

    errors = []
    checked_count = 0

    for model_name_raw, mae_str, rmse_str, r2_str, skill_str in matches:
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
        expected_r2 = float(data.get("r2", 0.0))
        expected_skill = float(data.get("skill_score", 0.0))

        actual_mae = float(mae_str)
        actual_rmse = float(rmse_str)
        actual_r2 = float(r2_str)
        actual_skill = float(skill_str)

        if abs(actual_mae - expected_mae) > 0.0001:
            errors.append(f"{model_name} MAE mismatch: README={actual_mae:.4f} vs JSON={expected_mae:.4f}")
        if abs(actual_rmse - expected_rmse) > 0.0001:
            errors.append(f"{model_name} RMSE mismatch: README={actual_rmse:.4f} vs JSON={expected_rmse:.4f}")
        if abs(actual_r2 - expected_r2) > 0.0001:
            errors.append(f"{model_name} R2 mismatch: README={actual_r2:.4f} vs JSON={expected_r2:.4f}")
        if abs(actual_skill - expected_skill) > 0.0001:
            errors.append(f"{model_name} Skill Score mismatch: README={actual_skill:.4f} vs JSON={expected_skill:.4f}")

        checked_count += 1

    if errors:
        print("=======================================================")
        print("README METRICS VERIFICATION FAILED:")
        for err in errors:
            print(" -", err)
        print("=======================================================")
        sys.exit(1)

    print(f"[PASS] README §12 metrics table matches on-disk JSON artifacts to 4 decimal places ({checked_count} models checked).")
    sys.exit(0)


if __name__ == "__main__":
    main()
