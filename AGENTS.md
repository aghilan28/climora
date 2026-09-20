# AGENTS.md — Permanent Project Rules & Engineering Standards

## 1. Project Identity & Purpose
- **Name**: CLIMORA AI — Advanced Climate Intelligence Platform
- **Type**: Interactive web-based Machine Learning climate intelligence platform
- **Stack**: Python 3.11/3.13 · Streamlit · scikit-learn · XGBoost · LightGBM · PyTorch (LSTM) · Isolation Forest · NumPy · pandas · matplotlib · Plotly · pydeck
- **Core Identity**: Climate Intelligence + Machine Learning. Not a weather app, not a chatbot, not a GIS product, not a generic analytics dashboard, not a GenAI application.

## 2. Fundamental Engineering & Honesty Rules
- **No Fake Data or Hardcoded Metrics**: Every number shown in the UI, written in docs, or reported must be traceable to a function call that computed it from real data.
- **Reality Over Plan**: When a plan and reality conflict, reality wins. Report empirical findings plainly. A mediocre real metric is a pass; a beautiful fake metric is a project failure.
- **Data Provenance Discipline**: Label ERA5 accurately as a reanalysis product (never "satellite observations"). Label GISTEMP anomalies as °C relative to baseline (never plain "temperature").
- **Strict Anti-Leakage**: Feature scaling, imputers, and selectors must be fit ONLY on training data splits. Windows must never look into the future. Rolling features must set `center=False`.
- **Reproducibility**: Global `SEED = 42` in `config/settings.py`, threaded into all random number generators (NumPy, PyTorch, scikit-learn, XGBoost, LightGBM).

## 3. Architecture & Code Structure
- **Streamlit Layer (`dashboard/`, `app.py`)**: UI and presentation only. Reads state from `dashboard/state.py`. Zero ML training or direct file ingestion logic in presentation code.
- **Core Pipeline (`src/`)**: Decoupled, modular Python packages (`data`, `eda`, `features`, `splits`, `models`, `evaluation`, `risk`, `anomaly`, `explain`, `geo`, `utils`).
- **Data Loaders (`src/data/providers/`)**: Each provider implements a common protocol (fetch → parse → validate → cache).
- **Model Registry (`src/models/`)**: Unified interface (`ClimateModel`) for XGBoost, LightGBM, PyTorch LSTM, and Baselines. Native artifact persistence (`.json`, `.pt`, `.joblib`).

## 4. Quality & Testing Bar
- `pytest` must pass cleanly before declaring any phase or feature complete.
- Static analysis via `ruff check .` and `mypy src` must exit with 0 errors.
- Test suite includes anti-cheat verification (`test_no_hardcoded_results.py`, `test_no_placeholders.py`).

## 5. Scientific Integrity
- Never claim guaranteed predictions, operational climate safety certification, or absolute superiority without statistical evidence and baselines comparison (Seasonal Naive & Climatology).
