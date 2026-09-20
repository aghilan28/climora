# CLIMORA AI Architecture Documentation

## System Overview
CLIMORA AI is an Advanced Climate Intelligence Platform built using Python 3.11, Streamlit, scikit-learn, XGBoost, LightGBM, and PyTorch. The platform performs end-to-end time-series forecasting, risk classification, anomaly detection, and model-grounded explainability on global surface temperature anomalies and station panel reanalysis data.

```
Climate Data -> Ingestion -> Validation -> Cleaning -> Preprocessing -> EDA -> Feature Engineering
             -> Dataset Prep -> ML Models -> Evaluation -> Prediction -> Risk Classification
             -> Anomaly Detection -> Explainability -> Streamlit Dashboard
```

## Architectural Decoupling
- **Streamlit Presentation Layer (`dashboard/`, `app.py`)**: Responsible strictly for visual rendering, user interactions, and page lifecycle management. Uses `dashboard/state.py` for state transitions. Contains 0 ML training or direct ingestion logic.
- **Core Pipeline (`src/`)**: Decoupled, modular Python packages:
  - `src/data/`: Multi-provider ingestion, schema validation, cleaning, missing value handling, unit conversion, manifest tracking.
  - `src/eda/`: Visualizations for trend, seasonality, correlation, distributions, and outliers.
  - `src/features/`: Temporal, lag, rolling, difference, rate-of-change feature engineering and leak-free pipeline fitting.
  - `src/splits/`: Strict chronological time-series split management (`train`, `val`, `test`).
  - `src/models/`: Model wrappers adhering to `ClimateModel` protocol for Baselines, XGBoost, LightGBM, and PyTorch LSTM.
  - `src/evaluation/`: Metric computation (`MAE`, `RMSE`, `MAPE`, `R²`, skill score) and baseline comparison.
  - `src/risk/`: Dual-methodology risk scoring (Empirical quantile-based and Literature +1.5°C/+2.0°C baseline offset).
  - `src/anomaly/`: Multi-station and global anomaly detection via Isolation Forest.
  - `src/explain/`: Tree-based feature importances, SHAP values, and model-grounded narrative derivations.
  - `src/geo/`: Geographic boundaries, station network, risk surface, and pydeck visualization overlays.
  - `src/utils/`: Logging, hashing, file I/O, error handling, caching.

## State Machine
The system uses `AppState` in `dashboard/state.py` with 9 explicit stages:
1. `NOT_LOADED` (0)
2. `LOADED` (1)
3. `VALIDATED` (2)
4. `FEATURES_READY` (3)
5. `MODELS_NOT_TRAINED` (4)
6. `MODELS_TRAINED` (5)
7. `PREDICTION_READY` (6)
8. `ANOMALY_READY` (7)
9. `FULL_ANALYSIS_READY` (8)

Pages enforce state gating, preventing uncomputed metrics or incomplete analysis views.

## Technology Stack
- **Language & Runtime**: Python 3.11 / Python 3.13
- **Framework**: Streamlit
- **ML / Deep Learning**: scikit-learn 1.6, XGBoost, LightGBM, PyTorch (LSTM), Isolation Forest
- **Data & Scientific**: NumPy, pandas, SciPy, statsmodels, pyarrow
- **Visualization**: Plotly, Matplotlib, Seaborn, pydeck
- **Explainability**: SHAP
- **Code Quality & Testing**: pytest, pytest-cov, ruff, mypy
