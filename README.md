# CLIMORA AI — Advanced Climate Intelligence Platform

## 1. Project Overview
CLIMORA AI is an interactive Machine Learning climate intelligence platform built with Python 3.11/3.13 and Streamlit. It performs end-to-end climate time-series forecasting, station-level panel reanalysis, dual-methodology risk classification, Isolation Forest anomaly detection, and SHAP-grounded explainability.

## 2. Problem Statement
Global and regional climate anomalies present complex temporal and spatial patterns that demand rigorous analytical tools. Naive weather applications or black-box predictions lack transparency, reproducibility, and rigorous baseline comparisons. CLIMORA AI addresses this by providing an end-to-end machine learning system trained on historical observations and reanalysis data, evaluated against strict baseline benchmarks (Seasonal Naive and Climatology).

## 3. Objectives
- Maintain 100% data provenance and honesty (zero fake metrics or hardcoded UI numbers).
- Implement strict chronological time-series splitting (Train 1880–1999, Val 2000–2014, Test 2015–2025) with zero feature leakage.
- Train and evaluate multiple model families (Seasonal Naive, Climatology, XGBoost, LightGBM, PyTorch LSTM).
- Provide data-derived risk classification and multivariate anomaly detection across global and station-level panels.
- Deliver an interactive, responsive Streamlit dashboard with pydeck geo-visualizations and SHAP explanations.

## 4. Features
- **Live Overview Dashboard**: Key climate KPIs, historical trend sparklines, state lifecycle chips, and model status indicators.
- **Dataset Explorer**: Schema validation report, column missingness breakdown, raw data preview, and CSV upload ingestion.
- **Climate Analysis**: Interactive seasonal decomposition, trend line fitting, correlation matrices, and distribution boxplots.
- **Feature Engineering**: Feature registry with leakage notes, rolling statistics, lag generators, and transformation pipelines.
- **Interactive Predictions**: Real-time multi-model forecasting with uncertainty intervals and driver explanations.
- **Risk Analysis**: Dual methodology selection (Empirical quantile vs Parisian +1.5°C/+2.0°C policy offset) with risk maps.
- **Anomaly Detection**: Isolation Forest outlier ranking, timeline analysis, and contamination sensitivity benchmarking.
- **Model Performance**: Quantitative metric comparison tables, residual plots, forecast overlays, and artifact metadata inspect.
- **Model Explainability**: Global feature importances, SHAP waterfall plots, and LSTM gradient sensitivity.
- **Methodology & Audit**: Transparent documentation of datasets, baseline offsets, math formulas, and reproducibility hashes.

## 5. Architecture
```
Climate Data -> Ingestion -> Validation -> Cleaning -> Preprocessing -> EDA -> Feature Engineering
             -> Dataset Prep -> ML Models -> Evaluation -> Prediction -> Risk Classification
             -> Anomaly Detection -> Explainability -> Streamlit Dashboard
```

## 6. Technology Stack
- **Language**: Python 3.11 / 3.13
- **Web Interface**: Streamlit, pydeck
- **Machine Learning**: scikit-learn, XGBoost, LightGBM, PyTorch (LSTM), Isolation Forest, SHAP
- **Data Processing**: NumPy, pandas, SciPy, statsmodels, pyarrow
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Testing & Quality**: pytest, pytest-cov, ruff, mypy

## 7. Dataset Specifications

| Attribute | NASA GISS GISTEMP v4 | NOAA GML Mauna Loa CO₂ | CPC ERSSTv5 Niño Indices | Open-Meteo ERA5 Reanalysis |
| :--- | :--- | :--- | :--- | :--- |
| **Source URL** | `https://data.giss.nasa.gov/...` | `https://gml.noaa.gov/...` | `https://www.cpc.ncep.noaa.gov/...` | `https://archive-api.open-meteo.com/...` |
| **License** | NASA Public Domain | NOAA Public Domain | NOAA NWS CPC Public Domain | Open-Meteo / Copernicus |
| **Download Date** | 2026-09-20 | 2026-09-20 | 2026-09-20 | 2026-09-20 |
| **Format** | CSV | CSV | ASCII (Fixed-width) | JSON API |
| **Rows / Cols** | 1,740 / 4 | 822 / 11 | 918 / 11 | 366 per station / 14 |
| **Temporal Range**| 1880-01 to 2024-12 | 1958-03 to 2024-08 | 1950-01 to 2026-06 | 1940-01-01 to 2025-12-31 |
| **Coverage** | Global Mean | Mauna Loa Observatory | Niño 3.4 Region | 12 Indian Stations Panel |
| **Variables** | `anomaly_c` (°C) | `co2_ppm`, deseasonalized | `nino34_sst`, `nino34_anom` | Temp, Precip, RH, Radiation |
| **Target Variable**| Global Temperature Anomaly | CO₂ forcing covariate | ENSO forcing covariate | Station extremes |
| **Missing Values**| Sentinels (`***`, `****`) | Missing values handled | 0 missing | 0 missing |
| **Limitations** | Baseline 1951–1980 | Single observatory | Regional SST summary | Reanalysis product (not raw satellite) |

## 8. Installation
```bash
git clone https://github.com/aghilan28/climora.git
cd climora
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
python scripts/download_data.py
```

## 9. Environment Setup
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Key configuration parameters in `config/settings.py`:
- `CLIMORA_SEED`: Global random seed (default `42`).
- `CLIMORA_LOG_LEVEL`: Logging verbosity (`INFO` / `DEBUG`).

## 10. Running the Application
Launch the Streamlit web application:
```bash
streamlit run app.py
```
Access in browser at `http://localhost:8501`.

## 11. Training Models
To execute a complete headless training run across all model families:
```bash
python scripts/train_models.py
```
Trained model artifacts and metadata will be persisted to `models/trained/`, `models/metadata/`, and `models/metrics/`.

## 12. Model Evaluation
Evaluated test performance on pinned GISTEMP snapshot `737ddcdc78e8ec488d8eaafa316af6c1489d5cf2b4afa054056e6a4ddc3c5001` (Realised Test Window: `2015-01 .. 2024-07`, `n_test=115` months):

| Model Name | MAE (°C) | RMSE (°C) | $R^2$ | Skill Score vs Seasonal Naive |
| :--- | :--- | :--- | :--- | :--- |
| **Seasonal Naive Baseline** | **0.1680** | **0.2180** | **-0.5025** | **0.0000** |
| **PyTorch LSTM** | 0.3649 | 0.3961 | -3.9622 | -0.8170 |
| **LightGBM** | 0.4348 | 0.4634 | -5.7915 | -1.1257 |
| **XGBoost** | 0.5098 | 0.5353 | -8.0643 | -1.4557 |
| **Climatology Baseline** | 1.0405 | 1.0553 | -34.2198 | -3.8407 |

> **Scientific Insight**: On monthly global mean land-ocean temperature anomaly nowcasting, the 12-month Seasonal Naive persistence baseline achieves superior performance (MAE 0.1680 °C) compared to complex ML regression models (MAE ~0.43–0.51 °C). This empirical result demonstrates that month-over-month noise and strong annual persistence dominate short-term global mean anomaly dynamics.

## 13. Dashboard Usage
Navigating the left sidebar allows access to 10 specialized pages:
1. **Overview**: Key indicators, live state machine status, summary text.
2. **Dataset Explorer**: Schema audit report, column statistics, raw data inspect.
3. **Climate Analysis**: Decomposition, trend fitting, seasonality analysis.
4. **Feature Engineering**: Feature registry inspection, trailing window preview.
5. **Predictions**: Interactive multi-model prediction interface with uncertainty intervals.
6. **Risk Analysis**: Dual methodology selection (Empirical quantile vs Parisian +1.5°C/+2.0°C policy offset) with risk maps.
7. **Anomaly Detection**: Isolation Forest score rankings and station timelines.
8. **Model Performance**: Metrics matrix, residual plots, test slice overlays.
9. **Explainability**: SHAP waterfalls and LSTM sensitivity analysis.
10. **About / Methodology**: Complete technical citations, math derivations, and audit hashes.

## 14. Project Structure
```text
climora-ai/
├── app.py                        # Streamlit main entrypoint
├── .streamlit/config.toml        # Server & theme configuration
├── requirements.txt              # Dependency specifications
├── requirements-dev.txt          # Test and dev specifications
├── pyproject.toml                # Tool configurations (ruff, mypy, pytest)
├── setup.py                      # Package configuration
├── Makefile                      # Build automation tasks
├── README.md                     # Project documentation
├── LICENSE                       # MIT License
├── AGENTS.md                     # Permanent engineering rules
├── config/                       # Settings and YAML parameters
├── scripts/                      # Headless CLI utilities (download, train, evaluate, audit)
├── data/                         # Raw, processed, sample data & manifest
├── models/                       # Trained models, metrics, and metadata artifacts
├── src/                          # Modular Python source packages
├── dashboard/                    # Presentation layer and 10 Streamlit page modules
├── tests/                        # Automated unit, integration, and anti-cheat test suite
└── docs/                         # Detailed architecture and methodology documentation
```

## 15. Testing
Run the complete automated test suite:
```bash
pytest --cov=src --cov-fail-under=85
```
Run static analysis linters:
```bash
ruff check .
mypy src dashboard app.py
```

## 16. Deployment
Deployable to Streamlit Community Cloud, local Python virtual environments, or Docker containers. See `docs/deployment.md` for full step-by-step instructions.

## 17. Limitations
- Global mean surface anomaly does not capture microclimate urban heat island effects.
- High-frequency month-over-month noise limits machine learning advantage over 12-month seasonal naive persistence.
- ERA5 reanalysis data represents numerical model outputs constrained by observations, not direct point measurements.

## 18. Future Improvements
- Incorporate spatial gridded NetCDF/HDF5 satellite observations (e.g. CERES / MODIS).
- Implement spatial graph neural networks (GNNs) for regional station graph modeling.
- Expand multi-horizon direct multi-step forecasting ensembles.
