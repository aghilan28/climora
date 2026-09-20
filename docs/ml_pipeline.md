# CLIMORA AI Machine Learning Pipeline Documentation

## 1. Problem Formulation
- **Primary Task**: Time-series forecasting (regression) of global mean surface temperature anomaly (`anomaly_c`).
- **Target Horizons**: 
  - Direct 1-step ahead prediction ($h=1$).
  - Multi-step 12-month ahead horizon ($h=12$).
- **Splits**: Strictly chronological split to preserve time dependence:
  - **Train**: 1880-01 to 1999-12 (1,440 monthly observations)
  - **Validation**: 2000-01 to 2014-12 (180 monthly observations)
  - **Test**: 2015-01 to 2025-12 (132 monthly observations)

## 2. Models (`src/models/`)
1. **Baselines**:
   - `SeasonalNaiveModel`: Predicts anomaly from 12 months prior ($y_t = y_{t-12}$).
   - `ClimatologyModel`: Predicts historical monthly mean for month $m$ computed over training set.
2. **XGBoost (`xgboost_model.py`)**:
   - Booster with histogram tree method (`tree_method="hist"`).
   - Early stopping on validation split to prevent overfitting.
   - Pinned reproducibility seed (`seed=42`).
3. **LightGBM (`lightgbm_model.py`)**:
   - Gradient boosted decision trees using LightGBM regressor.
   - Hyperparameter tuned over validation set.
4. **PyTorch LSTM (`lstm_model.py`)**:
   - 2-layer LSTM network (hidden size 64, dropout 0.2).
   - Sequence window length $L=24$ months.
   - Scaler fit on training slice only prior to window generation.
   - Deterministic execution set via `torch.manual_seed(42)`.

## 3. Artifact Persistence (`src/models/artifacts.py`)
- Native persistence formats: `xgb_model.json`, `lgbm_model.txt`, `lstm_model.pt`.
- Complete JSON metadata files stored in `models/metadata/` containing training timestamps, hyperparameters, dataset SHA256 hashes, feature lists, seed parameters, and package versions.
- Evaluation metrics persisted in `models/metrics/`.
