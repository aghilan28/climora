# CLIMORA AI Explainability Documentation

## Model-Grounded Interpretability (`src/explain/`)

### 1. Tree Feature Importance (`src/explain/tree_importance.py`)
Extracts native importances for XGBoost and LightGBM models:
- **Gain**: Improvement in accuracy brought by a feature to the branches it split.
- **Weight**: Number of times a feature is used to split data across all trees.
- **Cover**: Relative quantity of observations concerned by splits using a feature.

### 2. SHAP Interpretability (`src/explain/shap_wrapper.py`)
Uses `shap.TreeExplainer` to compute exact additive feature attribution values ($\phi_i$):
- **Global Feature Importance**: Mean absolute SHAP values across evaluation dataset.
- **Local Predictions**: Individual prediction attribution waterfall breakdown showing exact signed impacts ($+\Delta\text{ °C}$ or $-\Delta\text{ °C}$) of input variables.
- Graceful degradation: If `shap` library is absent, falls back to native gain importances with an explicit notice.

### 3. LSTM Input Sensitivity (`src/explain/lst_sensitivity.py`)
Computes input window gradient sensitivity over PyTorch LSTM sequence dimensions ($L=24 \times F=27$):
- Calculates $\frac{\partial \hat{y}}{\partial X_{t, f}}$ via auto-differentiation backpropagation.
- Verifies gradient accuracy against finite-difference numerical checks.

### 4. Model-Grounded Narratives (`src/explain/narrative.py`)
Generates natural language explanation summaries derived strictly from computed SHAP or gain contribution arrays. Never invents claims not backed by array metrics. Emits `EXPLANATION UNAVAILABLE: <reason>` if contribution metrics are uncomputed.
