# ML Problem Definition & Mathematical Formulation

## 1. Task Specification

- **Primary Task**: Chronological time-series forecasting (regression) of global mean monthly surface temperature anomaly (°C).
- **Target Variable**: `anomaly_c` (monthly temperature anomaly in °C relative to the 1951–1980 baseline).
- **Forecast Horizons**:
  1. **1-Step Horizon ($t+1$)**: Predict next month's temperature anomaly $\hat{y}_{t+1}$.
  2. **Multi-Step Horizon ($h=12$)**: Direct multi-step 12-month-ahead forecast $\hat{y}_{t+12}$.

---

## 2. Chronological Dataset Split

| Split Slice | Date Range | Duration | Row Count | Percentage |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | `1880-01-01` to `1999-12-01` | 120 years | **1,440 months** | 81.6% |
| **Validation** | `2000-01-01` to `2014-12-01` | 15 years | **180 months** | 10.2% |
| **Test** | `2015-01-01` to `2025-12-01` | 11 years | **132 months** | 7.5% |
| **Partial 2026**| `2026-01-01` to `2026-12-01` | Partial year | **12 months** | Excluded |
| **Total** | `1880-01-01` to `2026-12-01` | 147 years | **1,764 months** | 100.0% |

---

## 3. Metric Formulations & Baseline Skill Score

Let $y_i$ be the ground truth temperature anomaly and $\hat{y}_i$ be the model prediction for test sample $i \in \{1, \dots, N\}$.

### Evaluation Metrics

1. **Mean Absolute Error (MAE)**:
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$

2. **Root Mean Squared Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$

3. **Mean Absolute Percentage Error (MAPE)**:
   $$\text{MAPE} = \frac{100\%}{N} \sum_{i=1}^N \left| \frac{y_i - \hat{y}_i}{y_i + \epsilon} \right|$$

4. **Coefficient of Determination ($R^2$)**:
   $$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$

5. **Skill Score vs. Seasonal Naive Baseline**:
   $$\text{Skill Score} = 1 - \frac{\text{RMSE}_{\text{model}}}{\text{RMSE}_{\text{SeasonalNaive}}}$$

---

## 4. Anti-Leakage Discipline

- All feature transformers (scalers, imputers) are fit strictly on the **Train slice** only and transformed on Val and Test slices.
- Rolling features set `center=False` and shift by 1.
- No target statistics are incorporated into feature engineering.
