# CLIMORA AI Feature Engineering Documentation

## 1. Feature Registry & Justifications (`src/features/registry.py`)

| Feature Name | Family | Window | Source Column | Scientific Justification | Leakage Property |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `month` | Calendar | N/A | `date` | Captures annual seasonal cycle | Static |
| `quarter` | Calendar | N/A | `date` | Captures quarterly seasonal shifts | Static |
| `month_sin` | Cyclical | 12 | `month` | Smooth periodic encoding of month ($\sin(2\pi m / 12)$) | Static |
| `month_cos` | Cyclical | 12 | `month` | Smooth periodic encoding of month ($\cos(2\pi m / 12)$) | Static |
| `anomaly_c_lag_1` | Lag | 1 mo | `anomaly_c` | Auto-regressive persistence from previous month | Trailing only |
| `anomaly_c_lag_3` | Lag | 3 mo | `anomaly_c` | Seasonal memory at 1-quarter lead | Trailing only |
| `anomaly_c_lag_6` | Lag | 6 mo | `anomaly_c` | Semi-annual cycle memory | Trailing only |
| `anomaly_c_lag_12` | Lag | 12 mo | `anomaly_c` | Annual cycle memory | Trailing only |
| `anomaly_c_roll_mean_12` | Rolling | 12 mo | `anomaly_c` | 1-year trailing trend moving average (`center=False`) | Trailing only |
| `anomaly_c_roll_std_12` | Rolling | 12 mo | `anomaly_c` | Short-term inter-annual climate variability | Trailing only |
| `anomaly_c_roll_min_12` | Rolling | 12 mo | `anomaly_c` | Minimum anomaly observed in trailing 12 months | Trailing only |
| `anomaly_c_roll_max_12` | Rolling | 12 mo | `anomaly_c` | Peak warm anomaly in trailing 12 months | Trailing only |
| `anomaly_c_roll_mean_60` | Rolling | 60 mo | `anomaly_c` | 5-year multi-annual climate trend | Trailing only |
| `anomaly_c_roll_std_60` | Rolling | 60 mo | `anomaly_c` | Decadal variability scale | Trailing only |
| `anomaly_c_diff_1` | Difference | 1 mo | `anomaly_c` | Month-over-month anomaly rate of change | Trailing only |
| `anomaly_c_diff_12` | Difference | 12 mo | `anomaly_c` | Year-over-year anomaly change | Trailing only |
| `co2_ppm_lag_1` | Covariate Lag | 1 mo | `co2_ppm` | Trailing global atmospheric CO₂ forcing | Trailing only |
| `co2_ppm_diff_12` | Covariate Diff | 12 mo | `co2_ppm` | Annual rate of change in atmospheric CO₂ concentration | Trailing only |
| `nino34_anom_lag_1` | ENSO Lag | 1 mo | `nino34_anom` | Trailing El Niño/Southern Oscillation index at 1-mo lead | Trailing only |
| `nino34_anom_lag_6` | ENSO Lag | 6 mo | `nino34_anom` | ENSO modulation of global temperature at 6-mo lead | Trailing only |

## 2. Anti-Leakage Rules
1. **Trailing Windows Only**: All rolling calculations set `center=False`.
2. **Train-Set Fitting Only**: Imputers, scalers (`StandardScaler`), and selectors are fit on the training split index range ($1880–1999$) and applied via `transform()` to validation and test splits.
3. **No Target Leakage**: Feature engineering operates strictly on historical/lagged fields. Future observations are isolated from training features.
