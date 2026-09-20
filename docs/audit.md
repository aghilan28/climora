# Audit Report — CLIMORA AI Environment & Ground Truth Data Probe

## Workspace & Environment State

- **Date**: 2026-09-20
- **Python Version**: `Python 3.11.9` (`C:\Users\Admin\Downloads\APPLICATION\python_env\python.exe`)
- **Git Version**: `2.47.3`
- **OS Platform**: Windows (x86_64)

### Installed Dependencies Audit

| Package | Installed Version | Status |
| :--- | :--- | :--- |
| `numpy` | `2.1.3` | Pre-installed |
| `pandas` | `2.2.3` | Pre-installed |
| `scipy` | `1.17.1` | Pre-installed |
| `scikit-learn` | `1.6.1` | Pre-installed |
| `torch` | `2.14.0` | Pre-installed |
| `matplotlib` | `3.11.2` | Pre-installed |
| `requests` | `2.34.2` | Pre-installed |
| `pydantic` / `pydantic-settings` | `2.13.5` / `2.15.0` | Pre-installed |
| `pytest` | `8.2.2` | Pre-installed |
| `joblib` | `1.4.2` | Pre-installed |
| `streamlit` | `1.64.0` | Installed |
| `xgboost` | `3.2.0` | Installed |
| `lightgbm` | `4.6.0` | Installed |
| `shap` | `0.48.0` | Installed |
| `statsmodels` | `0.14.6` | Installed |
| `pyarrow` | `25.0.1` | Installed |
| `pydeck` | `0.9.3` | Installed |
| `ruff` | `0.15.0` | Installed |
| `mypy` | `1.19.1` | Installed |
| `pytest-cov` | `7.0.0` | Installed |

---

## Data Source Probe Verification Table (D1–D7)

Each dataset endpoint was probed directly via HTTP client with the measured response status, exact byte size, and row/column structure recorded below:

| ID | Data Source | Exact URL / Endpoint | HTTP Status | Measured Size | Row / Col Count | Data Contract Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **D1** | NASA GISS GISTEMP v4 | `https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv` | **200 OK** | 12,887 bytes | 149 lines (1 title + 1 header + 147 years 1880-2026) | Title row at line 1 (`skiprows=1`). Anomalies parsed from decimal strings like `-.19`. Sentinels `***` converted to `NaN`. Baseline: 1951–1980. Partial year 2026 excluded from training target. |
| **D2** | Open-Meteo ERA5 | `https://archive-api.open-meteo.com/v1/archive?latitude=13.0827&longitude=80.2707&daily=...` | **200 OK** | 526 bytes (sample probe) | 365 daily rows per year per station | Endpoint `/v1/archive`. Verified daily variables: `temperature_2m_max`, `temperature_2m_min`, `precipitation_sum`, `relative_humidity_2m_mean`, `cloud_cover_mean`, `soil_temperature_0_to_7cm_mean`, `shortwave_radiation_sum`, `et0_fao_evapotranspiration`. Parameter `timezone=auto`. |
| **D3** | NOAA GML Mauna Loa CO₂ | `https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv` | **200 OK** | 38,732 bytes | 863 lines | 40 comment `#` header lines. Sentinels `ndays=-1`, `sdev=-9.99`, `unc=-9.99` mapped to `NaN`. Monthly average in ppm (1958–2026). |
| **D4** | OWID National CO₂ | `https://ourworldindata.org/grapher/annual-co2-emissions-per-country.csv?v=1&csvType=full&columnKey=all` | **200 OK** | 833,076 bytes | 50,532 rows x 4 cols | Columns: Entity, Code (ISO3), Year, Annual CO₂ emissions. Join key `Code` ↔ GeoJSON `ISO_A3`. |
| **D5a**| Natural Earth 110m | `https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson` | **200 OK** | 838,726 bytes | 177 features | World country boundary polygons (CRS84). Note: Disputed areas use `ISO_A3="-99"`. |
| **D5b**| Natural Earth 50m | `https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_1_states_provinces.geojson` | **200 OK** | 2,325,694 bytes | 4,647 features | Admin-1 state/province boundaries (for India regional overlays). |
| **D6** | NOAA GHCN-Daily | `https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt` | **200 OK** | 10,634,812 bytes | 132,503 station lines | Metadata only (`ghcnd-stations.txt`). 3,807 stations in India. Observation data taken from D2 (Open-Meteo ERA5) to avoid ~1GB compressed file memory crashes. |
| **D7** | CPC ERSSTv5 Niño | `https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii` | **200 OK** | 67,087 bytes | 919 lines (1950–2026) | 10 whitespace-separated columns (`YR`, `MON`, `NINO1+2`, `ANOM`, `NINO3`, `ANOM`, `NINO4`, `ANOM`, `NINO3.4`, `ANOM`). Baseline: 1991–2020. Requires SSL context configuration on Windows NOAA endpoint. |

---

## Deviations & Adaptations

1. **Python Executable**: Using standard Python environment at `C:\Users\Admin\Downloads\APPLICATION\python_env\python.exe` (v3.11.9) on Windows host.
2. **NOAA NOAA SSL Certificate Verification**: The NOAA NCEP D7 endpoint (`www.cpc.ncep.noaa.gov`) produces standard Python SSL verification warnings on Windows hosts unless explicitly configured with fallback SSL context or custom headers. Handled seamlessly in provider module.
3. **No Dead Ends Used**: Confirmed avoidance of Berkeley Earth S3 (403), Met Norway (404), Open-Meteo CO2 (DNS failure), GHCN compressed by_year (OOM risk), and CPC ONI php shell.

---

## Gate A Confirmation

- `docs/audit.md` created with verified probe status codes, byte sizes, line counts, and schema notes for D1–D7.
- Phase A Exit Gate: **PASSED**.
