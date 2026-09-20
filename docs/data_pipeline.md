# CLIMORA AI Data Pipeline Documentation

## 1. Data Ingestion Architecture (`src/data/providers/`)
Each data source is managed by a provider conforming to the `BaseProvider` protocol:
- **`NasaGissProvider`**: Ingests NASA GISTEMP v4 (`GLB.Ts+dSST.csv`). Skips title row (`skiprows=1`), parses leading decimals (e.g. `-.19`), maps sentinel `***` to `NaN`. Baseline is 1951–1980 mean surface temperature anomaly (°C).
- **`NOAACO2Provider`**: Ingests NOAA GML Mauna Loa CO₂ (`co2_mm_mlo.csv`). Skips comment header lines (`#`), handles sentinel missing flags (`ndays=-1`, `sdev=-9.99`, `unc=-9.99`) by converting to `NaN`.
- **`ERSSTNinoProvider`**: Ingests CPC ERSSTv5 Niño indices ascii file (`ersst5.nino.mth.91-20.ascii`). Uses fixed-width parsing to extract `NINO3.4` SST and anomaly series (°C vs 1991–2020 baseline).
- **`OpenMeteoProvider`**: Fetches ERA5 reanalysis daily panel data across Indian monitoring stations (temperature, precipitation sum, relative humidity, shortwave radiation, ET0). Correct variable parameter `precipitation_sum`.
- **`GeoJSONProvider`**: Ingests boundary polygons from Natural Earth (`ne_110m_admin_0_countries.geojson` and `ne_50m_admin_1_states_provinces.geojson`). Handles disputed code sentinel `-99`.

## 2. Ingestion Script (`scripts/download_data.py`)
Features:
- Idempotent execution: Verifies SHA256 checksums before downloading.
- Atomic file writes using `.part` temporary files.
- Exponential backoff retry loop (3 retries).
- `--offline` flag to enforce cache-only loading.
- Writes metadata entries to `data/manifest.json`.

## 3. Data Validation (`src/data/validation.py`)
`DataValidator` performs automated checks against strict schema boundaries:
- Required column presence and data types.
- Datetime monotonicity and continuity.
- Physical range validation:
  - Global anomaly: `[-2.5, 3.5]` °C
  - CO₂ concentration: `[280.0, 500.0]` ppm
  - Niño 3.4 index: `[-5.0, 5.0]` °C
- Duplicates detection and sentinel code auditing.
- Emits a structured `ValidationReport` with status `VALID`, `WARNING`, or `INVALID`.

## 4. Data Cleaning (`src/data/cleaning.py`)
`DataCleaner` executes deterministic cleaning ops:
- Deduplication of identical rows.
- Standardizing datetime fields into monthly UTC timestamps.
- Explicit sentinel-to-NaN conversions.
- Partial year handling (e.g. 2026 incomplete months excluded from training).
- Preserves genuine extreme values while removing physically impossible outliers.
