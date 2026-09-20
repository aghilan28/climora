# CLIMORA AI Dataset Documentation

## Primary Backbone (D1: NASA GISS GISTEMP v4)
- **Source URL**: `https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv`
- **License**: NASA Open Data (Public Domain)
- **Temporal Range**: 1880-01 to 2026-08 (1,752 monthly records loaded)
- **Variables**: `anomaly_c` (Global mean surface temperature anomaly in °C)
- **Baseline**: 1951–1980 mean
- **Missing Values**: 4 cells marked with `***` in partial latest year (converted to NaN)
- **File Integrity**: 12,887 bytes, 149 lines (1 title row + 1 header + 147 data years)

## Global Forcing Covariate (D3: NOAA GML Mauna Loa CO₂)
- **Source URL**: `https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv`
- **License**: Public Domain (NOAA GML Data Policy)
- **Temporal Range**: 1958-03 to 2026-08 (822 monthly records)
- **Variables**: `co2_ppm` (Monthly average CO₂ in ppm), `co2_deseasonalized_ppm`
- **Sentinels**: Missing flags `ndays=-1`, `sdev=-9.99`, `unc=-9.99` converted to NaN

## ENSO Covariate (D7: CPC ERSSTv5 Niño Indices)
- **Source URL**: `https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii`
- **License**: Public Domain (NOAA NWS CPC)
- **Temporal Range**: 1950-01 to 2026-06 (918 monthly records)
- **Variables**: `nino34_sst` (°C), `nino34_anom` (°C anomaly vs 1991–2020 baseline)
- **Sentinels**: 0 missing values present

## Spatial Station Panel (D2: Open-Meteo ERA5 Reanalysis)
- **Endpoint**: `https://archive-api.open-meteo.com/v1/archive`
- **License**: Open-Meteo Non-Commercial / Copernicus ERA5 Terms
- **Coverage**: 12 Indian stations (Chennai, Delhi, Mumbai, Kolkata, Bengaluru, Hyderabad, Ahmedabad, Jaipur, Lucknow, Patna, Guwahati, Thiruvananthapuram)
- **Temporal Range**: 1940-01-01 to 2025-12-31 (31,412 daily panel records per station)
- **Variables**: `temperature_2m_max`, `temperature_2m_min`, `temperature_2m_mean`, `precipitation_sum`, `relative_humidity_2m_mean`, `cloud_cover_mean`, `soil_temperature_0_to_7cm_mean`, `shortwave_radiation_sum`, `et0_fao_evapotranspiration`
- **Note**: ERA5 is a reanalysis product combining observational data with numerical modeling.

## Geographic Boundaries (D5: Natural Earth)
- **Source**: `ne_110m_admin_0_countries.geojson` and `ne_50m_admin_1_states_provinces.geojson`
- **License**: Public Domain (Natural Earth)
- **Join Key**: ISO 3166-1 alpha-3 code (`ISO_A3`)
