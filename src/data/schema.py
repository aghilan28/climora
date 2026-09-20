"""Canonical schema definitions, data types, units, and physical plausibility ranges."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ColumnSchema:
    name: str
    dtype: str
    unit: str
    min_val: float
    max_val: float
    description: str


GISTEMP_SCHEMA = {
    "date": ColumnSchema("date", "datetime64[ns]", "YYYY-MM-DD", 0.0, 0.0, "Observation month start date"),
    "year": ColumnSchema("year", "int64", "year", 1880, 2100, "Observation year"),
    "month": ColumnSchema("month", "int64", "month", 1, 12, "Observation month (1-12)"),
    "anomaly_c": ColumnSchema("anomaly_c", "float64", "°C", -2.5, 3.5, "Global mean surface temperature anomaly vs 1951-1980"),
}

OPEN_METEO_SCHEMA = {
    "date": ColumnSchema("date", "datetime64[ns]", "YYYY-MM-DD", 0.0, 0.0, "Daily date"),
    "temperature_2m_max": ColumnSchema("temperature_2m_max", "float64", "°C", -90.0, 60.0, "Maximum daily temperature at 2m"),
    "temperature_2m_min": ColumnSchema("temperature_2m_min", "float64", "°C", -90.0, 60.0, "Minimum daily temperature at 2m"),
    "temperature_2m_mean": ColumnSchema("temperature_2m_mean", "float64", "°C", -90.0, 60.0, "Mean daily temperature at 2m"),
    "precipitation_sum": ColumnSchema("precipitation_sum", "float64", "mm", 0.0, 2000.0, "Daily total precipitation sum"),
    "relative_humidity_2m_mean": ColumnSchema("relative_humidity_2m_mean", "float64", "%", 0.0, 100.0, "Mean relative humidity"),
    "cloud_cover_mean": ColumnSchema("cloud_cover_mean", "float64", "%", 0.0, 100.0, "Mean total cloud cover"),
    "soil_temperature_0_to_7cm_mean": ColumnSchema("soil_temperature_0_to_7cm_mean", "float64", "°C", -50.0, 70.0, "Mean soil temp 0-7cm"),
    "shortwave_radiation_sum": ColumnSchema("shortwave_radiation_sum", "float64", "MJ/m²", 0.0, 50.0, "Shortwave solar radiation sum"),
    "et0_fao_evapotranspiration": ColumnSchema("et0_fao_evapotranspiration", "float64", "mm", 0.0, 50.0, "Reference evapotranspiration"),
}

NOAA_CO2_SCHEMA = {
    "date": ColumnSchema("date", "datetime64[ns]", "YYYY-MM-DD", 0.0, 0.0, "Month start date"),
    "co2_ppm": ColumnSchema("co2_ppm", "float64", "ppm", 280.0, 550.0, "Monthly mean atmospheric CO2 mole fraction"),
    "co2_deseasonalized_ppm": ColumnSchema("co2_deseasonalized_ppm", "float64", "ppm", 280.0, 550.0, "Deseasonalized monthly mean CO2"),
}

ERSST_NINO_SCHEMA = {
    "date": ColumnSchema("date", "datetime64[ns]", "YYYY-MM-DD", 0.0, 0.0, "Month start date"),
    "nino34_sst": ColumnSchema("nino34_sst", "float64", "°C", 15.0, 35.0, "NINO3.4 region sea surface temperature"),
    "nino34_anom": ColumnSchema("nino34_anom", "float64", "°C", -5.0, 5.0, "NINO3.4 SST anomaly vs 1991-2020 baseline"),
}
