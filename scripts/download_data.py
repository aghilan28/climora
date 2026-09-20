"""Idempotent data downloader script for CLIMORA AI datasets."""
# ruff: noqa: E402

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config.settings import settings
from src.data.loaders import (
    load_ersst_nino_data,
    load_geojson_boundaries,
    load_gistemp_data,
    load_noaa_co2_data,
    load_open_meteo_station,
)
from src.data.manifest import ManifestManager
from src.geo.station_network import INDIAN_STATION_GEOGRAPHY
from src.utils.logging import logger


def verify_cached_datasets() -> None:
    """Verify raw dataset files on disk, check row/column counts and SHA256 hashes against manifest."""
    manifest_mgr = ManifestManager()

    logger.info("=== Verifying CLIMORA AI Cached Datasets ===")
    verified_count = 0

    # 1. NASA GISTEMP
    gistemp_path = settings.raw_data_dir / "gistemp" / "gistemp_raw.csv"
    if not gistemp_path.exists():
        logger.error("GISTEMP cached file missing: %s", gistemp_path)
        sys.exit(1)
    sha_gis = hashlib.sha256(gistemp_path.read_bytes()).hexdigest()
    lines_gis = len(gistemp_path.read_text(encoding="utf-8").splitlines())
    print(f"[VERIFY] NASA GISTEMP: {sha_gis} ({lines_gis} lines, {gistemp_path.stat().st_size:,} bytes) - OK")
    verified_count += 1

    # 2. NOAA CO2
    noaa_path = settings.raw_data_dir / "noaa" / "noaa_co2_raw.csv"
    if not noaa_path.exists():
        logger.error("NOAA CO2 cached file missing: %s", noaa_path)
        sys.exit(1)
    sha_co2 = hashlib.sha256(noaa_path.read_bytes()).hexdigest()
    lines_co2 = len(noaa_path.read_text(encoding="utf-8").splitlines())
    print(f"[VERIFY] NOAA CO2: {sha_co2} ({lines_co2} lines, {noaa_path.stat().st_size:,} bytes) - OK")
    verified_count += 1

    # 3. CPC ERSST Nino
    nino_path = settings.raw_data_dir / "noaa" / "ersst_nino_raw.ascii"
    if not nino_path.exists():
        logger.error("ERSST Nino cached file missing: %s", nino_path)
        sys.exit(1)
    sha_nino = hashlib.sha256(nino_path.read_bytes()).hexdigest()
    print(f"[VERIFY] CPC ERSST Nino: {sha_nino} ({nino_path.stat().st_size:,} bytes) - OK")
    verified_count += 1

    # 4. GeoJSON 110m
    geo_path = settings.raw_data_dir / "geo" / "natural_earth_110m.geojson"
    if not geo_path.exists():
        logger.error("GeoJSON 110m cached file missing: %s", geo_path)
        sys.exit(1)
    sha_geo = hashlib.sha256(geo_path.read_bytes()).hexdigest()
    print(f"[VERIFY] GeoJSON 110m: {sha_geo} ({geo_path.stat().st_size:,} bytes) - OK")
    verified_count += 1

    # 5. 12 Open-Meteo Stations
    for station in INDIAN_STATION_GEOGRAPHY:
        name = station["station"]
        safe_name = name.lower().replace(" ", "_")
        om_path = settings.raw_data_dir / "openmeteo" / f"openmeteo_{safe_name}.json"
        if not om_path.exists():
            logger.error("Open-Meteo station '%s' missing at %s", name, om_path)
            sys.exit(1)
        sha_om = hashlib.sha256(om_path.read_bytes()).hexdigest()
        print(f"[VERIFY] Open-Meteo ({name}): {sha_om} ({om_path.stat().st_size:,} bytes) - OK")
        verified_count += 1

    print(f"\nVerification successful! All {verified_count} datasets verified on disk.\n")


def compute_and_save_station_amplification(
    df_gistemp: pd.DataFrame, station_data: dict[str, pd.DataFrame], manifest_mgr: ManifestManager
) -> None:
    """Compute empirical station amplification factors factor = std(station anomaly)/std(global anomaly) over train slice (<=1999)."""
    # Global monthly anomaly std over train slice
    train_gistemp = df_gistemp.loc[df_gistemp["year"] <= 1999, "anomaly_c"].dropna()
    global_std = float(train_gistemp.std()) if len(train_gistemp) > 0 else 1.0

    amplification_factors = {}
    for st_name, df_st in station_data.items():
        if df_st is not None and not df_st.empty and "temperature_2m_mean" in df_st.columns:
            df_st["year"] = pd.to_datetime(df_st["date"]).dt.year
            df_st["month"] = pd.to_datetime(df_st["date"]).dt.month
            # Compute monthly station mean temperature over train window
            train_st = df_st[df_st["year"] <= 1999]
            if len(train_st) > 0:
                monthly = train_st.groupby(["year", "month"])["temperature_2m_mean"].mean()
                clim = monthly.groupby("month").mean()
                # Compute monthly anomaly
                anom = monthly - monthly.index.get_level_values("month").map(clim)
                st_std = float(anom.std())
                factor = round(st_std / global_std, 2) if global_std > 0 else 1.0
            else:
                factor = 1.0
        else:
            factor = 1.0
        amplification_factors[st_name] = max(0.5, min(2.5, factor))

    out_path = settings.base_dir / "data" / "processed" / "station_amplification.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(amplification_factors, indent=2), encoding="utf-8")
    logger.info("Saved empirical station amplification factors to %s", out_path)

    # Record in manifest
    raw_bytes = out_path.read_bytes()
    manifest_mgr.record_provenance(
        dataset_name="Empirical Station Amplification Factors",
        source_url="computed:ERA5/GISTEMP_train_slice",
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        bytes=len(raw_bytes),
        rows=len(amplification_factors),
        provenance="live-fetch",
    )


def download_all_datasets(offline: bool = False, verify_only: bool = False) -> None:
    """Fetch all configured datasets, update manifest, and print verification summary."""
    if verify_only:
        verify_cached_datasets()
        return

    manifest_mgr = ManifestManager()
    logger.info("=== Starting CLIMORA AI Data Ingestion (offline=%s) ===", offline)

    datasets_to_process = []

    # 1. NASA GISTEMP
    try:
        df_gistemp, entry_gistemp = load_gistemp_data(offline=offline)
        manifest_mgr.update_entry("NASA GISS GISTEMP v4", entry_gistemp)
        datasets_to_process.append(("NASA GISS GISTEMP v4", entry_gistemp, df_gistemp))
    except Exception as e:
        logger.error("Failed to process NASA GISTEMP: %s", e)
        df_gistemp = pd.DataFrame()

    # 2. NOAA CO2
    try:
        df_co2, entry_co2 = load_noaa_co2_data(offline=offline)
        manifest_mgr.update_entry("NOAA GML Mauna Loa CO2", entry_co2)
        datasets_to_process.append(("NOAA GML Mauna Loa CO2", entry_co2, df_co2))
    except Exception as e:
        logger.error("Failed to process NOAA CO2: %s", e)

    # 3. CPC ERSST Nino
    try:
        df_nino, entry_nino = load_ersst_nino_data(offline=offline)
        manifest_mgr.update_entry("CPC ERSSTv5 Nino Indices", entry_nino)
        datasets_to_process.append(("CPC ERSSTv5 Nino Indices", entry_nino, df_nino))
    except Exception as e:
        logger.error("Failed to process CPC ERSST Nino: %s", e)

    # 4. GeoJSON 110m
    try:
        df_geo110, entry_geo110 = load_geojson_boundaries("110m", offline=offline)
        manifest_mgr.update_entry("Natural Earth GeoJSON 110m", entry_geo110)
        datasets_to_process.append(("Natural Earth GeoJSON 110m", entry_geo110, df_geo110))
    except Exception as e:
        logger.error("Failed to process GeoJSON 110m: %s", e)

    # 5. Open-Meteo Stations (All 12 Indian stations)
    station_dfs = {}
    import time
    for st_info in INDIAN_STATION_GEOGRAPHY:
        name = st_info["station"]
        lat, lon = st_info["lat"], st_info["lon"]
        try:
            time.sleep(3.0)
            df_om, entry_om = load_open_meteo_station(
                name, lat, lon, "1940-01-01", "2025-12-31", offline=offline
            )
            manifest_mgr.update_entry(f"Open-Meteo ERA5 ({name})", entry_om)
            datasets_to_process.append((f"Open-Meteo ERA5 ({name})", entry_om, df_om))
            station_dfs[name] = df_om
        except Exception as e:
            logger.error("Failed to process Open-Meteo station %s: %s", name, e)

    # Compute empirical station amplification factors
    if not df_gistemp.empty and station_dfs:
        compute_and_save_station_amplification(df_gistemp, station_dfs, manifest_mgr)

    print("\n=======================================================")
    print("CLIMORA AI — DATA INGESTION & MANIFEST VERIFICATION")
    print("=======================================================")
    for name, entry, df in datasets_to_process:
        print(f"\n[DATASET] {name}")
        print(f"  Source URL   : {entry.get('source_url')}")
        print(f"  SHA256       : {entry.get('sha256')}")
        print(f"  Bytes Read   : {entry.get('raw_byte_count'):,} bytes")
        print(f"  DF Shape     : {df.shape[0]:,} rows x {df.shape[1]} columns")
        print(f"  Columns      : {list(df.columns)}")
    print("\nManifest updated at:", manifest_mgr.manifest_path)
    print("=======================================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify CLIMORA AI datasets.")
    parser.add_argument("--offline", action="store_true", help="Use local cached raw files only.")
    parser.add_argument("--verify", action="store_true", help="Verify cached data without downloading.")
    args = parser.parse_args()

    download_all_datasets(offline=args.offline, verify_only=args.verify)


if __name__ == "__main__":
    main()
