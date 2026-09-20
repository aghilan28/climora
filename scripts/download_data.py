"""Idempotent data downloader script for CLIMORA AI datasets."""

import argparse

from src.data.loaders import (
    load_ersst_nino_data,
    load_geojson_boundaries,
    load_gistemp_data,
    load_noaa_co2_data,
    load_open_meteo_station,
)
from src.data.manifest import ManifestManager
from src.utils.logging import logger


def download_all_datasets(offline: bool = False, verify_only: bool = False) -> None:
    """Fetch all configured datasets, update manifest, and print verification summary."""
    manifest_mgr = ManifestManager()

    logger.info("=== Starting CLIMORA AI Data Ingestion (offline=%s, verify_only=%s) ===", offline, verify_only)

    datasets_to_process = []

    # 1. NASA GISTEMP
    try:
        df_gistemp, entry_gistemp = load_gistemp_data(offline=offline or verify_only)
        manifest_mgr.update_entry("NASA GISS GISTEMP v4", entry_gistemp)
        datasets_to_process.append(("NASA GISS GISTEMP v4", entry_gistemp, df_gistemp))
    except Exception as e:
        logger.error("Failed to process NASA GISTEMP: %s", e)

    # 2. NOAA CO2
    try:
        df_co2, entry_co2 = load_noaa_co2_data(offline=offline or verify_only)
        manifest_mgr.update_entry("NOAA GML Mauna Loa CO2", entry_co2)
        datasets_to_process.append(("NOAA GML Mauna Loa CO2", entry_co2, df_co2))
    except Exception as e:
        logger.error("Failed to process NOAA CO2: %s", e)

    # 3. CPC ERSST Nino
    try:
        df_nino, entry_nino = load_ersst_nino_data(offline=offline or verify_only)
        manifest_mgr.update_entry("CPC ERSSTv5 Nino Indices", entry_nino)
        datasets_to_process.append(("CPC ERSSTv5 Nino Indices", entry_nino, df_nino))
    except Exception as e:
        logger.error("Failed to process CPC ERSST Nino: %s", e)

    # 4. GeoJSON 110m
    try:
        df_geo110, entry_geo110 = load_geojson_boundaries("110m", offline=offline or verify_only)
        manifest_mgr.update_entry("Natural Earth GeoJSON 110m", entry_geo110)
        datasets_to_process.append(("Natural Earth GeoJSON 110m", entry_geo110, df_geo110))
    except Exception as e:
        logger.error("Failed to process GeoJSON 110m: %s", e)

    # 5. Open-Meteo Stations (Primary sample: Chennai)
    try:
        df_om, entry_om = load_open_meteo_station("Chennai", 13.0827, 80.2707, "2020-01-01", "2020-12-31", offline=offline or verify_only)
        manifest_mgr.update_entry("Open-Meteo ERA5 (Chennai)", entry_om)
        datasets_to_process.append(("Open-Meteo ERA5 (Chennai)", entry_om, df_om))
    except Exception as e:
        logger.error("Failed to process Open-Meteo station Chennai: %s", e)

    print("\n=======================================================")
    print("CLIMORA AI — DATAINGESTION & MANIFEST VERIFICATION")
    print("=======================================================")
    for name, entry, df in datasets_to_process:
        print(f"\n[DATASET] {name}")
        print(f"  Source URL   : {entry.get('source_url')}")
        print(f"  SHA256       : {entry.get('sha256')}")
        print(f"  Bytes Read   : {entry.get('raw_byte_count'):,} bytes")
        print(f"  DF Shape     : {df.shape[0]:,} rows x {df.shape[1]} columns")
        print(f"  Missing Cells: {entry.get('missing_cells'):,}")
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
