"""Download one declared OSM extract and register it as an immutable raw snapshot."""

from __future__ import annotations

import argparse

from gis_concept_llm.raw import RawSourceManifest, download_and_register_raw_source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", required=True, help="City token, e.g. beijing")
    parser.add_argument("--snapshot-id", required=True, help="Snapshot token, e.g. osm-2026-09-20")
    parser.add_argument("--download-url", required=True, help="Exact OSM extract URL")
    parser.add_argument("--download-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--bbox", required=True, nargs=4, type=float, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    parser.add_argument("--source-crs", default="EPSG:4326")
    parser.add_argument("--root", default="data/raw")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args()

    manifest = RawSourceManifest(
        provider="osm", city=args.city, snapshot_id=args.snapshot_id,
        source_name="OpenStreetMap", source_url=args.download_url, download_date=args.download_date,
        bbox=tuple(args.bbox), source_crs=args.source_crs, license="ODbL-1.0",
    )
    target = download_and_register_raw_source(
        root=args.root, manifest=manifest, download_url=args.download_url, timeout_seconds=args.timeout_seconds,
    )
    print(f"downloaded and registered OSM source at {target}")


if __name__ == "__main__":
    main()
