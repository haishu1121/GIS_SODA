"""Register a local raw GIS snapshot without downloading or transforming it."""

from __future__ import annotations

import argparse
from pathlib import Path

from gis_concept_llm.raw import RawSourceManifest, register_raw_source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, help="Source provider token, e.g. osm")
    parser.add_argument("--city", required=True, help="City token, e.g. beijing")
    parser.add_argument("--snapshot-id", required=True, help="Snapshot token, e.g. osm-2026-09-20")
    parser.add_argument("--source-name", required=True, help="Human-readable source name")
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--download-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--bbox", required=True, nargs=4, type=float, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    parser.add_argument("--source-crs", required=True, help="CRS of the source file, e.g. EPSG:4326")
    parser.add_argument("--license", required=True)
    parser.add_argument("--source-path", help="Existing local file or directory to copy into raw storage")
    parser.add_argument("--root", default="data/raw", help="Raw data root")
    args = parser.parse_args()

    manifest = RawSourceManifest(
        provider=args.provider, city=args.city, snapshot_id=args.snapshot_id,
        source_name=args.source_name, source_url=args.source_url, download_date=args.download_date,
        bbox=tuple(args.bbox), source_crs=args.source_crs, license=args.license,
    )
    target = register_raw_source(root=args.root, manifest=manifest, source_path=args.source_path)
    print(f"registered raw source at {target}")


if __name__ == "__main__":
    main()
