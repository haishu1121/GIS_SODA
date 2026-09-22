"""Normalize one registered OSM PBF into a projected GeoPackage and GraphML."""

from __future__ import annotations

import argparse
from pathlib import Path

from gis_concept_llm.normalization import StudyBBox, normalize_osm_pbf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pbf", required=True)
    parser.add_argument("--city", required=True)
    parser.add_argument("--bbox", required=True, nargs=4, type=float, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    parser.add_argument("--analysis-crs", default="EPSG:32650")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    counts = normalize_osm_pbf(
        pbf_path=args.pbf,
        output_gpkg=output_dir / f"{args.city}_map.gpkg",
        output_graphml=output_dir / "network.graphml",
        output_manifest=output_dir / "map_manifest.json",
        city=args.city,
        study_bbox=StudyBBox(*args.bbox),
        analysis_crs=args.analysis_crs,
        overwrite=args.overwrite,
    )
    print("normalized layers: " + ", ".join(f"{name}={count}" for name, count in counts.items()))


if __name__ == "__main__":
    main()
