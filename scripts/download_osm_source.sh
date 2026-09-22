#!/usr/bin/env bash
# Download one declared OSM extract and register it as an immutable raw snapshot.
# Run from any directory; all options are forwarded to download_osm_source.py.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"

if [[ $# -eq 0 ]]; then
  cat <<'USAGE'
Usage:
  scripts/download_osm_source.sh \
    --city beijing \
    --snapshot-id osm-2026-09-20 \
    --download-url https://download.geofabrik.de/asia/china/beijing-latest.osm.pbf \
    --download-date 2026-09-20 \
    --bbox MIN_LON MIN_LAT MAX_LON MAX_LAT

Set PYTHON_BIN to override the Python executable. The source URL, city,
snapshot ID, date, and bbox are always explicit. Existing snapshots are never
overwritten.
USAGE
  exit 2
fi

export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${python_bin}" "${script_dir}/download_osm_source.py" "$@"
