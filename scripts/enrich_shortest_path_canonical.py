"""Add program-certified runner-up witnesses to one canonical shortest-path file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gis_concept_llm.canonical import enrich_shortest_path_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path", type=Path,
        default=Path("data/canonical/gis-concept-v1/records/beijing/network_shortest_path.jsonl"),
    )
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    enriched = [enrich_shortest_path_record(row) for row in rows]
    temporary = args.input_path.with_suffix(args.input_path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in enriched), encoding="utf-8")
    temporary.replace(args.input_path)
    print(f"enriched_shortest_path_records={len(enriched)} path={args.input_path}")


if __name__ == "__main__":
    main()
