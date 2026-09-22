"""Replace one SFT branch's shortest-path records with an audited replacement branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gis_concept_llm.sft import validate_sft_record


SPLITS = ("train", "validation", "test")
STYLES = ("qa", "minimal", "ooda")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dir", type=Path, default=Path("data/canonical/gis-concept-v1"))
    parser.add_argument("--main-root", type=Path, default=Path("data/sft/gis-concept-v1/llm_augmented"))
    parser.add_argument("--replacement-root", type=Path, required=True)
    parser.add_argument("--archive-id", default="batch-005-shortest-path-runner-up-replacement")
    args = parser.parse_args()

    canonical_rows = read_jsonl(args.canonical_dir / "records" / "beijing" / "network_shortest_path.jsonl")
    canonical_by_id = {row["scenario_id"]: row for row in canonical_rows}
    splits = {row["scenario_id"]: row["split"] for row in read_jsonl(args.canonical_dir / "splits" / "scenario_split.jsonl")}
    expected_by_split = {
        split: {scenario_id for scenario_id in canonical_by_id if splits[scenario_id] == split}
        for split in SPLITS
    }
    archive_root = args.main_root / "archive" / args.archive_id
    replacement_counts: dict[str, int] = {}

    for split in SPLITS:
        for style in STYLES:
            main_path = args.main_root / split / f"anonymous_{style}_en.jsonl"
            replacement_path = args.replacement_root / split / f"anonymous_{style}_en.jsonl"
            current = read_jsonl(main_path)
            replacement = read_jsonl(replacement_path)
            current_shortest = [row for row in current if row["scenario_id"] in canonical_by_id]
            replacement_ids = {row["scenario_id"] for row in replacement}
            if {row["scenario_id"] for row in current_shortest} != expected_by_split[split]:
                raise ValueError(f"main branch shortest-path IDs do not match canonical {split} split")
            if replacement_ids != expected_by_split[split]:
                raise ValueError(f"replacement branch shortest-path IDs do not match canonical {split} split")
            for row in replacement:
                if row["view"]["trace_style"] != style:
                    raise ValueError(f"replacement view style mismatch in {replacement_path}")
                validate_sft_record(row, canonical_by_id[row["scenario_id"]])
            by_id = {row["scenario_id"]: row for row in replacement}
            merged = [by_id[row["scenario_id"]] if row["scenario_id"] in by_id else row for row in current]
            archive_path = archive_root / split / f"anonymous_{style}_en.jsonl"
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            write_jsonl_atomic(archive_path, current_shortest)
            write_jsonl_atomic(main_path, merged)
            replacement_counts[f"{split}/{style}"] = len(replacement)

    manifest_path = args.main_root / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    history = manifest.setdefault("task_replacements", [])
    history.append({
        "task": "network_shortest_path",
        "replacement_source": str(args.replacement_root),
        "archive": str(archive_root),
        "records_by_split_and_style": replacement_counts,
        "prompt_version": "ooda-renderer/v4",
        "canonical_witness": "program-certified runner_up_path and runner_up_cost_m",
    })
    manifest["prompt_version"] = "mixed: ooda-renderer/v3; shortest_path=ooda-renderer/v4"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("replaced", replacement_counts)
    print("archived", archive_root)


if __name__ == "__main__":
    main()
