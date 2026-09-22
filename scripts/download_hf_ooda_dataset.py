"""Download only the active OODA train/validation split from Hugging Face.

The target is the local active SFT directory expected by
``scripts/run_qwen4b_lora_sft.sh``. This downloader never requests the
withheld test split or any raw, normalized, canonical, review, model, or
credential artifact.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "sft" / "gis-concept-v1" / "llm_augmented"
ALLOWED_PATTERNS = [
    "train/anonymous_ooda_en.jsonl",
    "validation/anonymous_ooda_en.jsonl",
    "export_manifest.json",
    "README.md",
]


def _validate(path: Path, expected_split: str) -> int:
    scenario_ids: set[str] = set()
    count = 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            scenario_id = row["scenario_id"]
            split = row["split"]
            trace_style = row["view"]["trace_style"]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid OODA record at {path}:{line_no}") from exc
        if split != expected_split or trace_style != "ooda":
            raise ValueError(f"unexpected split or trace style at {path}:{line_no}")
        if not isinstance(scenario_id, str) or not scenario_id or scenario_id in scenario_ids:
            raise ValueError(f"invalid or duplicate scenario_id at {path}:{line_no}")
        scenario_ids.add(scenario_id)
        count += 1
    if not count:
        raise ValueError(f"no records found in {path}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="haishu1121/GIS_SODA_OODA", help="Hugging Face Dataset ID")
    parser.add_argument("--revision", default="main", help="Dataset branch or revision")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="List the allowed files without downloading")
    args = parser.parse_args()

    print(f"Dataset: {args.repo_id}@{args.revision}")
    print(f"Local target: {args.output_dir}")
    print("Allowed files:")
    for pattern in ALLOWED_PATTERNS:
        print(f"  {pattern}")
    if args.dry_run:
        print("Dry run complete; no files were downloaded.")
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install the downloader dependency: python -m pip install -U huggingface_hub") from exc

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        allow_patterns=ALLOWED_PATTERNS,
        local_dir=args.output_dir,
        token=os.environ.get("HF_TOKEN"),
    )
    train_path = args.output_dir / "train" / "anonymous_ooda_en.jsonl"
    validation_path = args.output_dir / "validation" / "anonymous_ooda_en.jsonl"
    manifest_path = args.output_dir / "export_manifest.json"
    if not all(path.is_file() for path in (train_path, validation_path, manifest_path)):
        raise FileNotFoundError("download finished without all required active SFT files")
    train_count = _validate(train_path, "train")
    validation_count = _validate(validation_path, "validation")
    print(f"Download verified: {train_count} train and {validation_count} validation OODA scenarios.")


if __name__ == "__main__":
    main()
