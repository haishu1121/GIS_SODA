"""Download only a separate private held-out OODA test Dataset to the server."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "sft" / "gis-concept-v1" / "llm_augmented"
# Do not download the test repository README into the active SFT root: that
# root may already hold the training Dataset Card downloaded from a different
# private repository.
ALLOWED_PATTERNS = ["test/anonymous_ooda_en.jsonl"]


def _validate_test(path: Path) -> int:
    scenario_ids: set[str] = set()
    count = 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            scenario_id = record["scenario_id"]
            split = record["split"]
            trace_style = record["view"]["trace_style"]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid test record at {path}:{line_no}") from exc
        if split != "test" or trace_style != "ooda":
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
    parser.add_argument("--repo-id", required=True, help="Separate private held-out HF Dataset ID")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing local held-out test file")
    parser.add_argument("--dry-run", action="store_true", help="List the allowed files without downloading")
    args = parser.parse_args()
    target = args.output_dir / "test" / "anonymous_ooda_en.jsonl"
    print(f"Dataset: {args.repo_id}@{args.revision}")
    print(f"Local target: {target}")
    print("Allowed files:")
    for pattern in ALLOWED_PATTERNS:
        print(f"  {pattern}")
    if args.dry_run:
        print("Dry run complete; no files were downloaded.")
        return
    if target.exists() and not args.overwrite:
        raise FileExistsError(f"held-out test already exists: {target}; pass --overwrite to replace it")

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
    if not target.is_file():
        raise FileNotFoundError(f"download finished without the held-out test JSONL: {target}")
    count = _validate_test(target)
    print(f"Held-out test download verified: {count} OODA scenarios.")


if __name__ == "__main__":
    main()
