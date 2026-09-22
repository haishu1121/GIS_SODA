"""Upload only the held-out OODA test split to a separate private HF Dataset.

This utility intentionally cannot upload train, validation, raw GIS,
canonical, review, model, or credential files.  It refuses the active training
Dataset repository ID so test data cannot be mixed into the training release.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = PROJECT_ROOT / "data" / "sft" / "gis-concept-v1" / "llm_augmented" / "test" / "anonymous_ooda_en.jsonl"
TRAINING_REPO_ID = "haishu1121/GIS_SODA_OODA"


def _validate_test(path: Path) -> int:
    scenario_ids: set[str] = set()
    count = 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record: dict[str, Any] = json.loads(line)
            scenario_id = record["scenario_id"]
            split = record["split"]
            trace_style = record["view"]["trace_style"]
            messages = record["messages"]
            program_act = record["acts"]["program"]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid test record at {path}:{line_no}") from exc
        if split != "test" or trace_style != "ooda":
            raise ValueError(f"non-test or non-OODA record at {path}:{line_no}")
        if not isinstance(scenario_id, str) or not scenario_id or scenario_id in scenario_ids:
            raise ValueError(f"invalid or duplicate scenario_id at {path}:{line_no}")
        if not isinstance(messages, list) or len(messages) != 2 or not isinstance(program_act, dict):
            raise ValueError(f"invalid messages or program Act at {path}:{line_no}")
        scenario_ids.add(scenario_id)
        count += 1
    if not count:
        raise ValueError(f"no records found in {path}")
    return count


def _dataset_card(repo_id: str, count: int) -> str:
    return f"""---
language:
- en
tags:
- gis
- spatial-reasoning
- ooda
private: true
---

# GIS SODA OODA Held-out Test

This is a private, evaluation-only repository containing {count} anonymous
English GIS Concept OODA test scenarios.

It is not a training dataset. Do not merge it into the public or private
train/validation dataset, and do not use it for checkpoint selection,
hyperparameter tuning, or prompt design. It is retained only for one-time
post-training evaluation of a frozen experiment.

Dataset repository: `{repo_id}`
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="A separate private HF Dataset ID, e.g. haishu1121/GIS_SODA_OODA_TEST_PRIVATE")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--dry-run", action="store_true", help="Validate and list files without accessing Hugging Face")
    args = parser.parse_args()
    if args.repo_id == TRAINING_REPO_ID:
        parser.error(f"refusing the active training Dataset repo; use a separate private repo, not {TRAINING_REPO_ID}")
    if not TEST_PATH.is_file():
        raise FileNotFoundError(f"held-out test file is missing: {TEST_PATH}")
    count = _validate_test(TEST_PATH)
    print(f"Validated {count} held-out OODA test scenarios.")
    print(f"Only upload allowed: {TEST_PATH.relative_to(PROJECT_ROOT)} -> test/anonymous_ooda_en.jsonl")
    print("Dataset Card -> README.md")
    if args.dry_run:
        print("Dry run complete; no Hugging Face repository was created or modified.")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Install the uploader dependency: python -m pip install -U huggingface_hub") from exc
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=True, exist_ok=True)
    info = api.repo_info(repo_id=args.repo_id, repo_type="dataset", revision=args.revision)
    if getattr(info, "private", None) is not True:
        raise RuntimeError(f"refusing to upload held-out test data: {args.repo_id} is not confirmed private")
    api.upload_file(
        path_or_fileobj=str(TEST_PATH),
        path_in_repo="test/anonymous_ooda_en.jsonl",
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        commit_message="Upload held-out OODA test split",
    )
    api.upload_file(
        path_or_fileobj=_dataset_card(args.repo_id, count).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        commit_message="Add evaluation-only Dataset Card",
    )
    print(f"Private held-out test upload complete: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
