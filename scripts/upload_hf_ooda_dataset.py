"""Upload the active OODA train/validation SFT data to a Hugging Face Dataset.

This script is deliberately narrow: it uploads only the current anonymous
OODA train and validation JSONL files, the export manifest, and an English
Dataset Card. Test, raw GIS, normalized maps, canonical scenarios, review
artifacts, credentials, and model files cannot be selected as upload inputs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_SFT_ROOT = PROJECT_ROOT / "data" / "sft" / "gis-concept-v1" / "llm_augmented"


def _validate_ooda_jsonl(path: Path, *, expected_split: str) -> int:
    """Validate the public-facing training contract before any upload."""
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
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid record at {path}:{line_no}") from exc
        if split != expected_split:
            raise ValueError(f"unexpected split {split!r} at {path}:{line_no}")
        if trace_style != "ooda":
            raise ValueError(f"non-OODA record at {path}:{line_no}")
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError(f"missing scenario_id at {path}:{line_no}")
        if scenario_id in scenario_ids:
            raise ValueError(f"duplicate scenario_id in {path}: {scenario_id}")
        if not isinstance(messages, list) or len(messages) != 2:
            raise ValueError(f"invalid messages layout at {path}:{line_no}")
        scenario_ids.add(scenario_id)
        count += 1
    if not count:
        raise ValueError(f"no records found in {path}")
    return count


def _dataset_card(*, repo_id: str, train_count: int, validation_count: int) -> str:
    """Return the public Dataset Card; model-facing and repository text is English."""
    return f"""---
language:
- en
task_categories:
- text-generation
tags:
- gis
- spatial-reasoning
- ooda
pretty_name: GIS SODA OODA
size_categories:
- 1K<n<10K
---

# GIS SODA OODA

This repository contains the active anonymous English OODA supervision split
for GIS concept reasoning research.

## Contents

| Split | File | Scenarios |
| --- | --- | ---: |
| Train | `train/anonymous_ooda_en.jsonl` | {train_count} |
| Validation | `validation/anonymous_ooda_en.jsonl` | {validation_count} |

Each record has one user message containing program-rendered spatial facts and
one assistant message containing `Observe`, `Orient`, `Decide`, and a final
program-verified `Act`.

## Data construction

Scenarios are grounded in OpenStreetMap-derived Beijing map data. Spatial facts
and final `Act` answers are computed and validated by GIS, geometry, or graph
programs. LLM-generated content is limited to English question wording and
OODA expression; it is accepted only when its reported answer matches the
program-derived answer.

The published files contain only anonymous OODA train and validation views.
Raw GIS files, normalized maps, canonical scenarios, review artifacts, model
checkpoints, credentials, and the held-out test split are intentionally absent.

## Provenance and licence notice

OpenStreetMap is the upstream geographic source and requires attribution under
the Open Database License (ODbL). Consult `export_manifest.json` and the
project source manifests for the exact city snapshot, transformation pipeline,
and provenance. Users are responsible for complying with upstream licensing
and attribution obligations before redistribution or derivative use.

## Intended use and limitations

This dataset is intended for research on GIS concept supervision and spatial
reasoning. It is not a navigation product, an authoritative map, or a source
of real-world operational routing decisions. The test split is withheld and
must not be reconstructed from this repository.

## Repository

Dataset repository: `{repo_id}`
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Hugging Face Dataset ID, e.g. haishu1121/GIS_SODA_OODA")
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument("--private", dest="private", action="store_true", help="Create the Dataset repo as private (default)")
    visibility.add_argument("--public", dest="private", action="store_false", help="Create the Dataset repo as public")
    parser.set_defaults(private=True)
    parser.add_argument("--revision", default="main", help="Target branch name")
    parser.add_argument("--dry-run", action="store_true", help="Validate and list files without creating or uploading")
    args = parser.parse_args()

    train_path = ACTIVE_SFT_ROOT / "train" / "anonymous_ooda_en.jsonl"
    validation_path = ACTIVE_SFT_ROOT / "validation" / "anonymous_ooda_en.jsonl"
    manifest_path = ACTIVE_SFT_ROOT / "export_manifest.json"
    for path in (train_path, validation_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"required active SFT file is missing: {path}")

    train_count = _validate_ooda_jsonl(train_path, expected_split="train")
    validation_count = _validate_ooda_jsonl(validation_path, expected_split="validation")
    card = _dataset_card(repo_id=args.repo_id, train_count=train_count, validation_count=validation_count)
    uploads = [
        (train_path, "train/anonymous_ooda_en.jsonl"),
        (validation_path, "validation/anonymous_ooda_en.jsonl"),
        (manifest_path, "export_manifest.json"),
    ]

    print(f"Validated {train_count} train and {validation_count} validation OODA scenarios.")
    print("Files allowed for upload:")
    for source, destination in uploads:
        print(f"  {source.relative_to(PROJECT_ROOT)} -> {destination}")
    print("  generated Dataset Card -> README.md")
    if args.dry_run:
        print("Dry run complete; no Hugging Face repository was created or modified.")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Install the uploader dependency: python -m pip install -U huggingface_hub") from exc

    # HfApi uses HF_TOKEN when set, otherwise a token created by `hf auth login`.
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    for source, destination in uploads:
        api.upload_file(
            path_or_fileobj=str(source),
            path_in_repo=destination,
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            commit_message=f"Upload {destination}",
        )
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        commit_message="Add GIS SODA OODA Dataset Card",
    )
    print(f"Upload complete: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
