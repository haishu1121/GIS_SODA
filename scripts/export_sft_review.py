"""Export a current scenario-level JSON review package without regenerating SFT data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gis_concept_llm.sft import export_review_sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/sft/gis-concept-v1"))
    parser.add_argument("--mode", choices=("template", "llm_augmented"), required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    path = export_review_sample(args.output_dir, mode=args.mode, batch_id=args.batch_id, sample_size=args.sample_size, seed=args.seed)
    print("review", path)


if __name__ == "__main__":
    main()
