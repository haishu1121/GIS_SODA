"""Recover stored DeepSeek outputs after an SFT validator improvement."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gis_concept_llm.sft import SFTBuildConfig, recover_rejected_sft


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dir", type=Path, default=Path("data/canonical/gis-concept-v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/sft/gis-concept-v1"))
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--rejected-path", type=Path, required=True)
    args = parser.parse_args()
    result = recover_rejected_sft(
        SFTBuildConfig(
            canonical_dir=args.canonical_dir,
            output_dir=args.output_dir,
            mode="llm_augmented",
            batch_id=args.batch_id,
        ),
        rejected_path=args.rejected_path,
    )
    print("recovery", result)


if __name__ == "__main__":
    main()
