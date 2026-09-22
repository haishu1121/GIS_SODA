"""Build the template or DeepSeek-augmented GIS Concept SFT branch."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Make direct ``python scripts/build_sft_dataset.py`` execution work without
# requiring an editable installation or a manually-set PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gis_concept_llm.sft import DeepSeekClient, SFTBuildConfig, build_sft_dataset, export_review_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dir", type=Path, default=Path("data/canonical/gis-concept-v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/sft/gis-concept-v1"))
    parser.add_argument("--mode", choices=("template", "llm_augmented"), required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--model", help="Override the model configured in --deepseek-config.")
    parser.add_argument("--deepseek-config", type=Path, default=Path("config/deepseek.local.json"))
    parser.add_argument("--limit-per-task", type=int, help="Use a small smoke-test subset from every task file.")
    parser.add_argument("--llm-max-attempts", type=int, default=2, help="Maximum accepted-output attempts per canonical scenario.")
    parser.add_argument(
        "--topology-geometry-view", choices=("full", "simplified"), default="full",
        help="Use exact canonical GeoJSON or a separately rendered, topology-verified local simplified view.",
    )
    parser.add_argument(
        "--skills", help="Comma-separated canonical skills, for example topology_relation. Default: all skills.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--export-review-sample", type=int, metavar="N", help="Write a readable scenario-level review package after generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = None
    model = args.model or "deepseek-chat"
    if args.mode == "llm_augmented":
        # The local JSON config is the normal credential source. Environment
        # variables remain available only for deliberately ephemeral runs.
        if args.deepseek_config.is_file():
            client = DeepSeekClient.from_config(args.deepseek_config, model_override=args.model)
            model = client.model
        else:
            client = DeepSeekClient.from_environment(model=model)
    config = SFTBuildConfig(
        canonical_dir=args.canonical_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        batch_id=args.batch_id,
        model=model,
        limit_per_task=args.limit_per_task,
        llm_max_attempts=args.llm_max_attempts,
        topology_geometry_view=args.topology_geometry_view,
        skills=tuple(part.strip() for part in args.skills.split(",") if part.strip()) if args.skills else None,
        overwrite=args.overwrite,
    )
    counts = build_sft_dataset(config, client=client)
    print("written", counts)
    if args.export_review_sample:
        review_path = export_review_sample(args.output_dir, mode=args.mode, batch_id=args.batch_id, sample_size=args.export_review_sample)
        print("review_sample", review_path)


if __name__ == "__main__":
    main()
