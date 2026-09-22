"""Build language-free canonical GIS concept records from a normalized city map."""

from __future__ import annotations

import argparse
from pathlib import Path

from gis_concept_llm.canonical import CanonicalBuildConfig, build_canonical_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", required=True)
    parser.add_argument("--normalized-dir", required=True)
    parser.add_argument("--output-dir", default="data/canonical/gis-concept-v1")
    parser.add_argument("--per-task", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--max-graph-nodes", type=int, default=14)
    parser.add_argument("--max-graph-edges", type=int, default=28)
    parser.add_argument("--max-geometry-vertices", type=int, default=160)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    counts = build_canonical_dataset(CanonicalBuildConfig(
        city=args.city,
        normalized_dir=Path(args.normalized_dir),
        output_dir=Path(args.output_dir),
        per_task=args.per_task,
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        max_graph_nodes=args.max_graph_nodes,
        max_graph_edges=args.max_graph_edges,
        max_geometry_vertices=args.max_geometry_vertices,
        overwrite=args.overwrite,
    ))
    print("canonical records: " + ", ".join(f"{name}={count}" for name, count in counts.items()))


if __name__ == "__main__":
    main()
