"""Build only the MVP atomic concept data: direction, topology, connectivity."""

from __future__ import annotations

import argparse

from gis_concept_llm.generators import ConnectivityGenerator, DirectionGenerator, TopologyGenerator
from gis_concept_llm.io import write_jsonl
from gis_concept_llm.taxonomy import ConceptTaxonomy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-skill", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--structural-ood", action="store_true")
    parser.add_argument("--linguistic-ood", action="store_true")
    args = parser.parse_args()
    if args.per_skill < 1:
        parser.error("--per-skill must be positive")
    direction = DirectionGenerator(args.seed)
    topology = TopologyGenerator(args.seed + 1)
    connectivity = ConnectivityGenerator(args.seed + 2)
    examples = [direction.synthetic(args.linguistic_ood) for _ in range(args.per_skill)]
    examples += [topology.synthetic(args.structural_ood) for _ in range(args.per_skill)]
    examples += [connectivity.synthetic(args.structural_ood) for _ in range(args.per_skill)]
    taxonomy = ConceptTaxonomy.load_default()
    for example in examples:
        taxonomy.validate_example(example)
    print(f"wrote {write_jsonl(args.output, examples)} atomic examples to {args.output}")


if __name__ == "__main__":
    main()
