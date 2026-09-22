"""Generate synthetic smoke-test records or an extensible task skeleton."""

from __future__ import annotations

import argparse

from soda.generation import TASKS, SPODGenerator
from soda.io import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-task", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tasks", nargs="*", choices=list(TASKS), help="Default: all 17 paper task interfaces")
    args = parser.parse_args()
    if args.per_task < 1:
        parser.error("--per-task must be positive")
    examples = SPODGenerator(args.seed).generate(args.per_task, args.tasks)
    print(f"wrote {write_jsonl(args.output, examples)} synthetic examples to {args.output}")


if __name__ == "__main__":
    main()
