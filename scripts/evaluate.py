"""Evaluate exact final answers by task and difficulty tier."""

from __future__ import annotations

import argparse
from collections import defaultdict

from soda.io import read_jsonl
from soda.rewards import extract_answer, normalize_answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--predictor", choices=["oracle"], default="oracle", help="Replace this adapter with your ModelBackend in production.")
    args = parser.parse_args()
    examples = read_jsonl(args.data)
    groups: dict[str, list[bool]] = defaultdict(list)
    for example in examples:
        # Oracle deliberately tests the evaluator/data contract, not model quality.
        predicted = extract_answer(example.completion())
        correct = normalize_answer(predicted) == normalize_answer(example.answer)
        groups[f"task/{example.task}"].append(correct)
        groups[f"tier/{example.tier}"].append(correct)
        groups["overall"].append(correct)
    for name in sorted(groups):
        values = groups[name]
        print(f"{name:42} {sum(values) / len(values):.1%} ({sum(values)}/{len(values)})")


if __name__ == "__main__":
    main()
