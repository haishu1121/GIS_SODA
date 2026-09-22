"""Automatic portion of the paper's 5% batch quality-control gate."""

from __future__ import annotations

import argparse
import random

from soda.io import read_jsonl
from soda.rewards import extract_answer


def check(example) -> list[str]:
    errors: list[str] = []
    try:
        example.validate()
    except ValueError as exc:
        errors.append(str(exc))
    if extract_answer(example.completion()) != example.answer:
        errors.append("Act output does not reproduce the declared answer")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--sample-rate", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    examples = read_jsonl(args.data)
    count = max(1, round(len(examples) * args.sample_rate))
    sample = random.Random(args.seed).sample(examples, min(count, len(examples)))
    failures = [(example.id, check(example)) for example in sample if check(example)]
    rate = 1 - len(failures) / len(sample)
    print(f"automatic pass rate: {rate:.1%} ({len(sample) - len(failures)}/{len(sample)})")
    for identifier, reasons in failures:
        print(identifier + ": " + "; ".join(reasons))
    if rate < args.threshold:
        raise SystemExit(1)
    print("Automatic gate passed. Perform double-blind domain review before accepting a training batch.")


if __name__ == "__main__":
    main()
