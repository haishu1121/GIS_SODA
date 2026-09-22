"""Smoke-test evaluator; replace oracle with the production model backend."""

from __future__ import annotations

import argparse
import json

from gis_concept_llm.evaluation import evaluate_atomic
from gis_concept_llm.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    examples = read_jsonl(args.data)
    # The oracle validates pipeline correctness only, not model performance.
    by_question = {example.question: example.gold_answer for example in examples}
    report = evaluate_atomic(examples, lambda prompt: by_question[prompt.split("Question: ", 1)[1].split("\n", 1)[0]])
    print(json.dumps({"count": report.count, "answer_accuracy": report.answer_accuracy, "verifier_accuracy": report.verifier_accuracy, "by_slice": report.by_slice}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
