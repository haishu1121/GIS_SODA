from __future__ import annotations

import unittest

from gis_concept_llm.evaluation.sft_ooda import (
    extract_final_act,
    has_valid_ooda_schema,
    score_completion,
    summarize_scores,
)


class SftOodaEvaluationTests(unittest.TestCase):
    def _shortest_record(self) -> dict:
        return {
            "scenario_id": "beijing-network_shortest_path-000001",
            "example_id": "example-1",
            "acts": {"program": {"optimal_cost_m": 3.0, "path": ["n1", "n2", "n3"]}},
            "messages": [{"role": "user", "content": (
                "Question:\nTask: directed-network shortest path\n\nDirected edges:\n"
                "n1 -> n2, cost=1.000 m\nn2 -> n3, cost=2.000 m\n"
                "n1 -> n3, cost=9.000 m\nQuery source: n1\nQuery target: n3"
            )}],
        }

    def test_scores_a_valid_shortest_path_completion(self) -> None:
        completion = (
            "Observe:\nThe graph has directed costs.\n\nOrient:\nCompare legal route costs.\n\n"
            "Decide:\nn1 -> n2 -> n3 costs 3.000 m.\n\nAct:\n"
            '{"optimal_cost_m":3.0,"path":["n1","n2","n3"]}'
        )
        score = score_completion(self._shortest_record(), completion)
        self.assertTrue(score["act_exact"])
        self.assertTrue(score["ooda_schema_valid"])
        self.assertTrue(score["path_is_legal"])
        self.assertTrue(score["path_is_optimal"])
        self.assertTrue(score["gold_network_scene_valid"])

    def test_rejects_trailing_text_after_act(self) -> None:
        act, error, final = extract_final_act('Act:\n{"connected":true}\nextra')
        self.assertEqual(act, {"connected": True})
        self.assertFalse(final)
        self.assertIn("follows", error)
        self.assertFalse(has_valid_ooda_schema(
            'Observe:\na\nOrient:\nb\nDecide:\nc\nAct:\n{"connected":true}\nextra'
        ))

    def test_summary_includes_task_balanced_accuracy(self) -> None:
        rows = [
            {"task": "location_direction", "act_exact": True, "act_json_valid": True, "ooda_schema_valid": True},
            {"task": "network_connectivity", "act_exact": False, "act_json_valid": True, "ooda_schema_valid": False},
        ]
        summary = summarize_scores(rows)
        self.assertEqual(summary["overall"]["act_exact_accuracy"], 0.5)
        self.assertEqual(summary["macro_act_exact_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
