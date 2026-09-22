"""Tests for canonical-to-SFT rendering and DeepSeek response safeguards."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from gis_concept_llm.canonical import enrich_shortest_path_record
from gis_concept_llm.sft import DeepSeekClient, SFTBuildConfig, _make_views, _scene_text, _template_question, _topology_geometry_for_render, build_sft_dataset, export_ooda_review_markdown, export_review_sample, validate_sft_record


ROOT = Path(__file__).resolve().parents[1]


def canonical(name: str) -> dict:
    path = ROOT / "data" / "canonical" / "gis-concept-v1" / "records" / "beijing" / name
    return json.loads(path.read_text(encoding="utf-8").splitlines()[0])


def canonical_matching(name: str, predicate) -> dict:
    path = ROOT / "data" / "canonical" / "gis-concept-v1" / "records" / "beijing" / name
    return next(row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()) if predicate(row))


class SFTTests(unittest.TestCase):
    def test_deepseek_client_loads_local_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deepseek.local.json"
            path.write_text(json.dumps({
                "api_key": "test-key-not-a-real-secret",
                "model": "deepseek-chat",
                "endpoint": "https://api.deepseek.com/chat/completions",
                "timeout_s": 30,
            }), encoding="utf-8")
            client = DeepSeekClient.from_config(path)
        self.assertEqual(client.model, "deepseek-chat")
        self.assertEqual(client.timeout_s, 30)
        self.assertEqual(client.temperature, 0.7)

    def test_template_views_keep_program_act_and_validate(self) -> None:
        record = canonical("network_connectivity.jsonl")
        config = SFTBuildConfig(Path("canonical"), Path("sft"), "template", "batch-test")
        views = _make_views(record, "train", config, None)
        self.assertEqual(set(views), {"qa", "minimal", "ooda"})
        for view in views.values():
            self.assertEqual(view["acts"]["program"], record["gold"]["answer"])
            self.assertIsNone(view["acts"]["llm"])
            validate_sft_record(view, record)
        self.assertNotIn("Observe:", views["qa"]["messages"][1]["content"])
        self.assertTrue(views["ooda"]["messages"][0]["content"].startswith("Question:\n" + _scene_text(record)))
        self.assertIn("n1 -> n13", views["ooda"]["messages"][0]["content"])
        self.assertNotIn("N1 -> N13", views["ooda"]["messages"][0]["content"])

    def test_template_direction_uses_natural_compass_word(self) -> None:
        record = canonical("location_direction.jsonl")
        config = SFTBuildConfig(Path("canonical"), Path("sft"), "template", "batch-test")
        view = _make_views(record, "train", config, None)["ooda"]
        self.assertIn("south of p1", view["messages"][1]["content"])
        validate_sft_record(view, record)

    def test_validator_rejects_a_user_message_without_scene_facts(self) -> None:
        record = canonical("network_connectivity.jsonl")
        config = SFTBuildConfig(Path("canonical"), Path("sft"), "template", "batch-test")
        view = _make_views(record, "train", config, None)["ooda"]
        view["messages"][0]["content"] = _template_question(record)
        with self.assertRaisesRegex(ValueError, "preserve the program-rendered scene text"):
            validate_sft_record(view, record)

    def test_llm_answer_must_match_gold(self) -> None:
        record = canonical("location_direction.jsonl")
        config = SFTBuildConfig(Path("canonical"), Path("sft"), "llm_augmented", "batch-test")
        llm = {
            "question": "How far is p1 from p2 under the Euclidean metric?",
            "observe": "The points use one projected coordinate system.",
            "orient": "Compare the target with the reference.",
            "decide": "The displacement gives the required direction.",
            "act": {"direction": "N"},
        }
        with self.assertRaisesRegex(ValueError, "LLM Act differs"):
            _make_views(record, "train", config, llm)

    def test_llm_ooda_preserves_both_acts(self) -> None:
        record = canonical("location_distance.jsonl")
        config = SFTBuildConfig(Path("canonical"), Path("sft"), "llm_augmented", "batch-test")
        llm = {
            "question": "How far apart are p1 and p2 using straight-line distance?",
            "observe": "The two points are expressed in metres in one projected coordinate system.",
            "orient": "Use the Euclidean distance formula.",
            "decide": "Δx = 4674.512 m and Δy = 10568.091 m; the Euclidean distance is 11555.761 m.",
            "act": record["gold"]["answer"],
        }
        view = _make_views(record, "train", config, llm)["ooda"]
        self.assertEqual(view["acts"]["llm"], record["gold"]["answer"])
        self.assertEqual(view["acts"]["program"], record["gold"]["answer"])
        validate_sft_record(view, record)

    def test_direction_llm_must_use_natural_compass_words(self) -> None:
        record = canonical("location_direction.jsonl")
        config = SFTBuildConfig(Path("canonical"), Path("sft"), "llm_augmented", "batch-test")
        llm = {
            "question": "Where is p2 relative to p1 on an eight-way compass?",
            "observe": "Both points use the same projected coordinate system.",
            "orient": "Classify the displacement into an eight-way compass direction.",
            "decide": "Δx = 1311.651 m and Δy = -4246.960 m, placing p2 to the S of p1.",
            "act": record["gold"]["answer"],
        }
        with self.assertRaisesRegex(ValueError, "natural compass words"):
            _make_views(record, "train", config, llm)

    def test_shortest_path_llm_must_not_relabel_cost_as_length(self) -> None:
        record = canonical("network_shortest_path.jsonl")
        config = SFTBuildConfig(Path("canonical"), Path("sft"), "llm_augmented", "batch-test")
        llm = {
            "question": "Which directed route has the lowest cost from n1 to n2?",
            "observe": "Each directed edge has a listed cost in metres.",
            "orient": "Compare legal routes by adding their costs.",
            "decide": "n1 -> n10 -> n8 -> n2 has total length 112.781 m.",
            "act": record["gold"]["answer"],
        }
        with self.assertRaisesRegex(ValueError, "must not relabel cost"):
            _make_views(record, "train", config, llm)

    def test_negative_connectivity_accepts_controlled_synonym(self) -> None:
        record = canonical_matching("network_connectivity.jsonl", lambda row: not row["gold"]["answer"]["connected"])
        config = SFTBuildConfig(Path("canonical"), Path("sft"), "llm_augmented", "batch-test")
        llm = {
            "question": "Can n1 reach n2 through the directed network?",
            "observe": "Every edge may be followed only in its stated direction.",
            "orient": "Test reachability from n1 to n2 without optimizing path cost.",
            "decide": "Starting at n1 and following directed edges does not lead to n2, so no valid directed path connects them.",
            "act": record["gold"]["answer"],
        }
        view = _make_views(record, "train", config, llm)["ooda"]
        validate_sft_record(view, record)

    def test_simplified_topology_view_is_local_and_preserves_truth(self) -> None:
        record = canonical("object_topology.jsonl")
        geometry_a, geometry_b, metadata = _topology_geometry_for_render(record, "simplified")
        self.assertEqual(metadata["topology_geometry_view"], "simplified")
        self.assertTrue(metadata["relation_preserved"])
        self.assertTrue(metadata["de9im_preserved"])
        self.assertNotEqual(geometry_a, record["scene"]["geometry_a"])
        self.assertIn("local projected metres", _scene_text(record, topology_geometry_view="simplified"))
        config = SFTBuildConfig(Path("canonical"), Path("sft"), "template", "batch-test", topology_geometry_view="simplified")
        view = _make_views(record, "train", config, None)["ooda"]
        self.assertEqual(view["rendering"]["topology_geometry_view"], "simplified")
        validate_sft_record(view, record)

    def test_shortest_path_runner_up_is_program_witness_and_ooda_evidence(self) -> None:
        record = enrich_shortest_path_record(canonical("network_shortest_path.jsonl"))
        witness = record["gold"]["witness"]
        self.assertIn("runner_up_path", witness)
        self.assertGreater(witness["runner_up_cost_m"], record["gold"]["answer"]["optimal_cost_m"])
        config = SFTBuildConfig(Path("canonical"), Path("sft"), "template", "batch-test")
        ooda = _make_views(record, "train", config, None)["ooda"]
        completion = ooda["messages"][1]["content"]
        self.assertIn(" -> ".join(witness["runner_up_path"]), completion)
        self.assertIn(f"{witness['runner_up_cost_m']:.3f}", completion)
        validate_sft_record(ooda, record)

    def test_augmented_builder_writes_only_verified_records(self) -> None:
        record = canonical("network_shortest_path.jsonl")

        class FakeDeepSeek:
            def render(self, item: dict, *, scene_text: str | None = None, attempt: int = 1) -> dict:
                runner_up = item["gold"]["witness"]["runner_up_path"]
                runner_up_cost = item["gold"]["witness"]["runner_up_cost_m"]
                return {
                    "question": "Which directed route has the least total cost from n1 to n2?",
                    "observe": "Every listed road link has a permitted direction and a distance cost.",
                    "orient": "Compare legal directed routes by their accumulated cost.",
                    "decide": (
                        f"The selected route n1 -> n10 -> n8 -> n2 costs 112.781 m. "
                        f"The program-computed runner-up {' -> '.join(runner_up)} costs {runner_up_cost:.3f} m. "
                        f"Since 112.781 m < {runner_up_cost:.3f} m, the selected route is cheapest."
                    ),
                    "act": item["gold"]["answer"],
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "canonical"
            record_path = source / "records" / "beijing" / "network_shortest_path.jsonl"
            record_path.parent.mkdir(parents=True)
            record_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            split_path = source / "splits" / "scenario_split.jsonl"
            split_path.parent.mkdir(parents=True)
            split_path.write_text(json.dumps({"scenario_id": record["scenario_id"], "split": "train"}) + "\n", encoding="utf-8")
            output = root / "sft"
            stale_rejected = output / "llm_augmented" / "rejected" / "batch-test.jsonl"
            stale_rejected.parent.mkdir(parents=True)
            stale_rejected.write_text('{"stale":true}\n', encoding="utf-8")
            counts = build_sft_dataset(
                SFTBuildConfig(source, output, "llm_augmented", "batch-test", overwrite=True), client=FakeDeepSeek()
            )
            self.assertEqual(counts, {"train/minimal": 1, "train/ooda": 1, "train/qa": 1})
            saved = json.loads((output / "llm_augmented" / "train" / "anonymous_ooda_en.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(saved["acts"]["program"], record["gold"]["answer"])
            self.assertEqual(saved["acts"]["llm"], record["gold"]["answer"])
            self.assertFalse(stale_rejected.exists())
            review_path = export_review_sample(output, mode="llm_augmented", batch_id="batch-test", sample_size=50)
            review = json.loads(review_path.read_text(encoding="utf-8"))
            self.assertEqual(review["sampling"]["unit"], "scenario_id")
            self.assertEqual(review["sampling"]["selected_scenarios"], 1)
            self.assertEqual(set(review["records"][0]["views"]), {"qa", "minimal", "ooda"})
            ooda_path = export_ooda_review_markdown(output, mode="llm_augmented", batch_id="batch-test", sample_size=50)
            ooda_text = ooda_path.read_text(encoding="utf-8")
            self.assertIn("# OODA review — batch-test", ooda_text)
            self.assertIn("### OODA completion", ooda_text)
            self.assertIn("Observe:\n", ooda_text)


if __name__ == "__main__":
    unittest.main()
