import unittest

from gis_concept_llm.evaluation import evaluate_atomic, split_unseen_cities
from gis_concept_llm.generators import ConnectivityGenerator, DirectionGenerator, TopologyGenerator
from gis_concept_llm.taxonomy import ConceptTaxonomy
from gis_concept_llm.training import score_completion
from gis_concept_llm.verifiers import verifier_for


class GISConceptTests(unittest.TestCase):
    def test_mvp_examples_match_taxonomy_and_executable_truth(self):
        examples = [DirectionGenerator(1).synthetic(), TopologyGenerator(2).synthetic(), ConnectivityGenerator(3).synthetic()]
        taxonomy = ConceptTaxonomy.load_default()
        for example in examples:
            taxonomy.validate_example(example)
            self.assertTrue(verifier_for(example.task_type).verify(example, example.gold_answer).correct)

    def test_gis_truth_outweighs_ooda_format(self):
        example = DirectionGenerator(4).synthetic()
        wrong = "Observe: x\nOrient: y\nDecide: z\nAct: N" if example.gold_answer != "N" else "Observe: x\nOrient: y\nDecide: z\nAct: S"
        reward = score_completion(example, wrong)
        self.assertLess(reward.total, 0)
        self.assertEqual(reward.gis, -1.0)

    def test_atomic_evaluation_reports_ood_slices(self):
        examples = [DirectionGenerator(5).synthetic(linguistic_ood=True), TopologyGenerator(6).synthetic(structural_ood=True)]
        answers = {example.question: example.gold_answer for example in examples}
        report = evaluate_atomic(examples, lambda prompt: answers[prompt.split("Question: ", 1)[1].split("\n", 1)[0]])
        self.assertEqual(report.verifier_accuracy, 1.0)
        self.assertIn("linguistic_ood", report.by_slice)
        self.assertIn("structural_ood", report.by_slice)

    def test_unseen_city_split_has_no_leakage(self):
        generator = DirectionGenerator(7)
        alpha = generator.from_gis_points(point_a=(0, 0), point_b=(1, 1), name_a="A", name_b="B", city="Alpha", anonymous=True)
        beta = generator.from_gis_points(point_a=(0, 0), point_b=(-1, 1), name_a="C", name_b="D", city="Beta", anonymous=True)
        train, test = split_unseen_cities([alpha, beta], {"Beta"})
        self.assertEqual([item.city for item in train], ["Alpha"])
        self.assertEqual([item.city for item in test], ["Beta"])


if __name__ == "__main__":
    unittest.main()
