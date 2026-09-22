import unittest

from gis_concept_llm.real_gis import GISProvenance, RealGISGrounder, require_real_gis_metadata
from gis_concept_llm.verifiers import verifier_for


class RealGISGroundingTests(unittest.TestCase):
    def setUp(self):
        self.grounder = RealGISGrounder(GISProvenance(
            city="ExampleCity", source_name="Licensed example map", source_license="ODbL-1.0", crs="EPSG:3857",
            source_url="https://example.invalid/dataset", dataset_version="2026-09",
        ))

    def test_direction_has_named_and_anonymous_counterparts_with_lineage(self):
        named, anonymous = self.grounder.paired_direction(
            scenario_id="poi-pair-1",
            point_a={"id": "poi-1", "name": "Library", "coordinates": [0, 0]},
            point_b={"id": "poi-2", "name": "Museum", "coordinates": [2, 1]},
        )
        self.assertEqual(named.gold_answer, "NE")
        self.assertIn("Museum", named.question)
        self.assertNotIn("Museum", anonymous.question)
        self.assertEqual(named.metadata["source_license"], "ODbL-1.0")
        self.assertEqual(named.metadata["scenario_id"], "poi-pair-1")
        self.assertTrue(verifier_for("direction").verify(anonymous, anonymous.gold_answer).correct)

    def test_connectivity_keeps_real_node_references_but_hides_names_when_anonymous(self):
        named, anonymous = self.grounder.paired_connectivity(
            scenario_id="road-subgraph-1",
            nodes={"n1": {"id": "n1", "name": "North Gate"}, "n2": {"id": "n2", "name": "Central Junction"}, "n3": {"id": "n3", "name": "South Gate"}},
            edges=[("n1", "n2"), ("n2", "n3")], source="n1", target="n3",
        )
        self.assertEqual(named.gold_answer, "Connected")
        self.assertIn("North Gate", named.question)
        self.assertNotIn("North Gate", anonymous.question)
        self.assertEqual(anonymous.metadata["node_refs"]["n1"], "n1")
        self.assertTrue(verifier_for("connectivity").verify(named, named.gold_answer).correct)

    def test_provenance_is_mandatory(self):
        with self.assertRaises(ValueError):
            require_real_gis_metadata({"city": "ExampleCity", "crs": "EPSG:3857"})

    def test_geographic_source_requires_a_projected_analysis_crs(self):
        with self.assertRaises(ValueError):
            RealGISGrounder(GISProvenance(
                city="ExampleCity", source_name="Licensed example map", source_license="ODbL-1.0", crs="EPSG:4326",
            ))


if __name__ == "__main__":
    unittest.main()
