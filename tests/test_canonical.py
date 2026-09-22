import unittest

from gis_concept_llm.canonical import direction_8, validate_canonical


class CanonicalTests(unittest.TestCase):
    def test_eight_direction_binning(self):
        self.assertEqual(direction_8(1, 0), "E")
        self.assertEqual(direction_8(1, 1), "NE")
        self.assertEqual(direction_8(0, -1), "S")

    def test_validator_recomputes_connectivity(self):
        record = {
            "example_id": "x", "scenario_id": "x", "task": {"skill": "connectivity"}, "provenance": {},
            "scene": {
                "nodes": [{"id": "n1"}, {"id": "n2"}, {"id": "n3"}],
                "edges": [{"id": "e1", "from": "n1", "to": "n2", "cost_m": 1.0}, {"id": "e2", "from": "n2", "to": "n3", "cost_m": 1.0}],
                "query": {"source": "n1", "target": "n3"},
            },
            "gold": {"answer": {"connected": True}, "witness": {"path": ["n1", "n2", "n3"]}},
        }
        validate_canonical(record)
        record["gold"]["answer"] = {"connected": False}
        with self.assertRaises(ValueError):
            validate_canonical(record)

    def test_validator_uses_serialized_distance_coordinates(self):
        record = {
            "example_id": "x", "scenario_id": "x", "task": {"skill": "euclidean_distance"}, "provenance": {},
            "scene": {
                "points": [{"id": "p1", "x": 1.111, "y": 2.222}, {"id": "p2", "x": 4.444, "y": 6.666}],
                "query": {"reference": "p1", "target": "p2"},
            },
            "gold": {"answer": {"distance_m": 5.555}, "witness": {}},
        }
        validate_canonical(record)

    def test_validator_recomputes_shortest_path(self):
        record = {
            "example_id": "x", "scenario_id": "x", "task": {"skill": "shortest_path"}, "provenance": {},
            "scene": {
                "nodes": [{"id": "n1"}, {"id": "n2"}, {"id": "n3"}, {"id": "n4"}],
                "edges": [
                    {"id": "e1", "from": "n1", "to": "n2", "cost_m": 3.0}, {"id": "e2", "from": "n2", "to": "n4", "cost_m": 3.0},
                    {"id": "e3", "from": "n1", "to": "n3", "cost_m": 2.0}, {"id": "e4", "from": "n3", "to": "n4", "cost_m": 2.0},
                ],
                "query": {"source": "n1", "target": "n4"},
            },
            "gold": {"answer": {"optimal_cost_m": 4.0, "path": ["n1", "n3", "n4"]}, "witness": {}},
        }
        validate_canonical(record)


if __name__ == "__main__":
    unittest.main()
