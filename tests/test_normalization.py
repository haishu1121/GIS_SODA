import unittest

from gis_concept_llm.normalization import StudyBBox, directions_for, is_drivable


class NormalizationRuleTests(unittest.TestCase):
    def test_road_access_and_direction_rules(self):
        self.assertTrue(is_drivable({"highway": "residential"}))
        self.assertFalse(is_drivable({"highway": "footway"}))
        self.assertFalse(is_drivable({"highway": "primary", "access": "private"}))
        self.assertEqual(directions_for({"oneway": "yes"}), (True, False))
        self.assertEqual(directions_for({"oneway": "-1"}), (False, True))
        self.assertEqual(directions_for({"junction": "roundabout"}), (True, False))
        self.assertEqual(directions_for({}), (True, True))

    def test_study_bbox_rejects_invalid_bounds(self):
        with self.assertRaises(ValueError):
            StudyBBox(116.4, 39.9, 116.3, 40.0).validate()


if __name__ == "__main__":
    unittest.main()
