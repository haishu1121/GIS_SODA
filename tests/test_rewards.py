import unittest

from soda.rewards import conclusion_reward, ooda_format_reward, soda_reward


GOOD = """Observe: A is north of B.
Orient: Delta y is positive.
Decide: North matches the relation.
Act: Output N."""


class RewardTests(unittest.TestCase):
    def test_paper_reward_bounds(self):
        self.assertEqual(conclusion_reward(GOOD, "N"), 1.0)
        self.assertEqual(ooda_format_reward(GOOD), 0.4)
        self.assertEqual(soda_reward(GOOD, "N"), 1.4)

    def test_wrong_answer_keeps_format_credit(self):
        self.assertEqual(soda_reward(GOOD, "S"), -0.6)

    def test_no_reasoning_is_penalized(self):
        self.assertEqual(soda_reward("N", "N"), -1.8)


if __name__ == "__main__":
    unittest.main()
