import json
import unittest
from pathlib import Path

from soda.business import DeclarativeSpatialAdapter, DomainSpec, one_decision


ROOT = Path(__file__).resolve().parents[1]
GOOD = """Observe: forklift-7 is at [2, 3].
Orient: [2, 4] is in range and clear.
Decide: Move east one cell.
Act: {\"operation\":\"move\",\"entity_id\":\"forklift-7\",\"target\":[2,4]}"""


class BusinessAdapterTests(unittest.TestCase):
    def setUp(self):
        self.adapter = DeclarativeSpatialAdapter(DomainSpec.from_json(str(ROOT / "config" / "domain.example.json")))
        self.state = json.loads((ROOT / "examples" / "state.json").read_text(encoding="utf-8"))

    def test_valid_action(self):
        decision = one_decision(self.adapter, self.state, lambda _: GOOD)
        self.assertEqual(decision.violations, [])
        self.assertEqual(decision.action["operation"], "move")

    def test_blocked_action_is_not_validated_as_safe(self):
        bad = GOOD.replace("[2,4]", "[2,5]")
        decision = one_decision(self.adapter, self.state, lambda _: bad)
        self.assertIn("target is blocked", decision.violations)


if __name__ == "__main__":
    unittest.main()
