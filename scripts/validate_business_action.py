"""Validate an LLM OODA completion against the declarative business adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from soda.business import DeclarativeSpatialAdapter, DomainSpec, one_decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, help="DomainSpec JSON")
    parser.add_argument("--state", required=True, help="Current business state JSON")
    parser.add_argument("--completion", required=True, help="Text file containing one OODA completion")
    args = parser.parse_args()

    adapter = DeclarativeSpatialAdapter(DomainSpec.from_json(args.domain))
    state = json.loads(Path(args.state).read_text(encoding="utf-8"))
    completion = Path(args.completion).read_text(encoding="utf-8")
    decision = one_decision(adapter, state, lambda _: completion)
    print(json.dumps({"action": decision.action, "violations": decision.violations, "state_prompt": decision.state_text}, ensure_ascii=False, indent=2))
    if decision.violations:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
