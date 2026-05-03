from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rental_voice_agent.state import CallOutcome, EndReason, Viability


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    axis: str
    failure_mode: str | None
    request_fixture: str
    cleaner_turns: dict[int, str]
    gold_outcome: CallOutcome | None
    gold_viability: Viability
    gold_end_reason: EndReason

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scenario:
        return cls(
            scenario_id=data["scenario_id"],
            axis=data["axis"],
            failure_mode=data.get("failure_mode"),
            request_fixture=data["request_fixture"],
            cleaner_turns={
                int(turn): response
                for turn, response in data.get("cleaner_turns", {}).items()
            },
            gold_outcome=(
                None
                if data.get("gold_outcome") is None
                else CallOutcome.from_dict(data["gold_outcome"])
            ),
            gold_viability=data["gold_viability"],
            gold_end_reason=data["gold_end_reason"],
        )

    @classmethod
    def from_yaml_file(cls, path: Path) -> Scenario:
        return cls.from_dict(yaml.safe_load(path.read_text()))


def load_scenarios(scenario_dir: Path) -> list[Scenario]:
    return [
        Scenario.from_yaml_file(path) for path in sorted(scenario_dir.glob("*.yaml"))
    ]
