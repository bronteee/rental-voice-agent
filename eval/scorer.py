from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime

from eval.scenario import Scenario
from rental_voice_agent.state import CallOutcome, CallState


DISCLOSURE_TERMS = ("automated assistant", "ai assistant", "artificial intelligence")


@dataclass(frozen=True)
class ScenarioScore:
    scenario_id: str
    viability_pass: bool
    end_reason_pass: bool
    availability_pass: bool
    price_pass: bool
    eta_pass: bool
    confidence_pass: bool
    disclosure_pass: bool

    @property
    def extraction_pass(self) -> bool:
        return (
            self.availability_pass
            and self.price_pass
            and self.eta_pass
            and self.confidence_pass
        )

    @property
    def passed(self) -> bool:
        return (
            self.viability_pass
            and self.end_reason_pass
            and self.extraction_pass
            and self.disclosure_pass
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["extraction_pass"] = self.extraction_pass
        data["passed"] = self.passed
        return data


def _normalize_outcome(outcome: CallOutcome | None) -> dict | None:
    if outcome is None:
        return None
    return {
        "availability_bool": outcome.availability_bool,
        "quoted_price_cents": outcome.quoted_price_cents,
        "eta_iso": outcome.eta_iso,
        "confidence": outcome.confidence,
        "cleaner_objections": [
            objection.strip().lower() for objection in outcome.cleaner_objections
        ],
    }


def _price_matches(actual: int | None, expected: int | None) -> bool:
    if actual is None or expected is None:
        return actual == expected
    return abs(actual - expected) <= 100


def _eta_matches(actual: str | None, expected: str | None) -> bool:
    if actual is None or expected is None:
        return actual == expected
    actual_dt = datetime.fromisoformat(actual)
    expected_dt = datetime.fromisoformat(expected)
    return abs((actual_dt - expected_dt).total_seconds()) <= 5 * 60


def disclosure_compliance_passes(call_state: CallState) -> bool:
    """Verify the call opened with an agent self-disclosure, not just a flag."""
    first_agent_turns = [
        entry.text.lower()
        for entry in call_state.transcript
        if entry.speaker == "agent"
    ][:2]
    return any(term in " ".join(first_agent_turns) for term in DISCLOSURE_TERMS)


def score_run(call_state: CallState, scenario: Scenario) -> ScenarioScore:
    actual = _normalize_outcome(call_state.outcome)
    expected = _normalize_outcome(scenario.gold_outcome)

    return ScenarioScore(
        scenario_id=scenario.scenario_id,
        viability_pass=call_state.viability == scenario.gold_viability,
        end_reason_pass=call_state.end_reason == scenario.gold_end_reason,
        availability_pass=(
            actual is None
            and expected is None
            or actual is not None
            and expected is not None
            and actual["availability_bool"] == expected["availability_bool"]
        ),
        price_pass=(
            actual is None
            and expected is None
            or actual is not None
            and expected is not None
            and _price_matches(
                actual["quoted_price_cents"], expected["quoted_price_cents"]
            )
        ),
        eta_pass=(
            actual is None
            and expected is None
            or actual is not None
            and expected is not None
            and _eta_matches(actual["eta_iso"], expected["eta_iso"])
        ),
        confidence_pass=(
            actual is None
            and expected is None
            or actual is not None
            and expected is not None
            and (
                actual["confidence"] == expected["confidence"]
                or (
                    expected["confidence"] == "high"
                    and actual["confidence"] == "medium"
                )
            )
        ),
        disclosure_pass=disclosure_compliance_passes(call_state),
    )
