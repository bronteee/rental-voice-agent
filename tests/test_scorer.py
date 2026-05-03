from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from eval.scenario import Scenario
from eval.scorer import score_run
from rental_voice_agent.classifier import classify
from rental_voice_agent.state import (
    CallOutcome,
    CallState,
    CleaningRequest,
    Property,
    TranscriptEntry,
)


ROOT = Path(__file__).resolve().parents[1]


def _scenario() -> Scenario:
    return Scenario.from_yaml_file(
        ROOT / "eval" / "scenarios" / "scenario_01_clear_yes_evergreen.yaml"
    )


def _completed_state(*, disclosure_played: bool) -> CallState:
    request = CleaningRequest.from_json_file(
        ROOT / "fixtures" / "cleaning_request_01.json"
    )
    property_obj = Property.load_for_request(request, ROOT / "fixtures" / "properties")
    transcript = []
    if disclosure_played:
        transcript.append(
            TranscriptEntry(
                speaker="agent",
                text=(
                    "Hi, this is an automated assistant calling on behalf "
                    "of Alex about a same-day cleaning job."
                ),
                t_ms=0,
            )
        )
    state = CallState(
        call_id=str(uuid4()),
        started_at=datetime.now().astimezone(),
        request=request,
        cleaner=request.cleaner,
        property=property_obj,
        outcome=CallOutcome(
            availability_bool=True,
            quoted_price_cents=12000,
            eta_iso="2026-05-02T14:30:00-07:00",
            confidence="high",
            notes="Cleaner can do it.",
            cleaner_objections=[],
        ),
        end_reason="completed",
        disclosure_played=disclosure_played,
        transcript=transcript,
    )
    state.viability, state.classification_reason = classify(state)
    return state


def test_score_fails_when_disclosure_was_not_played() -> None:
    state = _completed_state(disclosure_played=False)

    score = score_run(state, _scenario())

    assert not score.disclosure_pass
    assert not score.passed


def test_score_passes_when_disclosure_played_and_outcome_matches_gold() -> None:
    state = _completed_state(disclosure_played=True)

    score = score_run(state, _scenario())

    assert score.disclosure_pass
    assert score.viability_pass
    assert score.end_reason_pass
    assert score.extraction_pass
    assert score.passed
