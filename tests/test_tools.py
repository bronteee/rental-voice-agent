from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest

from rental_voice_agent.state import CallState, CleaningRequest, Property
from rental_voice_agent.tools import create_call_tools


ROOT = Path(__file__).resolve().parents[1]


def _state() -> CallState:
    request = CleaningRequest.from_json_file(
        ROOT / "fixtures" / "cleaning_request_01.json"
    )
    property_obj = Property.load_for_request(request, ROOT / "fixtures" / "properties")
    return CallState(
        call_id=str(uuid4()),
        started_at=datetime.now().astimezone(),
        request=request,
        cleaner=request.cleaner,
        property=property_obj,
    )


def test_record_call_outcome_writes_state_and_logs_tool_call() -> None:
    state = _state()
    tools = create_call_tools(
        state,
        property_fixtures_dir=ROOT / "fixtures" / "properties",
    )

    result = asyncio.run(
        tools.record_call_outcome(
            availability_bool=True,
            quoted_price_cents=12000,
            eta_iso="2026-01-01T14:30:00+00:00",
            confidence="high",
            notes="Cleaner can make it before guest check-in.",
            cleaner_objections=[],
        )
    )

    assert "call end_call" in result
    assert "speaking to the cleaner" in result
    assert "cleaner-facing confirmation" in result
    assert "Capitol Hill Studio" in result
    assert "Standard turnover" in result
    assert state.outcome is not None
    assert state.outcome.availability_bool is True
    assert state.outcome.quoted_price_cents == 12000
    assert state.outcome.eta_iso == "2026-05-02T14:30:00-07:00"
    assert state.tool_calls[-1].name == "record_call_outcome"
    assert state.tool_calls[-1].args["cleaner_objections"] == []


def test_record_call_outcome_rejects_invalid_payload_without_overwriting_outcome() -> (
    None
):
    state = _state()
    tools = create_call_tools(
        state,
        property_fixtures_dir=ROOT / "fixtures" / "properties",
    )

    with pytest.raises(ValueError, match="quoted_price_cents"):
        asyncio.run(
            tools.record_call_outcome(
                availability_bool=True,
                quoted_price_cents=0,
                eta_iso=None,
                confidence="high",
                notes="Invalid price.",
                cleaner_objections=[],
            )
        )

    assert state.outcome is None
    assert state.tool_calls[-1].name == "record_call_outcome"
    assert state.tool_calls[-1].result is not None
    assert "quoted_price_cents" in state.tool_calls[-1].result


def test_lookup_property_details_returns_value_and_graceful_miss() -> None:
    state = _state()
    tools = create_call_tools(
        state,
        property_fixtures_dir=ROOT / "fixtures" / "properties",
    )

    assert (
        asyncio.run(tools.lookup_property_details("floor_type"))
        == "hardwood"
    )
    state.request.property_id = "missing_property"
    assert "don't have" in asyncio.run(
        tools.lookup_property_details("floor_type")
    )


def test_end_call_returns_stop_signal_and_sets_event() -> None:
    state = _state()
    tools = create_call_tools(
        state,
        property_fixtures_dir=ROOT / "fixtures" / "properties",
    )

    result = asyncio.run(tools.end_call("completed"))
    assert "Do not generate" in result
    assert state.end_reason == "completed"
    assert state.ended_at is not None
    assert tools.end_event.is_set()

    with pytest.raises(RuntimeError, match="already ended"):
        asyncio.run(
            tools.escalate_to_host("Budget exception.", "Cleaner asked for $225.")
        )


def test_terminal_tool_waits_for_prior_speech_before_setting_event() -> None:
    state = _state()
    tools = create_call_tools(
        state,
        property_fixtures_dir=ROOT / "fixtures" / "properties",
    )
    waited = False

    async def wait_for_playout() -> None:
        nonlocal waited
        assert not tools.end_event.is_set()
        assert state.end_reason is None
        waited = True

    result = asyncio.run(tools.end_call_after_playout("completed", wait_for_playout))

    assert waited
    assert "Do not generate" in result
    assert state.end_reason == "completed"
    assert tools.end_event.is_set()


def test_escalate_to_host_sets_end_event() -> None:
    state = _state()
    tools = create_call_tools(
        state,
        property_fixtures_dir=ROOT / "fixtures" / "properties",
    )

    result = asyncio.run(
        tools.escalate_to_host("Budget exception.", "Cleaner asked for $225.")
    )
    assert "Do not generate" in result
    assert state.end_reason == "escalation"
    assert tools.end_event.is_set()
