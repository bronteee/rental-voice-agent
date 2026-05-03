from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from rental_voice_agent.config import Settings
from rental_voice_agent.live_judge import (
    judge_error_path_for_snapshot,
    judge_path_for_snapshot,
    write_live_call_judge,
)
from rental_voice_agent.state import (
    CallState,
    CleaningRequest,
    Property,
    TranscriptEntry,
)


ROOT = Path(__file__).resolve().parents[1]


def _settings_without_api_key() -> Settings:
    return Settings(
        openai_api_key=None,
        openai_realtime_model="gpt-realtime",
        openai_judge_model="gpt-4o-mini",
        livekit_url=None,
        livekit_api_key=None,
        livekit_api_secret=None,
        twilio_account_sid=None,
        twilio_auth_token=None,
        twilio_phone_number=None,
        livekit_sip_outbound_trunk_id=None,
        extraction_retry_limit=2,
        off_topic_redirect_limit=2,
        max_agent_turns=30,
        max_call_duration_seconds=180,
    )


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
        transcript=[
            TranscriptEntry(
                speaker="agent",
                text="Hi, this is an automated assistant calling for Alex.",
                t_ms=0,
            )
        ],
        end_reason="completed",
    )


def test_judge_sidecar_paths_are_beside_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "state-id.json"

    assert judge_path_for_snapshot(snapshot) == tmp_path / "state-id.judge.json"
    assert judge_error_path_for_snapshot(snapshot) == (
        tmp_path / "state-id.judge_error.json"
    )


def test_live_judge_is_best_effort_when_api_key_missing(tmp_path: Path) -> None:
    snapshot = tmp_path / "call-123.json"

    result = asyncio.run(
        write_live_call_judge(_state(), snapshot, _settings_without_api_key())
    )

    assert result is None
    error_payload = json.loads(judge_error_path_for_snapshot(snapshot).read_text())
    assert "OPENAI_API_KEY" in error_payload["error"]
