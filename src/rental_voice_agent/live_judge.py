from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from rental_voice_agent.config import Settings
from rental_voice_agent.state import CallState


logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """
You are judging a proof-of-concept outbound voice agent call.
Evaluate conversation quality only. Do not re-score deterministic business facts
such as budget/deadline classification except when they affect the conversation.
Return strict JSON with the requested keys and concise evidence.
""".strip()

JUDGE_RUBRIC = {
    "disclosure": (
        "Did the agent clearly identify itself as automated or AI at the start?"
    ),
    "required_field_collection": (
        "Did the agent collect availability before deadline, price, and ETA, or "
        "gracefully handle that a field was impossible to collect?"
    ),
    "ambiguity_and_caveats": (
        "Did the agent clarify or preserve important caveats such as traffic, "
        "conditional availability, or deadline risk?"
    ),
    "cleaner_facing_close": (
        "Did the agent close as if speaking to the cleaning business, not as if "
        "summarizing to the host?"
    ),
    "no_overcommitment": (
        "Did the agent avoid committing to booking, payment, access details, or "
        "host-only decisions?"
    ),
    "naturalness": "Was the call concise, polite, and reasonably natural?",
}


async def write_live_call_judge(
    state: CallState,
    snapshot_path: Path,
    settings: Settings,
) -> Path | None:
    """Write an LLM judge sidecar for a finalized live call snapshot.

    The sidecar is intentionally best-effort: the call state snapshot is the
    primary artifact and should never be hidden by a post-call judge failure.
    """
    judge_path = judge_path_for_snapshot(snapshot_path)
    error_path = judge_error_path_for_snapshot(snapshot_path)
    if not settings.openai_api_key:
        _write_error(error_path, "OPENAI_API_KEY is not configured")
        return None

    try:
        payload = await judge_live_call(state, settings=settings)
        judge_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if error_path.exists():
            error_path.unlink()
        logger.info("[judge] wrote %s", judge_path)
        return judge_path
    except Exception as exc:  # noqa: BLE001 - judge must not break live calls
        logger.warning("[judge] failed: %r", exc)
        _write_error(error_path, repr(exc))
        return None


async def judge_live_call(
    state: CallState,
    *,
    settings: Settings,
) -> dict[str, Any]:
    client = AsyncOpenAI(api_key=settings.openai_api_key, timeout=20.0)
    response = await client.chat.completions.create(
        model=settings.openai_judge_model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(_judge_input(state), indent=2, sort_keys=True),
            },
        ],
    )
    raw = response.choices[0].message.content
    if raw is None:
        raise RuntimeError("judge returned an empty response")
    payload = json.loads(raw)
    payload["judge_model"] = settings.openai_judge_model
    payload["judged_at"] = datetime.now().astimezone().isoformat()
    payload["call_id"] = state.call_id
    return payload


def judge_path_for_snapshot(snapshot_path: Path) -> Path:
    return snapshot_path.with_name(f"{snapshot_path.stem}.judge.json")


def judge_error_path_for_snapshot(snapshot_path: Path) -> Path:
    return snapshot_path.with_name(f"{snapshot_path.stem}.judge_error.json")


def _judge_input(state: CallState) -> dict[str, Any]:
    return {
        "rubric": JUDGE_RUBRIC,
        "expected_output_shape": {
            "overall_pass": "boolean",
            "scores": {
                key: "0 or 1"
                for key in (
                    "disclosure",
                    "required_field_collection",
                    "ambiguity_and_caveats",
                    "cleaner_facing_close",
                    "no_overcommitment",
                    "naturalness",
                )
            },
            "missing_or_risky": ["short strings"],
            "evidence": {
                "disclosure": "short evidence string",
                "required_field_collection": "short evidence string",
                "ambiguity_and_caveats": "short evidence string",
                "cleaner_facing_close": "short evidence string",
                "no_overcommitment": "short evidence string",
                "naturalness": "short evidence string",
            },
            "summary": "one or two sentences",
        },
        "request": {
            "request_id": state.request.request_id,
            "host_first_name": state.request.host_first_name,
            "property_id": state.request.property_id,
            "deadline": state.request.deadline.isoformat(),
            "max_budget_cents": state.request.max_budget_cents,
            "cleaner_name": state.cleaner.name,
        },
        "property": {
            "name": state.property.name,
            "summary_short": state.property.summary_short,
            "special_instructions": state.property.special_instructions,
            "parking_notes": state.property.parking_notes,
        },
        "transcript": [
            {"speaker": entry.speaker, "text": entry.text, "t_ms": entry.t_ms}
            for entry in state.transcript
        ],
        "tool_calls": [
            {
                "name": entry.name,
                "args": entry.args,
                "result": entry.result,
                "t_ms": entry.t_ms,
            }
            for entry in state.tool_calls
        ],
        "outcome": None if state.outcome is None else state.outcome.__dict__,
        "viability": state.viability,
        "classification_reason": state.classification_reason,
        "end_reason": state.end_reason,
    }


def _write_error(path: Path, error: str) -> None:
    payload = {
        "error": error,
        "judged_at": datetime.now().astimezone().isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
