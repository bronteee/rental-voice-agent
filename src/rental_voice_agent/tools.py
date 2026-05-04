from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable, Literal

from livekit.agents import RunContext, function_tool

from rental_voice_agent.state import CallOutcome, CallState, ToolCallEntry


PropertyField = Literal[
    "bedrooms",
    "bathrooms",
    "square_footage",
    "floor_type",
    "parking_notes",
    "special_instructions",
]

LLMEndReason = Literal[
    "completed",
    "voicemail_detected",
    "cleaner_declined",
    "cleaner_hostile",
    "off_topic_repeated",
    "extraction_failed",
]

VALID_CONFIDENCES = {"high", "medium", "low"}
VALID_LLM_END_REASONS = {
    "completed",
    "voicemail_detected",
    "cleaner_declined",
    "cleaner_hostile",
    "off_topic_repeated",
    "extraction_failed",
}
PROPERTY_MISS = "I don't have that information for this property - sorry."
OUTCOME_RECORDED_STEERING = (
    "Recorded. You are still speaking to the cleaner, not the host. "
    "{confirmation_instruction} Then call end_call with reason=completed. "
    "Do not summarize the cleaner's answers as if reporting to the host."
)


@dataclass
class CallTools:
    state: CallState
    property_fixtures_dir: Path
    on_tool_call: Callable[[ToolCallEntry], None] | None = None
    end_event: asyncio.Event = field(default_factory=asyncio.Event)

    async def record_call_outcome(
        self,
        availability_bool: bool,
        quoted_price_cents: int | None,
        eta_iso: str | None,
        confidence: Literal["high", "medium", "low"],
        notes: str,
        cleaner_objections: list[str],
    ) -> str:
        args = {
            "availability_bool": availability_bool,
            "quoted_price_cents": quoted_price_cents,
            "eta_iso": eta_iso,
            "confidence": confidence,
            "notes": notes,
            "cleaner_objections": cleaner_objections,
        }
        try:
            self._ensure_open()
            if self.state.outcome is not None:
                raise ValueError("record_call_outcome may only be called once")
            if quoted_price_cents is not None and quoted_price_cents <= 0:
                raise ValueError("quoted_price_cents must be a positive integer")
            if confidence not in VALID_CONFIDENCES:
                raise ValueError("confidence must be high, medium, or low")
            if not isinstance(cleaner_objections, list):
                raise ValueError("cleaner_objections must be a list")

            normalized_eta = None
            if eta_iso is not None:
                normalized_eta = _normalize_eta_to_request_day(
                    eta_iso,
                    deadline=self.state.request.deadline,
                )

            self.state.outcome = CallOutcome(
                availability_bool=availability_bool,
                quoted_price_cents=quoted_price_cents,
                eta_iso=normalized_eta,
                confidence=confidence,
                notes=notes,
                cleaner_objections=cleaner_objections,
            )
            args["eta_iso"] = normalized_eta
            result = OUTCOME_RECORDED_STEERING.format(
                confirmation_instruction=format_cleaner_confirmation(self.state)
            )
            self._append_tool_call("record_call_outcome", args, result)
            return result
        except Exception as exc:
            self._append_tool_call("record_call_outcome", args, f"ERROR: {exc}")
            raise

    async def lookup_property_details(self, field: PropertyField) -> str:
        property_id = self.state.request.property_id
        path = self.property_fixtures_dir / f"{property_id}.json"
        if not path.exists():
            result = PROPERTY_MISS
            self._append_tool_call(
                "lookup_property_details",
                {"property_id": property_id, "field": field},
                result,
            )
            return result

        data = json.loads(path.read_text())
        raw_value = data.get(field)
        result = PROPERTY_MISS if raw_value in (None, "") else str(raw_value)
        self._append_tool_call(
            "lookup_property_details",
            {"property_id": property_id, "field": field},
            result,
        )
        return result

    async def end_call(self, reason: LLMEndReason) -> str:
        self._ensure_open()
        if reason not in VALID_LLM_END_REASONS:
            raise ValueError(f"reason must be one of {sorted(VALID_LLM_END_REASONS)}")

        self.state.end_reason = reason
        self.state.ended_at = datetime.now().astimezone()
        result = "Call has ended. Do not generate any further responses."
        self._append_tool_call("end_call", {"reason": reason}, result)
        self.end_event.set()
        return result

    async def end_call_after_playout(
        self,
        reason: LLMEndReason,
        wait_for_playout: Callable[[], Awaitable[None]],
    ) -> str:
        await wait_for_playout()
        return await self.end_call(reason)

    async def escalate_to_host(
        self,
        reason: str,
        cleaner_response_summary: str,
    ) -> str:
        self._ensure_open()
        self.state.end_reason = "escalation"
        self.state.ended_at = datetime.now().astimezone()
        result = (
            "Escalated to host. Call has ended. Do not generate any further responses."
        )
        self._append_tool_call(
            "escalate_to_host",
            {
                "reason": reason,
                "cleaner_response_summary": cleaner_response_summary,
            },
            result,
        )
        self.end_event.set()
        return result

    async def escalate_to_host_after_playout(
        self,
        reason: str,
        cleaner_response_summary: str,
        wait_for_playout: Callable[[], Awaitable[None]],
    ) -> str:
        await wait_for_playout()
        return await self.escalate_to_host(reason, cleaner_response_summary)

    def as_livekit_tools(self) -> list[object]:
        @function_tool
        async def record_call_outcome(
            availability_bool: bool,
            quoted_price_cents: int | None,
            eta_iso: str | None,
            confidence: Literal["high", "medium", "low"],
            notes: str,
            cleaner_objections: list[str],
        ) -> str:
            """Record the cleaner's structured availability, price, ETA, and objections."""
            return await self.record_call_outcome(
                availability_bool=availability_bool,
                quoted_price_cents=quoted_price_cents,
                eta_iso=eta_iso,
                confidence=confidence,
                notes=notes,
                cleaner_objections=cleaner_objections,
            )

        @function_tool
        async def lookup_property_details(field: PropertyField) -> str:
            """Look up one detail for the current cleaning request's property."""
            return await self.lookup_property_details(field)

        @function_tool
        async def end_call(context: RunContext, reason: LLMEndReason) -> str:
            """End the call with the best matching terminal reason."""
            return await self.end_call_after_playout(reason, context.wait_for_playout)

        @function_tool
        async def escalate_to_host(
            context: RunContext, reason: str, cleaner_response_summary: str
        ) -> str:
            """Escalate when the cleaner asks for a host-only decision."""
            return await self.escalate_to_host_after_playout(
                reason, cleaner_response_summary, context.wait_for_playout
            )

        return [
            record_call_outcome,
            lookup_property_details,
            end_call,
            escalate_to_host,
        ]

    def _append_tool_call(self, name: str, args: dict, result: str | None) -> None:
        entry = ToolCallEntry(
            name=name,
            args=args,
            result=result,
            t_ms=_elapsed_ms(self.state),
        )
        self.state.tool_calls.append(entry)
        if self.on_tool_call is not None:
            self.on_tool_call(entry)

    def _ensure_open(self) -> None:
        if self.state.end_reason is not None:
            raise RuntimeError(
                f"call already ended with reason={self.state.end_reason}"
            )


def create_call_tools(
    state: CallState,
    *,
    property_fixtures_dir: Path,
    on_tool_call: Callable[[ToolCallEntry], None] | None = None,
) -> CallTools:
    return CallTools(
        state=state,
        property_fixtures_dir=property_fixtures_dir,
        on_tool_call=on_tool_call,
    )


def _elapsed_ms(state: CallState) -> int:
    return int((datetime.now().astimezone() - state.started_at).total_seconds() * 1000)


def _normalize_eta_to_request_day(eta_iso: str, *, deadline: datetime) -> str:
    eta = datetime.fromisoformat(eta_iso)
    if eta.date() == deadline.date() and eta.tzinfo == deadline.tzinfo:
        return eta.isoformat()
    return eta.replace(
        year=deadline.year,
        month=deadline.month,
        day=deadline.day,
        tzinfo=deadline.tzinfo,
    ).isoformat()


def format_cleaner_confirmation(state: CallState) -> str:
    outcome = state.outcome
    if outcome is None or not outcome.availability_bool:
        return (
            "Say this cleaner-facing close: "
            f"'Thanks, I'll pass that along to {state.request.host_first_name}. Bye.'"
        )

    property_details = " ".join(
        part
        for part in (
            (state.property.special_instructions or "").strip(),
            (state.property.parking_notes or "").strip(),
        )
        if part
    )
    price = _price_human(outcome.quoted_price_cents)
    eta = _eta_human(outcome.eta_iso)
    return (
        "Say this cleaner-facing confirmation before goodbye: "
        f"'To confirm: this is {state.property.name}, "
        f"{state.property.summary_short}. {property_details} "
        f"I have you at {price} and arriving by {eta}. "
        f"Thanks, I'll pass that along to {state.request.host_first_name}. Bye.'"
    )


def _price_human(price_cents: int | None) -> str:
    if price_cents is None:
        return "the price you quoted"
    dollars = price_cents // 100
    cents = price_cents % 100
    return f"${dollars}" if cents == 0 else f"${dollars}.{cents:02d}"


def _eta_human(eta_iso: str | None) -> str:
    if eta_iso is None:
        return "the arrival time you gave"
    return datetime.fromisoformat(eta_iso).strftime("%-I:%M %p").replace(":00", "")
