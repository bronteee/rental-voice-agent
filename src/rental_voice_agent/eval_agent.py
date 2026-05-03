from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from livekit.agents import Agent, AgentSession
from livekit.agents.llm import RealtimeError
from livekit.plugins.openai import realtime

from eval.cleaner_bot import DeterministicCleanerBot
from eval.scenario import Scenario
from rental_voice_agent.config import load_settings
from rental_voice_agent.state import (
    CallState,
    CleaningRequest,
    Property,
    TranscriptEntry,
)
from rental_voice_agent.tools import create_call_tools


async def run_text_eval_agent(
    scenario: Scenario,
    *,
    fixtures_dir: Path,
    prompts_dir: Path,
) -> CallState:
    request = CleaningRequest.from_json_file(fixtures_dir / scenario.request_fixture)
    property_obj = Property.load_for_request(request, fixtures_dir / "properties")
    state = CallState(
        call_id=str(uuid4()),
        started_at=datetime.now().astimezone(),
        request=request,
        cleaner=request.cleaner,
        property=property_obj,
    )
    settings = load_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for real eval agent mode")

    disclosure = _render_disclosure(
        prompts_dir / "disclosure_v1.md", request, property_obj
    )
    instructions = _render_system_prompt(
        prompts_dir / "system_v1.md", request, property_obj
    )
    state.transcript.append(TranscriptEntry("agent", disclosure, _elapsed_ms(state)))
    state.disclosure_played = True
    call_tools = create_call_tools(
        state,
        property_fixtures_dir=fixtures_dir / "properties",
    )

    model = realtime.RealtimeModel(
        model=settings.openai_realtime_model,
        voice="marin",
        modalities=["text"],
        api_key=settings.openai_api_key,
        temperature=0.6,
    )
    session = AgentSession(llm=model, max_tool_steps=4)
    session_errors: list[str] = []

    @session.on("error")
    def _capture_session_error(event: object) -> None:
        session_errors.append(repr(event))

    agent = Agent(instructions=instructions, tools=call_tools.as_livekit_tools())

    try:
        await session.start(agent)
        cleaner_bot = DeterministicCleanerBot(scenario.cleaner_turns)
        cleaner_lines: list[str] = []
        for turn in sorted(scenario.cleaner_turns):
            cleaner_text = cleaner_bot.next_response(turn)
            if cleaner_text is None:
                continue
            cleaner_lines.append(f"turn {turn}: {cleaner_text}")
            state.transcript.append(
                TranscriptEntry("cleaner", cleaner_text, _elapsed_ms(state))
            )

        if session_errors:
            raise RuntimeError(f"AgentSession error: {session_errors[-1]}")

        # Branch by scenario type:
        # - extraction (failure_mode is None): force record_call_outcome and
        #   verify the agent extracts the correct structured fields.
        # - terminal classification (failure_mode set): force end_call and
        #   verify the agent picks the correct terminal reason from the
        #   cleaner transcript. Does NOT exercise the orchestrator's
        #   redirect-counter or ring-timeout logic — see LEARNINGS.md.
        if scenario.failure_mode is None:
            user_input = (
                "Offline eval replay. Extract the final cleaner outcome from this "
                "scripted cleaner transcript and call record_call_outcome exactly "
                "once. Include cleaner_objections as an empty list if no explicit "
                "objections or conditions appear. Do not ask follow-up questions.\n\n"
                + "\n".join(cleaner_lines)
            )
            tools = ["record_call_outcome"]
        else:
            user_input = (
                "Offline eval replay. Read this scripted cleaner transcript and "
                "decide how the call should end. Call end_call exactly once with "
                "the appropriate reason. Do not call any other tool.\n\n"
                + "\n".join(cleaner_lines)
            )
            tools = ["end_call"]

        handle = session.generate_reply(
            user_input=user_input,
            tool_choice="required",
            tools=tools,
            input_modality="text",
        )
        await handle.wait_for_playout()
        await session.drain()

        # `update_chat_ctx timed out` and similar Realtime errors fire on a
        # background task that wait_for_playout() does not propagate. Surface
        # them as a raised RealtimeError so the runner's retry loop sees the
        # failure as transient instead of recording an empty `tool_calls=[]`
        # state as a real eval result.
        if session_errors and not state.tool_calls:
            raise RealtimeError(
                f"AgentSession captured error and produced no tool calls: "
                f"{session_errors[-1]}"
            )

        # In extraction mode, end_call cannot fire (it's not in the allowed
        # tool list), so the orchestrator sets end_reason="completed" once
        # outcome is captured. In terminal-classification mode, end_call
        # fires directly and sets end_reason itself.
        if (
            scenario.failure_mode is None
            and state.outcome is not None
            and state.end_reason is None
        ):
            state.end_reason = "completed"
            state.ended_at = datetime.now().astimezone()

        if state.end_reason is None:
            state.end_reason = "turn_limit_exceeded"
            state.ended_at = datetime.now().astimezone()
    finally:
        await session.aclose()
        await model.aclose()

    return state


def _render_disclosure(
    path: Path, request: CleaningRequest, property_obj: Property
) -> str:
    return (
        path.read_text()
        .format(
            cleaner_name=request.cleaner.name,
            host_first_name=request.host_first_name,
            property_short_name=property_obj.name,
        )
        .strip()
    )


def _render_system_prompt(
    path: Path, request: CleaningRequest, property_obj: Property
) -> str:
    return path.read_text().format(
        cleaner_name=request.cleaner.name,
        host_first_name=request.host_first_name,
        property_short_name=property_obj.name,
        deadline_human=_deadline_human(request.deadline),
        property_summary_short=property_obj.summary_short,
        max_budget_human=f"${request.max_budget_cents // 100}",
        property_id=request.property_id,
    )


def _deadline_human(deadline: datetime) -> str:
    return deadline.strftime("%-I:%M %p").replace(":00", "")


def _elapsed_ms(state: CallState) -> int:
    return int((datetime.now().astimezone() - state.started_at).total_seconds() * 1000)
