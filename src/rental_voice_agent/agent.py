from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from livekit import api
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from openai.types import realtime as openai_realtime
from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad
from livekit.plugins.openai import realtime

from rental_voice_agent.classifier import classify
from rental_voice_agent.config import load_settings
from rental_voice_agent.live_judge import write_live_call_judge
from rental_voice_agent.state import (
    CallState,
    CleaningRequest,
    Property,
    Speaker,
    ToolCallEntry,
    TranscriptEntry,
)
from rental_voice_agent.tools import create_call_tools


logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = ROOT / "fixtures"
PROMPTS_DIR = ROOT / "prompts"
STATE_SNAPSHOTS_DIR = ROOT / "state_snapshots"


async def entrypoint(ctx: JobContext) -> None:
    metadata = _job_metadata(ctx)
    logger.info("[job] accepted metadata=%s", json.dumps(metadata, sort_keys=True))
    request_path = _path_from_metadata_or_env(
        metadata,
        "request_fixture",
        "RENTAL_VOICE_REQUEST_FIXTURE",
        FIXTURES_DIR / "cleaning_request_01.json",
    )
    snapshot_dir = _path_from_metadata_or_env(
        metadata,
        "snapshot_dir",
        "RENTAL_VOICE_STATE_SNAPSHOT_DIR",
        STATE_SNAPSHOTS_DIR,
    )
    await run_agent(
        ctx,
        request_path=request_path,
        prompts_dir=PROMPTS_DIR,
        fixtures_dir=FIXTURES_DIR,
        snapshot_dir=snapshot_dir,
        cleaner_phone_override=metadata.get("cleaner_phone_override"),
    )


async def run_agent(
    ctx: JobContext,
    *,
    request_path: Path,
    prompts_dir: Path,
    fixtures_dir: Path,
    snapshot_dir: Path,
    cleaner_phone_override: str | None = None,
) -> CallState:
    """Run the LiveKit room/audio agent for a single cleaning request."""
    settings = load_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required to run the LiveKit voice agent")

    request = CleaningRequest.from_json_file(request_path)
    # Dev-override (per CLI dispatch metadata) so the dialed number IS
    # state.cleaner.phone — single source of truth.
    if cleaner_phone_override:
        request.cleaner.phone = cleaner_phone_override
    property_obj = Property.load_for_request(request, fixtures_dir / "properties")
    state = CallState(
        call_id=str(uuid4()),
        started_at=datetime.now().astimezone(),
        request=request,
        cleaner=request.cleaner,
        property=property_obj,
    )
    _print_call_start(
        state,
        request_path,
        snapshot_dir,
        model_name=settings.openai_realtime_model,
    )
    call_tools = create_call_tools(
        state,
        property_fixtures_dir=fixtures_dir / "properties",
        on_tool_call=_print_tool_call,
    )

    disclosure = render_disclosure(
        prompts_dir / "disclosure_v1.md", request, property_obj
    )
    instructions = render_system_prompt(
        prompts_dir / "system_v1.md", request, property_obj
    )

    model = _build_live_realtime_model(
        model_name=settings.openai_realtime_model,
        api_key=settings.openai_api_key,
    )
    _print_live_audio_config(model)
    # ivr_detection=False per SPEC §3: the spec says we DETECT voicemail/IVR
    # and TERMINATE; we do not navigate them. With ivr_detection=True, LiveKit's
    # AMD pipeline can auto-classify a real human as MACHINE_IVR, attach an
    # IVRActivity that injects DTMF tools, and on silence fire an instruction-
    # less generate_reply() that improvises (observed: agent speaking Spanish,
    # no transcripts captured because the auto-replies bypass our state-capture
    # handlers). Voicemail detection is handled by the system prompt + the
    # `end_call(reason="voicemail_detected")` tool path instead.
    session = AgentSession(llm=model, max_tool_steps=4, ivr_detection=False)
    done = asyncio.Event()

    _attach_state_capture(session, state, done)

    async def _on_shutdown(reason: str) -> None:
        _finalize_shutdown(state, snapshot_dir, reason)

    ctx.add_shutdown_callback(_on_shutdown)

    try:
        logger.info("[room] connecting")
        await ctx.connect()
        logger.info("[room] connected; waiting for participant")
        participant = await ctx.wait_for_participant()
        participant_identity = getattr(participant, "identity", "<unknown>")
        logger.info(
            f"[room] participant joined identity={participant_identity}; "
            "starting agent session"
        )
        agent = Agent(instructions=instructions, tools=call_tools.as_livekit_tools())
        await session.start(agent, room=ctx.room)
        logger.info("[room] agent session started")

        # Give AEC convergence a head start before TTS hits the wire — reduces
        # echo-as-speech false positives during the warmup window. See LEARNINGS.md.
        # Do not append this to the transcript until LiveKit emits the actual
        # assistant item; otherwise logs can claim disclosure audio played while
        # the SIP leg was still ringing.
        await asyncio.sleep(1.5)
        logger.info("[disclosure] generating scripted opening")
        disclosure_handle = session.generate_reply(
            user_input="Read the opening disclosure to the cleaner now.",
            instructions=(
                f"Say exactly this opening disclosure and nothing else:\n\n{disclosure}"
            ),
            input_modality="text",
        )
        await disclosure_handle.wait_for_playout()
        state.disclosure_played = True
        logger.info("[disclosure] playout complete")

        # Event-driven wait: end_event set by end_call/escalate_to_host tools,
        # done set by session close/error, or duration timeout fires.
        terminal_tool_fired = False
        end_task = asyncio.create_task(call_tools.end_event.wait())
        done_task = asyncio.create_task(done.wait())
        try:
            finished, pending = await asyncio.wait(
                [end_task, done_task],
                timeout=settings.max_call_duration_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            terminal_tool_fired = end_task in finished
            for task in pending:
                task.cancel()
            if not finished and state.end_reason is None:
                state.end_reason = "duration_exceeded"
                state.ended_at = datetime.now().astimezone()
        finally:
            for task in (end_task, done_task):
                if not task.done():
                    task.cancel()

        if _should_drain_session(terminal_tool_fired=terminal_tool_fired):
            await session.drain()
        else:
            logger.info("[room] terminal tool fired; skipping session drain")
    except Exception:
        if state.end_reason is None:
            state.end_reason = "infra_failure"
            state.ended_at = datetime.now().astimezone()
        raise
    finally:
        if state.end_reason is None:
            state.end_reason = "mid_call_disconnect"
            state.ended_at = datetime.now().astimezone()
        await session.aclose()
        await model.aclose()
        await _hangup_room(ctx, settings)
        snapshot_path = _finalize_state(state, snapshot_dir)
        await write_live_call_judge(state, snapshot_path, settings)

    return state


async def _hangup_room(ctx: JobContext, settings) -> None:
    """Force-hang up the call by deleting the LiveKit room.

    `session.aclose()` only closes the agent session — the SIP participant
    (the cleaner's phone leg) stays in the room until the room's empty_timeout
    expires (5 min). Deleting the room evicts all participants immediately.
    """
    room_name = getattr(getattr(ctx, "room", None), "name", None)
    if not room_name:
        return
    try:
        lkapi = api.LiveKitAPI(
            settings.livekit_url,
            settings.livekit_api_key,
            settings.livekit_api_secret,
        )
        try:
            await lkapi.room.delete_room(api.DeleteRoomRequest(room=room_name))
            logger.info("[room] deleted %s", room_name)
        finally:
            await lkapi.aclose()
    except Exception as exc:  # noqa: BLE001 — best-effort hangup
        logger.warning("[room] delete_room failed: %r", exc)


def main() -> None:
    settings = load_settings()
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="rental-voice-agent",
            ws_url=settings.livekit_url,
            api_key=settings.livekit_api_key,
            api_secret=settings.livekit_api_secret,
        )
    )


def _build_live_realtime_model(
    *,
    model_name: str,
    api_key: str,
) -> realtime.RealtimeModel:
    # Use ordinary server VAD for the live SIP path. Semantic VAD reduced early
    # echo false positives, but in live demo calls it correlated with missing
    # transcript events. Speakerphone demo calls can feed the agent's own TTS
    # back through the phone mic as cleaner audio, so keep VAD conservative and
    # disable barge-in interruption to avoid truncating the disclosure mid-turn.
    return realtime.RealtimeModel(
        model=model_name,
        voice="marin",
        modalities=["audio"],
        api_key=api_key,
        temperature=0.6,
        input_audio_transcription=openai_realtime.AudioTranscription(
            model="gpt-4o-transcribe",
        ),
        turn_detection=ServerVad(
            type="server_vad",
            interrupt_response=False,
            threshold=0.7,
            silence_duration_ms=700,
            prefix_padding_ms=300,
        ),
    )


def render_disclosure(
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


def render_system_prompt(
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


def _attach_state_capture(
    session: AgentSession,
    state: CallState,
    done: asyncio.Event,
) -> None:
    last_user_speech_end: list[float | None] = [None]

    @session.on("user_input_transcribed")
    def _capture_user_transcript(event: object) -> None:
        transcript = getattr(event, "transcript", "").strip()
        is_final = bool(getattr(event, "is_final", False))
        if not is_final:
            if transcript:
                logger.info("[transcript:cleaner:interim] %s", transcript)
            else:
                logger.info("[transcript:cleaner:pending] awaiting final")
            return
        if transcript:
            _append_transcript_once(state, "cleaner", transcript)
        else:
            logger.info("[transcript:cleaner:empty] final transcript was empty")

    @session.on("conversation_item_added")
    def _capture_agent_transcript(event: object) -> None:
        item = getattr(event, "item", None)
        if getattr(item, "role", None) != "assistant":
            return
        text = _message_text(getattr(item, "content", ""))
        if text:
            _append_transcript_once(state, "agent", text)

    @session.on("user_state_changed")
    def _capture_user_state(event: object) -> None:
        if (
            getattr(event, "old_state", None) == "speaking"
            and getattr(event, "new_state", None) == "listening"
        ):
            last_user_speech_end[0] = getattr(event, "created_at", None)

    @session.on("agent_state_changed")
    def _capture_turn_latency(event: object) -> None:
        if getattr(event, "new_state", None) != "speaking":
            return
        created_at = getattr(event, "created_at", None)
        if isinstance(created_at, int | float) and last_user_speech_end[0] is not None:
            latency_ms = int((created_at - last_user_speech_end[0]) * 1000)
            if latency_ms >= 0:
                state.turn_latencies_ms.append(latency_ms)
            last_user_speech_end[0] = None

    @session.on("close")
    def _capture_close(_: object) -> None:
        done.set()

    @session.on("error")
    def _capture_error(event: object) -> None:
        if state.end_reason is None:
            state.end_reason = "infra_failure"
            state.ended_at = datetime.now().astimezone()
        _append_transcript_once(state, "system", f"AgentSession error: {event!r}")
        done.set()


def _finalize_shutdown(state: CallState, snapshot_dir: Path, reason: str) -> None:
    if state.end_reason is None:
        state.end_reason = "mid_call_disconnect"
        state.ended_at = datetime.now().astimezone()
    _append_transcript_once(state, "system", f"Job shutdown: {reason}")
    _finalize_state(state, snapshot_dir)


def _finalize_state(state: CallState, snapshot_dir: Path) -> Path:
    if state.ended_at is None:
        state.ended_at = datetime.now().astimezone()
    if state.viability is None:
        viability, reason = classify(state)
        state.viability = viability
        state.classification_reason = reason
    snapshot_path = snapshot_dir / f"{state.call_id}.json"
    state.write_json(snapshot_path)
    logger.info(
        "[state] "
        f"end_reason={state.end_reason} viability={state.viability} "
        f"snapshot={snapshot_path}"
    )
    return snapshot_path


def _append_transcript_once(
    state: CallState,
    speaker: Speaker,
    text: str,
) -> None:
    if (
        state.transcript
        and state.transcript[-1].speaker == speaker
        and state.transcript[-1].text == text
    ):
        return
    entry = TranscriptEntry(
        speaker=speaker,
        text=text,
        t_ms=_elapsed_ms(state),
    )
    state.transcript.append(entry)
    logger.info("[transcript:%s] %s", entry.speaker, entry.text)


def _print_tool_call(entry: ToolCallEntry) -> None:
    args = json.dumps(entry.args, sort_keys=True)
    logger.info("[tool:%s] args=%s result=%s", entry.name, args, entry.result)


def _print_call_start(
    state: CallState,
    request_path: Path,
    snapshot_dir: Path,
    *,
    model_name: str,
) -> None:
    logger.info(
        "[job] "
        f"call_id={state.call_id} model={model_name} "
        f"request={request_path} snapshot_dir={snapshot_dir} "
        f"cleaner_phone={state.cleaner.phone}"
    )


def _print_live_audio_config(model: realtime.RealtimeModel) -> None:
    turn_detection = model._opts.turn_detection
    transcription = model._opts.input_audio_transcription
    vad_type = None if turn_detection is None else turn_detection.type
    interrupt_response = (
        None if turn_detection is None else turn_detection.interrupt_response
    )
    transcription_model = None if transcription is None else transcription.model
    logger.info(
        "[audio] "
        f"vad={vad_type} interrupt_response={interrupt_response} "
        f"input_audio_transcription={transcription_model}"
    )


def _should_drain_session(*, terminal_tool_fired: bool) -> bool:
    return not terminal_tool_fired


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(str(part) for part in content if part).strip()
    return ""


def _job_metadata(ctx: JobContext) -> dict:
    raw = getattr(ctx.job, "metadata", "")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("[state] ignoring non-JSON dispatch metadata: %r", raw)
        return {}
    return data if isinstance(data, dict) else {}


def _path_from_metadata_or_env(
    metadata: dict,
    metadata_key: str,
    env_name: str,
    default: Path,
) -> Path:
    raw = metadata.get(metadata_key) or os.getenv(env_name)
    if not raw:
        return default
    path = Path(str(raw))
    return path if path.is_absolute() else ROOT / path


def _path_from_env(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if not raw:
        return default
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def _deadline_human(deadline: datetime) -> str:
    return deadline.strftime("%-I:%M %p").replace(":00", "")


def _elapsed_ms(state: CallState) -> int:
    return int((datetime.now().astimezone() - state.started_at).total_seconds() * 1000)


if __name__ == "__main__":
    main()
