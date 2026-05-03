from __future__ import annotations

import asyncio
import json

from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, function_tool
from livekit.plugins.openai import realtime

from rental_voice_agent.config import load_settings


async def main() -> None:
    load_dotenv(override=True)
    settings = load_settings()
    if not settings.openai_api_key:
        raise SystemExit("OPENAI_API_KEY is not set")

    tool_payloads: list[dict] = []

    @function_tool
    async def record_call_outcome(
        availability_bool: bool,
        quoted_price_cents: int | None,
        eta_iso: str | None,
        confidence: str,
        notes: str,
    ) -> str:
        tool_payloads.append(
            {
                "availability_bool": availability_bool,
                "quoted_price_cents": quoted_price_cents,
                "eta_iso_present": bool(eta_iso),
                "confidence": confidence,
                "notes_present": bool(notes),
            }
        )
        return "Recorded. No more follow-up is needed."

    model = realtime.RealtimeModel(
        model=settings.openai_realtime_model,
        voice="marin",
        modalities=["text"],
        api_key=settings.openai_api_key,
        temperature=0.6,
    )
    session = AgentSession(llm=model, max_tool_steps=3)
    agent = Agent(
        instructions=(
            "You extract cleaner call outcomes. When the user gives the cleaner's "
            "answer, call record_call_outcome exactly once. Use ISO-8601 times. "
            "Do not ask a follow-up in this smoke test."
        ),
        tools=[record_call_outcome],
    )

    try:
        await session.start(agent)
        handle = session.generate_reply(
            user_input=(
                "Evergreen Turnovers says their team can be at the Capitol Hill "
                "Studio by 2:30 PM today for $120. Deadline is "
                "2026-05-02T15:00:00-07:00."
            ),
            input_modality="text",
        )
        await handle.wait_for_playout()
        await session.drain()
    finally:
        await session.aclose()
        await model.aclose()

    if not tool_payloads:
        raise SystemExit("LiveKit text spike did not call record_call_outcome")

    print("LiveKit text spike tool payload:")
    print(json.dumps(tool_payloads[0], indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
