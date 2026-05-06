# Rental outbound voice agent POC

Eval-first proof of concept for an outbound voice agent that calls a backup cleaner after a same-day short-term-rental cleaning cancellation.

## Demo link
https://www.loom.com/share/21fb8f4e44644821885a270b9de803e0

## Design
[Design doc](DESIGN.md)

## Presentation
[Presentation slides](PRESENTATION.html)

## Local Setup

```bash
uv sync
cp .env.example .env
```

Fill `.env` with account credentials as they become available:

- `OPENAI_API_KEY`
- `LIVEKIT_URL`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`
- `LIVEKIT_SIP_OUTBOUND_TRUNK_ID`
- `RENTAL_VOICE_TEST_PHONE_NUMBER` for manual live-call tests

## Manual LiveKit Call Test

Run the agent worker in one terminal:

```bash
uv run rental-voice-agent dev
```

Place a manual outbound call from another terminal:

```bash
uv run rental-voice-agent call --to +15551234567
```

The worker terminal prints live `[transcript:*]`, `[tool:*]`, final `[state]`,
and post-call `[judge]` lines as the call runs. The final state snapshot is
written to `state_snapshots/{call_id}.json`; the live conversation-quality
judge writes a sidecar at `state_snapshots/{call_id}.judge.json`.

## Current Shape

- `src/rental_voice_agent/` contains the agent, state, tool, classifier, and config modules.
- `eval/` contains the offline extraction/classification harness. It replays
  scripted cleaner transcripts for fast regression checks; it is not a
  turn-by-turn synthetic conversation simulator.
- `fixtures/` contains structured cleaning request and property fixtures.
- `prompts/` contains versioned prompt/disclosure templates.

The eval harness runs with `uv run python -m eval`.
