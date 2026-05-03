from __future__ import annotations

from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad

from rental_voice_agent.agent import _build_live_realtime_model, _should_drain_session


def test_live_realtime_model_uses_server_vad_for_transcript_reliability() -> None:
    model = _build_live_realtime_model(
        model_name="gpt-realtime",
        api_key="test-api-key",
    )

    turn_detection = model._opts.turn_detection  # noqa: SLF001
    assert turn_detection is not None
    assert isinstance(turn_detection, ServerVad)
    assert turn_detection.type == "server_vad"
    assert turn_detection.interrupt_response is False
    assert turn_detection.threshold == 0.7
    assert turn_detection.silence_duration_ms == 700
    assert turn_detection.prefix_padding_ms == 300
    assert model._opts.input_audio_transcription is not None  # noqa: SLF001
    assert model._opts.input_audio_transcription.model == "gpt-4o-transcribe"  # noqa: SLF001


def test_live_agent_skips_drain_after_terminal_tool() -> None:
    assert _should_drain_session(terminal_tool_fired=True) is False
    assert _should_drain_session(terminal_tool_fired=False) is True
