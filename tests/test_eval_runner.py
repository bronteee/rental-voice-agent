from __future__ import annotations

import asyncio
from dataclasses import dataclass

from livekit.agents.llm import RealtimeError

from eval import runner
from eval.scenario import Scenario


def _scenario() -> Scenario:
    return Scenario(
        scenario_id="clear_yes_evergreen_01",
        axis="clear_yes",
        failure_mode=None,
        request_fixture="cleaning_request_01.json",
        cleaner_turns={},
        gold_outcome=None,
        gold_viability="viable",
        gold_end_reason="completed",
    )


@dataclass
class FakeState:
    tool_calls: list[object]
    end_reason: str | None


def test_retry_backoff_grows_and_caps() -> None:
    assert runner._retry_backoff_seconds(1, jitter=0.0) == 1.5
    assert runner._retry_backoff_seconds(2, jitter=0.0) == 3.75
    assert runner._retry_backoff_seconds(3, jitter=0.0) == 9.375
    assert runner._retry_backoff_seconds(4, jitter=0.0) == 12.0


def test_empty_tool_calls_with_turn_limit_retry_as_infra_flake(monkeypatch) -> None:
    attempts = 0
    sleeps: list[float] = []

    async def fake_run_text_eval_agent(*args, **kwargs) -> FakeState:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return FakeState(tool_calls=[], end_reason="turn_limit_exceeded")
        return FakeState(tool_calls=[object()], end_reason="completed")

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(runner, "EVAL_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(runner, "run_text_eval_agent", fake_run_text_eval_agent)
    monkeypatch.setattr(runner.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(runner, "_retry_backoff_seconds", lambda attempt: 3.0)

    state = asyncio.run(runner._run_scenario_with_retry(_scenario()))

    assert attempts == 2
    assert sleeps == [3.0]
    assert state.tool_calls


def test_realtime_errors_retry_with_backoff(monkeypatch) -> None:
    attempts = 0
    sleeps: list[float] = []

    async def fake_run_text_eval_agent(*args, **kwargs) -> FakeState:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RealtimeError("update_chat_ctx timed out.")
        return FakeState(tool_calls=[object()], end_reason="completed")

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(runner, "EVAL_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(runner, "run_text_eval_agent", fake_run_text_eval_agent)
    monkeypatch.setattr(runner.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(runner, "_retry_backoff_seconds", lambda attempt: 4.0)

    state = asyncio.run(runner._run_scenario_with_retry(_scenario()))

    assert attempts == 2
    assert sleeps == [4.0]
    assert state.tool_calls
