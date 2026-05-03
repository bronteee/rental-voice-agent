from __future__ import annotations

import asyncio
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path

from livekit.agents.llm import RealtimeError

from eval.leaderboard import (
    append_leaderboard,
    append_scenario_history,
    utc_run_id,
    write_config,
    write_leaderboard_md,
    write_results,
    write_summary,
)
from eval.scenario import Scenario, load_scenarios
from eval.scorer import ScenarioScore, score_run
from rental_voice_agent.classifier import classify
from rental_voice_agent.eval_agent import run_text_eval_agent


# Sequential Realtime sessions occasionally hit rate limits or
# `update_chat_ctx timed out` in the LiveKit OpenAI plugin under burst load.
# Retry whole scenarios with exponential backoff before recording failure.
EVAL_MAX_ATTEMPTS = int(os.getenv("RENTAL_VOICE_EVAL_MAX_ATTEMPTS", "4"))
EVAL_RETRY_INITIAL_BACKOFF_S = float(
    os.getenv("RENTAL_VOICE_EVAL_RETRY_INITIAL_S", "1.5")
)
EVAL_RETRY_BACKOFF_MULTIPLIER = float(
    os.getenv("RENTAL_VOICE_EVAL_RETRY_MULTIPLIER", "2.5")
)
EVAL_RETRY_MAX_BACKOFF_S = float(os.getenv("RENTAL_VOICE_EVAL_MAX_S", "12"))
EVAL_RETRY_JITTER = float(os.getenv("RENTAL_VOICE_EVAL_RETRY_JITTER", "0.25"))
EVAL_SCENARIO_SPACING_S = float(
    os.getenv("RENTAL_VOICE_EVAL_SCENARIO_SPACING_S", "1.0")
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = ROOT / "fixtures"
SCENARIO_DIR = ROOT / "eval" / "scenarios"
EVAL_RUNS_DIR = ROOT / "eval_runs"
PROMPTS_DIR = ROOT / "prompts"


def main() -> None:
    asyncio.run(_main())


async def _main() -> None:
    scenarios = load_scenarios(SCENARIO_DIR)
    if not scenarios:
        raise SystemExit(f"No scenarios found in {SCENARIO_DIR}")

    run_id = utc_run_id(datetime.now())
    run_dir = EVAL_RUNS_DIR / "runs" / run_id
    snapshot_dir = run_dir / "state_snapshots"
    run_dir.mkdir(parents=True, exist_ok=True)

    scenario_scores: list[tuple[Scenario, ScenarioScore]] = []
    for index, scenario in enumerate(scenarios):
        if index > 0 and EVAL_SCENARIO_SPACING_S > 0:
            await asyncio.sleep(EVAL_SCENARIO_SPACING_S)
        state = await _run_scenario_with_retry(scenario)
        viability, reason = classify(state)
        state.viability = viability
        state.classification_reason = reason
        state.write_json(snapshot_dir / f"{scenario.scenario_id}.json")
        scenario_scores.append((scenario, score_run(state, scenario)))

    scores = [score for _, score in scenario_scores]
    write_config(run_dir, git_sha=_git_sha())
    write_results(run_dir, scores)
    write_summary(run_dir, scores)
    append_leaderboard(EVAL_RUNS_DIR, run_id, scores)
    append_scenario_history(EVAL_RUNS_DIR, run_id, scenario_scores)
    write_leaderboard_md(EVAL_RUNS_DIR)

    passed = sum(score.passed for score in scores)
    print(f"Eval run {run_id}: {passed}/{len(scores)} scenarios passed")
    print(f"Summary: {run_dir / 'summary.md'}")


async def _run_scenario_with_retry(scenario: Scenario):
    last_exc: Exception | None = None
    last_state = None
    for attempt in range(1, EVAL_MAX_ATTEMPTS + 1):
        try:
            state = await run_text_eval_agent(
                scenario,
                fixtures_dir=FIXTURES_DIR,
                prompts_dir=PROMPTS_DIR,
            )
            if not _looks_like_realtime_infra_failure(state):
                return state
            last_state = state
            print(
                f"[eval] {scenario.scenario_id} attempt {attempt}/"
                f"{EVAL_MAX_ATTEMPTS} produced no tool calls "
                f"(end_reason={state.end_reason}); treating as transient "
                "Realtime infra failure",
                flush=True,
            )
        except RealtimeError as exc:
            last_exc = exc
            print(
                f"[eval] {scenario.scenario_id} attempt {attempt}/"
                f"{EVAL_MAX_ATTEMPTS} hit RealtimeError: {exc!r}",
                flush=True,
            )
        if attempt < EVAL_MAX_ATTEMPTS:
            await asyncio.sleep(_retry_backoff_seconds(attempt))
    if last_state is not None:
        return last_state
    raise RuntimeError(
        f"scenario {scenario.scenario_id} failed after {EVAL_MAX_ATTEMPTS} attempts"
    ) from last_exc


def _looks_like_realtime_infra_failure(state) -> bool:
    return not state.tool_calls and state.end_reason == "turn_limit_exceeded"


def _retry_backoff_seconds(attempt: int, *, jitter: float = EVAL_RETRY_JITTER) -> float:
    base = EVAL_RETRY_INITIAL_BACKOFF_S * (
        EVAL_RETRY_BACKOFF_MULTIPLIER ** (attempt - 1)
    )
    capped = min(base, EVAL_RETRY_MAX_BACKOFF_S)
    if jitter <= 0:
        return capped
    return capped * random.uniform(1 - jitter, 1 + jitter)


def _git_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()
