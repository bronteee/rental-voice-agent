from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

from eval.scorer import ScenarioScore
from eval.scenario import Scenario
from rental_voice_agent.config import load_settings


def utc_run_id(now: datetime) -> str:
    return now.astimezone().isoformat(timespec="seconds").replace(":", "-")


def write_config(run_dir: Path, *, git_sha: str | None) -> None:
    settings = load_settings()
    root = Path(__file__).resolve().parents[1]
    system_prompt = root / "prompts" / "system_v1.md"
    disclosure_template = root / "prompts" / "disclosure_v1.md"
    config = {
        "model_id": settings.openai_realtime_model,
        "git_sha": git_sha,
        "system_prompt_sha": _file_sha256(system_prompt),
        "disclosure_template_sha": _file_sha256(disclosure_template),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def write_results(run_dir: Path, scores: list[ScenarioScore]) -> None:
    payload = [score.to_dict() for score in scores]
    (run_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")


def write_summary(run_dir: Path, scores: list[ScenarioScore]) -> None:
    total = len(scores)
    passed = sum(score.passed for score in scores)
    viability = sum(score.viability_pass for score in scores)
    extraction = sum(score.extraction_pass for score in scores)
    disclosure = sum(score.disclosure_pass for score in scores)
    lines = [
        "# Eval Run Summary",
        "",
        f"- Scenarios: {total}",
        f"- Overall pass: {passed}/{total}",
        f"- Viability pass: {viability}/{total}",
        f"- Extraction pass: {extraction}/{total}",
        f"- Disclosure pass: {disclosure}/{total}",
        "",
        "| Scenario | Overall | Viability | Extraction | Disclosure | End reason |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for score in scores:
        lines.append(
            "| "
            f"{score.scenario_id} | "
            f"{_mark(score.passed)} | "
            f"{_mark(score.viability_pass)} | "
            f"{_mark(score.extraction_pass)} | "
            f"{_mark(score.disclosure_pass)} | "
            f"{_mark(score.end_reason_pass)} |"
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")


def append_leaderboard(
    eval_runs_dir: Path, run_id: str, scores: list[ScenarioScore]
) -> None:
    path = eval_runs_dir / "leaderboard.csv"
    write_header = not path.exists()
    total = len(scores)
    metrics = {
        "overall_pass_rate": _rate(sum(score.passed for score in scores), total),
        "viability_accuracy": _rate(
            sum(score.viability_pass for score in scores), total
        ),
        "extraction_accuracy": _rate(
            sum(score.extraction_pass for score in scores), total
        ),
        "disclosure_accuracy": _rate(
            sum(score.disclosure_pass for score in scores), total
        ),
        "end_reason_accuracy": _rate(
            sum(score.end_reason_pass for score in scores), total
        ),
    }
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_id", "metric", "value"])
        if write_header:
            writer.writeheader()
        for metric, value in metrics.items():
            writer.writerow({"run_id": run_id, "metric": metric, "value": value})


def write_leaderboard_md(eval_runs_dir: Path) -> None:
    """Pivot ``leaderboard.csv`` into a human-readable cross-run table.

    Produces one row per run with one column per metric. The metric set is
    discovered from the CSV so older runs (pre-disclosure) render as blank
    cells rather than disappearing or showing 0.
    """
    csv_path = eval_runs_dir / "leaderboard.csv"
    md_path = eval_runs_dir / "leaderboard.md"
    if not csv_path.exists():
        md_path.write_text("# Eval Leaderboard\n\n_No runs recorded yet._\n")
        return

    rows: dict[str, dict[str, str]] = {}
    metrics_seen: list[str] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            run_id = row["run_id"]
            metric = row["metric"]
            value = row["value"]
            rows.setdefault(run_id, {})[metric] = value
            if metric not in metrics_seen:
                metrics_seen.append(metric)

    metric_order = [
        "overall_pass_rate",
        "viability_accuracy",
        "extraction_accuracy",
        "disclosure_accuracy",
        "end_reason_accuracy",
    ]
    metric_columns = [m for m in metric_order if m in metrics_seen] + [
        m for m in metrics_seen if m not in metric_order
    ]

    sorted_runs = sorted(rows.keys())
    header = "| run_id | " + " | ".join(metric_columns) + " |"
    divider = "|---|" + "|".join("---:" for _ in metric_columns) + "|"
    body_lines = []
    for run_id in sorted_runs:
        cells = [rows[run_id].get(metric, "") for metric in metric_columns]
        body_lines.append(f"| {run_id} | " + " | ".join(cells) + " |")

    md = [
        "# Eval Leaderboard",
        "",
        f"_Auto-generated from `leaderboard.csv` — {len(sorted_runs)} runs._",
        "",
        header,
        divider,
        *body_lines,
        "",
    ]
    md_path.write_text("\n".join(md))


def append_scenario_history(
    eval_runs_dir: Path,
    run_id: str,
    scenario_scores: list[tuple[Scenario, ScenarioScore]],
) -> None:
    path = eval_runs_dir / "scenario_history.csv"
    write_header = not path.exists()
    fieldnames = [
        "run_id",
        "scenario_id",
        "axis",
        "gold_viability",
        "viability_pass",
        "extraction_pass",
        "disclosure_pass",
        "end_reason_pass",
        "passed",
    ]
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for scenario, score in scenario_scores:
            writer.writerow(
                {
                    "run_id": run_id,
                    "scenario_id": scenario.scenario_id,
                    "axis": scenario.axis,
                    "gold_viability": scenario.gold_viability,
                    "viability_pass": score.viability_pass,
                    "extraction_pass": score.extraction_pass,
                    "disclosure_pass": score.disclosure_pass,
                    "end_reason_pass": score.end_reason_pass,
                    "passed": score.passed,
                }
            )


def _rate(count: int, total: int) -> str:
    return "0.000" if total == 0 else f"{count / total:.3f}"


def _mark(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()
