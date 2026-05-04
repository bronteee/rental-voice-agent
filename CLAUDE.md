# CLAUDE.md — Project Instructions

Project-scoped instructions for AI assistants working in this repo.
Higher-priority rules (e.g. `~/.claude/CLAUDE.md`) still apply.

---

## Validation protocol — required before claiming changes are done

After **any** code change, run all four checks below. Do not report a change
as complete or commit it until each one passes (or you have explicitly
acknowledged a known-failing baseline — see "Known baseline" below).

Run them in this order — fastest-feedback first, most-expensive last:

```bash
# 1. Lint — ~1s
uv run ruff check .

# 2. Format check — ~1s
uv run ruff format --check .

# 3. Type check — ~5s
uv run ty check

# 4. Unit tests — ~2s
uv run pytest -q

# 5. Eval harness — ~30-90s, hits OpenAI Realtime API (costs tokens)
uv run python -m eval
```

### When to skip the eval

Skip step 5 only if the change is **strictly limited to**:

- Documentation (`*.md`).
- Comments without code semantics changes.
- Test-only files (`tests/**`).
- Spec, build plan, or learnings docs.

Any change touching `src/rental_voice_agent/**`, `eval/**`, `prompts/**`, or
`fixtures/**` requires the eval to run. Cost is ~$0.10–0.30 per full run;
it's worth catching agent-behavior regressions immediately rather than
discovering them at demo time.

### How to handle failures

- **Lint / format / type errors you introduced**: fix before continuing.
- **Pre-existing failures**: see "Known baseline" — call them out explicitly
  in your report so the user knows they were not caused by current work.
- **Eval regressions**: do not paper over. Report which scenarios regressed,
  show the diff in `eval_runs/runs/<latest>/summary.md`, and surface the
  most likely cause before iterating.

### How to report

After running all checks, summarize concisely:

> Validation: ruff ✅ · ruff format ✅ · ty ✅ · pytest 7/7 ✅ · eval 7/7 ✅

Or if anything failed or was skipped, say so explicitly with the reason.

---

## Known baseline (as of 2026-05-03)

All five validation steps pass cleanly. No documented baseline failures.
Any failure in `ruff check`, `ruff format --check`, `ty check`, `pytest`,
or `eval` is a regression you introduced — investigate before continuing.

---

## Other project conventions

- **Eval-first**: every change to agent behavior is validated against the
  eval harness before being declared done. See `SPEC.md §4` and
  `keep_in_mind.md` for the rationale.
- **Tests resolve `eval/` and `src/rental_voice_agent/` automatically** via
  `[tool.pytest.ini_options] pythonpath = ["."]` in `pyproject.toml`.
  No `PYTHONPATH=.` prefix needed.
- **The eval harness drives the LiveKit Realtime session in text mode**
  (see `SPEC.md §4.8`). It does not exercise the STT path. Live calls
  are the only ASR validation; budget time for at least one live call
  before any demo.
- **Tool boundary is the determinism boundary**: the LLM extracts facts;
  deterministic Python (Viability Classifier, validators, state writes)
  makes decisions. Do not let LLM logic creep across this line — see
  `SPEC.md §5.1` and `SPEC.md §7`.
