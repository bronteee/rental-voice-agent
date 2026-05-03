from __future__ import annotations

from datetime import datetime

from rental_voice_agent.state import CallState, Viability


UNREACHABLE_END_REASONS = {
    "cleaner_unreachable",
    "voicemail_detected",
    "mid_call_disconnect",
    "infra_failure",
}


def classify(state: CallState) -> tuple[Viability, str]:
    outcome = state.outcome

    if outcome is None:
        if state.end_reason in UNREACHABLE_END_REASONS:
            return "unreachable", f"end_reason={state.end_reason}"
        return "unclear", "no recorded call outcome"

    if outcome.availability_bool is False:
        return "declined", "cleaner declined availability"

    if (
        outcome.quoted_price_cents is not None
        and outcome.quoted_price_cents > state.request.max_budget_cents
    ):
        return "over_budget", "quoted price exceeds max budget"

    if outcome.eta_iso is not None:
        eta = datetime.fromisoformat(outcome.eta_iso)
        if eta > state.request.deadline:
            return "past_deadline", "ETA is after deadline"

    if outcome.confidence == "low":
        return "unclear", "low extraction confidence"

    return "viable", "available, in budget, before deadline"
