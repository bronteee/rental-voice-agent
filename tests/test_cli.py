from __future__ import annotations

from rental_voice_agent.cli import _build_parser


def test_manual_call_waits_until_answered_by_default() -> None:
    args = _build_parser().parse_args(["call"])

    assert args.wait_until_answered is True


def test_manual_call_can_disable_answer_wait_for_telephony_debugging() -> None:
    args = _build_parser().parse_args(["call", "--no-wait-until-answered"])

    assert args.wait_until_answered is False
