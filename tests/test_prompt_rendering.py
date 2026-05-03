from __future__ import annotations

from pathlib import Path

from rental_voice_agent.agent import render_disclosure, render_system_prompt
from rental_voice_agent.eval_agent import (
    _render_disclosure as render_eval_disclosure,
)
from rental_voice_agent.eval_agent import (
    _render_system_prompt as render_eval_system_prompt,
)
from rental_voice_agent.state import CleaningRequest, Property


ROOT = Path(__file__).resolve().parents[1]


def test_prompts_address_cleaning_business_not_person() -> None:
    request = CleaningRequest.from_json_file(ROOT / "fixtures/cleaning_request_01.json")
    property_obj = Property.load_for_request(request, ROOT / "fixtures" / "properties")

    disclosure = render_disclosure(
        ROOT / "prompts" / "disclosure_v1.md", request, property_obj
    )
    system_prompt = render_system_prompt(
        ROOT / "prompts" / "system_v1.md", request, property_obj
    )

    assert "is this Evergreen Turnovers?" in disclosure
    assert "Hi Evergreen" not in disclosure
    assert "Cleaning business: Evergreen Turnovers" in system_prompt
    assert "known backup cleaning business" in system_prompt
    assert "I have your team at" in system_prompt


def test_live_and_eval_prompt_rendering_stay_aligned() -> None:
    request = CleaningRequest.from_json_file(ROOT / "fixtures/cleaning_request_01.json")
    property_obj = Property.load_for_request(request, ROOT / "fixtures" / "properties")

    disclosure_path = ROOT / "prompts" / "disclosure_v1.md"
    system_prompt_path = ROOT / "prompts" / "system_v1.md"

    assert render_disclosure(disclosure_path, request, property_obj) == (
        render_eval_disclosure(disclosure_path, request, property_obj)
    )
    assert render_system_prompt(system_prompt_path, request, property_obj) == (
        render_eval_system_prompt(system_prompt_path, request, property_obj)
    )
