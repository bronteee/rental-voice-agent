from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv(override=True)


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    openai_realtime_model: str
    openai_judge_model: str
    livekit_url: str | None
    livekit_api_key: str | None
    livekit_api_secret: str | None
    twilio_account_sid: str | None
    twilio_auth_token: str | None
    twilio_phone_number: str | None
    livekit_sip_outbound_trunk_id: str | None
    extraction_retry_limit: int
    off_topic_redirect_limit: int
    max_agent_turns: int
    max_call_duration_seconds: int


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value in (None, "") else int(value)


def load_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_realtime_model=os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime"),
        openai_judge_model=os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o-mini"),
        livekit_url=os.getenv("LIVEKIT_URL"),
        livekit_api_key=os.getenv("LIVEKIT_API_KEY"),
        livekit_api_secret=os.getenv("LIVEKIT_API_SECRET"),
        twilio_account_sid=os.getenv("TWILIO_ACCOUNT_SID"),
        twilio_auth_token=os.getenv("TWILIO_AUTH_TOKEN"),
        twilio_phone_number=os.getenv("TWILIO_PHONE_NUMBER"),
        livekit_sip_outbound_trunk_id=os.getenv("LIVEKIT_SIP_OUTBOUND_TRUNK_ID"),
        extraction_retry_limit=_int_env("EXTRACTION_RETRY_LIMIT", 2),
        off_topic_redirect_limit=_int_env("OFF_TOPIC_REDIRECT_LIMIT", 2),
        max_agent_turns=_int_env("MAX_AGENT_TURNS", 30),
        max_call_duration_seconds=_int_env("MAX_CALL_DURATION_SECONDS", 180),
    )
