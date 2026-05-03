from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal


SCHEMA_VERSION = 2

Confidence = Literal["high", "medium", "low"]
Speaker = Literal["agent", "cleaner", "system"]
EndReason = Literal[
    "completed",
    "voicemail_detected",
    "cleaner_declined",
    "cleaner_unreachable",
    "cleaner_hostile",
    "off_topic_repeated",
    "extraction_failed",
    "mid_call_disconnect",
    "turn_limit_exceeded",
    "duration_exceeded",
    "infra_failure",
    "escalation",
]
Viability = Literal[
    "viable",
    "declined",
    "over_budget",
    "past_deadline",
    "unclear",
    "unreachable",
]


@dataclass
class CleanerProfile:
    cleaner_id: str
    name: str
    phone: str
    priority: int

    @classmethod
    def from_dict(cls, data: dict) -> CleanerProfile:
        return cls(
            cleaner_id=data["cleaner_id"],
            name=data["name"],
            phone=data["phone"],
            priority=int(data["priority"]),
        )


@dataclass
class CleaningRequest:
    request_id: str
    host_first_name: str
    property_id: str
    deadline: datetime
    max_budget_cents: int
    cleaner: CleanerProfile

    @classmethod
    def from_dict(cls, data: dict) -> CleaningRequest:
        return cls(
            request_id=data["request_id"],
            host_first_name=data["host_first_name"],
            property_id=data["property_id"],
            deadline=datetime.fromisoformat(data["deadline"]),
            max_budget_cents=int(data["max_budget_cents"]),
            cleaner=CleanerProfile.from_dict(data["cleaner"]),
        )

    @classmethod
    def from_json_file(cls, path: Path) -> CleaningRequest:
        return cls.from_dict(json.loads(path.read_text()))


@dataclass
class Property:
    """All property data lives here. The cleaning request only references it by id."""

    property_id: str
    name: str
    summary_short: str
    bedrooms: int | None = None
    bathrooms: int | None = None
    square_footage: int | None = None
    floor_type: str | None = None
    parking_notes: str | None = None
    special_instructions: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> Property:
        return cls(
            property_id=data["property_id"],
            name=data["name"],
            summary_short=data["summary_short"],
            bedrooms=_optional_int(data.get("bedrooms")),
            bathrooms=_optional_int(data.get("bathrooms")),
            square_footage=_optional_int(data.get("square_footage")),
            floor_type=data.get("floor_type"),
            parking_notes=data.get("parking_notes"),
            special_instructions=data.get("special_instructions"),
        )

    @classmethod
    def from_json_file(cls, path: Path) -> Property:
        return cls.from_dict(json.loads(path.read_text()))

    @classmethod
    def load_for_request(
        cls, request: CleaningRequest, properties_dir: Path
    ) -> Property:
        return cls.from_json_file(properties_dir / f"{request.property_id}.json")


@dataclass
class TranscriptEntry:
    speaker: Speaker
    text: str
    t_ms: int

    @classmethod
    def from_dict(cls, data: dict) -> TranscriptEntry:
        return cls(speaker=data["speaker"], text=data["text"], t_ms=int(data["t_ms"]))


@dataclass
class ToolCallEntry:
    name: str
    args: dict
    result: str | None
    t_ms: int

    @classmethod
    def from_dict(cls, data: dict) -> ToolCallEntry:
        return cls(
            name=data["name"],
            args=data["args"],
            result=data.get("result"),
            t_ms=int(data["t_ms"]),
        )


@dataclass
class CallOutcome:
    availability_bool: bool
    quoted_price_cents: int | None
    eta_iso: str | None
    confidence: Confidence
    notes: str
    cleaner_objections: list[str]

    @classmethod
    def from_dict(cls, data: dict) -> CallOutcome:
        return cls(
            availability_bool=bool(data["availability_bool"]),
            quoted_price_cents=(
                None
                if data.get("quoted_price_cents") is None
                else int(data["quoted_price_cents"])
            ),
            eta_iso=data.get("eta_iso"),
            confidence=data["confidence"],
            notes=data.get("notes", ""),
            cleaner_objections=list(data.get("cleaner_objections", [])),
        )


@dataclass
class CallState:
    call_id: str
    started_at: datetime
    request: CleaningRequest
    cleaner: CleanerProfile
    property: Property
    schema_version: int = SCHEMA_VERSION
    twilio_call_sid: str | None = None
    ended_at: datetime | None = None
    transcript: list[TranscriptEntry] = field(default_factory=list)
    tool_calls: list[ToolCallEntry] = field(default_factory=list)
    outcome: CallOutcome | None = None
    end_reason: EndReason | None = None
    turn_latencies_ms: list[int] = field(default_factory=list)
    viability: Viability | None = None
    classification_reason: str | None = None
    disclosure_played: bool = False

    def to_dict(self) -> dict:
        data = asdict(self)
        data["started_at"] = self.started_at.isoformat()
        data["ended_at"] = None if self.ended_at is None else self.ended_at.isoformat()
        data["request"]["deadline"] = self.request.deadline.isoformat()
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n")

    @classmethod
    def from_dict(cls, data: dict) -> CallState:
        schema_version = int(data["schema_version"])
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"CallState schema version mismatch: file is v{schema_version}, "
                f"code is v{SCHEMA_VERSION}. Migrate the snapshot or check out the "
                f"matching code revision (per SPEC §9.4)."
            )
        return cls(
            schema_version=schema_version,
            call_id=data["call_id"],
            twilio_call_sid=data.get("twilio_call_sid"),
            started_at=datetime.fromisoformat(data["started_at"]),
            ended_at=(
                None
                if data.get("ended_at") is None
                else datetime.fromisoformat(data["ended_at"])
            ),
            request=CleaningRequest.from_dict(data["request"]),
            cleaner=CleanerProfile.from_dict(data["cleaner"]),
            property=Property.from_dict(data["property"]),
            transcript=[
                TranscriptEntry.from_dict(entry) for entry in data.get("transcript", [])
            ],
            tool_calls=[
                ToolCallEntry.from_dict(entry) for entry in data.get("tool_calls", [])
            ],
            outcome=(
                None
                if data.get("outcome") is None
                else CallOutcome.from_dict(data["outcome"])
            ),
            end_reason=data.get("end_reason"),
            turn_latencies_ms=list(data.get("turn_latencies_ms", [])),
            viability=data.get("viability"),
            classification_reason=data.get("classification_reason"),
            disclosure_played=bool(data.get("disclosure_played", False)),
        )

    @classmethod
    def from_json(cls, raw: str) -> CallState:
        return cls.from_dict(json.loads(raw))

    @classmethod
    def from_json_file(cls, path: Path) -> CallState:
        return cls.from_json(path.read_text())


def _optional_int(value: int | str | None) -> int | None:
    if value is None:
        return None
    return int(value)
