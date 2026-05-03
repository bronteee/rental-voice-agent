from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from livekit import api

from rental_voice_agent.config import load_settings
from rental_voice_agent.state import CleaningRequest


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AGENT_NAME = "rental-voice-agent"
DEFAULT_REQUEST_FIXTURE = ROOT / "fixtures" / "cleaning_request_01.json"
DEFAULT_SNAPSHOT_DIR = ROOT / "state_snapshots"
LIVEKIT_WORKER_COMMANDS = {"console", "dev", "start", "download-files"}


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] in LIVEKIT_WORKER_COMMANDS:
        _run_worker_cli(argv)
        return

    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "worker":
        _run_worker_cli(args.worker_args)
    elif args.command == "call":
        asyncio.run(_place_call(args))
    else:
        parser.print_help()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rental-voice-agent",
        description="Manual CLI for the Rental outbound voice agent POC.",
    )
    subparsers = parser.add_subparsers(dest="command")

    worker = subparsers.add_parser(
        "worker",
        help="Run the LiveKit worker. You can also call dev/start/console directly.",
    )
    worker.add_argument(
        "worker_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to LiveKit, e.g. dev or start.",
    )

    call = subparsers.add_parser(
        "call",
        help="Create a LiveKit room, dispatch the agent, and dial a phone number.",
    )
    call.add_argument(
        "--to",
        default=None,
        help=(
            "Override the cleaner's phone in the request fixture (E.164). "
            "Falls back to $RENTAL_VOICE_TEST_PHONE_NUMBER. If neither is set, the "
            "phone in the request fixture is dialed as-is (production behavior)."
        ),
    )
    call.add_argument(
        "--room",
        default=None,
        help="Room name to use. Defaults to a timestamped rental-manual-* room.",
    )
    call.add_argument(
        "--agent-name",
        default=DEFAULT_AGENT_NAME,
        help=f"LiveKit agent name to dispatch. Default: {DEFAULT_AGENT_NAME}.",
    )
    call.add_argument(
        "--request-fixture",
        default=str(DEFAULT_REQUEST_FIXTURE),
        help="CleaningRequest fixture for the worker to load.",
    )
    call.add_argument(
        "--snapshot-dir",
        default=str(DEFAULT_SNAPSHOT_DIR),
        help="Directory where the worker writes final CallState snapshots.",
    )
    call.add_argument(
        "--trunk-id",
        default=None,
        help="LiveKit outbound SIP trunk id. Defaults to LIVEKIT_SIP_OUTBOUND_TRUNK_ID.",
    )
    answer_group = call.add_mutually_exclusive_group()
    answer_group.add_argument(
        "--wait-until-answered",
        dest="wait_until_answered",
        action="store_true",
        default=True,
        help=(
            "Block until LiveKit reports the SIP call has been answered before "
            "dispatching the agent. Default for manual calls."
        ),
    )
    answer_group.add_argument(
        "--no-wait-until-answered",
        dest="wait_until_answered",
        action="store_false",
        help=(
            "Dispatch after SIP participant creation even if the phone may still "
            "be ringing. Useful only for low-level telephony debugging."
        ),
    )
    return parser


def _run_worker_cli(worker_args: list[str]) -> None:
    from rental_voice_agent.agent import main as agent_main

    forwarded = worker_args or ["dev"]
    sys.argv = [sys.argv[0], *forwarded]
    agent_main()


async def _place_call(args: argparse.Namespace) -> None:
    settings = load_settings()
    trunk_id = args.trunk_id or settings.livekit_sip_outbound_trunk_id
    room_name = args.room or _default_room_name()

    _require("LIVEKIT_URL", settings.livekit_url)
    _require("LIVEKIT_API_KEY", settings.livekit_api_key)
    _require("LIVEKIT_API_SECRET", settings.livekit_api_secret)
    _require("LIVEKIT_SIP_OUTBOUND_TRUNK_ID", trunk_id)

    # Single source of truth for the dial target: request.cleaner.phone.
    # --to / RENTAL_VOICE_TEST_PHONE_NUMBER are explicit dev overrides — they patch
    # the request before dialing, so the snapshot's state.cleaner.phone
    # matches the number that was actually dialed.
    request_path = Path(_absolute_path(args.request_fixture))
    request = CleaningRequest.from_json_file(request_path)
    override = args.to or _env("RENTAL_VOICE_TEST_PHONE_NUMBER")
    phone_number = override or request.cleaner.phone
    _require(
        "request.cleaner.phone or --to or RENTAL_VOICE_TEST_PHONE_NUMBER",
        phone_number,
    )
    if override and override != request.cleaner.phone:
        print(
            f"[call] dev override: dialing {override} instead of "
            f"request.cleaner.phone={request.cleaner.phone}"
        )

    metadata_payload: dict[str, str] = {
        "request_fixture": str(request_path),
        "snapshot_dir": _absolute_path(args.snapshot_dir),
    }
    if override:
        metadata_payload["cleaner_phone_override"] = override
    metadata = json.dumps(metadata_payload, sort_keys=True)

    lkapi = api.LiveKitAPI(
        settings.livekit_url,
        settings.livekit_api_key,
        settings.livekit_api_secret,
    )
    try:
        existing_rooms = await lkapi.room.list_rooms(
            api.ListRoomsRequest(names=[room_name])
        )
        if existing_rooms.rooms:
            print(f"[call] using existing room: {room_name}")
        else:
            print(f"[call] creating room: {room_name}")
            await lkapi.room.create_room(
                api.CreateRoomRequest(
                    name=room_name,
                    empty_timeout=300,
                    max_participants=2,
                )
            )

        print(f"[call] dialing {phone_number} via outbound trunk {trunk_id}")
        sip_info = await lkapi.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=room_name,
                sip_trunk_id=trunk_id,
                sip_call_to=phone_number,
                participant_identity="manual-cleaner-phone",
                participant_name="Manual Cleaner Phone",
                wait_until_answered=args.wait_until_answered,
            )
        )
        print(f"[call] SIP participant: {sip_info.participant_identity}")
        print(f"[call] SIP call id: {sip_info.sip_call_id}")
        print(f"[call] dispatching agent {args.agent_name!r}")
        dispatch = await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=args.agent_name,
                room=room_name,
                metadata=metadata,
            )
        )
        print(f"[call] dispatch id: {dispatch.id}")
        print("[call] Watch the worker terminal for live transcript/tool output.")
    finally:
        await lkapi.aclose()


def _default_room_name() -> str:
    return "rental-manual-" + datetime.now().strftime("%Y%m%d-%H%M%S")


def _absolute_path(raw_path: str) -> str:
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    return str(path)


def _env(name: str) -> str | None:
    import os

    return os.getenv(name)


def _require(name: str, value: object | None) -> None:
    if value in (None, ""):
        raise SystemExit(f"Missing required value: {name}")


if __name__ == "__main__":
    main()
