from __future__ import annotations

import asyncio
import json
import os

import websockets
from dotenv import load_dotenv


async def main() -> None:
    load_dotenv(override=True)
    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")

    url = f"wss://api.openai.com/v1/realtime?model={model}"
    headers = {"Authorization": f"Bearer {api_key}"}
    async with websockets.connect(url, additional_headers=headers) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "output_modalities": ["text"],
                        "instructions": "Reply with exactly: realtime-ok",
                    },
                }
            )
        )
        await ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Say realtime-ok.",
                            }
                        ],
                    },
                }
            )
        )
        await ws.send(
            json.dumps(
                {
                    "type": "response.create",
                    "response": {"output_modalities": ["text"]},
                }
            )
        )

        output: list[str] = []
        while True:
            event = json.loads(await asyncio.wait_for(ws.recv(), timeout=20))
            event_type = event.get("type")
            if event_type in {"response.output_text.delta", "response.text.delta"}:
                output.append(event.get("delta", ""))
            elif event_type == "response.done":
                break
            elif event_type == "error":
                raise SystemExit(f"Realtime error: {event}")

        text = "".join(output).strip()
        print(f"OpenAI Realtime smoke response: {text}")


if __name__ == "__main__":
    asyncio.run(main())
