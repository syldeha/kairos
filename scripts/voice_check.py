"""End-to-end check of the voice demo, without a browser.

A participant's lines are synthesised with Gradium (another voice) and streamed
to the console's /ws/audio in real time, as a microphone would; silence is
streamed in between, like an open microphone. Kairos's voice coming back is
counted. At the end, the transcript and the interventions are printed.

    python scripts/voice_check.py        (the console must be running)
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import time
from pathlib import Path

import httpx
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kairos.config import Settings  # noqa: E402

BASE = "http://127.0.0.1:8765"
LINES = [  # (seconds of silence before, what the participant says)
    (2.0, "Bonjour Kairos. Je prépare un voyage à Lisbonne en février avec un petit budget."),
    (6.0, "Tu sais combien de temps dure le vol depuis Paris ?"),
    (12.0, "D'accord. Et il fait quel temps là-bas en février ?"),
    (14.0, "Merci, c'est noté."),
]
PARTICIPANT_VOICE = "6oIkS98REoVZ1dEw"  # Apolline; Kairos speaks with Gaspard
CHUNK = 3840  # 80 ms at 24 kHz, 16-bit


async def synthesise(settings: Settings, text: str) -> bytes:
    url = settings.gradium_url.rstrip("/") + "/tts"
    async with websockets.connect(url, additional_headers={"x-api-key": settings.gradium_api_key}) as ws:
        await ws.send(json.dumps({"type": "setup", "model_name": "default", "voice_id": PARTICIPANT_VOICE,
                                  "output_format": "pcm_24000"}))
        await ws.recv()
        for word in text.split():
            await ws.send(json.dumps({"type": "text", "text": word}))
        await ws.send(json.dumps({"type": "end_of_stream"}))
        audio = b""
        async for raw in ws:
            m = json.loads(raw)
            if m["type"] == "audio":
                audio += base64.b64decode(m["audio"])
            elif m["type"] in ("end_of_stream", "error"):
                break
        return audio


async def main() -> None:
    settings = Settings()
    clips = [await synthesise(settings, text) for _, text in LINES]
    print("participant lines synthesised:", [f"{len(c) / 48000:.1f}s" for c in clips])
    async with httpx.AsyncClient(timeout=30) as http:
        r = await http.post(f"{BASE}/api/start", json={"source": "live", "voice": True, "memory": "",
                                                       "context": "Je prépare un voyage."})
        r.raise_for_status()
        await asyncio.sleep(1.0)
        received = {"bytes": 0, "stops": 0, "chunks": 0}
        async with websockets.connect("ws://127.0.0.1:8765/ws/audio", max_size=2**23) as ws:
            async def listen() -> None:
                async for message in ws:
                    if isinstance(message, bytes):
                        received["bytes"] += len(message)
                        received["chunks"] += 1
                    elif json.loads(message).get("type") == "stop":
                        received["stops"] += 1

            listener = asyncio.create_task(listen())
            t0 = time.monotonic()
            silence = b"\x00" * CHUNK

            sent = 0  # chunks sent: paced on the clock like a microphone (sleep(0.08) drifts on Windows)

            async def stream(pcm: bytes) -> None:
                nonlocal sent
                for i in range(0, len(pcm), CHUNK):
                    await ws.send(pcm[i:i + CHUNK].ljust(CHUNK, b"\x00"))
                    sent += 1
                    await asyncio.sleep(max(0.0, t0 + sent * 0.08 - time.monotonic()))

            for (gap, text), clip in zip(LINES, clips):
                await stream(silence * int(gap / 0.08))
                print(f"{time.monotonic() - t0:6.1f}s participant: {text}")
                await stream(clip)
            await stream(silence * int(12 / 0.08))
            listener.cancel()
        state = (await http.get(f"{BASE}/api/state")).json()
        await http.post(f"{BASE}/api/stop")
    print("\n-- transcript")
    for line in state["transcript"]:
        print(f"  {line['t']:6.1f} {line['speaker']}: {line['text']}")
    print("\n-- interventions")
    for iv in state["interventions"]:
        print(f"  {iv['t']:6.1f} [{iv['reason']}] after speech {iv.get('after_speech_s')} s · "
              f"answer ready {iv.get('question_to_answer_s')} s after the question · {iv['text'][:80]}")
    print("\n-- decisions")
    for d in state["decisions"]:
        print(f"  {d['t']:6.1f} {'SPEAK' if d['speak'] else 'wait '} {d['why']}")
    print("\n-- voice", state.get("voice"))
    print(f"-- Kairos audio received: {received['bytes'] / 48000:.1f} s in {received['chunks']} chunks, "
          f"{received['stops']} stop(s)")
    print("-- metrics", state["metrics"])
    print("-- timings", {k: v["mean_s"] for k, v in state["timings"].items()})


if __name__ == "__main__":
    asyncio.run(main())
