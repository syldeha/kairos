"""Replay a recorded microphone through Gradium's speech-to-text at several look-ahead delays.

    python scripts/stt_replay.py                      # latest recording in runs/voice/
    python scripts/stt_replay.py runs/voice/x.wav 10 24 48

The console records the microphone of every voice meeting (24 kHz, 16-bit mono WAV).
Each delay is a number of 80 ms frames Gradium listens ahead before writing a word:
more is more accurate, and later. Prints the transcript for each, one line per turn.
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import time
import wave
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kairos.config import Settings  # noqa: E402

CHUNK = 3840  # 80 ms


async def transcribe(settings: Settings, pcm: bytes, delay: int, language: str = "fr") -> tuple[str, float]:
    url = settings.gradium_url.rstrip("/") + "/asr"
    lines, words, ended, quiet = [], [], False, 0
    t0 = time.perf_counter()
    async with websockets.connect(url, additional_headers={"x-api-key": settings.gradium_api_key}) as ws:
        await ws.send(json.dumps({"type": "setup", "model_name": "default", "input_format": "pcm",
                                  "json_config": {"language": language, "delay_in_frames": delay}}))
        await ws.recv()

        async def send() -> None:
            # a little silence first (the first word is not lost), and after, so the last words come out
            audio = b"\x00" * CHUNK * 6 + pcm + b"\x00" * CHUNK * (delay + 10)
            for i in range(0, len(audio), CHUNK):
                await ws.send(json.dumps({"type": "audio", "audio": base64.b64encode(audio[i:i + CHUNK]).decode()}))
                if i // CHUNK % 25 == 0:
                    await asyncio.sleep(0.02)  # faster than real time, without flooding
            await ws.send(json.dumps({"type": "end_of_stream"}))

        sender = asyncio.create_task(send())
        async for raw in ws:
            m = json.loads(raw)
            if m["type"] == "text":
                words.append(m["text"])
                ended = False
            elif m["type"] == "step":
                quiet = quiet + 1 if m["vad"][-1]["inactivity_prob"] > 0.5 else 0
                if words and not ended and quiet >= 3:  # the console's rule for the end of a turn
                    lines.append(" ".join(words))
                    words, ended = [], True
            elif m["type"] in ("end_of_stream", "error"):
                if m["type"] == "error":
                    lines.append(f"[error: {m.get('message')}]")
                break
        sender.cancel()
    if words:
        lines.append(" ".join(words))
    return "\n".join(lines), time.perf_counter() - t0


async def main() -> None:
    args = sys.argv[1:]
    path = Path(args.pop(0)) if args and args[0].endswith(".wav") else \
        max((ROOT / "runs" / "voice").glob("*.wav"), key=lambda p: p.stat().st_mtime)
    delays = [int(a) for a in args] or [10, 24, 48]
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == 24_000 and w.getsampwidth() == 2 and w.getnchannels() == 1, "24 kHz 16-bit mono"
        pcm = w.readframes(w.getnframes())
    print(f"{path.name}: {len(pcm) / 48000:.1f} s of audio\n")
    settings = Settings()
    for delay in delays:
        text, seconds = await transcribe(settings, pcm, delay)
        print(f"== delay {delay} frames ({delay * 0.08:.1f} s)  [{seconds:.1f} s to transcribe]")
        print(text or "(nothing)")
        print()


if __name__ == "__main__":
    asyncio.run(main())
