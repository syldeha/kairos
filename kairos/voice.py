"""Kairos's voice: Gradium text-to-speech, streamed to the console's loudspeakers.

The speaker decides what to say and when to stop; this module only turns text
into audio and sends it to whoever listens (the console page, through a sink).
A connection is opened ahead of time, so the next utterance does not pay for
the handshake.

The echo guard remembers what Kairos is saying and until when its audio plays:
the speech-to-text source drops words that are only Kairos heard back through
the microphone.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
from typing import Callable, Protocol

from .sources.gradium import words_of
from .speakable import speakable

log = logging.getLogger("kairos.voice")

SAMPLE_RATE = 24_000
BYTES_PER_S = SAMPLE_RATE * 2
#: What Kairos says the instant it is asked something and its answer is not ready: it takes the turn, like a
#: person, instead of leaving three seconds of silence. Neutral openers: the answer follows without repeating them.
CUES = {"French": ["Alors…", "Hmm, voyons…", "Oui…", "Attends…"],
        "English": ["So…", "Hmm, let's see…", "Right…", "One sec…"]}
VOICES = {  # Gradium flagship French voices
    "iEu63s1rhn_kegTr": "Gaspard",
    "6oIkS98REoVZ1dEw": "Apolline",
    "YKeBw3OV1RgpdhLh": "Jules",
    "YhIHaAfQ0cQPDV9R": "Solène",
}


class AudioSink(Protocol):
    def send_audio(self, pcm: bytes) -> None: ...
    def stop(self) -> None: ...


class EchoGuard:
    """What Kairos is saying, and until when it may come back through the microphone."""

    def __init__(self, now: Callable[[], float], tail_s: float = 1.8, match: float = 0.6) -> None:
        self._now = now
        self.tail_s = tail_s  # speech-to-text delay plus the room's tail
        self.match = match
        self._words: set[str] = set()
        self._until = float("-inf")

    def speaking(self, text: str, audio_end: float) -> None:
        self._words |= set(words_of(text))
        self._until = max(self._until, audio_end + self.tail_s)

    def stopped(self) -> None:
        self._until = min(self._until, self._now() + self.tail_s)

    def active(self, now: float) -> bool:
        return now < self._until

    def __call__(self, text: str, now: float) -> bool:
        """With text: is it only Kairos heard back? With "": is Kairos's voice still in the room?"""
        if not self.active(now):
            self._words.clear()
            return False
        words = words_of(text)
        if not text:
            return True
        if len(words) == 1 and len(words[0]) < 4:
            return False  # "tu", "et", "oui": too common to prove it is Kairos heard back
        return bool(words) and sum(w in self._words for w in words) / len(words) >= self.match


class GradiumVoice:
    def __init__(self, api_key: str, url: str, voice_id: str, now: Callable[[], float],
                 echo: EchoGuard | None = None, language: str = "French") -> None:
        self.language = language
        self.api_key = api_key
        self.url = url.rstrip("/") + "/tts"
        self.voice_id = voice_id
        self.now = now
        self.echo = echo
        self.sink: AudioSink | None = None
        self.error: str | None = None
        self.first_audio_s: list[float] = []  # time from text to first audio, per utterance
        self._task: asyncio.Task | None = None
        self._warm: asyncio.Task | None = None
        self._audio_end = 0.0

    async def _open(self):
        import websockets

        ws = await websockets.connect(self.url, additional_headers={"x-api-key": self.api_key}, max_size=2**23)
        await ws.send(json.dumps({"type": "setup", "model_name": "default", "voice_id": self.voice_id,
                                  "output_format": "pcm_24000",
                                  # Meeting Sidecar's settings: Gradium's own French text normalisation, and a
                                  # livelier pace (padding_bonus below 0 shortens pauses, ~20 % faster).
                                  "json_config": {"rewrite_rules": "fr" if self.language == "French" else "en",
                                                  "temp": 0.9, "cfg_coef": 2.2, "padding_bonus": -0.5}}))
        ready = json.loads(await ws.recv())
        if ready.get("type") != "ready":
            await ws.close()
            raise RuntimeError(f"Gradium TTS refused the setup: {ready}")
        return ws

    def warm_up(self) -> None:
        """Open the next connection now, so speaking starts sooner."""
        if self._warm is None or self._warm.done():
            self._warm = asyncio.create_task(self._open())

    async def _connection(self):
        warm, self._warm = self._warm, None
        if warm is not None:
            try:
                ws = await warm
                if ws.state.name == "OPEN":
                    return ws
            except Exception:
                pass
        return await self._open()

    async def say(self, text: str, timeout_s: float = 3.0) -> bool:
        """Start saying `text`; returns once the audio has started (True) or could not (False)."""
        self.stop()
        text = speakable(text, self.language)  # no "slash", no "étoile": symbols are not read out
        started = asyncio.Event()
        self._task = asyncio.create_task(self._stream(text, started))
        try:
            await asyncio.wait_for(started.wait(), timeout_s)
            return True
        except asyncio.TimeoutError:
            return False

    async def _stream(self, text: str, started: asyncio.Event) -> None:
        asked = self.now()
        ws = None
        try:
            ws = await self._connection()
            for word in text.split():
                await ws.send(json.dumps({"type": "text", "text": word}))
            await ws.send(json.dumps({"type": "end_of_stream"}))
            async for raw in ws:
                message = json.loads(raw)
                if message.get("type") == "audio":
                    pcm = base64.b64decode(message["audio"])
                    if not started.is_set():
                        self.first_audio_s.append(round(self.now() - asked, 2))
                        started.set()
                    # Audio arrives faster than it plays: it queues up in the page.
                    self._audio_end = max(self._audio_end, self.now()) + len(pcm) / BYTES_PER_S
                    if self.echo is not None:
                        self.echo.speaking(text, self._audio_end)
                    if self.sink is not None:
                        self.sink.send_audio(pcm)
                elif message.get("type") == "error":
                    self.error = str(message.get("message"))
                    log.error("gradium tts: %s", self.error)
                    break
                elif message.get("type") == "end_of_stream":
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = f"Gradium TTS : {type(exc).__name__}: {exc}"
            log.exception("tts failed")
        finally:
            started.set()
            if ws is not None:
                with contextlib.suppress(Exception):
                    await ws.close()
            self.warm_up()

    async def prepare_cues(self) -> None:
        """Synthesize the turn-taking openers once, before anyone asks: said the instant they are needed."""
        if getattr(self, "_cues", None) is not None:
            return
        self._cues: list[bytes] = []
        self._cue_texts: list[str] = []
        self._cue_turn = 0
        for text in CUES.get(self.language, CUES["English"]):
            try:
                pcm = await self._synthesize(text)
            except Exception:
                continue
            pcm = _trim_silence(pcm)
            if pcm:
                self._cues.append(pcm)
                self._cue_texts.append(text)

    async def _synthesize(self, text: str) -> bytes:
        ws = await self._open()
        await ws.send(json.dumps({"type": "text", "text": text}))
        await ws.send(json.dumps({"type": "end_of_stream"}))
        pcm = b""
        async for raw in ws:
            message = json.loads(raw)
            if message.get("type") == "audio":
                pcm += base64.b64decode(message["audio"])
            elif message.get("type") in ("end_of_stream", "error"):
                break
        await ws.close()
        return pcm

    async def cue(self) -> str:
        """Take the turn at once while the answer is being prepared ("Alors…", "Hmm, voyons…"): a person
        answering a question starts with a sound within half a second, not after three seconds of silence."""
        if self.sink is None or (self._task is not None and not self._task.done()):
            return ""
        await self.prepare_cues()
        if not self._cues:
            return ""
        turn = self._cue_turn % len(self._cues)  # a different one each time
        self._cue_turn += 1
        self.saying = self._cue_texts[turn]  # what the audio about to start says (for subtitles)
        self.sink.send_audio(self._cues[turn])
        self._audio_end = max(self._audio_end, self.now()) + len(self._cues[turn]) / BYTES_PER_S
        return self._cue_texts[turn]

    def wait_remaining(self) -> float:
        """Seconds of audio still to play (0 when silent), or while the audio is still being produced."""
        if self._task is not None and not self._task.done():
            return max(0.08, self._audio_end - self.now())
        return max(0.0, self._audio_end - self.now())

    def stop(self) -> None:
        """Stop at once: no more audio, and the page drops what it has queued."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None
        if self._audio_end > self.now():  # audio still queued in the page, even if all of it was sent
            if self.sink is not None:
                self.sink.stop()
            if self.echo is not None:
                self.echo.stopped()
        self._audio_end = min(self._audio_end, self.now())

    def close(self) -> None:
        self.stop()
        if self._warm is not None:
            warm, self._warm = self._warm, None

            async def _close() -> None:
                with contextlib.suppress(Exception):
                    ws = await warm
                    await ws.close()
            asyncio.create_task(_close())


def _trim_silence(pcm: bytes, threshold: int = 500, keep_s: float = 0.06) -> bytes:
    """Cut the silence the synthesizer pads around a short sound: an opener must not delay the answer."""
    import array
    samples = array.array("h", pcm[: len(pcm) // 2 * 2])
    loud = [i for i, x in enumerate(samples) if abs(x) > threshold]
    if not loud:
        return b""
    keep = int(SAMPLE_RATE * keep_s)
    start, end = max(0, loud[0] - keep), min(len(samples), loud[-1] + keep)
    return samples[start:end].tobytes()

