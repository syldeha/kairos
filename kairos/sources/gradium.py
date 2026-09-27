"""Live audio source: a microphone streamed to Gradium's speech-to-text.

Gradium sends words as they are recognised (`text`, with the time the word
started in the audio) and, every 80 ms, the probability that the speaker has
finished at four horizons (`step`: 0.5, 1, 2 and 3 s, the same as HORIZONS).
This source turns them into the events every other source produces:

- each word extends the current segment: a SpeechPartial;
- a segment ends when the voice activity says so (Gradium's turn-taking recipe:
  the longest horizon above 0.5 for three steps, or a sentence end and the
  shortest horizon above 0.5 for three steps). The source then asks Gradium to
  flush the words still in its pipeline and commits the segment: a SpeechFinal;
- every step becomes a VadStep.

One microphone gives one voice: the speaker is `speaker` (None when several
people share the microphone). Kairos's own voice may come back through the
microphone when there are no headphones: words matching what Kairos is saying,
while it says it, are dropped (the echo guard).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import re
import time
from typing import AsyncIterator, Callable

from ..clock import Clock
from ..contracts import HORIZONS, SpeechEvent, SpeechFinal, SpeechPartial, VadStep

log = logging.getLogger("kairos.gradium")

SENTENCE_END = (".", "?", "!")
END_STEPS = 3          # consecutive steps above the threshold that end a segment
END_P = 0.5
FLUSH_TIMEOUT_S = 1.2  # commit the segment even if the flush confirmation is late
QUIET_TICK_S = 0.08    # without audio (microphone muted), time still passes
ROTATE_S = 280.0       # Gradium closes a session at 300 s: renew it before


def words_of(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.lower())


class GradiumSource:
    def __init__(self, api_key: str, url: str, language: str = "fr", speaker: str | None = "Vous",
                 delay_in_frames: int = 10, first_segment: int = 1,
                 is_echo: Callable[[str, float], bool] | None = None,
                 keywords: tuple[str, ...] = ("Kairos",), keyword_boost: float = 3.0) -> None:
        self.api_key = api_key
        #: words the recognizer should favour (Gradium's custom dictionary): "Kairos", not "Kéros"
        self.keywords = tuple(dict.fromkeys(k for k in keywords if k.strip()))[:500]
        self.keyword_boost = keyword_boost
        self.url = url.rstrip("/") + "/asr"
        self.language = language
        self.speaker = speaker
        self.delay_in_frames = delay_in_frames
        self.is_echo = is_echo or (lambda text, t: False)
        self.offset = 0.0
        self.error: str | None = None
        self.connected = False
        self._clock: Clock | None = None
        self._now: Callable[[], float] = lambda: 0.0
        self._audio: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=200)
        self._events: asyncio.Queue[SpeechEvent | None] = asyncio.Queue()
        self._stopped = asyncio.Event()
        self._segment_ids = iter(range(first_segment, 10**9))
        self._origin: float | None = None  # meeting time of audio second 0
        # the segment being spoken
        self._segment: int | None = None
        self._words: list[str] = []
        self._t_start = 0.0
        self._t_end = 0.0
        self._above = 0
        self._flush_id = 0
        self._flushing: tuple[int, float] | None = None  # (flush id, meeting time asked)
        self._last_human_word = float("-inf")
        self._ws = None
        self._carry: bytes | None = None  # a chunk held over a session renewal
        self.rotations = 0
        #: where the time goes: word heard -> word received, and last word -> line committed (seconds)
        self.stt_lag_s: list[float] = []
        self.commit_s: list[float] = []

    # -- wiring ------------------------------------------------------------------------

    def attach(self, clock: Clock, now: Callable[[], float]) -> None:
        self._clock, self._now = clock, now

    def push_audio(self, pcm: bytes) -> None:
        """24 kHz, 16-bit mono PCM from the microphone (any length; 80 ms chunks are ideal)."""
        if self._origin is None:
            self._origin = self._now()
        with contextlib.suppress(asyncio.QueueFull):  # a stalled connection drops audio, not the meeting
            self._audio.put_nowait(pcm)

    def yield_floor(self, seconds: float) -> None:
        """Live participants simply wait for Kairos."""

    def set_speaking(self, speaking: bool, t: float) -> None:
        """Typing in the console still works in a voice meeting; the scribe tracks it."""

    def stop(self) -> None:
        self._stopped.set()
        self._audio.put_nowait(None)

    # -- the protocol, independent of the connection (tested directly) -----------------

    def handle(self, message: dict) -> list[SpeechEvent]:
        """One Gradium message in, the resulting speech events out. May ask for a flush (self.flush_wanted)."""
        now = self._now()
        kind = message.get("type")
        if kind == "text":
            return self._on_word(str(message.get("text", "")), float(message.get("start_s", 0.0)), now)
        if kind == "end_text":
            if self._segment is not None:
                self._t_end = self._audio_time(float(message.get("stop_s", 0.0)))
                self.stt_lag_s = (self.stt_lag_s + [round(now - self._t_end, 2)])[-50:]
            return []
        if kind == "step":
            return self._on_step(message, now)
        if kind == "flushed":
            if self._flushing and message.get("flush_id") == self._flushing[0]:
                return self._commit(now)
            return []
        if kind == "error":
            self.error = str(message.get("message", "error"))
            log.error("gradium: %s", self.error)
        return []

    def flush_due(self) -> int | None:
        """The flush id to send, once, when a segment is ending."""
        if self._flushing and self._flushing[1] == -1.0:
            self._flushing = (self._flushing[0], self._now())
            return self._flushing[0]
        return None

    def overdue(self) -> list[SpeechEvent]:
        """The flush confirmation is late: commit what was heard."""
        if self._flushing and self._flushing[1] >= 0 and self._now() - self._flushing[1] > FLUSH_TIMEOUT_S:
            return self._commit(self._now())
        return []

    def _audio_time(self, seconds: float) -> float:
        return (self._origin or 0.0) + seconds

    def _on_word(self, text: str, start_s: float, now: float) -> list[SpeechEvent]:
        text = text.strip()
        if not text:
            return []
        if self.is_echo(text, now):
            return []  # Kairos hearing itself
        self._last_human_word = now
        if self._segment is None:
            self._segment = next(self._segment_ids)
            self._words, self._t_start = [], min(self._audio_time(start_s), now)
        self._words.append(text)
        self._above = 0
        return [SpeechPartial(t=now, segment=self._segment, speaker=self.speaker, text=" ".join(self._words),
                              t_start=self._t_start)]

    def _on_step(self, message: dict, now: float) -> list[SpeechEvent]:
        vad = message.get("vad") or []
        by_horizon = {round(float(v["horizon_s"]), 2): float(v["inactivity_prob"]) for v in vad}
        p = tuple(by_horizon.get(h, by_horizon.get(float(h), 0.0)) for h in HORIZONS)
        # While Kairos talks through the loudspeakers, the detector hears Kairos: only a human word counts.
        echo_window = self.is_echo("", now)
        speaking = p[0] < END_P and (not echo_window or now - self._last_human_word < 1.0)
        events: list[SpeechEvent] = [VadStep(t=now, speaking=speaking, p_silence=p)]
        if self._segment is not None and self._flushing is None:
            sentence_done = " ".join(self._words).rstrip().endswith(SENTENCE_END)
            ending = p[-1] > END_P or (sentence_done and p[0] > END_P)
            self._above = self._above + 1 if ending else 0
            if self._above >= END_STEPS:
                self._flush_id += 1
                self._flushing = (self._flush_id, -1.0)  # to send
        return events

    def _commit(self, now: float) -> list[SpeechEvent]:
        self._flushing = None
        if self._segment is None or not self._words:
            self._segment = None
            return []
        event = SpeechFinal(t=now, segment=self._segment, speaker=self.speaker, text=" ".join(self._words),
                            t_start=self._t_start, t_end=max(self._t_end, self._t_start))
        self.commit_s = (self.commit_s + [round(now - event.t_end, 2)])[-50:]
        self._segment, self._words, self._above = None, [], 0
        return [event]

    # -- the connection ------------------------------------------------------------------

    async def events(self) -> AsyncIterator[SpeechEvent]:
        connection = asyncio.create_task(self._connect())
        try:
            while not self._stopped.is_set():
                try:
                    event = await asyncio.wait_for(self._events.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    # No audio (microphone not started or muted): time passes, the room is quiet.
                    yield VadStep(t=self._now(), speaking=False, p_silence=(1.0,) * len(HORIZONS))
                    continue
                if event is None:
                    break
                yield event
        finally:
            connection.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await connection
            for event in self._commit(self._now()):
                yield event

    async def _connect(self) -> None:
        """Keep a connection open for the whole meeting. Gradium ends a session after 300 s: the source
        renews it before (ROTATE_S), commits the line in progress, and carries on; after an error it retries."""
        while not self._stopped.is_set():
            try:
                await self._session()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.error = f"Gradium : {type(exc).__name__}: {exc}"
                log.exception("gradium connection failed; retrying")
                await asyncio.sleep(1.0)

    async def _session(self) -> None:
        import websockets

        async with websockets.connect(self.url, additional_headers={"x-api-key": self.api_key},
                                      max_size=2**22) as ws:
            config = {"language": self.language, "delay_in_frames": self.delay_in_frames}
            if self.keywords:
                config["keywords"] = {"words": list(self.keywords), "boost": self.keyword_boost}
            await ws.send(json.dumps({"type": "setup", "model_name": "default", "input_format": "pcm",
                                      "json_config": config}))
            ready = json.loads(await ws.recv())
            if ready.get("type") != "ready":
                raise RuntimeError(f"Gradium did not accept the setup: {ready}")
            self.connected, self.error = True, None
            self._origin = None  # this connection's audio clock starts with its first chunk
            sender = asyncio.create_task(self._send(ws, time.monotonic()))
            try:
                async for raw in ws:
                    message = json.loads(raw)
                    if message.get("type") == "end_of_stream":
                        break
                    for event in self.handle(message):
                        self._events.put_nowait(event)
                    flush = self.flush_due()
                    if flush is not None:
                        await ws.send(json.dumps({"type": "flush", "flush_id": flush}))
                    for event in self.overdue():
                        self._events.put_nowait(event)
            finally:
                sender.cancel()
                self.connected = False
            # The session ended (renewal or stop): what was heard so far becomes a committed line.
            for event in self._commit(self._now()):
                self._events.put_nowait(event)
            self.rotations += 1

    async def _send(self, ws, opened: float) -> None:
        while True:
            pcm, self._carry = (self._carry, None) if self._carry is not None else (await self._audio.get(), None)
            if pcm is None:
                await ws.send(json.dumps({"type": "end_of_stream"}))
                return
            if time.monotonic() - opened > ROTATE_S:
                self._carry = pcm  # sent first on the next connection
                await ws.send(json.dumps({"type": "end_of_stream"}))
                return
            if self._origin is None:
                self._origin = self._now()
            await ws.send(json.dumps({"type": "audio", "audio": base64.b64encode(pcm).decode()}))
