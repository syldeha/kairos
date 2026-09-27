"""Live source: no script, only people typing in the console and Kairos.

Emits voice activity every 80 ms until stopped. Nobody is speaking unless a
participant is typing (the runtime marks that); the probability that the
silence will last grows with the silence, like a detector hearing a turn end.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from ..clock import Clock
from ..contracts import HORIZONS, SpeechEvent, VadStep


class LiveSource:
    def __init__(self, clock: Clock, step_s: float = 0.08) -> None:
        self._clock = clock
        self._step_s = step_s
        self._stopped = asyncio.Event()
        self.offset = 0.0
        self.speaking_since: float | None = None
        self.silent_since = 0.0

    def yield_floor(self, seconds: float) -> None:
        """Nothing to push back: live participants simply wait for Kairos."""

    def stop(self) -> None:
        self._stopped.set()

    def set_speaking(self, speaking: bool, t: float) -> None:
        if speaking:
            self.speaking_since = t
        else:
            self.speaking_since = None
            self.silent_since = t

    async def events(self) -> AsyncIterator[SpeechEvent]:
        start = self._clock.now()
        while not self._stopped.is_set():
            t = round(self._clock.now() - start, 3)
            silence = 0.0 if self.speaking_since is not None else max(0.0, t - self.silent_since)
            # Right after someone stops, a turn end is likely but not certain; it firms up quickly.
            p = 0.1 if self.speaking_since is not None else min(0.95, 0.55 + silence)
            yield VadStep(t=t, speaking=False, p_silence=(p,) * len(HORIZONS))
            await self._clock.sleep(self._step_s)
