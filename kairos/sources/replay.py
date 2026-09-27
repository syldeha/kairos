"""Replay a scripted meeting word by word, as a streaming speech-to-text would deliver it.

Script format, one line per utterance (heard-meeting's fixture format, with
signed gaps):

    +0.8  Léa: Quinze euros, c'est pas un peu cher ?
    +-0.3 Camille: Oui mais...        (negative gap: starts before the previous line ends)

`+N` is the silence, in seconds, between the end of the previous line and the
start of this one.

`build_timeline` is pure: it turns a script into timed events (partials,
finals, voice activity every 80 ms) plus the ground truth of who spoke when.
`ReplaySource` then paces those events on a clock, and can yield the floor to
the AI (closed loop): the rest of the meeting is pushed back while it speaks.

The turn-end signal is SIMULATED. The probability of silence at each horizon is
derived from the true timeline, blurred with noise, and fooled by mid-sentence
pauses the way a real model partly is. The text simulation tests the decision
logic given such a signal; it does not measure the quality of a real detector
(that is v2: Gradium's semantic VAD and/or Smart Turn).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, replace
from typing import AsyncIterator, Iterable, Sequence

from ..clock import Clock
from ..contracts import HORIZONS, SpeechEvent, SpeechFinal, SpeechPartial, VadStep

LINE = re.compile(r"^\+(-?\d+(?:[.,]\d+)?)\s+([^:]+?):\s*(.+)$")
PAUSE_AFTER = (",", ";", ":")


@dataclass(frozen=True, slots=True)
class ScriptLine:
    gap: float
    speaker: str
    text: str


@dataclass(frozen=True, slots=True)
class ReplayParams:
    words_per_s: float = 2.75        # conversational speech rate
    comma_pause_s: float = 0.25      # breath after a comma: silence that does not end the turn
    final_delay_s: float = 0.5       # the recogniser commits a segment after this much silence
    vad_step_s: float = 0.08
    tail_s: float = 3.0              # voice activity keeps being reported after the last word
    anonymous: bool = False          # one room microphone: speakers are not identified
    noise: float = 0.08              # standard deviation of the noise on silence probabilities
    pause_false_alarm: float = 0.35  # how much a mid-sentence pause fools the detector
    seed: int = 0


@dataclass(frozen=True, slots=True)
class Gap:
    """A silence between two stretches of speech: the moments the AI could take the floor."""

    start: float
    end: float
    before: str
    after: str | None  # None after the last word of the meeting

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def shift(self) -> bool:
        """True when someone else spoke after the silence (a turn change), False for a pause."""
        return self.after is not None and self.after != self.before


@dataclass(frozen=True, slots=True)
class Timeline:
    events: tuple[SpeechEvent, ...]
    #: Ground truth: (start, end, speaker) for every stretch of actual sound.
    speech: tuple[tuple[float, float, str], ...]

    def gaps(self, min_s: float = 0.2) -> list[Gap]:
        return find_gaps(self.speech, min_s)


def parse_script(text: str) -> list[ScriptLine]:
    lines = []
    for number, raw in enumerate(text.splitlines(), start=1):
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        m = LINE.match(raw)
        if not m:
            raise ValueError(f"line {number}: expected '+N  Name: text', got {raw!r}")
        lines.append(ScriptLine(float(m.group(1).replace(",", ".")), m.group(2).strip(), m.group(3).strip()))
    return lines


def build_timeline(lines: Sequence[ScriptLine], params: ReplayParams = ReplayParams()) -> Timeline:
    word_s = 1.0 / params.words_per_s
    speech_events: list[SpeechEvent] = []
    speech: list[tuple[float, float, str]] = []
    pauses: list[tuple[float, float]] = []
    prev_end = 0.0

    for seg_id, line in enumerate(lines):
        start = max(0.0, prev_end + line.gap)
        speaker = None if params.anonymous else line.speaker
        words = line.text.split()
        t = run_start = start
        for i, word in enumerate(words):
            t += word_s
            speech_events.append(SpeechPartial(t=_r(t), segment=seg_id, speaker=speaker,
                                               text=" ".join(words[: i + 1]), t_start=_r(start)))
            if word.endswith(PAUSE_AFTER) and i < len(words) - 1:
                speech.append((_r(run_start), _r(t), line.speaker))
                pauses.append((_r(t), _r(t + params.comma_pause_s)))
                t += params.comma_pause_s
                run_start = t
        speech.append((_r(run_start), _r(t), line.speaker))
        speech_events.append(SpeechFinal(t=_r(t + params.final_delay_s), segment=seg_id, speaker=speaker,
                                         text=line.text, t_start=_r(start), t_end=_r(t)))
        prev_end = t

    events = speech_events + vad_steps(speech, pauses, params)
    events.sort(key=lambda e: e.t)  # stable: at equal times, speech comes before voice activity
    return Timeline(events=tuple(events), speech=tuple(sorted(speech)))


def find_gaps(speech: Iterable[tuple[float, float, str]], min_s: float = 0.2) -> list[Gap]:
    runs = sorted(speech)
    gaps: list[Gap] = []
    covered_until, last_speaker = None, None
    for start, end, speaker in runs:
        if covered_until is not None and start - covered_until >= min_s:
            gaps.append(Gap(covered_until, start, last_speaker, speaker))
        if covered_until is None or end >= covered_until:
            covered_until, last_speaker = end, speaker
    if covered_until is not None:
        gaps.append(Gap(covered_until, float("inf"), last_speaker, None))
    return gaps


class ReplaySource:
    """Paces timeline events on a clock. Supports yielding the floor to the AI."""

    def __init__(self, events: Sequence[SpeechEvent], clock: Clock) -> None:
        self._events = events
        self._clock = clock
        self._next = 0
        self._offset = 0.0
        #: The delay applied when each segment started: speech already heard is never moved.
        self._segment_offset: dict[int, float] = {}
        self._last_word_offset: dict[int, float] = {}

    @property
    def offset(self) -> float:
        return self._offset

    def yield_floor(self, seconds: float) -> None:
        """Push every event not yet delivered back by `seconds` (closed loop)."""
        self._offset += max(0.0, seconds)

    async def events(self) -> AsyncIterator[SpeechEvent]:
        start = self._clock.now()
        while self._next < len(self._events):
            event = self._events[self._next]
            wait = start + event.t + self._offset - self._clock.now()
            if wait > 1e-9:
                await self._clock.sleep(wait)
                continue  # the offset may have changed while we slept
            self._next += 1
            yield self._shifted(event)

    def _shifted(self, event: SpeechEvent) -> SpeechEvent:
        """Delivery time moves with the offset; the time a segment started keeps the offset it had then."""
        if isinstance(event, SpeechPartial):
            spoken = self._segment_offset.setdefault(event.segment, self._offset)
            self._last_word_offset[event.segment] = self._offset
            return replace(event, t=_r(event.t + self._offset), t_start=_r(event.t_start + spoken))
        if isinstance(event, SpeechFinal):
            spoken = self._segment_offset.get(event.segment, self._offset)
            # Words after a pause may have been pushed back: the end is at least the last partial's time.
            return replace(event, t=_r(event.t + self._offset), t_start=_r(event.t_start + spoken),
                           t_end=_r(event.t_end + self._last_word_offset.get(event.segment, spoken)))
        return _shift(event, self._offset)


# -- internals -----------------------------------------------------------------------

def vad_steps(speech: list[tuple[float, float, str]], pauses: list[tuple[float, float]],
               params: ReplayParams) -> list[VadStep]:
    rng = random.Random(params.seed)
    end = max((e for _, e, _ in speech), default=0.0) + params.tail_s
    steps = []
    for k in range(int(end / params.vad_step_s) + 1):
        t = _r(k * params.vad_step_s)
        speaking = _inside(speech, t)
        in_pause = not speaking and any(a <= t < b for a, b in pauses)
        probs = []
        for h in HORIZONS:
            truth = not _inside(speech, t + h)
            p = 0.8 if truth else 0.15
            if in_pause:
                p += params.pause_false_alarm
            p += rng.gauss(0.0, params.noise)
            probs.append(round(min(0.98, max(0.02, p)), 2))
        steps.append(VadStep(t=t, speaking=speaking, p_silence=tuple(probs)))
    return steps


def _inside(speech: list[tuple[float, float, str]], t: float) -> bool:
    return any(a <= t < b for a, b, _ in speech)


def _shift(event: SpeechEvent, dt: float) -> SpeechEvent:
    if dt == 0:
        return event
    if isinstance(event, SpeechFinal):
        return replace(event, t=_r(event.t + dt), t_start=_r(event.t_start + dt), t_end=_r(event.t_end + dt))
    if isinstance(event, SpeechPartial):
        return replace(event, t=_r(event.t + dt), t_start=_r(event.t_start + dt))
    return replace(event, t=_r(event.t + dt))


def _r(t: float) -> float:
    return round(t, 3)
