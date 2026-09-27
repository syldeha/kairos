"""The Scribe: plain code, no model. Turns speech events into the transcript and the room state.

Partials are written to the board at once, as provisional segments: the rest
of the system can use what is being said before the sentence ends. A final
replaces the provisional segment; afterwards only the cleaner may correct its words (live speech).
"""

from __future__ import annotations

from dataclasses import replace

from ..board import Board
from ..contracts import RoomState, Segment, SpeechEvent, SpeechFinal, SpeechPartial, VadStep

AI_SEGMENT_BASE = 100_000


class Scribe:
    def __init__(self, board: Board) -> None:
        self.board = board
        self._segments: dict[int, Segment] = {}
        self._room = RoomState()
        self._vad_speaking = False
        self._live_speaking = False
        self._ai_count = 0
        self._ai_open: int | None = None
        #: segment -> what the recognizer heard, for lines the cleaner corrected
        self.raw: dict[int, str] = {}

    def on_event(self, event: SpeechEvent) -> None:
        if isinstance(event, SpeechPartial):
            self._segments[event.segment] = Segment(event.segment, event.speaker, event.text,
                                                    event.t_start, event.t, final=False)
            self._publish_transcript()
            self._update_room(t=event.t, last_speaker=event.speaker or self._room.last_speaker)
        elif isinstance(event, SpeechFinal):
            self._segments[event.segment] = Segment(event.segment, event.speaker, event.text,
                                                    event.t_start, event.t_end, final=True)
            self._publish_transcript()
            self._update_room(t=event.t)
        elif isinstance(event, VadStep):
            self._vad_speaking = event.speaking
            self._set_speaking(event.t, p_silence=event.p_silence)

    def revise(self, segment: int, text: str) -> None:
        """A committed line corrected from its context (live speech recognition): the raw text is kept."""
        current = self._segments.get(segment)
        if current is None or not current.final:
            return
        self.raw.setdefault(segment, current.text)
        self._segments[segment] = replace(current, text=text)
        self._publish_transcript()

    def ai_line(self, text: str, t_start: float, t_end: float, final: bool, speaker: str) -> None:
        """Kairos's own words go to the transcript too, so every agent knows what it said."""
        if self._ai_open is None:
            self._ai_count += 1
            self._ai_open = AI_SEGMENT_BASE + self._ai_count
        self._segments[self._ai_open] = Segment(self._ai_open, speaker, text, t_start, t_end, final)
        if final:
            self._ai_open = None
        self._publish_transcript()

    def set_live_speaking(self, speaking: bool, t: float) -> None:
        """A participant typing in the console counts as someone speaking."""
        self._live_speaking = speaking
        self._set_speaking(t)

    def _set_speaking(self, t: float, **changes) -> None:
        speaking = self._vad_speaking or self._live_speaking
        silence_since = self._room.silence_since
        if speaking:
            silence_since = None
        elif self._room.speaking or silence_since is None:
            silence_since = t
        self._update_room(t=t, speaking=speaking, silence_since=silence_since, **changes)

    def _publish_transcript(self) -> None:
        ordered = sorted(self._segments.values(), key=lambda s: (s.t_start, s.id))
        self.board.publish("transcript", tuple(ordered))

    def _update_room(self, **changes) -> None:
        changes["t"] = max(changes.get("t", 0.0), self._room.t)
        self._room = replace(self._room, **changes)
        self.board.publish("room", self._room)
