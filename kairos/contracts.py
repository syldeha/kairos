"""The shapes that cross component boundaries.

Every event carries its own time `t`, in seconds since the meeting started.
Components never read a wall clock to reason about the conversation: time
comes from the events, which is what makes a replay deterministic.

The speech events mirror what a streaming speech-to-text service with semantic
voice activity detection sends (Gradium: interim text, final text, and a
silence probability at several horizons every 80 ms). The text simulation and
the future audio source therefore produce the same events.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

#: Horizons (seconds) at which the voice activity detector predicts silence.
HORIZONS: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)


# -- speech events --------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SpeechPartial:
    """Interim transcript of the segment being spoken. Replaced by the next one."""

    t: float
    segment: int
    speaker: str | None  # None when voices are not separated (one room microphone)
    text: str            # everything heard so far in this segment
    t_start: float


@dataclass(frozen=True, slots=True)
class SpeechFinal:
    """The committed transcript of a segment. Never edited afterwards."""

    t: float
    segment: int
    speaker: str | None
    text: str
    t_start: float
    t_end: float


@dataclass(frozen=True, slots=True)
class VadStep:
    """Voice activity, sent every 80 ms: is anyone speaking, and will it be quiet soon?"""

    t: float
    speaking: bool
    #: Probability that the room is silent `h` seconds from now, per horizon in HORIZONS.
    p_silence: tuple[float, ...]

    def p(self, horizon: float) -> float:
        return self.p_silence[HORIZONS.index(horizon)]


SpeechEvent = SpeechPartial | SpeechFinal | VadStep


# -- board contents ---------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Segment:
    """One line of the transcript. `final` is False while the segment is still being spoken."""

    id: int
    speaker: str | None
    text: str
    t_start: float
    t_end: float
    final: bool


@dataclass(frozen=True, slots=True)
class RoomState:
    t: float = 0.0
    speaking: bool = False
    last_speaker: str | None = None
    #: When the room last fell silent; None while someone is speaking.
    silence_since: float | None = 0.0
    p_silence: tuple[float, ...] = (0.0,) * len(HORIZONS)

    @property
    def silence_s(self) -> float:
        if self.speaking or self.silence_since is None:
            return 0.0
        return max(0.0, self.t - self.silence_since)


class ThoughtStatus(StrEnum):
    DRAFT = "draft"
    READY = "ready"
    PENDING = "pending"    # its topic moved on before it could be said
    SPOKEN = "spoken"
    STALE = "stale"        # someone said it, or it is too old


@dataclass(frozen=True, slots=True)
class Thought:
    id: str
    topic: str
    content: str
    utterance: str               # the sentence, ready to be said
    transition: str | None       # the lead-in when it is chained after another thought
    importance: float            # 1-5, scored when generated (slow path)
    relevance: float             # 0-1, similarity with the current discussion (recomputed continuously)
    fit_now: float               # 0-1, "coherent to say now", precomputed by the judge
    already_said: float          # 0-1, "someone already said it", precomputed by the judge
    status: ThoughtStatus
    stimuli: tuple[str, ...]
    version: int
    created_at: float
    answers: int | None = None   # id of the segment this thought answers, when someone addressed the AI
    judged_line: int | None = None  # the human line its fit_now/already_said were judged against
    note: str = ""               # why its status last changed, shown in the console
    ack: bool = False            # "let me look that up": the question stays open until the finding is said
    #: idea · answer · correction · finding · question (a worker asking the room for missing details)
    kind: str = "idea"
    brief: str | None = None     # the worker brief it serves or comes from
    chosen: float | None = None  # Jev rater: probability it is the one to say at the next pause
    value: float | None = None   # Jev rater: 0 no value, 1 useful, 2 decision-changing


@dataclass(frozen=True, slots=True)
class Finding:
    """What the researcher looked up on the web, and what it found."""

    id: str
    question: str        # the question as it came up in the meeting
    query: str           # what was sent to the search engine (no names, nothing internal)
    segment: int | None  # the line that raised it
    status: str          # "searching" | "done" | "failed"
    started_at: float
    answer: str = ""
    sources: tuple[str, ...] = ()
    seconds: float = 0.0  # wall-clock time of the search


@dataclass(frozen=True, slots=True)
class Signals:
    """What the judge read in the last committed line."""

    addressed: float = 0.0              # probability that the last line addresses the AI
    addressed_segment: int | None = None
    answered: bool = True
    topic: str = ""                     # what the room talks about now (the topic tracker)
    previous_topic: str = ""            # the subject before the last change
    topic_since: float | None = None
    last_final: int | None = None
    #: The latest human line the judge has seen, and how many of its words: judgments follow speech
    #: as it arrives, so they can be ready before the line is committed.
    judged_segment: int | None = None
    judged_words: int = 0
    #: Jev rater: the line its choice was made on, and the probability of "say nothing"
    rater_line: int | None = None
    rater_none: float = 0.0
    #: the line the checker is reading right now: a correction may be on its way
    checking: int | None = None


@dataclass(frozen=True, slots=True)
class AiState:
    speaking: bool = False
    text: str = ""                      # what has been said so far in the current intervention
    started_at: float | None = None
    last_spoke_at: float | None = None
    total_speech_s: float = 0.0
    interventions: int = 0


@dataclass(frozen=True, slots=True)
class Decision:
    t: float
    speak: bool
    why: str
    reason: str | None = None           # "asked" | "important" | "pending"
    primary: str | None = None          # thought ids
    chain: str | None = None
    confidence: float = 0.0
    score: float = 0.0


@dataclass(frozen=True, slots=True)
class UtterancePlan:
    t: float
    reason: str
    parts: tuple[tuple[str, str], ...]  # (thought id, text to say), in order
