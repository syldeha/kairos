"""The speaker: says an utterance plan word by word, and yields when interrupted.

- Between two chained thoughts it leaves a short pause: if someone starts
  talking there, the second thought goes back to PENDING.
- A listening signal ("mm", "yeah") does not stop it.
- Anything else stops it at once, without asking any model. If nothing
  follows within `false_start_s`, it was a false start: Kairos resumes with
  "as I was saying". Otherwise it yields and what it did not say goes back to PENDING.

The words appear on screen at speech rate; with a voice (Gradium text-to-speech) they are also heard:
the audio starts with each thought, stops when Kairos is cut off, and restarts when it resumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

from .agents.common import AI_NAME
from .agents.scribe import Scribe
from .backchannel import is_backchannel
from .board import Board
from .clock import Clock
from .contracts import SpeechFinal, SpeechPartial, ThoughtStatus, UtterancePlan

RESUME = {"French": "donc je disais,", "English": "as I was saying,"}
MAX_CUTS = 2  # a thought cut off this many times is dropped


@dataclass(frozen=True, slots=True)
class SpeakerParams:
    words_per_s: float = 2.75
    chain_pause_s: float = 0.3
    false_start_s: float = 1.5
    language: str = "English"


@dataclass(slots=True)
class SpeechOutcome:
    plan: UtterancePlan
    t_start: float
    t_end: float = 0.0
    text: str = ""
    said: list[str] = field(default_factory=list)   # thought ids fully said
    cut: list[str] = field(default_factory=list)    # thought ids not (fully) said
    interruptions: int = 0
    resumed: int = 0
    yielded: bool = False


class Speaker:
    def __init__(self, board: Board, scribe: Scribe, clock: Clock, now: Callable[[], float],
                 params: SpeakerParams = SpeakerParams(),
                 on_word: Callable[[float], None] | None = None, voice=None) -> None:
        self.voice = voice  # optional GradiumVoice: say(text), stop(), wait_done()
        self.board = board
        self.scribe = scribe
        self.clock = clock
        self.now = now
        self.params = params
        self.on_word = on_word or (lambda seconds: None)
        self.speaking = False
        self._interrupted = False
        self._heard_after = 0  # human words heard since the interruption
        self._cuts: dict[str, int] = {}  # thought id -> times Kairos was cut off saying it

    def hear(self, event: SpeechPartial | SpeechFinal) -> None:
        """Human speech arriving while Kairos holds the floor (or has just stopped)."""
        if not self.speaking:
            return
        if self._interrupted:
            self._heard_after += 1
        elif isinstance(event, SpeechPartial) and not is_backchannel(event.text):
            self._interrupted = True
            self._heard_after = 0

    def barge_in(self) -> None:
        """The page heard someone talk over Kairos and already silenced the audio: stop the words too."""
        if self.speaking and not self._interrupted:
            self._interrupted = True
            self._heard_after = 0

    async def say(self, plan: UtterancePlan) -> SpeechOutcome:
        outcome = SpeechOutcome(plan, t_start=self.now())
        self.speaking, self._interrupted = True, False
        spoken: list[str] = []
        word_s = 1.0 / self.params.words_per_s
        self._publish_ai(True, "", outcome.t_start)
        try:
            for index, (thought_id, text) in enumerate(plan.parts):
                if index > 0:
                    await self.clock.sleep(self.params.chain_pause_s)
                    if self._interrupted:  # someone took the pause: keep the rest for later
                        outcome.cut.extend(tid for tid, _ in plan.parts[index:])
                        outcome.interruptions += 1
                        outcome.yielded = True
                        break
                words = text.split()
                await self._voice_say(text)
                w = 0
                while w < len(words):
                    if self._interrupted:
                        outcome.interruptions += 1
                        if await self._false_start():
                            outcome.resumed += 1
                            if w <= 3 or outcome.resumed > 1:
                                # Barely started, or already resumed once: the sentence again, plainly, without
                                # a second "donc je disais" ("… donc je disais, C'est surtout donc donc je disais").
                                words = _without_resume(words, self.params.language)
                                words = _lower_first(words[_sentence_start(words, min(w, len(words) - 1)):]) \
                                    if w > 3 else words
                                w = 0
                            else:
                                # Restart the sentence that was cut, as a person would: "donc je disais, le moins
                                # cher est…", never "le moins cher connu donc je disais, est…".
                                words = RESUME.get(self.params.language, RESUME["English"]).split() + \
                                    _lower_first(words[_sentence_start(words, w):])
                                w = 0
                            self._publish_ai(True, " ".join(spoken), outcome.t_start)
                            await self._voice_say(" ".join(words))
                        else:
                            outcome.cut.extend(tid for tid, _ in plan.parts[index:])
                            outcome.yielded = True
                            break
                    self.on_word(word_s)  # closed loop: the room waits while Kairos speaks
                    await self.clock.sleep(word_s)
                    spoken.append(words[w])
                    w += 1
                    self._publish_ai(True, " ".join(spoken), outcome.t_start)
                if outcome.yielded:
                    break
                await self._voice_tail()  # the words are on screen; let the audio finish
                outcome.said.append(thought_id)
        finally:
            if self.voice is not None and outcome.yielded:
                self.voice.stop()
            self.speaking = False
            outcome.t_end = self.now()
            outcome.text = " ".join(spoken)
            self._finish(outcome)
        return outcome

    async def _voice_say(self, text: str) -> None:
        if self.voice is not None:
            await self.voice.say(text)  # returns when the audio starts: the words follow it

    async def _voice_tail(self) -> None:
        """Audio can run a little behind the words: wait for it, unless someone takes the floor."""
        if self.voice is None:
            return
        for _ in range(60):
            if self._interrupted:
                self.voice.stop()
                return
            if self.voice.wait_remaining() <= 0:
                return
            await self.clock.sleep(0.08)

    async def _false_start(self) -> bool:
        """Stop, listen for `false_start_s`: True if the other person stopped (resume), False to yield."""
        if self.voice is not None:
            self.voice.stop()
        self._publish_ai(False, None, None)
        await self.clock.sleep(self.params.false_start_s)
        human_still_talking = self.board.snapshot().room.speaking or self._heard_after > 1
        self._interrupted = False
        self._heard_after = 0
        return not human_still_talking

    def _publish_ai(self, speaking: bool, text: str | None, started_at: float | None) -> None:
        ai = self.board.snapshot().ai
        changes = {"speaking": speaking}
        if text is not None:
            changes["text"] = text
            self.scribe.ai_line(text or "…", started_at, self.now(), final=False, speaker=AI_NAME)
        if started_at is not None:
            changes["started_at"] = started_at
        self.board.publish("ai", replace(ai, **changes))

    def _finish(self, outcome: SpeechOutcome) -> None:
        duration = max(0.0, outcome.t_end - outcome.t_start)
        if outcome.text:
            self.scribe.ai_line(outcome.text, outcome.t_start, outcome.t_end, final=True, speaker=AI_NAME)
        ai = self.board.snapshot().ai
        self.board.publish("ai", replace(ai, speaking=False, text=outcome.text, last_spoke_at=outcome.t_end,
                                         total_speech_s=ai.total_speech_s + duration,
                                         interventions=ai.interventions + (1 if outcome.text else 0)))
        changes = {tid: {"status": ThoughtStatus.SPOKEN, "note": f"dite à {outcome.t_start:.0f} s"}
                   for tid in outcome.said}
        for tid in outcome.cut:
            if tid in changes:
                continue
            self._cuts[tid] = self._cuts.get(tid, 0) + 1
            if self._cuts[tid] >= MAX_CUTS:
                # Cut off twice: the room is not interested right now. Kairos lets it go.
                changes[tid] = {"status": ThoughtStatus.STALE, "note": "coupée deux fois : abandonnée"}
            else:
                changes[tid] = {"status": ThoughtStatus.PENDING, "note": "coupée : en attente"}
        if changes:
            self.board.update_thoughts(changes, by="orateur")
        thoughts = {t.id: t for t in self.board.snapshot().thoughts}
        real_answer = any(tid in thoughts and not thoughts[tid].ack for tid in outcome.said)
        if outcome.plan.reason == "asked" and real_answer:
            # A "let me look that up" does not answer: the question stays open for the finding.
            self.board.publish("signals", replace(self.board.snapshot().signals, answered=True))


def _sentence_start(words: list[str], w: int) -> int:
    """Index of the first word of the sentence that word `w` belongs to."""
    start = 0
    for i in range(min(w, len(words))):
        if words[i].endswith((".", "?", "!", ":")):
            start = i + 1
    return start


#: words that start a sentence without being names: lowercased after "donc je disais,"
_COMMON = {"le", "la", "les", "l'", "un", "une", "des", "du", "pour", "je", "j'ai", "il", "elle", "on", "nous", "vous",
           "ce", "c'est", "cette", "ces", "en", "à", "au", "aux", "avec", "dans", "d'après", "selon", "oui", "non",
           "the", "a", "an", "for", "it", "it's", "we", "you", "this", "that", "there", "in", "on", "with", "yes", "no"}


def _lower_first(words: list[str]) -> list[str]:
    if words and words[0].lower().replace("’", "'") in _COMMON:  # "C’est" as well as "C'est"
        return [words[0].lower()] + words[1:]
    return words


def _without_resume(words: list[str], language: str) -> list[str]:
    """The words of the sentence without the "donc je disais," a first resume put in front of it."""
    prefix = RESUME.get(language, RESUME["English"]).split()
    return words[len(prefix):] if words[:len(prefix)] == prefix else words
