"""Helpers shared by the background agents."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Awaitable, Callable

from ..board import BoardSnapshot
from ..clock import Clock
from ..contracts import Segment, Thought, ThoughtStatus

log = logging.getLogger("kairos")

AI_NAME = "Kairos"

_ENGLISH = {"the", "you", "is", "are", "what", "i", "to", "and", "can", "do", "it", "that", "would", "like", "have",
            "my", "for", "there", "how", "where", "yeah", "don't", "want", "this", "of"}
_FRENCH = {"le", "la", "les", "je", "tu", "vous", "est", "et", "de", "des", "un", "une", "que", "pour", "pas", "on",
           "il", "ça", "c'est", "moi", "nous", "quoi", "comment", "oui", "non", "du", "au", "en"}


def language_of(text: str, default: str | None = "French") -> str | None:
    """The language a person just spoke, French or English: Kairos replies in it."""
    words = re.findall(r"[a-zà-ÿ']+", text.lower())
    english, french = sum(w in _ENGLISH for w in words), sum(w in _FRENCH for w in words)
    if english >= 2 and english > french:
        return "English"
    if french >= 2 and french > english:
        return "French"
    return default


def spoken_language(lines: list[str], default: str) -> str:
    """The language the room speaks now: the latest line where it can be told. A short line ("We're going
    with my boys.", "On part samedi.") does not reset it to the meeting's default."""
    for text in reversed(lines[-4:]):
        language = language_of(text, None)
        if language is not None:
            return language
    return default


ACTIVE = (ThoughtStatus.READY, ThoughtStatus.PENDING)


def is_ai(segment: Segment) -> bool:
    return segment.speaker == AI_NAME


def render_transcript(snap: BoardSnapshot, last: int = 12) -> str:
    lines = []
    for s in snap.transcript[-last:]:
        who = s.speaker or "Someone"
        marker = "" if s.final else " …(still speaking)"
        lines.append(f"L{s.id} [{s.t_start:.1f}s] {who}: {s.text}{marker}")
    return "\n".join(lines) or "(nothing said yet)"


def render_findings(snap: BoardSnapshot) -> str:
    done = [f for f in snap.findings if f.status == "done"]
    return "\n".join(f"- {f.question}: {f.answer}" for f in done) or "(none)"


def last_human_final(snap: BoardSnapshot) -> Segment | None:
    finals = [s for s in snap.transcript if s.final and not is_ai(s)]
    return max(finals, key=lambda s: s.t_end) if finals else None


def last_human_segment(snap: BoardSnapshot) -> Segment | None:
    """The latest line spoken by a person, committed or still being spoken."""
    humans = [s for s in snap.transcript if not is_ai(s)]
    return max(humans, key=lambda s: (s.t_start, s.id)) if humans else None


def active_thoughts(snap: BoardSnapshot) -> list[Thought]:
    return [t for t in snap.thoughts if t.status in ACTIVE]


MAX_ACTIVE = 5  # the reservoir: a few distinct ideas ready to say, not a pile


MAX_PENDING = 2  # points set aside for later: a couple, not a backlog


def enforce_cap(board, max_active: int = MAX_ACTIVE) -> None:
    """Keep at most `max_active` thoughts ready or pending, and at most MAX_PENDING pending.

    What leaves is what matters least right now: the lowest current score (importance, relevance to the
    discussion, freshness), pending points first, decisive ones last. Answers to a question still open are kept.
    """
    from ..decide.policy import PolicyParams, score  # the decider's own measure of what matters now

    from dataclasses import replace

    snap = board.snapshot()
    now, params = snap.room.t, PolicyParams()

    def worth(t: Thought) -> float:
        # A thought not yet measured gets the benefit of the doubt: a new idea is not evicted for being new.
        return score(t if t.judged_line is not None else replace(t, relevance=params.relevance_high), now, params)

    open_question = snap.signals.addressed_segment if not snap.signals.answered else None
    candidates = [t for t in active_thoughts(snap) if t.answers is None or t.answers != open_question]
    drop: dict[str, dict] = {}
    # Decisive thoughts (corrections, known constraints) leave last: they age, but stay true.
    pending = sorted((t for t in candidates if t.status == ThoughtStatus.PENDING),
                     key=lambda t: (t.importance >= 5, worth(t)))
    for t in pending[:max(0, len(pending) - MAX_PENDING)]:
        drop[t.id] = {"status": ThoughtStatus.STALE, "note": "trop de points en attente : le moins utile sort"}
    remaining = [t for t in candidates if t.id not in drop]
    overflow = len(remaining) - max_active
    if overflow > 0:
        ranked = sorted(remaining, key=lambda t: (t.importance >= 5, t.status != ThoughtStatus.PENDING, worth(t)))
        for t in ranked[:overflow]:
            drop[t.id] = {"status": ThoughtStatus.STALE, "note": "réservoir plein (5 max) : le moins utile maintenant sort"}
    if drop:
        board.update_thoughts(drop, by="réservoir")


LEAD_INS = ("pour ", "et pour", "du côté", "côté ", "sur ", "et sur", "concernant", "quant à", "à propos",
            "au sujet", "d'ailleurs", "going back", "back to", "on the ", "regarding", "about ", "as for", "for the ",
            # openings that already frame the sentence
            "petite précision", "précision", "attention", "au fait", "juste pour", "par ailleurs", "rappel",
            "oui", "non", "ok", "note", "careful", "quick note", "by the way", "just to")


def with_lead_in(transition: str | None, utterance: str) -> str:
    """Prefix the lead-in, unless the sentence already opens with one or already names its subject
    (no "Du côté de la bêta, Sur la bêta," nor "Sur l'Alfama, justement : Attention à l'Alfama")."""
    if not transition or utterance.lower().lstrip().startswith(LEAD_INS):
        return utterance
    subject = {w for w in re.findall(r"[\w-]+", transition.lower()) if len(w) >= 4}
    opening = set(re.findall(r"[\w-]+", " ".join(utterance.lower().split()[:8])))
    if subject & opening - {"pour", "revenir", "justement", "going", "back"}:
        return utterance
    return f"{transition} {utterance}"


def spoken_text(thought: Thought) -> str:
    """What Kairos would actually say: with its lead-in when it comes back to a past topic."""
    if thought.status == ThoughtStatus.PENDING:
        return with_lead_in(thought.transition, thought.utterance)
    return thought.utterance


class Worker:
    """Runs `fn` in the background when kicked. Kicks during a run schedule exactly one more run."""

    def __init__(self, name: str, fn: Callable[[], Awaitable[None]], clock: Clock, delay_s: float = 0.0) -> None:
        self.name = name
        self._fn = fn
        self._clock = clock
        self._delay_s = delay_s
        self._event = asyncio.Event()
        self.runs = 0
        self.errors = 0

    def kick(self) -> None:
        self._event.set()

    async def loop(self) -> None:
        while True:
            await self._event.wait()
            self._event.clear()
            if self._delay_s:
                await self._clock.sleep(self._delay_s)  # let a burst of lines land as one run
            try:
                await self._fn()
                self.runs += 1
            except Exception:
                self.errors += 1
                log.exception("%s failed", self.name)


def recent_ack(snap: BoardSnapshot, within_s: float = 8.0) -> bool:
    """Kairos already said (or is about to say) "je regarde": a second one sounds like a machine."""
    now = snap.room.t
    return any(t.ack and (t.status in ACTIVE or (t.status == ThoughtStatus.SPOKEN and now - t.created_at <= within_s))
               for t in snap.thoughts)

