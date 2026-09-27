"""The topic tracker: what the room is talking about now, and when it changes.

A conversation drifts: people test the assistant, then book train tickets, then talk about a friend's
arrival. Without a notion of the current subject, Kairos reads the new talk through the old one
("before using this case to test the agent, ...") and keeps ideas the room has left.

Two steps, so that the check costs nothing on most lines:
1. Jev, in the dispatcher's request on each sentence, answers one more yes/no question: do the last
   five lines talk about a different subject than the current topic?
2. Only when Jev says yes (or when there is no topic yet), the writing model names the subject of the
   last five lines and confirms the change. Then the ideas of the old subject leave the reservoir,
   and the thinkers, the notes and the background research start from the new one.
"""

from __future__ import annotations

from dataclasses import replace

from ..board import Board
from ..contracts import ThoughtStatus
from ..llm import LLM
from .common import active_thoughts, is_ai

NAME = """Kairos, an AI assistant, attends a live conversation. Say what the people are talking about in their latest lines.
Return one JSON object: {"topic": "the subject most of the latest lines are about, in a few words, in {language}", "changed": true}.
The subject covers most of the latest lines, broadly enough to include them all: an isolated aside ("À Londres, c'est ailleurs"), a question to the assistant or one unclear line is not a subject.
"changed": true only when most of the latest lines are about a different subject than the previous topic (another trip, another problem, another plan), false when they continue it, add details to it, drift inside it or come back to it.
Name the subject as the speakers would, from what they actually say, never through the previous topic: "billets de train pour Rotterdam à 5", not "un cas de test pour l'agent"."""


class TopicTracker:
    def __init__(self, board: Board, llm: LLM, language: str = "French", window: int = 5,
                 min_gap_s: float = 15.0) -> None:
        self.board = board
        self.llm = llm
        self.language = language
        self.window = window
        self.min_gap_s = min_gap_s  # a topic just named is not replaced at once: one aside is not a new subject
        self.history: list[tuple[float, str]] = []  # (time, topic), oldest first
        self.checks = 0
        #: a subject named from its first lines is named again once it has a few more (at this line count)
        self._refine_after: int | None = None

    @property
    def topic(self) -> str:
        return self.history[-1][1] if self.history else ""

    def wants_name(self) -> bool:
        """No topic yet: name one as soon as a few lines were said. A new subject: name it again once it
        has lines of its own (named from its first, often messy, line it is "la photo des groupes tickets")."""
        count = _human_lines(self.board.snapshot())
        if not self.history:
            return count >= 3
        return self._refine_after is not None and count >= self._refine_after

    async def check(self) -> bool:
        """Name the subject of the last lines. Returns True when the topic changed (not on the first naming,
        nor when a new subject is only renamed)."""
        snap = self.board.snapshot()
        now = snap.room.t
        count = _human_lines(snap)
        lines = [s for s in snap.transcript if s.final and not is_ai(s)][-self.window:]
        if len(lines) < 3:
            return False
        refine = bool(self.history) and self._refine_after is not None and count >= self._refine_after
        if self.history and not refine and now - self.history[-1][0] < self.min_gap_s:
            return False
        self.checks += 1
        user = (f"Previous topic: {self.topic or '(none yet)'}\n\nLatest lines (latest last):\n"
                + "\n".join(f"{s.speaker or 'Someone'}: {s.text}" for s in lines))
        raw = await self.llm.json(NAME.replace("{language}", self.language), user, purpose="topic", temperature=0.2)
        topic = str(raw.get("topic") or "").strip()
        if not topic:
            return False
        first = not self.history
        if refine:
            # Same subject, better name: nothing leaves the reservoir.
            self._refine_after = None
            since = self.history[-1][0]
            self.history[-1] = (since, topic)
            self.board.publish("signals", replace(self.board.snapshot().signals, topic=topic))
            return False
        if not first and raw.get("changed") is False:
            return False
        previous = self.topic
        self._refine_after = None if first else count + 3  # a stable first subject keeps its name
        self.history.append((now, topic))
        signals = self.board.snapshot().signals
        self.board.publish("signals", replace(signals, topic=topic, previous_topic=previous, topic_since=round(now, 1)))
        if first:
            return False
        self._retire(topic, before=now)
        return True

    def _retire(self, topic: str, before: float) -> None:
        """The room left the old subject: its ideas leave the reservoir. Answers owed, corrections (a wrong
        figure stays wrong) and the workers' questions and results stay: they are about what people asked."""
        changes = {t.id: {"status": ThoughtStatus.STALE, "note": f"le sujet a changé : « {topic} »"}
                   for t in active_thoughts(self.board.snapshot())
                   if t.kind == "idea" and t.answers is None and t.importance < 5 and t.created_at < before}
        if changes:
            self.board.update_thoughts(changes, by="sujet")


def _human_lines(snap) -> int:
    return sum(1 for s in snap.transcript if s.final and not is_ai(s))
