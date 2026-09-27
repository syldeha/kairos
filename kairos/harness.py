"""The harness: rules the decider cannot bend (heard-meeting's pattern).

A decision to speak becomes an utterance plan only if it has a valid reason,
fits the budget of unsolicited interventions, and stays short. Every refusal
is a reason to stay silent.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .agents.common import with_lead_in
from .board import BoardSnapshot
from .contracts import Decision, ThoughtStatus, UtterancePlan

REASONS = ("asked", "important", "pending")


@dataclass(frozen=True, slots=True)
class HarnessParams:
    # At most one unprompted intervention per this many seconds. 30 s (long AMI meetings) kept a thought Jev picked
    # at 0.81 unsaid for 23 s of silence in a live conversation.
    unsolicited_gap_s: float = 10.0
    urgent_score: float = 0.55        # ...unless Jev picks the thought above the speaking threshold: said at once
    max_words: int = 45


class Harness:
    def __init__(self, params: HarnessParams = HarnessParams()) -> None:
        self.params = params
        self._last_unsolicited: float | None = None
        self._before_last: float | None = None
        self.refusals: list[tuple[float, str]] = []

    def check(self, decision: Decision, snap: BoardSnapshot) -> UtterancePlan | None:
        why = self._refusal(decision, snap)
        if why:
            self.refusals.append((decision.t, why))
            return None
        thoughts = {t.id: t for t in snap.thoughts}
        primary = thoughts[decision.primary]
        first = primary.utterance
        if primary.status == ThoughtStatus.PENDING:
            first = with_lead_in(primary.transition, primary.utterance)
        parts = [(primary.id, _cut(first, self.params.max_words))]
        if decision.chain and decision.chain in thoughts:
            chained = thoughts[decision.chain]
            text = with_lead_in(chained.transition, chained.utterance)
            if _words(parts[0][1]) + _words(text) <= self.params.max_words:
                parts.append((chained.id, text))
        if decision.reason != "asked":
            self._before_last, self._last_unsolicited = self._last_unsolicited, decision.t
        return UtterancePlan(decision.t, decision.reason or "", tuple(parts))

    def forgive(self, t: float) -> None:
        """The intervention started at `t` was cut off in its first words ("Le vol…"): it does not count."""
        if self._last_unsolicited == t:
            self._last_unsolicited = self._before_last

    def _refusal(self, decision: Decision, snap: BoardSnapshot) -> str | None:
        if not decision.speak:
            return "no decision to speak"
        if decision.reason not in REASONS:
            return f"reason must be one of {REASONS}"
        if decision.primary not in {t.id for t in snap.thoughts}:
            return "the thought no longer exists"
        if snap.room.speaking or snap.ai.speaking:
            return "the floor closed before speaking"
        primary = next((t for t in snap.thoughts if t.id == decision.primary), None)
        decisive = primary is not None and primary.importance >= 5
        if decision.reason != "asked" and self._last_unsolicited is not None \
                and decision.score < self.params.urgent_score and not decisive:
            wait = self.params.unsolicited_gap_s - (decision.t - self._last_unsolicited)
            if wait > 0:
                return f"budget: next unprompted intervention in {wait:.0f} s"
        return None


def decisive_only(snap: BoardSnapshot) -> BoardSnapshot:
    """The board as the decider would see it if only the thoughts the budget lets through existed:
    corrections and decisive facts (importance 5) and answers to a question."""
    return replace(snap, thoughts=tuple(
        t for t in snap.thoughts
        if t.status not in (ThoughtStatus.READY, ThoughtStatus.PENDING) or t.importance >= 5 or t.answers is not None))


def _words(text: str) -> int:
    return len(text.split())


def _cut(text: str, max_words: int) -> str:
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words]).rstrip(",;:") + "."
