"""Relevance: how close each thought is to what is being discussed right now.

One embedding per committed line (the last three human lines together) and
one per new thought. Cheap enough to recompute after every line.
"""

from __future__ import annotations

from ..board import Board
from ..contracts import ThoughtStatus
from ..llm import LLM, cosine
from .common import active_thoughts, is_ai


class Relevance:
    def __init__(self, board: Board, llm: LLM, window: int = 3) -> None:
        self.board = board
        self.llm = llm
        self.window = window
        self._vectors: dict[str, list[float]] = {}
        self._discussion: list[float] | None = None
        self._discussion_key: tuple = ()

    async def novel(self, texts: list[str], threshold: float = 0.80, active_threshold: float = 0.65,
                    decisive: list[bool] | None = None) -> list[list[float] | None]:
        """For each candidate thought: its vector if it is new, None if Kairos already had this idea.

        Two checks: against every thought Kairos ever had (spoken and stale included), and, more strictly,
        against the thoughts still in the reservoir, so that the reservoir holds different ideas, not
        paraphrases of one. Candidates are also compared with each other. A decisive candidate (a correction)
        may repeat what Kairos said before in other words: someone just repeated the mistake.
        """
        if not texts:
            return []
        vectors = await self.llm.embed(texts, purpose="embedding")
        snap = self.board.snapshot()
        active = {t.id for t in active_thoughts(snap)}
        spoken = {t.id for t in snap.thoughts if t.status == ThoughtStatus.SPOKEN}
        kept: list[list[float]] = []
        out: list[list[float] | None] = []
        decisive = decisive or [False] * len(vectors)
        for v, strong in zip(vectors, decisive):
            # Ideas set aside without ever being said only block near-copies: a thought dropped by mistake
            # (a false "outdated") can come back when the room needs it again.
            history = [u for tid, u in self._vectors.items() if tid not in active and tid not in spoken]
            threshold = max(threshold, 0.9)
            said = [u for tid, u in self._vectors.items() if tid in spoken]
            current = [u for tid, u in self._vectors.items() if tid in active] + kept
            # What Kairos already said is held to the reservoir's strict scale: no coming back in other words.
            # A correction only has to differ from a near-copy (0.80): "12 000 €, not 15 000" after someone
            # says 15 000 repeats the reminder Kairos gave, and is needed.
            said_threshold = 0.80 if strong else active_threshold
            if any(cosine(v, u) >= threshold for u in history) or \
                    any(cosine(v, u) >= active_threshold for u in current) or \
                    any(cosine(v, u) >= said_threshold for u in said):
                out.append(None)
            else:
                kept.append(v)
                out.append(v)
        return out

    async def retire_similar(self, text: str, threshold: float, note: str, keep: tuple[str, ...] = ()) -> int:
        """Mark as stale every active thought saying what `text` just said. Returns how many."""
        if not text.strip():
            return 0
        vector = (await self.llm.embed([text], purpose="embedding"))[0]
        changes = {t.id: {"status": ThoughtStatus.STALE, "note": note}
                   for t in active_thoughts(self.board.snapshot())
                   if t.id not in keep and t.id in self._vectors and cosine(vector, self._vectors[t.id]) >= threshold}
        if changes:
            self.board.update_thoughts(changes, by="pertinence")
        return len(changes)

    def remember(self, thought_id: str, vector: list[float]) -> None:
        self._vectors[thought_id] = vector

    async def run_once(self) -> None:
        snap = self.board.snapshot()
        # The last lines, including the one still being spoken: relevance follows speech as it arrives.
        recent = [s for s in snap.transcript if not is_ai(s)][-self.window:]
        key = tuple((s.id, len(s.text.split())) for s in recent)
        todo = [t for t in active_thoughts(snap) if t.id not in self._vectors]
        texts, what = [], []
        if recent and key != self._discussion_key:
            texts.append(" ".join(s.text for s in recent))
            what.append(None)
        for t in todo:
            texts.append(t.utterance)  # what Kairos would say: the text later compared with what was said
            what.append(t.id)
        if not texts:
            return
        vectors = await self.llm.embed(texts, purpose="embedding")
        for thought_id, vector in zip(what, vectors):
            if thought_id is None:
                self._discussion, self._discussion_key = vector, key
            else:
                self._vectors[thought_id] = vector
        if self._discussion is None:
            return
        changes = {t.id: {"relevance": round(cosine(self._vectors[t.id], self._discussion), 3)}
                   for t in active_thoughts(self.board.snapshot()) if t.id in self._vectors}
        if changes:
            self.board.update_thoughts(changes, by="pertinence")
