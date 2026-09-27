"""Context research: Kairos reads up on what the room is working on, before anyone asks.

In a real work meeting nobody says "Kairos, look this up": people discuss a subject. Without
outside knowledge the thinkers only have general sense, and produce generic advice that the
judge rightly rejects. This worker gives them material: when the subject of the meeting settles
or changes (read from the notes and the latest lines), it writes one or two targeted web queries
about what the group is deciding right now, runs them (Exa), and stores the results as sourced
facts in Kairos's knowledge (the findings). It never writes the reservoir: the thinkers turn the
facts into thoughts, and Jev still decides whether and when they are worth saying.

Limits: at most one run per `min_interval_s`, at most `max_searches` searches per meeting, never
the same query twice, nothing about the team itself (its budget, people, internal plans).
"""

from __future__ import annotations

import itertools

from ..board import Board
from ..contracts import Finding
from ..llm import LLM, cosine
from .common import render_findings, render_transcript

PLAN = """Kairos, an AI assistant, silently attends a work meeting. Decide whether to read up on the web now, in the background, so that it can later contribute concrete, sourced facts to the discussion.
Return one JSON object: {"subject": "what the group is working on or deciding right now, in a few words", "queries": ["..."]}.
Give 0 to 2 queries. Each is a short, specific web query for outside knowledge that would help the group decide what it is deciding right now: studies, statistics, market data, what users prefer, how existing products do it, norms, prices. Not generic ("remote control design"), not about this team, its budget or its people, not already covered by what Kairos found before. Write the queries in English or in the meeting's language, whichever finds better sources.
Give no query when the discussion is small talk, logistics of the meeting itself, or already covered."""


class ContextResearch:
    def __init__(self, board: Board, llm: LLM, search, language: str = "English",
                 min_interval_s: float = 90.0, max_searches: int = 6) -> None:
        self.board = board
        self.llm = llm
        self.search = search
        self.language = language
        self.min_interval_s = min_interval_s
        self.max_searches = max_searches
        self.searches = 0
        self.log: list[tuple[float, str, str]] = []  # (time, subject, query)
        self._last_run: float | None = None
        self._queries: list[list[float]] = []
        self._ids = itertools.count(1)

    def new_subject(self) -> None:
        """The room changed subject: read up on the new one without waiting for the interval."""
        self._last_run = None

    async def run_once(self) -> None:
        snap = self.board.snapshot()
        now = snap.room.t
        if self.search is None or self.searches >= self.max_searches:
            return
        if self._last_run is not None and now - self._last_run < self.min_interval_s:
            return
        if sum(1 for s in snap.transcript if s.final) < 4:
            return  # not enough said yet to know the subject
        self._last_run = now
        memory = "\n".join(f"- {m}" for m in snap.long_term) or "(none)"
        user = (f"Current topic (tracked from the latest lines): {snap.signals.topic or '(unknown)'}\n\n"
                f"Meeting notes:\n{snap.notes or '(none yet)'}\n\nKairos's memory:\n{memory}\n\n"
                f"What Kairos already found:\n{render_findings(snap)}\n\n"
                f"Latest lines:\n{render_transcript(snap, last=12)}")
        plan = await self.llm.json(PLAN, user, purpose="context plan", temperature=0.2)
        subject = str(plan.get("subject") or "").strip()
        queries = [str(q).strip() for q in plan.get("queries") or [] if str(q).strip()][:2]
        for query in queries:
            if self.searches >= self.max_searches or not await self._new(query):
                continue
            self.searches += 1
            self.log.append((now, subject, query))
            await self._run(subject, query)

    async def _new(self, query: str) -> bool:
        vector = (await self.llm.embed([query], purpose="embedding"))[0]
        if any(cosine(vector, past) >= 0.85 for past in self._queries):
            return False
        self._queries.append(vector)
        return True

    async def _run(self, subject: str, query: str) -> None:
        snap = self.board.snapshot()
        fid = f"ctx{next(self._ids)}"
        pending = Finding(id=fid, question=f"contexte : {subject}", query=query, segment=None, status="searching",
                          started_at=snap.room.t)
        self.board.publish("findings", snap.findings + (pending,))
        try:
            # Facts, not advice: the thinkers turn facts into contributions; advice from a web page is noise.
            result = await self.search.search(
                query, f"What do studies, statistics or market data say about: {query}? Give one or two concrete "
                       f"facts with figures and their source, no advice.", self.language)
            status = "done" if result.answer else "failed"
            answer, sources, seconds = result.answer, tuple(result.sources), round(result.seconds, 2)
        except Exception as exc:
            status, answer, sources, seconds = "failed", f"{type(exc).__name__}", (), 0.0
        snap = self.board.snapshot()
        done = Finding(id=fid, question=f"contexte : {subject}", query=query, segment=None, status=status,
                       started_at=pending.started_at, answer=answer, sources=sources, seconds=seconds)
        self.board.publish("findings", tuple(f for f in snap.findings if f.id != fid) + (done,))
