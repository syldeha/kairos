"""The notes agent: keeps the short-term memory of the meeting.

Rewrites the whole notes from the previous version and the lines committed
since (heard-meeting's pattern). The thinkers and the judge read these notes
instead of the full transcript. A failed pass leaves the previous notes in place.
"""

from __future__ import annotations

import re

from ..board import Board
from ..llm import LLM
from .common import is_ai, language_of, render_findings, render_transcript

SYSTEM = """You keep the live notes of a meeting attended by an AI assistant named Kairos. You receive your previous notes and the latest transcript. Return one JSON object: {"notes": "the notes as plain markdown text"}.

The notes are short and specific, under 300 words:
- one line on where the discussion stands right now
- the topics discussed, with the claims made and who made them
- open questions nobody answered yet
- decisions taken
- what Kairos said, if anything
- results of Kairos's searches (flights, facts) with their key figures and sources
- the key figures and constraints the room works with (prices, budgets, targets, counts), kept even when the
  discussion moves on: the checker relies on them

Never invent. Keep earlier points unless clearly superseded. Write in {language}."""


class NotesAgent:
    def __init__(self, board: Board, llm: LLM, language: str = "French") -> None:
        self.board = board
        self.llm = llm
        self.language = language
        self._last_line: int | None = None

    async def run_once(self) -> None:
        snap = self.board.snapshot()
        finals = [s for s in snap.transcript if s.final]
        if not finals or finals[-1].id == self._last_line:
            return
        # The subject tracked from the last lines leads the "where the discussion stands" line: notes written
        # before a change of subject otherwise keep describing the old one.
        topic = (f"CURRENT TOPIC (tracked live; the first line of the notes must describe it): {snap.signals.topic}"
                 + (f" (before: {snap.signals.previous_topic})" if snap.signals.previous_topic else "") + "\n\n"
                 if snap.signals.topic else "")
        user = (f"{topic}RESULTS OF KAIROS'S SEARCHES:{chr(10)}{render_findings(snap)}{chr(10)}{chr(10)}"
                f"PREVIOUS NOTES:\n{snap.notes or '(none yet)'}\n\n"
                f"LATEST TRANSCRIPT:\n{render_transcript(snap, last=20)}")
        # The language people speak now, never one the model guesses (English AMI notes came out in Dutch).
        spoken = " ".join(s.text for s in finals[-6:] if not is_ai(s))
        system = SYSTEM.replace("{language}", language_of(spoken, self.language) or self.language)
        result = await self.llm.json(system, user, purpose="notes", temperature=0.2)
        notes = re.sub(r"</?markdown>", "", str(result.get("notes") or "")).strip()
        if len(notes) >= 20:
            self.board.publish("notes", notes)
            self._last_line = finals[-1].id
