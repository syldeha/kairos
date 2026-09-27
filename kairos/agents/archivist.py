"""The archivist: turns a finished meeting into lines of long-term memory.

Run at the end of a meeting, never during it. Its output is what Kairos
remembers next time: decisions, who committed to what, figures, ideas raised,
questions left open.
"""

from __future__ import annotations

from ..llm import LLM

SYSTEM = """You write the long-term memory an AI assistant, Kairos, keeps from a meeting it attended.
Return one JSON object: {"memories": ["...", "..."]}.

Write at most 14 short, self-contained lines, each useful in a later meeting of the same team:
- decisions taken, with figures
- who committed to do what (use the speaker labels given)
- ideas and options raised, and whether they were accepted or rejected
- constraints and requirements stated
- questions left open
Never invent. Each line must make sense on its own, without the transcript. Write in {language}."""

NOTES_SYSTEM = """You keep the live notes of a meeting attended by an AI assistant named Kairos. Kairos joins now: write the notes of what has been said so far, from the transcript. Return one JSON object: {"notes": "the notes as plain markdown text"}.
The notes are short and specific, under 250 words: where the discussion stands, topics with claims and who made them, open questions, decisions. Never invent. Write in {language}."""


async def archive(transcript: str, llm: LLM, language: str = "English", label: str = "") -> list[str]:
    result = await llm.json(SYSTEM.replace("{language}", language), transcript, purpose="archivist", temperature=0.2)
    memories = [str(m).strip() for m in result.get("memories") or [] if str(m).strip()]
    return [f"{label}{m}" for m in memories]


async def notes_so_far(transcript: str, llm: LLM, language: str = "English") -> str:
    result = await llm.json(NOTES_SYSTEM.replace("{language}", language), transcript, purpose="notes", temperature=0.2)
    return str(result.get("notes") or "").strip()
