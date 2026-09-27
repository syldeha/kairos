"""Is a line addressed to Kairos? Decided in code first, confirmed by the judge.

    strong  the line names Kairos, or it is a question and only one human is in the meeting:
            the fast lane starts at once (answer, research if needed)
    weak    a question with "tu" / "vous" / "you" in a group: it may be aimed at Kairos or at a
            colleague, so only the judge's confirmation opens the fast lane
    None    not addressed
"""

from __future__ import annotations

import re

#: The assistant's name, with the spellings a speech recogniser is likely to produce.
NAME = re.compile(r"\b(kairos|kaïros|cairos|kairo|kyros|chiros|kéros|keros|kiros)\b", re.IGNORECASE)

QUESTION_STARTS = (
    # French
    "est-ce", "est ce", "tu sais", "sais-tu", "tu peux", "peux-tu", "tu pourrais", "pourrais-tu", "tu connais",
    "connais-tu", "tu penses", "penses-tu", "vous savez", "savez-vous", "vous pouvez", "pouvez-vous",
    "combien", "quand", "où", "comment", "pourquoi", "quel ", "quelle", "quels", "qui ", "qu'est", "y a-t-il",
    "c'est quoi", "c'est combien",
    # English
    "do you", "can you", "could you", "would you", "what", "when", "where", "how", "why", "which", "who ",
    "is there", "are there", "does ", "is it",
)
SECOND_PERSON = re.compile(r"\b(tu|toi|te|vous|you|your)\b|\bt'", re.IGNORECASE)


def is_question(text: str) -> bool:
    t = text.strip().lower()
    return t.endswith("?") or t.startswith(QUESTION_STARTS)


def names_someone_else(text: str, others: set[str]) -> bool:
    """ "Hugo, tu avais une idée ?" is for Hugo, not for Kairos."""
    words = set(re.findall(r"[\w'-]+", text.lower()))
    return any(name.lower() in words for name in others if name)


def address_strength(text: str, humans: int, others: set[str] = frozenset()) -> str | None:
    if NAME.search(text):
        return "strong"
    if not is_question(text) or names_someone_else(text, others):
        return None
    if humans <= 1:
        return "strong"  # alone with Kairos: every question is for it
    if SECOND_PERSON.search(text):
        return "weak"
    return None
