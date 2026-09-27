"""Notes and Board for the Atlas interface, with Atlas's method (github.com/KpihX/atlas, core/coordinator.py).

Kairos's own notes are a short working memory for its agents. This agent keeps what people read:
1. structured shared memory (synthesis, participants, topics, findings, ideas, hypotheses, questions, decisions,
   recommendations, commitments, current work), updated from the new lines and what Kairos found;
2. the board as a *desired state*: the model rebuilds the complete board it wants (one durable thesis per card,
   a stable concept_key, the existing cards each one replaces, evidence for every merge, explicit retirements),
3. a review of that proposal against fragmentation and destructive merges,
4. a reconciliation in code: create, update, merge only with proof of equivalence, retire only when asked;
   a card the proposal leaves out survives.
It only reads Kairos's board; it never speaks.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from ..agents.common import AI_NAME
from .protocol import BoardOperation, BoardProposal, Card, NotesDocument, now_iso

if TYPE_CHECKING:
    from ..runtime import Runtime

#: Rewrite after this many new lines, or a new finding, and never more often than every MIN_GAP_S seconds.
EVERY_LINES = 3
MIN_GAP_S = 15.0

NOTES_SYSTEM = (
    "You are the note-keeper of Kairos, an AI assistant attending a live conversation. Update the structured "
    "shared memory. Return one JSON object only, with keys: synthesis, participants, topics, findings, ideas, "
    "hypotheses, questions, decisions, recommendations, commitments, current_work, source_ids. Every value is an "
    "array of strings. Synthesis contains at most four short thematic paragraphs and captures the current "
    "understanding, not chronology. Do not include Kairos introductions, capability answers, note-taking, or "
    "requests merely asking Kairos to speak. Participants contains only unique, confirmed human attendees "
    "established by the conversation as actual speakers; never list Kairos, unknown speakers, transcription "
    "artifacts, or merely mentioned people. Topics contains durable subjects, not fragments. Hypotheses contains "
    "uncertain claims. Findings contains evidence-backed results (what Kairos's searches found, with figures and "
    "source). Ideas contains proposals raised in the room. Questions contains only substantive unresolved "
    "questions. Decisions contains accepted choices. Recommendations contains suggested next steps that nobody "
    "has committed to. Commitments contains only explicit participant agreements, with owner when known; never "
    "convert Kairos's advice into a commitment. Current_work contains what Kairos is still searching. Do not copy "
    "one item into multiple sections. Merge repetition, replace obsolete items, preserve important facts unless "
    "superseded, and never invent. The transcript comes from live speech recognition: understand misheard words by "
    "context and never record a transcription artifact as a fact. Keep source IDs only in source_ids. Write in "
    "{language}."
)

BOARD_SYSTEM = (
    "You curate the live board of a conversation Kairos, an AI assistant, attends. Reconstruct the complete "
    "desired live board from shared memory and existing cards. Return desired_cards and explicit retirements, "
    "never incremental operations. Card count is not an optimization target: neither maximize nor minimize it. "
    "Each desired card is exactly one durable thesis: one finding, idea, question, decision, or suggestion. "
    "Preserve genuinely independent concepts even when they concern the same project, depend on each other, or "
    "support a common goal. Combine cards only when resolving either card would necessarily and fully resolve the "
    "other because they are mutually substitutable expressions of the same thesis. Relatedness, shared entities, "
    "causal links, implementation dependencies, or one concept operationalizing another never establish "
    "equivalence. For every multi-source card, provide merge_evidence answering whether the sources have the same "
    "resolution criterion, are mutually substitutable, and whether independent value would be lost. Omitted "
    "existing cards survive unchanged. Retire a card explicitly only when it is contradicted, unsupported, "
    "completed with no continuing relevance, or fully superseded by stronger evidence. For every desired card, "
    "source_card_ids lists every existing card it replaces; use an empty list only for a materially new concept. "
    "Every existing card ID must appear in at most one desired card. Keep stable concept_key values when the "
    "thesis survives. Titles are specific and bodies are concise current syntheses. Create a card only for a "
    "meaningful durable concept that can independently be discussed, answered, accepted, rejected, or acted on. "
    "Do not turn transcription uncertainty, incidental names, generic process advice, or inferred coordination "
    "duties into cards. When evidence is insufficient, preserve existing cards and create nothing. Never invent "
    "facts. Write titles and bodies in {language}. Return JSON matching the schema given."
)

REVIEW = (
    "Review the proposed desired board against the original cards and shared memory. Return a complete final "
    "object with desired_cards and explicit retirements, not commentary. Card count is neutral. Reject both "
    "fragmentation and destructive compression. Preserve separate cards whenever they can change, be answered, be "
    "accepted, be rejected, or be acted on independently. A clarification normally updates one card; it does not "
    "justify merging adjacent questions, risks, ideas, or actions. A merge is valid only for true semantic "
    "equivalence: the same resolution criterion, mutual substitutability, and no independent value lost. Every "
    "multi-source card must include merge_evidence that proves all three conditions; otherwise preserve its "
    "sources separately. Omission never deletes an existing card. Use retirements only with an explicit "
    "evidence-based reason. Every existing source card ID may appear in at most one final card. A final card may "
    "have empty source_card_ids only when no existing card covers that independently valuable thesis. Preserve "
    "full concept coverage."
)

SCHEMA = {
    "desired_cards": [{
        "kind": "idea|question|decision|suggestion|finding",
        "concept_key": "stable semantic identity for this one thesis",
        "title": "durable title",
        "body": "concise current synthesis",
        "source_card_ids": ["all existing card ids represented by this card"],
        "merge_evidence": {"same_resolution": True, "mutually_substitutable": True,
                           "loses_independent_value": False,
                           "rationale": "required only when combining multiple source cards"},
    }],
    "retirements": [{"card_id": "explicitly obsolete existing card id",
                     "reason": "why this concept no longer belongs on the live board"}],
}

SECTIONS = ("synthesis", "decisions", "commitments", "questions", "findings", "ideas", "recommendations",
            "hypotheses", "current_work", "topics", "participants")
HEADINGS = {
    "fr": {"synthesis": "Synthèse", "decisions": "Décisions", "commitments": "Engagements",
           "questions": "Questions ouvertes", "findings": "Informations trouvées", "ideas": "Idées",
           "recommendations": "Recommandations", "hypotheses": "Hypothèses", "current_work": "En cours",
           "topics": "Sujets", "participants": "Participants"},
    "en": {"synthesis": "Summary", "decisions": "Decisions", "commitments": "Commitments",
           "questions": "Open questions", "findings": "Findings", "ideas": "Ideas",
           "recommendations": "Recommendations", "hypotheses": "Hypotheses", "current_work": "In progress",
           "topics": "Topics", "participants": "Participants"},
}


class MeetingMemory:
    def __init__(self) -> None:
        self.document = NotesDocument()
        self.cards: list[Card] = []
        self.version = 0
        self.cursor = 0  # human lines integrated into the notes
        self._findings = 0
        self._last = 0.0
        self._running = False

    def due(self, rt: Runtime) -> bool:
        if self._running or time.monotonic() - self._last < MIN_GAP_S:
            return False
        snap = rt.board.snapshot()
        lines = sum(1 for s in snap.transcript if s.final and s.speaker != AI_NAME)
        findings = sum(1 for f in snap.findings if f.status == "done")
        return lines - self.cursor >= EVERY_LINES or (findings > self._findings and lines > 0)

    async def update(self, rt: Runtime, language: str) -> None:
        self._running = True
        try:
            snap = rt.board.snapshot()
            lines = [s for s in snap.transcript if s.final]
            humans = sum(1 for s in lines if s.speaker != AI_NAME)
            findings = [f for f in snap.findings if f.status == "done"]
            name = "French" if language == "fr" else "English"
            await self._notes(rt, lines, findings, name)
            await self._board(rt, lines, findings, name)
            self.version += 1
            self.cursor = humans
            self._findings = len(findings)
        finally:
            self._last = time.monotonic()
            self._running = False

    # -- 1. structured shared memory -------------------------------------------------------------------------

    async def _notes(self, rt: Runtime, lines: list[Any], findings: list[Any], language: str) -> None:
        payload = {
            "current_memory": self.document.model_dump(mode="json"),
            "transcript": [{"id": f"L{s.id}", "speaker": s.speaker or "Vous", "text": s.text} for s in lines[-40:]],
            "kairos_findings": [{"question": f.question, "answer": f.answer[:500]} for f in findings[-10:]],
        }
        raw = await rt.llm.json(NOTES_SYSTEM.format(language=language), json.dumps(payload, ensure_ascii=False),
                                purpose="meeting notes", temperature=0.2)
        fields = {key: _strings(raw.get(key)) for key in NotesDocument.model_fields}
        self.document = NotesDocument(**fields)

    # -- 2-4. the board as a desired state, reviewed, reconciled in code --------------------------------------

    async def _board(self, rt: Runtime, lines: list[Any], findings: list[Any], language: str) -> None:
        payload = json.dumps({
            "cards": [card.model_dump(mode="json") for card in self.cards],
            "shared_memory": self.document.model_dump(mode="json"),
            "recent_transcript": [{"id": f"L{s.id}", "speaker": s.speaker or "Vous", "text": s.text}
                                  for s in lines[-20:]],
            "completed_searches": [{"question": f.question, "answer": f.answer[:400]} for f in findings[-8:]],
            "schema": SCHEMA,
        }, ensure_ascii=False)
        system = BOARD_SYSTEM.format(language=language)
        proposal = _proposal(await rt.llm.json(system, payload, purpose="board", temperature=0.2))
        if proposal is None:
            return
        reviewed = _proposal(await rt.llm.json(
            system + "\n\n" + REVIEW,
            payload + "\n\nProposed desired board:\n" + proposal.model_dump_json(),
            purpose="board", temperature=0.1))
        self.cards = apply(self.cards, reconcile(self.cards, reviewed or proposal))

    def markdown(self, language: str) -> str:
        """The notes as the Notes view shows them."""
        headings = HEADINGS.get(language, HEADINGS["en"])
        parts = []
        for key in SECTIONS:
            items = getattr(self.document, key)
            if not items:
                continue
            body = "\n\n".join(items) if key == "synthesis" else "\n".join(f"- {item}" for item in items)
            parts.append(f"## {headings[key]}\n\n{body}")
        return "\n\n".join(parts)


def _proposal(raw: dict[str, Any]) -> BoardProposal | None:
    """A valid proposal: no duplicate concept keys, no existing card claimed twice. Else None."""
    try:
        proposal = BoardProposal.model_validate(raw)
    except ValidationError:
        return None
    keys = [concept.concept_key for concept in proposal.desired_cards]
    claimed = [card_id for concept in proposal.desired_cards for card_id in concept.source_card_ids]
    if len(keys) != len(set(keys)) or len(claimed) != len(set(claimed)):
        return None
    return proposal


def reconcile(existing: list[Card], proposal: BoardProposal) -> list[BoardOperation]:
    """Atlas's reconciliation: the desired board becomes create, update, merge (proven) and delete operations."""
    by_id = {card.id: card for card in existing}
    claimed: set[str] = set()
    operations: list[BoardOperation] = []
    for concept in proposal.desired_cards:
        sources = list(dict.fromkeys(i for i in concept.source_card_ids if i in by_id and i not in claimed))
        if len(sources) > 1 and (concept.merge_evidence is None or not concept.merge_evidence.proves_equivalence):
            continue  # a merge without proof of equivalence: the sources stay as they are
        if not sources:
            match = next((c.id for c in existing if c.id not in claimed and c.concept_key == concept.concept_key), None)
            if match is not None:
                sources = [match]
        if not sources:
            operations.append(BoardOperation(action="create", kind=concept.kind, concept_key=concept.concept_key,
                                             title=concept.title, body=concept.body, reason="new durable concept"))
            continue
        claimed.update(sources)
        target = by_id[sources[0]]
        if len(sources) > 1:
            operations.append(BoardOperation(action="merge", card_id=target.id, merge_ids=sources[1:],
                                             kind=concept.kind, concept_key=concept.concept_key, title=concept.title,
                                             body=concept.body, reason="semantically equivalent cards"))
        elif (target.kind, target.concept_key, target.title, target.body) != (
                concept.kind, concept.concept_key, concept.title, concept.body):
            operations.append(BoardOperation(action="update", card_id=target.id, kind=concept.kind,
                                             concept_key=concept.concept_key, title=concept.title, body=concept.body,
                                             reason="refines the existing concept"))
    for retirement in proposal.retirements:
        if retirement.card_id in by_id and retirement.card_id not in claimed:
            operations.append(BoardOperation(action="delete", card_id=retirement.card_id, reason=retirement.reason))
    return operations


def apply(cards: list[Card], operations: list[BoardOperation]) -> list[Card]:
    result = {card.id: card for card in cards}
    for op in operations:
        fields = {"kind": op.kind, "concept_key": op.concept_key, "title": op.title, "body": op.body,
                  "updated_at": now_iso()}
        if op.action == "create":
            card = Card(**{key: value for key, value in fields.items() if key != "updated_at"})
            result[card.id] = card
        elif op.action in {"update", "merge"} and op.card_id in result:
            result[op.card_id] = result[op.card_id].model_copy(update=fields)
            for merged in op.merge_ids:
                result.pop(merged, None)
        elif op.action == "delete":
            result.pop(op.card_id or "", None)
    return list(result.values())


def _strings(value: object) -> list[str]:
    return [str(item).strip() for item in value if str(item).strip()][:8] if isinstance(value, list) else []
