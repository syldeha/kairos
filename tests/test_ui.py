"""The Atlas interface on Kairos: the state it reads, the voice it plays, the sessions it lists."""

from __future__ import annotations

import time
from typing import Any

from kairos.ui import server
from kairos.ui.project import Clock, now_utc, project
from kairos.ui.protocol import AtlasState, SessionStatus
from kairos.ui.server import END_OF_SPEECH_S, Store, Voice


def test_without_a_meeting_the_interface_still_gets_a_complete_state() -> None:
    state = project(None, session_id=None, status=SessionStatus.IDLE, title="Nouvelle session", language="fr",
                    clock=Clock(now_utc()))
    data = state.model_dump(mode="json")
    assert AtlasState.model_validate(data).session_status == SessionStatus.IDLE
    assert data["identity_name"] == "Kairos" and data["protocol_version"] == 11


class Outbox:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    def send(self, message: dict[str, Any]) -> None:
        self.sent.append(message)

    def saying(self) -> str:
        return "Le budget est de 210 euros, pas 180."


def test_each_sentence_kairos_says_is_one_speech_of_the_atlas_protocol() -> None:
    outbox = Outbox()
    voice = Voice(outbox)  # type: ignore[arg-type]
    voice.send_audio(b"\x00\x01" * 10)
    voice.send_audio(b"\x00\x01" * 10)
    kinds = [message["type"] for message in outbox.sent]
    assert kinds == ["speech.authorized", "speech.subtitle", "speech.audio.chunk", "speech.audio.chunk"]
    first = outbox.sent[0]["speech_id"]
    voice.last_audio = time.monotonic() - END_OF_SPEECH_S - 0.1
    voice.tick()
    assert [m["type"] for m in outbox.sent[-2:]] == ["speech.audio.end", "speech.subtitle"]
    voice.send_audio(b"\x00\x01")  # Kairos resumes: a new speech
    assert outbox.sent[-2]["type"] == "speech.subtitle" and outbox.sent[-3]["speech_id"] != first
    voice.stop()  # cut off: the page drops what it queued
    assert outbox.sent[-1]["type"] == "speech.stop"


def test_sessions_are_kept_listed_and_deleted(tmp_path: Any) -> None:
    store = Store(tmp_path / "sessions.db")
    state = project(None, session_id="session_1", status=SessionStatus.CLOSED, title="Samedi", language="fr",
                    clock=Clock(now_utc()))
    store.save(state)
    assert [item.title for item in store.summaries()] == ["Samedi"]
    assert store.load("session_1") is not None
    assert store.delete("session_1") and store.summaries() == []


def test_the_page_protocol_version_matches_the_backend() -> None:
    source = (server.PROJECT_ROOT / "frontend" / "src" / "protocol.ts").read_text(encoding="utf-8")
    assert f"CLIENT_PROTOCOL_VERSION = {server.PROTOCOL_VERSION};" in source


def test_a_session_cut_by_a_restart_is_listed_as_closed(tmp_path: Any) -> None:
    store = Store(tmp_path / "sessions.db")
    state = project(None, session_id="session_2", status=SessionStatus.LISTENING, title="Nouvelle session",
                    language="fr", clock=Clock(now_utc()))
    state.topic = "un voyage à Groningen"
    store.save(state)
    reopened = Store(tmp_path / "sessions.db")
    [summary] = reopened.summaries()
    assert summary.status == SessionStatus.CLOSED and summary.title == "un voyage à Groningen"


# -- Notes and Board, Atlas's method -----------------------------------------------------------------------

def test_the_board_is_reconciled_from_a_desired_state_and_merges_only_with_proof():
    from kairos.ui.memory import apply, reconcile
    from kairos.ui.protocol import BoardProposal, Card
    flight = Card(kind="finding", concept_key="flight_paris_lisbon", title="Vol", body="99 euros")
    hotel = Card(kind="finding", concept_key="hotel_lisbon", title="Hôtel", body="Novotel")
    budget = Card(kind="question", concept_key="budget", title="Budget ?", body="À fixer")
    proposal = BoardProposal.model_validate({
        "desired_cards": [
            {"kind": "finding", "concept_key": "flight_paris_lisbon", "title": "Vol Paris Lisbonne",
             "body": "Direct à 105 euros", "source_card_ids": [flight.id]},
            {"kind": "finding", "concept_key": "stay", "title": "Séjour", "body": "Vol et hôtel",
             "source_card_ids": [hotel.id, budget.id]},  # a merge with no evidence: refused
            {"kind": "decision", "concept_key": "dates", "title": "Dates", "body": "Du 16 au 19 octobre",
             "source_card_ids": []},
        ],
        "retirements": [],
    })
    cards = apply([flight, hotel, budget], reconcile([flight, hotel, budget], proposal))
    by_key = {card.concept_key: card for card in cards}
    assert by_key["flight_paris_lisbon"].body == "Direct à 105 euros" and by_key["flight_paris_lisbon"].id == flight.id
    assert {"hotel_lisbon", "budget", "dates"} <= set(by_key)  # unproven merge kept both; the new concept added


class NotesLLM:
    def __init__(self):
        self.calls = []

    async def json(self, system, user, *, purpose, temperature=0.4):
        self.calls.append(purpose)
        if purpose == "meeting notes":
            return {"synthesis": ["Trois amis préparent un week-end à Lisbonne."], "decisions": ["Partir le 16 octobre."],
                    "findings": ["Vol direct Paris Lisbonne à 105 euros (Jinko)."], "participants": ["Claude", "Léa"]}
        return {"desired_cards": [{"kind": "decision", "concept_key": "dates", "title": "Dates du week-end",
                                   "body": "Du 16 au 19 octobre", "source_card_ids": []}], "retirements": []}


def test_notes_and_board_follow_the_meeting_and_reach_the_interface():
    import asyncio as aio
    from kairos.board import Board
    from kairos.contracts import Segment
    from kairos.ui.memory import MeetingMemory

    class Runtime:
        def __init__(self):
            self.board, self.llm = Board(), NotesLLM()
            self.board.publish("transcript", tuple(Segment(i, "Claude", f"ligne {i}", i, i + 1, True) for i in range(3)))

    rt, memory = Runtime(), MeetingMemory()
    assert memory.due(rt)
    aio.run(memory.update(rt, "fr"))
    assert rt.llm.calls == ["meeting notes", "board", "board"]  # notes, board proposal, review
    assert memory.cursor == 3 and memory.version == 1 and memory.cards[0].title == "Dates du week-end"
    text = memory.markdown("fr")
    assert "## Synthèse" in text and "## Décisions" in text and "- Partir le 16 octobre." in text
    assert not memory.due(rt)  # nothing new since
