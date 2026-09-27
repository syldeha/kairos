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
