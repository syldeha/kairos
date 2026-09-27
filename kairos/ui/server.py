"""Kairos behind the Atlas interface.

    python -m kairos.ui.server            # then open http://127.0.0.1:8787

Kairos's runtime runs the meeting exactly as in its own console (the board, Jev, the thinkers, the judge, the
speaker that yields and resumes, the workers); this server only speaks the Atlas frontend's protocol (version 11):
the page streams the microphone and plays Kairos's voice over one WebSocket (/v1/live), and receives Kairos's
state translated into an `AtlasState` several times per second. Sessions are kept in SQLite.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import PROJECT_ROOT
from ..server.app import Session, StartRequest
from .project import PROJECT_ID, PROTOCOL_VERSION, Clock, now_utc, project
from .protocol import AtlasState, SessionStatus, SessionSummary, new_id

log = logging.getLogger("kairos.ui")

FRONTEND = Path(os.environ.get("KAIROS_FRONTEND_DIR", PROJECT_ROOT / "frontend" / "dist"))
DATABASE = Path(os.environ.get("KAIROS_DB", Path.home() / ".local" / "share" / "kairos" / "sessions.db"))
DISPLAY_NAME = "Kairos"
LANGUAGES = {"fr": "French", "en": "English"}
SAMPLE_RATE = 24_000
#: No new audio for this long: Kairos's sentence is fully sent (it plays on in the page).
END_OF_SPEECH_S = 0.45


class RenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=80)


class Store:
    """One JSON snapshot per session, as the Atlas interface lists and reopens them."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("create table if not exists kairos_sessions (session_id text primary key, payload text)")
        self._db.commit()
        self._close_interrupted()

    def _close_interrupted(self) -> None:
        """A session still marked live when the server starts was cut by a restart: it is over."""
        for (payload,) in self._db.execute("select payload from kairos_sessions").fetchall():
            state = AtlasState.model_validate_json(payload)
            if state.session_status in {SessionStatus.LISTENING, SessionStatus.STARTING, SessionStatus.FINALIZING}:
                state.session_status = SessionStatus.CLOSED
                if state.title == "Nouvelle session":
                    state.title = (state.topic or (state.transcript[0].text if state.transcript else "Session"))[:80]
                self.save(state)

    def save(self, state: AtlasState) -> None:
        if state.session_id is None:
            return
        self._db.execute(
            "insert or replace into kairos_sessions (session_id, payload) values (?, ?)",
            (state.session_id, state.model_dump_json()),
        )
        self._db.commit()

    def load(self, session_id: str) -> AtlasState | None:
        row = self._db.execute("select payload from kairos_sessions where session_id = ?", (session_id,)).fetchone()
        return AtlasState.model_validate_json(row[0]) if row else None

    def delete(self, session_id: str) -> bool:
        cursor = self._db.execute("delete from kairos_sessions where session_id = ?", (session_id,))
        self._db.commit()
        return cursor.rowcount > 0

    def summaries(self) -> list[SessionSummary]:
        out: list[SessionSummary] = []
        for (payload,) in self._db.execute("select payload from kairos_sessions"):
            state = AtlasState.model_validate_json(payload)
            out.append(_summary(state))
        return sorted(out, key=lambda item: item.updated_at, reverse=True)


def _summary(state: AtlasState) -> SessionSummary:
    return SessionSummary(
        session_id=state.session_id or "",
        title=state.title,
        status=state.session_status,
        preview=state.transcript[-1].text[:120] if state.transcript else "",
        utterance_count=len(state.transcript),
        created_at=state.created_at,
        updated_at=state.updated_at,
    )


class Voice:
    """Kairos's voice as the Atlas protocol's speeches: one per sentence Kairos says."""

    def __init__(self, bridge: Bridge) -> None:
        self.bridge = bridge
        self.speech_id: str | None = None
        self.sequence = 0
        self.last_audio = 0.0

    def send_audio(self, pcm: bytes) -> None:
        if self.speech_id is None:
            self.speech_id = new_id("say")
            self.sequence = 0
            text = self.bridge.saying()
            self.bridge.send(
                {"type": "speech.authorized", "speech_id": self.speech_id, "text": text,
                 "reason": "direct_address", "audio": None}
            )
            if text:
                self.bridge.send(
                    {"type": "speech.subtitle", "speech_id": self.speech_id, "text": text, "segment_index": 0,
                     "start_s": 0.0, "stop_s": 0.0, "final": False}
                )
        self.bridge.send(
            {"type": "speech.audio.chunk", "speech_id": self.speech_id, "sequence": self.sequence,
             "data_base64": base64.b64encode(pcm).decode(), "format": "pcm_24000", "sample_rate": SAMPLE_RATE}
        )
        self.sequence += 1
        self.last_audio = time.monotonic()

    def stop(self) -> None:
        """Kairos was cut off: the page drops the audio it has queued."""
        if self.speech_id is not None:
            self.bridge.send({"type": "speech.stop", "speech_id": self.speech_id, "reason": "yielded"})
        self.speech_id = None

    def tick(self) -> None:
        if self.speech_id is not None and time.monotonic() - self.last_audio > END_OF_SPEECH_S:
            self.bridge.send({"type": "speech.audio.end", "speech_id": self.speech_id})
            self.bridge.send({"type": "speech.subtitle", "speech_id": self.speech_id, "text": "", "final": True})
            self.speech_id = None


class Bridge:
    """One meeting at a time, as in Kairos's console; every connected page sees it."""

    def __init__(self, store: Store) -> None:
        self.store = store
        self.session = Session()
        self.session_id: str | None = None
        self.status = SessionStatus.IDLE
        self.title = "Nouvelle session"
        self.language = "fr"
        self.clock = Clock(now_utc())
        self.sockets: set[WebSocket] = set()
        self.outbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.voice = Voice(self)
        self.audio_frames = 0
        self.viewing: AtlasState | None = None  # a past session reopened from the library
        self._saved = 0.0

    # -- state --------------------------------------------------------------------------------------------

    def state(self) -> AtlasState:
        if self.viewing is not None and self.session.runtime is None:
            return self.viewing
        state = project(
            self.session.runtime, session_id=self.session_id, status=self.status, title=self.title,
            language=self.language, clock=self.clock, audio_frames=self.audio_frames,
        )
        if self.session.runtime is not None and self.session.task is not None and self.session.task.done():
            state.session_status = SessionStatus.CLOSED
        return state

    def sessions(self) -> list[SessionSummary]:
        summaries = {item.session_id: item for item in self.store.summaries()}
        if self.session_id is not None and self.session.runtime is not None:
            summaries[self.session_id] = _summary(self.state())
        return sorted(summaries.values(), key=lambda item: item.updated_at, reverse=True)

    def snapshot(self) -> dict[str, Any]:
        return {
            "type": "state.snapshot",
            "state": self.state().model_dump(mode="json"),
            "sessions": [item.model_dump(mode="json") for item in self.sessions()],
        }

    def saying(self) -> str:
        rt = self.session.runtime
        if rt is None or not rt.interventions:
            return ""
        return rt.interventions[-1].planned

    def send(self, message: dict[str, Any]) -> None:
        self.outbox.put_nowait(message)

    # -- commands -----------------------------------------------------------------------------------------

    async def start(self, language: str, *, resume: AtlasState | None = None) -> None:
        """A new meeting; resuming a past one, Kairos starts with what its notes remember."""
        await self.stop()
        memory = None
        if resume is not None:
            remembered = [line for line in resume.notes.splitlines() if line.strip()]
            memory = "\n".join([f"Contexte de la réunion : {resume.title}", *remembered])
        self.viewing = None
        self.language = language if language in LANGUAGES else "fr"
        self.session_id = new_id("session")
        self.title = "Nouvelle session"
        self.clock = Clock(now_utc())
        self.status = SessionStatus.STARTING
        await self.session.start(
            StartRequest(source="live", mode="open", judge="jev", search="exa", role="discreet", prime=False,
                         voice=True, stt_delay_frames=16, language=LANGUAGES[self.language], memory=memory)
        )
        if self.session.runtime is not None and self.session.runtime.voice is not None:
            self.session.runtime.voice.sink = self.voice
        self.status = SessionStatus.LISTENING

    async def stop(self) -> None:
        if self.session.runtime is None:
            return
        self.status = SessionStatus.FINALIZING
        await self.session.stop()
        self.status = SessionStatus.CLOSED
        final = self.state()
        final.session_status = SessionStatus.CLOSED
        self.title = self._title(final)
        final.title = self.title
        self.store.save(final)
        self.viewing = final
        self.session.runtime = None

    def open(self, session_id: str) -> None:
        stored = self.store.load(session_id)
        if stored is not None and self.session.runtime is None:
            self.viewing = stored

    async def say(self, text: str) -> None:
        if self.session.runtime is not None:
            await self.session.say(text)

    def barge_in(self) -> None:
        rt = self.session.runtime
        if rt is not None and rt.speaker is not None:
            rt.speaker.barge_in()  # the page already stopped the audio; Kairos resumes or yields

    def push_audio(self, pcm: bytes) -> None:
        rt = self.session.runtime
        if rt is not None and rt.live_source is not None:
            rt.live_source.push_audio(pcm)
            self.audio_frames += 1

    def _title(self, state: AtlasState) -> str:
        if self.title != "Nouvelle session":
            return self.title
        return (state.topic or (state.transcript[0].text if state.transcript else "Session"))[:80]

    # -- loops --------------------------------------------------------------------------------------------

    async def broadcast_loop(self) -> None:
        """Voice messages as they come; the state at most ~8 times per second; a saved copy every 10 s."""
        last_version, last_state, last_partial = -1, 0.0, ""
        while True:
            try:
                message = await asyncio.wait_for(self.outbox.get(), timeout=0.05)
                await self._broadcast(message)
                continue
            except TimeoutError:
                pass
            self.voice.tick()
            now = time.monotonic()
            if self.session.version != last_version and now - last_state >= 0.12:
                last_version, last_state = self.session.version, now
                snapshot = self.snapshot()
                partial = snapshot["state"].get("partial") or ""
                if partial != last_partial:
                    # What is being said right now, word by word (the transcript strip and the live line).
                    last_partial = partial
                    await self._broadcast({"type": "transcript.partial", "text": partial})
                await self._broadcast(snapshot)
            if self.session.runtime is not None and now - self._saved > 10:
                self._saved = now
                with contextlib.suppress(Exception):
                    self.store.save(self.state())

    async def _broadcast(self, message: dict[str, Any]) -> None:
        for socket in list(self.sockets):
            try:
                await socket.send_json(message)
            except Exception:
                self.sockets.discard(socket)


def create_app(store: Store | None = None) -> FastAPI:
    bridge = Bridge(store or Store(DATABASE))
    app = FastAPI(title=DISPLAY_NAME)
    app.state.bridge = bridge

    @app.on_event("startup")
    async def startup() -> None:
        app.state.loop_task = asyncio.create_task(bridge.broadcast_loop())

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await bridge.stop()
        app.state.loop_task.cancel()

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"status": "ok", "session": bridge.status}

    @app.get("/v1/bootstrap")
    async def bootstrap() -> dict[str, object]:
        return {
            "project_id": PROJECT_ID,
            "display_name": DISPLAY_NAME,
            "companion_name": DISPLAY_NAME,
            "protocol_version": PROTOCOL_VERSION,
            "active_model": "kairos",
            "models": ["kairos"],
            "state": bridge.state().model_dump(mode="json"),
            "sessions": [item.model_dump(mode="json") for item in bridge.sessions()],
        }

    @app.patch("/v1/sessions/{session_id}")
    async def rename(session_id: str, request: RenameRequest) -> dict[str, bool]:
        if session_id == bridge.session_id:
            bridge.title = request.title
            if bridge.viewing is not None:
                bridge.viewing.title = request.title
        stored = bridge.store.load(session_id)
        if stored is None and session_id != bridge.session_id:
            raise HTTPException(404, "Session not found")
        if stored is not None:
            stored.title = request.title
            bridge.store.save(stored)
        bridge.session.bump()
        return {"updated": True}

    @app.delete("/v1/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete(session_id: str) -> Response:
        if session_id == bridge.session_id and bridge.session.runtime is not None:
            raise HTTPException(409, "Stop the session first")
        if not bridge.store.delete(session_id):
            raise HTTPException(404, "Session not found")
        if bridge.viewing is not None and bridge.viewing.session_id == session_id:
            bridge.viewing = None
        bridge.session.bump()
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get("/v1/sessions/{session_id}/export")
    async def export(session_id: str) -> Response:
        state = bridge.state() if session_id == bridge.session_id else bridge.store.load(session_id)
        if state is None:
            raise HTTPException(404, "Session not found")
        lines = [f"# {state.title}", "", "## Transcript", ""]
        lines += [f"- **{u.speaker}**: {u.text}" for u in state.transcript]
        lines += ["", f"## Said by {DISPLAY_NAME}", ""] + [f"- {s.text}" for s in state.speeches]
        lines += ["", "## Notes", "", state.notes or "(none)"]
        return Response(
            content="\n".join(lines), media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{session_id}.md"'},
        )

    @app.websocket("/v1/live")
    async def live(socket: WebSocket) -> None:
        await socket.accept()
        bridge.sockets.add(socket)
        await socket.send_json({"type": "backend.hello", "project_id": PROJECT_ID, "protocol_version": PROTOCOL_VERSION})
        await socket.send_json(bridge.snapshot())
        try:
            while True:
                frame = await socket.receive()
                if frame.get("type") == "websocket.disconnect":
                    break
                if frame.get("bytes") is not None:
                    bridge.push_audio(frame["bytes"])
                    continue
                text = frame.get("text")
                if isinstance(text, str):
                    await handle(bridge, socket, text)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            bridge.sockets.discard(socket)

    if FRONTEND.is_dir() and (FRONTEND / "index.html").is_file():
        app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
    return app


async def handle(bridge: Bridge, socket: WebSocket, text: str) -> None:
    try:
        message = json.loads(text)
    except json.JSONDecodeError:
        return
    kind = message.get("type")
    try:
        if kind == "client.hello" and message.get("protocol_version") != PROTOCOL_VERSION:
            await socket.send_json({"type": "protocol.error", "code": "PROTOCOL_MISMATCH",
                                    "message": f"Backend protocol {PROTOCOL_VERSION}. Reload the page."})
        elif kind == "session.start":
            await bridge.start(str(message.get("language") or bridge.language))
        elif kind == "session.command" and message.get("command") in {"stop", "pause"}:
            await bridge.stop()
        elif kind == "session.command" and message.get("command") == "resume":
            await bridge.start(bridge.language, resume=bridge.viewing)
        elif kind == "session.open":
            bridge.open(str(message.get("session_id", "")))
        elif kind == "session.language":
            bridge.language = str(message.get("language") or bridge.language)
        elif kind == "transcript.inject":
            await bridge.say(str(message.get("text", "")).strip())
        elif kind == "playback.interrupted":
            bridge.barge_in()
        # floor.changed, playback.started/finished, voice.mode, capture.changed, board.curate: Kairos hears the
        # room through its own speech detection and tracks its own audio; nothing to do.
    except Exception as error:  # shown in the page
        log.exception("command failed")
        await socket.send_json({"type": "protocol.error", "code": "COMMAND_FAILED",
                                "message": f"{type(error).__name__}: {error}"})
    bridge.session.bump()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(), host="127.0.0.1", port=int(os.environ.get("KAIROS_PORT", "8787")),
                log_level="warning")


if __name__ == "__main__":
    main()
