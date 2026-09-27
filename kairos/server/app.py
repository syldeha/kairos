"""The console: watch Kairos think and decide during a live replay, and take part in it.

    python -m kairos.server.app            # then open http://127.0.0.1:8765

One meeting at a time. The page receives a state snapshot over a WebSocket
several times per second; typing in the page injects your words into the
meeting at speech rate, so you can ask Kairos something or talk over it.

With the voice on (Direct only), the page streams the microphone over a second
WebSocket (/ws/audio) to Gradium's speech-to-text, and plays Kairos's voice
(Gradium text-to-speech) sent back on the same socket.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import time
import wave
from dataclasses import asdict, replace
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..agents.common import AI_NAME
from ..config import PROJECT_ROOT, Settings
from ..decide.policy import score
from ..report import agent_timings, metrics, rows
from ..runtime import RunConfig, Runtime
from ..session import (FIXTURES, default_language, default_memory, load_memory, load_timeline, make_judge, make_llm,
                       make_search, prime_ami)
from ..sources.ami import DATA_DIR
from ..sources.gradium import GradiumSource
from ..travel import JinkoFlights
from ..voice import VOICES, EchoGuard, GradiumVoice

WEB = PROJECT_ROOT / "web" / "index.html"
LIVE_SPEAKER = "Vous"
log = logging.getLogger("kairos.server")


class StartRequest(BaseModel):
    source: str
    mode: str = "closed"
    speed: float = 1.0
    judge: str = "jev"  # with the LLM judge behind it if Jev fails
    search: str = "exa"
    role: str = "discreet"
    prime: bool = True
    context: str = ""          # what the meeting is about, typed in the console
    memory: str | None = None  # what Kairos knows, one line per memory; None = the meeting's default
    anonymous: bool = False
    voice: bool = False        # Direct only: microphone and loudspeakers through Gradium
    voice_id: str = "iEu63s1rhn_kegTr"
    stt_delay_frames: int = 16  # Gradium's look-ahead, 80 ms each: lower is faster, higher is more accurate
    language: str | None = None  # "French" or "English"; None = the meeting's default


class SayRequest(BaseModel):
    text: str
    speaker: str | None = None  # another participant typing (a second person in the room); default "Vous"


class Session:
    def __init__(self) -> None:
        self.runtime: Runtime | None = None
        self.task: asyncio.Task | None = None
        self.source = ""
        self.error: str | None = None
        self.version = 0
        self._segments = itertools.count(500_000)

    async def start(self, req: StartRequest) -> None:
        await self.stop()
        settings = Settings()
        llm = make_llm(settings)
        judge = make_judge(req.judge, settings, llm)
        timeline = load_timeline(req.source, req.anonymous)
        language = req.language if req.language in ("French", "English") else default_language(req.source)
        config = RunConfig(mode=req.mode, speed=max(req.speed, 0.5), language=language, role=req.role)
        if req.memory is not None:
            memory = [line.strip() for line in req.memory.splitlines() if line.strip()]
        else:
            memory = load_memory(default_memory(req.source))
        if req.context.strip():
            memory.insert(0, f"Contexte de la réunion : {req.context.strip()}")
        notes = ""
        if req.prime and req.source.startswith("ami:"):
            extra, notes = await prime_ami(req.source, llm, config.language)
            # Memories of the previous meeting are what Kairos remembers (its memory across meetings); the summary of
            # this meeting's beginning is context only: it goes to the notes, never into what Kairos knows for sure.
            memory += extra
        live_source = voice = None
        if req.voice:
            if timeline is not None:
                raise ValueError("La voix ne fonctionne qu'en mode Direct.")
            if not settings.gradium_api_key:
                raise ValueError("GRADIUM_API_KEY manque dans .env.")
            config = replace(config, speed=1.0)  # a voice meeting happens in real time
            now = lambda: self.runtime.meeting_time() if self.runtime and self.runtime.clock else 0.0  # noqa: E731
            echo = EchoGuard(now)
            live_source = GradiumSource(settings.gradium_api_key, settings.gradium_url,
                                        language="fr" if config.language == "French" else "en",
                                        speaker=None if req.anonymous else LIVE_SPEAKER,
                                        delay_in_frames=max(7, min(48, req.stt_delay_frames)), is_echo=echo)
            voice = GradiumVoice(settings.gradium_api_key, settings.gradium_url,
                                 req.voice_id if req.voice_id in VOICES else "iEu63s1rhn_kegTr", now, echo,
                                 language=config.language)
        self.runtime = Runtime(timeline, memory, llm, judge, config, on_change=self.bump,
                               initial_notes=notes, search=make_search(req.search, settings, llm),
                               live_source=live_source, voice=voice,
                               flights=JinkoFlights(settings.jinko_api_key, settings.jinko_url)
                               if settings.jinko_api_key else None)
        self.source, self.error = req.source, None
        self.task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        try:
            await self.runtime.run()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # shown in the page
            log.exception("run failed")
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.bump()

    async def stop(self) -> None:
        if self.runtime:
            self.runtime.stop()  # a live meeting ends cleanly
            await asyncio.sleep(0.2)
        if self.task and not self.task.done():
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self.task
        self.bump()

    def bump(self) -> None:
        self.version += 1

    async def say(self, text: str, speaker: str | None = None) -> None:
        if not self.runtime or not self.task or self.task.done():
            raise HTTPException(409, "Start a meeting first.")
        asyncio.create_task(self.runtime.say_live(speaker or LIVE_SPEAKER, text, next(self._segments)))

    def state(self) -> dict:
        rt = self.runtime
        if rt is None or rt.clock is None:
            return {"running": False, "error": self.error, "source": self.source}
        snap = rt.board.snapshot()
        now = snap.room.t
        rank = {"ready": 0, "pending": 1, "spoken": 2, "stale": 3, "draft": 4}
        thoughts = sorted(snap.thoughts, key=lambda t: (rank.get(str(t.status), 5), -score(t, now, rt.config.policy)
                          if str(t.status) in ("ready", "pending") else -t.created_at))[:16]
        decisions = sorted(rt.decisions.values(), key=lambda d: d.t)[-10:]
        m = metrics(rt)
        return {
            "running": bool(self.task and not self.task.done()),
            "finished": rt.finished,
            "error": self.error,
            "source": self.source,
            "mode": rt.config.mode,
            "t": round(now, 2),
            "offset": round(rt.source.offset, 2) if rt.source else 0.0,
            "room": {"speaking": snap.room.speaking, "silence": round(snap.room.silence_s, 2),
                     "p": list(snap.room.p_silence), "last": snap.room.last_speaker},
            "ai": {"speaking": snap.ai.speaking, "text": snap.ai.text,
                   "share": round(snap.ai.total_speech_s / now, 3) if now else 0.0,
                   "interventions": snap.ai.interventions},
            "signals": asdict(snap.signals),
            "transcript": [{"id": s.id, "speaker": s.speaker or "?", "text": s.text, "t": round(s.t_start, 1),
                            "final": s.final, "ai": s.speaker == AI_NAME,
                            "live": s.speaker == LIVE_SPEAKER, "raw": rt.scribe.raw.get(s.id)}
                           for s in snap.transcript[-60:]],
            "topics": [{"t": round(t, 1), "topic": topic} for t, topic in (rt.topic.history if rt.topic else [])],
            "thoughts": [{"id": t.id, "topic": t.topic, "utterance": t.utterance, "status": str(t.status),
                          "importance": t.importance, "relevance": t.relevance, "fit": t.fit_now,
                          "said": t.already_said, "answers": t.answers, "note": t.note,
                          "kind": t.kind, "chosen": t.chosen, "value": t.value, "brief": t.brief,
                          "fresh": t.judged_line == snap.signals.judged_segment,
                          "score": round(score(t, now, rt.config.policy), 2)} for t in thoughts],
            "decisions": [{"t": round(d.t, 1), "speak": d.speak, "why": d.why} for d in decisions],
            "interventions": [asdict(r) for r in rows(rt)],
            "notes": snap.notes,
            "findings": [{"id": f.id, "question": f.question, "query": f.query, "status": f.status,
                          "answer": f.answer, "sources": list(f.sources), "seconds": f.seconds}
                         for f in snap.findings],
            "timings": agent_timings(rt),
            "reservoir_log": [{"t": round(t, 1), "id": tid, "event": e, "by": by, "why": why, "thought": u}
                              for t, tid, e, by, why, u in rt.board.log[-80:]],
            "rater": {"line": snap.signals.rater_line, "none": snap.signals.rater_none},
            "judgements": [{"t": round(t, 1), "line": line, "words": words, "scores": scores}
                           for t, line, words, scores in rt.judge_agent.history],
            "flows": self._flows(rt),
            "context": [{"t": round(t, 1), "subject": subj, "query": q} for t, subj, q in (rt.context.log if rt.context else [])],
            "briefs": [{"id": b.id, "kind": b.kind, "status": b.status, "line": b.line, "segment": b.segment,
                        "details": b.details, "missing": b.missing, "question": b.question, "result": b.result,
                        "error": b.error, "history": [[round(t, 1), h] for t, h in b.history]}
                       for b in rt.supervisor.briefs[-8:]],
            "dispatch": [{"t": round(d.t, 1), "segment": d.segment, "text": d.text[:90],
                          "p": {k: round(v, 2) for k, v in d.p.items()}, "actions": d.actions, "seconds": d.seconds}
                         for d in (rt.dispatcher.log[-10:] if rt.dispatcher else [])],
            "role": rt.config.role,
            "judge": {"name": rt.judge.name, "error": getattr(rt.judge, "last_error", None),
                      "answered": getattr(rt.judge, "answered", None)},
            "voice": self._voice_state(rt),
            "metrics": {k: m[k] for k in ("llm_calls", "cost_usd", "in_natural_gap", "mean_latency_s",
                                          "harness_refusals", "interrupted", "resumed", "web_searches")},
        }


    @staticmethod
    def _flows(rt: Runtime) -> list[dict]:
        """One timeline per request: the sentence, Jev's decision, the brief, the question, the answers, the search,
        the result in the reservoir, the moment it was said. The delay between hops shows where fluidity breaks."""
        snap = rt.board.snapshot()
        lines = {s.id: s for s in snap.transcript}
        dispatch = {d.segment: d for d in (rt.dispatcher.log if rt.dispatcher else [])}
        flows = []
        for b in rt.supervisor.briefs[-6:]:
            events = []
            line = lines.get(b.segment)
            if line is not None:
                events.append((line.t_end, "phrase", f"{line.speaker or '?'} : « {line.text[:80]} »"))
            d = dispatch.get(b.segment)
            if d is not None:
                events.append((d.t, "Jev", f"{', '.join(d.actions) or 'rien'} ({d.seconds} s)"))
            events += [(t, "demande", h) for t, h in b.history]
            for t, tid, event, by, why, text in rt.board.log:
                if tid.endswith(b.id) or tid.endswith(b.id + "n1") or f"{b.id}n" in tid:
                    events.append((t, "réservoir", f"{event} · « {text[:70]} »"))
            for seg in sorted(s for s in rt.supervisor._owned if s in lines and s != b.segment and b.last_line and s <= b.last_line
                              and lines[s].t_start >= (line.t_start if line else 0)):
                events.append((lines[seg].t_end, "réponse", f"« {lines[seg].text[:70]} »"))
            events.sort(key=lambda e: e[0])
            out, previous = [], None
            for t, kind, text in events:
                out.append({"t": round(t, 1), "dt": round(t - previous, 1) if previous is not None else None,
                            "kind": kind, "text": text})
                previous = t
            flows.append({"id": b.id, "kind": b.kind, "status": b.status, "events": out,
                          "result_said": any(tid == f"r{b.id}" and ev == "spoken" for _, tid, ev, *_ in rt.board.log)})
        return flows

    @staticmethod
    def _voice_state(rt: Runtime) -> dict | None:
        if rt.voice is None:
            return None
        src, voice = rt.live_source, rt.voice
        return {"connected": src.connected, "error": src.error or voice.error, "name": VOICES.get(voice.voice_id),
                "first_audio_s": voice.first_audio_s[-1] if voice.first_audio_s else None,
                "listening": src._origin is not None,
                "stt_lag_s": _mean(src.stt_lag_s), "commit_s": _mean(src.commit_s),
                "first_audio_mean_s": _mean(voice.first_audio_s)}


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 2) if xs else None


def sources() -> list[dict]:
    out = [{"id": "live", "label": "Direct · vous et Kairos, sans script"}]
    out += [{"id": p.name, "label": f"Script · {p.stem}"} for p in sorted(FIXTURES.glob("*.txt"), reverse=True)
            if not p.name.startswith("kairos_")]
    for meeting in sorted(d.name for d in DATA_DIR.glob("*") if d.is_dir()):
        out.append({"id": f"ami:{meeting}:540:180", "label": f"AMI {meeting} · 3 min from 9:00"})
        out.append({"id": f"ami:{meeting}:0:600", "label": f"AMI {meeting} · first 10 min"})
    return out


def create_app() -> FastAPI:
    app = FastAPI(title="Kairos console")
    session = Session()

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(WEB)

    @app.get("/api/sources")
    async def list_sources() -> list[dict]:
        return sources()

    @app.get("/api/memory")
    async def default_memory_for(source: str) -> dict:
        """The memory that goes with a meeting, to prefill the console (you can edit it before starting)."""
        return {"memory": "\n".join(load_memory(default_memory(source)))}

    @app.get("/api/state")
    async def get_state() -> dict:
        return session.state()

    @app.post("/api/start")
    async def start(req: StartRequest) -> dict:
        try:
            await session.start(req)
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            raise HTTPException(400, str(exc))
        return {"ok": True}

    @app.post("/api/stop")
    async def stop() -> dict:
        await session.stop()
        return {"ok": True}

    @app.post("/api/say")
    async def say(req: SayRequest) -> dict:
        if not req.text.strip():
            raise HTTPException(400, "Nothing to say.")
        await session.say(req.text.strip(), (req.speaker or "").strip() or None)
        return {"ok": True}

    @app.websocket("/ws/audio")
    async def audio(socket: WebSocket) -> None:
        """Microphone in (24 kHz 16-bit mono PCM, binary frames); Kairos's voice out (same format),
        and {"type": "stop"} when Kairos is cut off: the page drops the audio it has queued."""
        await socket.accept()
        rt = session.runtime
        if rt is None or rt.live_source is None:
            await socket.close(code=1011, reason="Démarrez d'abord une réunion Direct avec la voix.")
            return
        queue: asyncio.Queue = asyncio.Queue()
        stop = object()

        class Sink:
            def send_audio(self, pcm: bytes) -> None:
                queue.put_nowait(pcm)

            def stop(self) -> None:
                queue.put_nowait(stop)

        sink = Sink()
        if rt.voice is not None:
            rt.voice.sink = sink

        async def pump() -> None:
            while True:
                item = await queue.get()
                if item is stop:
                    await socket.send_json({"type": "stop"})
                else:
                    await socket.send_bytes(item)

        pumping = asyncio.create_task(pump())
        # The microphone is kept (local file) so the transcription can be replayed and tuned afterwards.
        folder = PROJECT_ROOT / "runs" / "voice"
        folder.mkdir(parents=True, exist_ok=True)
        recording = wave.open(str(folder / time.strftime("%Y%m%d-%H%M%S.wav")), "wb")
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(24_000)
        try:
            while True:
                message = await socket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes"):
                    rt.live_source.push_audio(message["bytes"])
                    recording.writeframes(message["bytes"])
                elif message.get("text") and '"barge_in"' in message["text"] and rt.speaker is not None:
                    rt.speaker.barge_in()  # the page already stopped the audio
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            recording.close()
            pumping.cancel()
            if rt.voice is not None and rt.voice.sink is sink:
                rt.voice.sink = None

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        try:
            sent = -1
            while True:
                if session.version != sent:
                    sent = session.version
                    await socket.send_json(session.state())
                await asyncio.sleep(0.12)  # at most ~8 updates per second
        except (WebSocketDisconnect, RuntimeError):
            pass

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(), host="127.0.0.1", port=8765, log_level="warning")


if __name__ == "__main__":
    main()
