"""Kairos's board, seen through the Atlas interface: one `AtlasState` built from a running meeting.

Kairos keeps its own state (the board, the decisions, the briefs, the calls); this module only translates it,
at every snapshot, into the shape the Atlas frontend reads. Nothing here decides anything.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from ..agents.common import AI_NAME
from .protocol import (
    Activity,
    AgentRun,
    AtlasState,
    Decision,
    DecisionRecord,
    Health,
    NotesDocument,
    PipelineMetrics,
    PolicyDecision,
    ProcessStatus,
    ReservoirEvent,
    SessionStatus,
    Speech,
    Task,
    Thought,
    TraceEvent,
    Utterance,
)

if TYPE_CHECKING:
    from ..runtime import Runtime
    from .memory import MeetingMemory

PROTOCOL_VERSION = 11
PROJECT_ID = "kairos"

#: Which Atlas agent a Kairos model call belongs to, for the Flow and the Monitor.
AGENTS: dict[str, str] = {
    "answer": "speaker",
    "thinker": "thinkers",
    "checker": "checker",
    "cleaner": "checker",
    "topic": "topic",
    "judge": "rater",
    "rater": "rater",
    "dispatcher": "coordinator",
    "brief filler": "worker",
    "brief result": "worker",
    "search summary": "worker",
    "research plan": "anticipation",
    "context plan": "anticipation",
    "notes": "notes",
    "meeting notes": "notes",
    "board": "coordinator",  # the Board curator node of the Flow
    "archivist": "notes",
    "annotate": "naming",
}

#: A brief's status in Kairos, as an Atlas task.
TASK_STATUS: dict[str, str] = {
    "needs_details": "queued",
    "asked": "queued",
    "running": "running",
    "done": "done",
    "failed": "failed",
    "expired": "canceled",
}

LANGUAGE = {"French": "fr", "English": "en"}


def utterance_id(segment: int | str | None) -> str:
    return f"L{segment}" if segment is not None else ""


class Clock:
    """Meeting seconds as wall-clock timestamps: the meeting started at `started`."""

    def __init__(self, started: datetime) -> None:
        self.started = started

    def iso(self, t: float | None) -> str:
        return (self.started + timedelta(seconds=max(0.0, t or 0.0))).isoformat()


def project(
    rt: Runtime | None,
    *,
    session_id: str | None,
    status: SessionStatus,
    title: str,
    language: str,
    clock: Clock,
    audio_frames: int = 0,
    memory: MeetingMemory | None = None,
) -> AtlasState:
    state = AtlasState(
        project_id=PROJECT_ID,
        protocol_version=PROTOCOL_VERSION,
        process_status=ProcessStatus.READY,
        session_status=status,
        session_id=session_id,
        title=title,
        identity_name=AI_NAME,
        language=language,
        capture_mode="microphone",
    )
    if rt is None or rt.clock is None:
        return state
    snap = rt.board.snapshot()
    humans = [s for s in snap.transcript if s.final and s.speaker != AI_NAME]
    state.transcript = [
        Utterance(
            id=utterance_id(s.id),
            text=s.text,
            speaker=s.speaker or "unknown",
            language=language,
            committed_at=clock.iso(s.t_end),
        )
        for s in humans[-200:]
    ]
    partial = next((s for s in reversed(snap.transcript) if not s.final and s.speaker != AI_NAME), None)
    state.partial = partial.text if partial is not None else ""
    state.floor_busy = snap.room.speaking
    state.room_epoch = len(humans)
    state.thoughts = [_thought(t, clock) for t in snap.thoughts[-40:]]
    state.reservoir_log = [
        ReservoirEvent(thought_id=tid, event=event, by=by, why=why, utterance=text, at=clock.iso(t))
        for t, tid, event, by, why, text in rt.board.log[-150:]
    ]
    state.speeches = sorted(
        [_speech(i, clock) for i in rt.interventions[-60:]]
        + [Speech(text=text, reason="direct_address", status="finished", source_ids=[utterance_id(segment)],
                  created_at=clock.iso(t)) for t, segment, text in getattr(rt, "cues", [])[-20:]],
        key=lambda speech: speech.created_at,
    )
    state.policy_decisions = _policy(rt, clock)
    state.trace = _trace(rt, clock)
    state.tasks = [_task(b, clock) for b in rt.supervisor.briefs[-30:]]
    state.agent_runs = _agent_runs(rt, clock)
    state.decisions = _decisions(rt, clock)
    state.activities = _activities(rt, humans, clock)
    if rt.topic is not None and rt.topic.history:
        state.topic = rt.topic.history[-1][1]
        state.topic_since = clock.iso(rt.topic.history[-1][0])
        if len(rt.topic.history) > 1:
            state.previous_topic = rt.topic.history[-2][1]
    if memory is not None and memory.version > 0:
        # What people read: structured notes and the board, written by the Notes & Board agent.
        state.notes = memory.markdown(language)
        state.notes_document = memory.document
        state.notes_version = memory.version
        state.notes_cursor = memory.cursor
        state.cards = list(memory.cards)
    else:
        state.notes = snap.notes
        state.notes_document = _notes(snap.notes, [f.answer for f in snap.findings if f.status == "done"])
        state.notes_version = 0
        state.notes_cursor = 0
    state.voice_mode = "active"
    state.health = _health(rt)
    state.pipeline = PipelineMetrics(
        audio_frames=audio_frames,
        stt_turns=len(humans),
        stt_last_fragment=state.partial,
        decisions=len(rt.decisions),
        agent_runs=len(rt.llm.tracer.calls) if hasattr(rt.llm, "tracer") else 0,
    )
    state.created_at = clock.iso(0)
    state.updated_at = clock.iso(snap.room.t)
    return state


def _thought(t: Any, clock: Clock) -> Thought:
    kind = t.kind if t.kind in {"idea", "answer", "correction", "finding", "question"} else "idea"
    status = str(t.status) if str(t.status) in {"ready", "pending", "spoken", "stale"} else "stale"
    return Thought(
        id=t.id,
        kind=kind,
        topic=t.topic,
        utterance=t.utterance,
        importance=min(5.0, max(1.0, float(t.importance))),
        status=status,
        owed=t.answers is not None,
        chosen=t.chosen,
        fit=t.fit_now,
        already_said=t.already_said,
        rated_utterance_id=utterance_id(t.judged_line) if t.judged_line is not None else None,
        source_ids=list(t.stimuli),
        task_id=t.brief,
        created_ts=t.created_at,
        created_at=clock.iso(t.created_at),
        note=t.note,
    )


def _speech(intervention: Any, clock: Clock) -> Speech:
    outcome = intervention.outcome
    if outcome is None:
        status = "playing"
    elif outcome.yielded:
        status = "interrupted"
    else:
        status = "finished"
    reason = {"asked": "direct_address", "important": "reservoir_thought", "pending": "reservoir_thought"}.get(
        intervention.reason, "reservoir_thought"
    )
    primary = intervention.decision.primary or ""
    return Speech(
        text=intervention.text or intervention.planned,
        reason=reason,
        status=status,
        source_ids=[tid for tid in primary.split(",") if tid],
        created_at=clock.iso(intervention.t),
    )


def _policy(rt: Runtime, clock: Clock) -> list[PolicyDecision]:
    decisions = (getattr(rt, "decision_log", None) or sorted(rt.decisions.values(), key=lambda d: d.t))[-150:]
    return [
        PolicyDecision(
            speak=d.speak,
            reason=d.why,
            thought_id=(d.primary or "").split(",")[0] or None,
            at=clock.iso(d.t),
        )
        for d in decisions
    ]


def _trace(rt: Runtime, clock: Clock) -> list[TraceEvent]:
    """Each line from hearing to speaking: the line, Jev's reading, the briefs it opened, what was said."""
    snap = rt.board.snapshot()
    events: list[TraceEvent] = []
    lines = [s for s in snap.transcript if s.final and s.speaker != AI_NAME][-60:]
    known = {s.id for s in lines}
    for s in lines:
        events.append(
            TraceEvent(utterance_id=utterance_id(s.id), stage="heard", detail=s.text, at=clock.iso(s.t_end))
        )
    for d in rt.dispatcher.log[-80:] if rt.dispatcher else []:
        if d.segment not in known:
            continue
        probabilities = ", ".join(f"{k} {v:.2f}" for k, v in sorted(d.p.items(), key=lambda kv: -kv[1])[:4])
        events.append(
            TraceEvent(
                utterance_id=utterance_id(d.segment),
                stage="jev",
                ms=int(d.seconds * 1000),
                detail=f"{', '.join(d.actions) or 'nothing to start'} ({probabilities})",
                at=clock.iso(d.t),
            )
        )
    for b in rt.supervisor.briefs[-20:]:
        for t, what in b.history:
            if b.segment in known:
                events.append(
                    TraceEvent(utterance_id=utterance_id(b.segment), stage="worker", detail=what, at=clock.iso(t))
                )
    for i in rt.interventions[-40:]:
        segment = snap.signals.addressed_segment if i.reason == "asked" else None
        events.append(
            TraceEvent(
                utterance_id=utterance_id(segment) if segment in known else utterance_id(_line_before(lines, i.t)),
                stage="voice",
                ms=int((i.after_speech_s or 0) * 1000),
                detail=f"{'cut off' if i.outcome and i.outcome.yielded else 'said'}: {i.text or i.planned}",
                at=clock.iso(i.t),
            )
        )
    events.sort(key=lambda e: e.at)
    return events[-300:]


def _line_before(lines: list[Any], t: float) -> int | None:
    before = [s.id for s in lines if s.t_end <= t]
    return before[-1] if before else None


def _task(b: Any, clock: Clock) -> Task:
    status = TASK_STATUS.get(b.status, "queued")
    return Task(
        id=b.id,
        mission_key=b.kind,
        tool="jinko_flight_calendar" if b.kind == "flights" else "exa_search",
        summary=b.line[:300],
        status=status,
        phase="complete" if status in {"done", "failed", "canceled"} else "executing"
        if status == "running"
        else "queued",
        result={"answer": b.result, "sources": list(b.sources),
                "details": {k: v.get("value") for k, v in b.details.items()}} if b.result or b.details else None,
        error=b.error or None,
        created_at=clock.iso(b.created_at),
        completed_at=clock.iso(b.history[-1][0]) if status in {"done", "failed", "canceled"} and b.history else None,
    )


def _agent_runs(rt: Runtime, clock: Clock) -> list[AgentRun]:
    calls = getattr(getattr(rt.llm, "tracer", None), "calls", [])
    now = clock.iso(rt.board.snapshot().room.t)
    runs: list[AgentRun] = []
    for call in calls[-80:]:
        agent = AGENTS.get(call.purpose)
        if agent is None:
            continue
        runs.append(
            AgentRun(
                agent=agent,
                summary=call.purpose,
                status="done" if call.ok else "failed",
                duration_ms=int(call.seconds * 1000),
                started_at=now,
                completed_at=now,
            )
        )
    return runs


def _decisions(rt: Runtime, clock: Clock) -> list[DecisionRecord]:
    records: list[DecisionRecord] = []
    for d in rt.dispatcher.log[-40:] if rt.dispatcher else []:
        route = "investigate" if d.actions else "capture"
        records.append(
            DecisionRecord(
                utterance_id=utterance_id(d.segment),
                context={"new_utterance": d.text, "probabilities": d.p},
                result=Decision(route=route, rationale=", ".join(d.actions)),
                decided_at=clock.iso(d.t),
            )
        )
    return records


def _activities(rt: Runtime, humans: list[Any], clock: Clock) -> list[Activity]:
    """What just happened, for the strip at the bottom of the page: a line heard, a search, what Kairos said."""
    events: list[tuple[float, str, str]] = [(s.t_end, "heard", f"Heard: {s.text}") for s in humans[-10:]]
    for b in rt.supervisor.briefs[-10:]:
        events += [(t, "worker", f"Search {b.id}: {what}") for t, what in b.history[-3:]]
    for i in rt.interventions[-10:]:
        events.append((i.t, "speech", f"Kairos: {i.text or i.planned}"))
    events.sort(key=lambda event: event[0])
    return [Activity(kind=kind, summary=summary[:200], created_at=clock.iso(t)) for t, kind, summary in events[-20:]]


def _notes(text: str, findings: list[str]) -> NotesDocument:
    lines = [line.strip("-• ").strip() for line in text.splitlines() if line.strip("-• ").strip()]
    return NotesDocument(synthesis=lines[:12], findings=findings[-8:])


def _health(rt: Runtime) -> dict[str, Any]:
    health = {"generator": Health(status="ok"), "decision": Health(status="ok")}
    source, voice = rt.live_source, rt.voice
    if source is not None:
        health["stt"] = Health(status="ok" if source.connected else "down", detail=source.error or "")
    if voice is not None:
        health["tts"] = Health(status="down" if voice.error else "ok", detail=voice.error or "")
    return health


def now_utc() -> datetime:
    return datetime.now(UTC)
