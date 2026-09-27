"""Metrics from a finished run, matched against the real timeline of the meeting."""

from __future__ import annotations

from dataclasses import dataclass

from .agents.common import is_ai
from .runtime import Runtime


@dataclass(frozen=True, slots=True)
class InterventionRow:
    t: float
    reason: str
    moment: str       # "turn change" | "same speaker's pause" | "end of meeting" | "during speech"
    latency_s: float
    text: str
    interrupted: bool
    resumed: bool
    yielded: bool
    after_speech_s: float | None = None
    thought_ready_s: float | None = None
    question_to_answer_s: float | None = None


def classify_moment(runtime: Runtime, original_t: float) -> str:
    if runtime.timeline is None:
        return "live"
    for gap in runtime.timeline.gaps(min_s=0.15):
        if gap.start - 0.05 <= original_t <= gap.end:
            if gap.after is None:
                return "end of meeting"
            if gap.shift:
                return "turn change"
            return "long pause" if gap.duration >= 1.5 else "same speaker's pause"
    return "during speech"


def rows(runtime: Runtime) -> list[InterventionRow]:
    out = []
    for iv in runtime.interventions:
        o = iv.outcome
        out.append(InterventionRow(
            t=iv.t, reason=iv.reason, moment=classify_moment(runtime, iv.original_t), latency_s=iv.latency_s,
            text=iv.text or iv.planned, interrupted=bool(o and o.interruptions),
            resumed=bool(o and o.resumed), yielded=bool(o and o.yielded),
            after_speech_s=iv.after_speech_s, thought_ready_s=iv.thought_ready_s,
            question_to_answer_s=iv.question_to_answer_s))
    return out


def _extra_tracers(runtime: Runtime) -> list:
    """Tracers other than the writing model's: the judge (Jev), the web search, the flight search."""
    candidates = (getattr(runtime.judge, "tracer", None), getattr(runtime.search, "tracer", None),
                  getattr(getattr(getattr(runtime, "supervisor", None), "flights", None), "tracer", None))
    out = []
    for t in candidates:
        if t is not None and t is not runtime.llm.tracer and all(t is not o for o in out):
            out.append(t)
    return out


def agent_timings(runtime: Runtime) -> dict[str, dict]:
    """Wall-clock seconds per kind of model call: where the time goes."""
    tracers = [runtime.llm.tracer] + _extra_tracers(runtime)
    by: dict[str, list[float]] = {}
    for tracer in tracers:
        for c in tracer.calls:
            by.setdefault(c.purpose, []).append(c.seconds)
    return {k: {"n": len(v), "mean_s": round(sum(v) / len(v), 2), "last_s": round(v[-1], 2),
                "max_s": round(max(v), 2)} for k, v in by.items()}


def metrics(runtime: Runtime) -> dict:
    table = rows(runtime)
    snap = runtime.board.snapshot()
    duration = max(snap.room.t, 1e-9)
    natural = [r for r in table if r.moment in ("turn change", "end of meeting", "live", "long pause")]
    tracer = runtime.llm.tracer
    extra = _extra_tracers(runtime)
    calls = list(tracer.calls) + [c for t in extra for c in t.calls]
    findings = snap.findings
    found = [f for f in findings if f.status == "done"]
    return {
        "meeting_s": round(duration, 1),
        "human_lines": sum(1 for s in snap.transcript if s.final and not is_ai(s)),
        "interventions": len(table),
        "by_reason": {r: sum(1 for x in table if x.reason == r) for r in sorted({x.reason for x in table})},
        "in_natural_gap": f"{len(natural)}/{len(table)}" if table else "0/0",
        "in_same_speaker_pause": sum(1 for r in table if r.moment == "same speaker's pause"),
        "mean_latency_s": round(sum(r.latency_s for r in table) / len(table), 2) if table else None,
        "ai_share": round(snap.ai.total_speech_s / duration, 3),
        "interrupted": sum(1 for r in table if r.interrupted),
        "resumed": sum(1 for r in table if r.resumed),
        "harness_refusals": len(runtime.harness.refusals),
        "ruminations_dropped": runtime.thinkers.ruminations,
        "llm_calls": len(calls),
        "llm_failures": sum(1 for c in calls if not c.ok),
        "calls_by_purpose": tracer.summary(),
        # Token cost only: OpenAI also bills each web search call, which is not counted here.
        "cost_usd": round(tracer.cost_usd() + sum(t.cost_usd() for t in extra), 4),
        "cost_not_counted_for": sorted({m for t in [tracer] + extra for m in t.unpriced()}),
        "web_searches": f"{len(found)} found / {len(findings)} tried",
        "mean_search_s": round(sum(f.seconds for f in found) / len(found), 2) if found else None,
        "thoughts": {s: sum(1 for t in snap.thoughts if t.status == s) for s in
                     sorted({t.status for t in snap.thoughts})},
    }
