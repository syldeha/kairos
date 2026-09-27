"""Run Kairos on a meeting from the command line, then print what happened.

    python -m kairos.run fixtures/reunion_produit.txt                   # French script, closed loop, instant
    python -m kairos.run ami:ES2002b:540:180                            # AMI excerpt: start 540 s, 180 s long
    python -m kairos.run ami:ES2002b --mode offline                     # decisions only, vs real turn changes
    python -m kairos.run fixtures/reunion_produit.txt --speed 1         # real time, agents in the background
    python -m kairos.run fixtures/reunion_produit.txt --listen          # transcript only, no model calls

A JSON report of every run is saved in runs/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .agents.common import is_ai
from .agents.scribe import Scribe
from .board import Board
from .clock import RealClock, SimClock
from .config import PROJECT_ROOT, Settings
from .contracts import SpeechFinal, SpeechPartial, VadStep
from .report import metrics, rows
from .runtime import MODES, RunConfig, Runtime
from .session import (default_language, default_memory, load_memory, load_timeline, make_judge, make_llm, make_search,
                      prime_ami)
from .sources.replay import ReplaySource

RUNS_DIR = PROJECT_ROOT / "runs"


async def run(args) -> dict:
    timeline = load_timeline(args.source, args.anonymous)
    settings = Settings()
    llm = make_llm(settings)
    judge = make_judge(args.judge, settings, llm)
    config = RunConfig(mode=args.mode, speed=args.speed, language=args.language or default_language(args.source))
    memory = load_memory(args.memory or default_memory(args.source))
    notes = ""
    if args.prime and args.source.startswith("ami:"):
        extra, notes = await prime_ami(args.source, llm, config.language)
        # Memories of the previous meeting are what Kairos remembers (its memory across meetings); the summary of
        # this meeting's beginning is context only: it goes to the notes, never into what Kairos knows for sure.
        memory += extra
        print("── Kairos's memory ───────────────────────────────────")
        for line in memory:
            print(f"  - {line}")
        print("── Notes prepared before joining ─────────────────────")
        print("  " + notes.replace("\n", "\n  "))
    total = max((e for _, e, _ in timeline.speech), default=0.0)
    last_shown = {"t": -10.0}

    def progress() -> None:
        snap = runtime.board.snapshot()
        if snap.room.t - last_shown["t"] >= 10.0:  # one line per 10 s of meeting
            last_shown["t"] = snap.room.t
            print(f"  … {snap.room.t:5.0f}/{total:.0f} s de réunion · {runtime.human_lines()} phrases · "
                  f"{len(runtime.interventions)} interventions · {llm.tracer.cost_usd():.3f} $", flush=True)

    runtime = Runtime(timeline, memory, llm, judge, config, on_change=progress, initial_notes=notes,
                      search=make_search(args.search, settings, llm))
    started = time.perf_counter()
    await runtime.run()
    elapsed = time.perf_counter() - started

    snap = runtime.board.snapshot()
    print("\n── Transcript ─────────────────────────────────────────")
    for s in snap.transcript:
        if not s.final:
            continue
        prefix = ">>" if is_ai(s) else "  "
        print(f"{prefix} {s.t_start:6.1f}s {s.speaker or '?'}: {s.text}")

    print("\n── Interventions ──────────────────────────────────────")
    table = rows(runtime)
    if not table:
        print("  (Kairos stayed silent)")
    for r in table:
        flags = " interrupted" if r.interrupted else ""
        flags += " · resumed" if r.resumed else ""
        flags += " · yielded" if r.yielded else ""
        print(f"  {r.t:6.1f}s [{r.reason}] {r.moment}, after {r.latency_s:.2f}s of silence{flags}")
        print(f"          « {r.text} »")

    report = metrics(runtime)
    report["wall_clock_s"] = round(elapsed, 1)
    print("\n── Metrics ────────────────────────────────────────────")
    for key, value in report.items():
        print(f"  {key}: {value}")

    RUNS_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = RUNS_DIR / f"{stamp}-{args.mode}.json"
    out.write_text(json.dumps({
        "args": vars(args), "metrics": report, "interventions": [asdict(r) for r in table],
        "decisions": [asdict(d) for d in runtime.decisions.values()],
        "thoughts": [dict(asdict(t), status=str(t.status)) for t in snap.thoughts],
        "notes": snap.notes,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport saved: {out.relative_to(PROJECT_ROOT)}")
    return report


async def listen(args) -> None:
    """Milestone J1 view: the transcript as it arrives, no model involved."""
    timeline = load_timeline(args.source, args.anonymous)
    clock = SimClock() if args.speed == 0 else RealClock(args.speed)
    board = Board()
    scribe = Scribe(board)
    width = 0
    async for event in ReplaySource(timeline.events, clock).events():
        scribe.on_event(event)
        if isinstance(event, SpeechPartial) and args.speed > 0:
            text = f"{event.t_start:6.1f}s {event.speaker or '?'}: {event.text}…"
            sys.stdout.write("\r" + text.ljust(width))
            sys.stdout.flush()
            width = len(text)
        elif isinstance(event, SpeechFinal):
            sys.stdout.write("\r" + " " * width + f"\r{event.t_start:6.1f}s {event.speaker or '?'}: {event.text}\n")
            width = 0
        elif isinstance(event, VadStep):
            pass
    gaps = [g for g in timeline.gaps(min_s=0.3) if g.after is not None]
    shifts = sum(1 for g in gaps if g.shift)
    print(f"\n{len(gaps)} silences ≥ 0.3 s: {shifts} turn changes, {len(gaps) - shifts} pauses of the same speaker.")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="kairos.run", description=__doc__.splitlines()[0])
    parser.add_argument("source", help="script file, or ami:MEETING[:start_s[:duration_s]]")
    parser.add_argument("--mode", choices=MODES, default="closed")
    parser.add_argument("--speed", type=float, default=0.0, help="0 = instant (lockstep), 1 = real time")
    parser.add_argument("--memory", help="Kairos's long-term memory file (defaults to the one for the source)")
    parser.add_argument("--language", help="language of the meeting (default: French for scripts, English for AMI)")
    parser.add_argument("--judge", choices=("llm", "jev"), default=Settings().judge)
    parser.add_argument("--search", choices=("exa", "openai", "tavily", "off"), default=Settings().search,
                        help="web search for the researcher sub-agent")
    parser.add_argument("--anonymous", action="store_true", help="hide speaker names (one room microphone)")
    parser.add_argument("--listen", action="store_true", help="only replay the transcript (no model calls)")
    parser.add_argument("--prime", action="store_true",
                        help="AMI: remember the team's previous meeting and take notes of what was said before")
    args = parser.parse_args(argv)
    if args.listen:
        asyncio.run(listen(args))
    else:
        asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
