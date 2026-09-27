"""The answer key: where should Kairos have spoken, and did the pipeline get there?

    python -m kairos.opportunities ami:ES2002b:300:300 runs/<report>.json

1. Annotate: a stronger model reads the excerpt with the same memory Kairos had and
   marks every moment where a well-informed assistant should have spoken, with what.
   Cached next to the meeting data, so each excerpt is annotated once.
2. Diagnose: for each marked moment, find what the pipeline had at that time in the
   run report: a similar thought? its judge scores? was the floor open? The first
   stage that failed is where to look.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .config import Settings
from .contracts import SpeechFinal
from .decide.policy import PolicyParams
from .llm import OpenAILLM, cosine
from .session import default_memory, load_memory, load_timeline, prime_ami
from .sources.ami import DATA_DIR

ANNOTATE_SYSTEM = """You build the answer key for evaluating an AI meeting assistant named Kairos.
Kairos attends the meeting below. It knows ONLY what is in its memory (given), and it can look facts up on the public web.
Mark every moment where a well-informed, polite colleague in Kairos's position SHOULD have spoken, right after a given line.

A moment counts only if speaking would clearly help the room:
- answering a question asked out loud that nobody answered, when Kairos knows or could look up the answer;
- a fact from Kairos's memory that the room is missing right now (a decision, a figure, a constraint), or that contradicts what is being said;
- a public fact (web) that would change what the room is deciding right now.
Do not mark generic advice, opinions, summaries, or anything the room already said. When in doubt, do not mark it.
Most meetings have few such moments.

Return one JSON object:
{"opportunities": [{"line": 12, "kind": "answer" | "memory" | "web" | "correction",
  "what_to_say": "what Kairos should say, one or two sentences", "why": "why the room needs it then",
  "value": 1-3}]}
"line" is the number after L of the line right after which Kairos should speak. value 3 = the room clearly loses something without it."""


def excerpt_lines(spec: str) -> list[SpeechFinal]:
    timeline = load_timeline(spec)
    return sorted((e for e in timeline.events if isinstance(e, SpeechFinal)), key=lambda e: e.t_start)


async def annotate(spec: str, memory: list[str], notes: str, llm: OpenAILLM) -> list[dict]:
    meeting, start = spec.split(":")[1], spec.split(":")[2]
    cache = DATA_DIR / meeting / f"opportunities_{start}_{spec.split(':')[3]}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    lines = excerpt_lines(spec)
    transcript = "\n".join(f"L{e.segment} [{e.t_start:.1f}s] {e.speaker}: {e.text}" for e in lines)
    user = ("KAIROS'S MEMORY:\n" + "\n".join(f"- {m}" for m in memory) +
            f"\n\nNOTES KAIROS HAD WHEN IT JOINED (what was said before this excerpt):\n{notes or '(none)'}" +
            f"\n\nTRANSCRIPT OF THE EXCERPT:\n{transcript}")
    result = await llm.json(ANNOTATE_SYSTEM, user, purpose="annotate", temperature=0.0)
    ops = [o for o in result.get("opportunities") or [] if isinstance(o, dict) and "line" in o]
    by_id = {e.segment: e for e in lines}
    for o in ops:
        line = by_id.get(int(str(o["line"]).lstrip("L")))
        o["line"] = line.segment if line else o["line"]
        o["t"] = round(line.t_end, 1) if line else None
        o["line_text"] = f"{line.speaker}: {line.text}" if line else ""
    cache.write_text(json.dumps(ops, ensure_ascii=False, indent=2), encoding="utf-8")
    return ops


async def diagnose(ops: list[dict], report: dict, llm: OpenAILLM, window_s: float = 8.0) -> list[dict]:
    """For each opportunity: the closest thought Kairos had around that moment, and where it stopped."""
    thoughts = report["thoughts"]
    texts = [o["what_to_say"] for o in ops] + [t["content"] for t in thoughts]
    vectors = await llm.embed(texts, purpose="embedding") if texts else []
    op_vecs, th_vecs = vectors[:len(ops)], vectors[len(ops):]
    p = PolicyParams()
    out = []
    for o, ov in zip(ops, op_vecs):
        t = o.get("t") or 0.0
        gaps = [d for d in report["decisions"] if t - 0.5 <= d["t"] <= t + window_s]
        near = [(cosine(ov, tv), th) for th, tv in zip(thoughts, th_vecs) if th["created_at"] <= t + window_s]
        similarity, best = max(near, key=lambda x: x[0], default=(0.0, None))
        if best is None or similarity < 0.55:
            stage = "no similar thought: the thinkers/researcher never had this idea"
        elif best["fit_now"] < p.fit_min:
            stage = f"thought existed, judge said not coherent now (fit {best['fit_now']:.2f})"
        elif best["already_said"] > p.said_max:
            stage = f"thought existed, judge said already said ({best['already_said']:.2f})"
        elif not gaps:
            stage = "thought ready, but the floor never opened after that line"
        else:
            stage = f"thought ready, score too low (importance {best['importance']:.0f}, relevance {best['relevance']:.2f})"
        out.append({**o, "similarity": round(similarity, 2), "closest_thought": best["utterance"] if best else "",
                    "floor_open_after": [f"{d['t']:.1f}s: {d['why']}" for d in gaps], "stopped_at": stage})
    return out


async def main_async(spec: str, report_path: Path) -> None:
    settings = Settings()
    annotator = OpenAILLM(settings.openai_api_key, "gpt-4o", settings.embedding_model)
    memory = load_memory(default_memory(spec))
    extra, notes = await prime_ami(spec, annotator, "English")
    ops = await annotate(spec, memory + extra, notes, annotator)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = await diagnose(ops, report, annotator)
    print(f"{len(rows)} occasion(s) marquée(s) par l'annotateur · Kairos a parlé {len(report['interventions'])} fois\n")
    for r in rows:
        print(f"── L{r['line']} à {r['t']}s · {r['kind']} · valeur {r['value']}/3")
        print(f"   Après : {r['line_text'][:160]}")
        print(f"   Aurait dû dire : {r['what_to_say']}")
        print(f"   Pourquoi : {r['why']}")
        print(f"   Pensée la plus proche (similarité {r['similarity']}) : {r['closest_thought'][:160] or '—'}")
        print(f"   Blanc disponible : {'; '.join(r['floor_open_after']) or 'aucun dans les 8 s'}")
        print(f"   → Bloqué à : {r['stopped_at']}\n")
    print(f"Coût de l'annotation et du diagnostic : {annotator.tracer.cost_usd():.3f} $")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="kairos.opportunities")
    parser.add_argument("source", help="ami:MEETING:start:duration (the same excerpt as the run)")
    parser.add_argument("report", type=Path, help="the run's JSON report in runs/")
    args = parser.parse_args()
    asyncio.run(main_async(args.source, args.report))


if __name__ == "__main__":
    main()
