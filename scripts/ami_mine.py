"""Find the AMI passages where an assistant could really have helped.

    python scripts/ami_mine.py [--zip data/ami_zip/ami_public_manual_1.6.2.zip] [--keep 60]

1. Extracts the word files of every meeting (once) into data/ami/<meeting>/.
2. Code only: scores 2-minute windows on cues of an information need (questions, costs, prices,
   figures, "do you know", "I'm not sure", "remember"...), keeps the 2 best per meeting, 60 overall.
3. The writing model rates each kept window as an evaluator would: where could Kairos, an assistant
   with no opinions of its own but able to look things up, calculate and remember, have added
   something important that nobody in the room did? Each moment gets a time, a type, what Kairos
   should have said and a value; the passage gets an overall score.

Writes runs/ami_mining.json and prints the best passages.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kairos.config import Settings  # noqa: E402
from kairos.session import make_llm  # noqa: E402
from kairos.sources.ami import DATA_DIR, read_words  # noqa: E402

CUES = re.compile(r"\b(how much|how many|cost|costs|price|prices|euro|euros|pound|pounds|cents|percent|budget|"
                  r"do you know|does anyone know|anyone know|i don't know|i'm not sure|not sure|remember|what was|"
                  r"what's the|total|add up|calculate|figure|figures|statistic|survey|research|market)\b", re.I)
NUMBER = re.compile(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve|fifty|hundred|thousand)\b", re.I)

RATE = """You evaluate whether an AI assistant named Kairos, silently attending this real meeting, could have added something important.
Kairos can: answer a factual question nobody answers, look something up on the web, do a calculation (sums, costs, budgets), recall a figure or decision said earlier, correct a wrong figure, point out a forgotten constraint. Kairos must NOT give opinions, design preferences or generic advice, and must not answer what a participant answers well a few seconds later.
Return one JSON object:
{"score": 0-5, "summary": "what the passage is about", "moments": [{"t": seconds from the passage start, "type": "unanswered_question|calculation|wrong_figure|lookup|recall|forgotten_constraint", "trigger": "the words that open the moment", "kairos": "what Kairos should say, one or two sentences", "why": "why nobody in the room did it", "value": 1-5}]}
score: 5 = several clear, valuable moments where a person in the room would have welcomed it; 3 = one real moment; 0-1 = nothing Kairos should say (then moments is empty). Be strict: most passages score 0-2."""


def utterances(words, gap: float = 0.6):
    out, open_by = [], {}
    for w in words:
        cur = open_by.get(w.speaker)
        if cur is None or w.start - cur["end"] > gap:
            cur = {"speaker": w.speaker, "start": w.start, "end": w.end, "words": []}
            out.append(cur)
            open_by[w.speaker] = cur
        cur["words"].append(w.text)
        cur["end"] = w.end
    for u in out:
        u["text"] = " ".join(u["words"])
    return sorted(out, key=lambda u: u["start"])


def windows(utts, length: float = 120.0, step: float = 60.0):
    if not utts:
        return []
    end = utts[-1]["end"]
    result, t = [], 0.0
    while t < end:
        inside = [u for u in utts if t <= u["start"] < t + length]
        text = " ".join(u["text"] for u in inside)
        cues = len(CUES.findall(text))
        questions = sum(1 for u in inside if u["text"].rstrip().endswith("?"))
        numbers = len(NUMBER.findall(text))
        substantive = sum(1 for u in inside if len(u["text"].split()) > 4)
        score = 2 * cues + questions + 0.3 * numbers + (0 if substantive >= 6 else -5)
        result.append((score, t))
        t += step
    return result


def extract_all(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as archive:
        names = [n for n in archive.namelist() if n.startswith("words/") and n.endswith(".words.xml")]
        meetings = sorted({Path(n).name.split(".")[0] for n in names})
        for n in names:
            target = DATA_DIR / Path(n).name.split(".")[0] / Path(n).name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(n))
    return meetings


async def rate(llm, meeting, start, utts, sem):
    lines = [f"[{u['start'] - start:5.1f}s] {u['speaker']}: {u['text']}" for u in utts
             if start <= u["start"] < start + 120 and len(u["text"].split()) > 1]
    async with sem:
        try:
            r = await llm.json(RATE, f"Meeting {meeting}, passage from {start:.0f} s:\n" + "\n".join(lines),
                               purpose="mining", temperature=0.2)
        except Exception as exc:
            return {"meeting": meeting, "start": start, "score": -1, "error": str(exc)[:100]}
    return {"meeting": meeting, "start": start, **r}


async def main_async(args) -> None:
    meetings = extract_all(Path(args.zip))
    candidates = []
    for m in meetings:
        utts = utterances(read_words(DATA_DIR / m))
        best, taken = sorted(windows(utts), reverse=True), []
        for score, t in best:
            if all(abs(t - x) >= 120 for _, x in taken):
                taken.append((score, t))
            if len(taken) == 2:
                break
        candidates += [(s, m, t, utts) for s, t in taken]
    candidates.sort(key=lambda c: c[0], reverse=True)
    kept = candidates[:args.keep]
    print(f"{len(meetings)} meetings, {len(candidates)} candidate windows, rating the best {len(kept)}", flush=True)
    llm = make_llm(Settings())
    sem = asyncio.Semaphore(6)
    rated = await asyncio.gather(*(rate(llm, m, t, utts, sem) for _, m, t, utts in kept))
    rated.sort(key=lambda r: r.get("score", -1), reverse=True)
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs" / "ami_mining.json").write_text(json.dumps(rated, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"cost ${llm.tracer.cost_usd():.3f}")
    for r in rated[:15]:
        print(f"\n{r['score']}  {r['meeting']} from {r['start']:.0f} s  | {r.get('summary', '')[:110]}")
        for mo in r.get("moments", []):
            print(f"    {mo.get('t')} s [{mo.get('type')}] v{mo.get('value')} « {str(mo.get('trigger'))[:70]} »"
                  f" -> {str(mo.get('kairos'))[:110]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default=str(ROOT / "data" / "ami_zip" / "ami_public_manual_1.6.2.zip"))
    parser.add_argument("--keep", type=int, default=60)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
