"""AMI benchmark: real passages where an assistant could help (or should stay quiet), replayed live, scored.

    python scripts/ami_bench.py [case ...]      (the console must be running)

The passages were mined from the 171 AMI meetings (scripts/ami_mine.py) and validated by reading
them. Each case lists the moments where Kairos should say something, with a time window (seconds
in the replay) and what it should convey; the control cases expect silence. After each replay a
judge (the writing model) classifies every intervention:

    hit        conveys an expected moment's content, inside its window
    useful     not expected, but a person in the room would welcome it
    redundant  repeats what someone just said, or comes after the room solved it
    wrong      incorrect, or misleading
    noise      generic, off topic, or an opinion Kairos should not give

Scores: recall (expected moments hit), precision ((hit + useful) / interventions), per case and overall.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kairos.config import Settings  # noqa: E402
from kairos.session import make_llm  # noqa: E402

B = "http://127.0.0.1:8765"

CASES = {
    "TS3007a-units": dict(
        source="ami:TS3007a:960:80",
        moments=[dict(id="units", window=[54, 95],
                      expect="Points out that two million remotes cannot bring 50 million euros of profit, and gives "
                             "the right order: 50 million divided by the margin per remote (about 11 million at a "
                             "4.50 euro margin, or 4 million at 12.50 euros).")]),
    "ES2013a-units": dict(
        source="ami:ES2013a:490:95",
        moments=[dict(id="units", window=[14, 70],
                      expect="Gives or confirms that 4 million units must be sold (50 million / 12.50 euros), "
                             "before the room settles it itself at about 84 s.")]),
    # Re-read after round 3: in this final meeting, 12.50 euros was the target and B estimates the actual cost of
    # their design at about 22; "twenty two?" may be right. No required moment: a wrong "correction" is the risk.
    "TS3011d-cost": dict(source="ami:TS3011d:1455:75", moments=[]),
    "ES2002a-units": dict(
        source="ami:ES2002a:460:95",
        moments=[dict(id="units", window=[30, 95],
                      expect="Works out that making 50 million euros with a 12.50 euro margin per remote (25 - 12.50) "
                             "requires selling 4 million remotes.")]),
    "TS3008d-buttons": dict(
        source="ami:TS3008d:1640:75",
        moments=[dict(id="share", window=[38, 75],
                      expect="Works out that 18 buttons at 50 cents cost 9 euros, which is about 72 percent of the "
                             "12.50 euro budget, not 80 percent.")]),
    "TS3005b-speech": dict(
        source="ami:TS3005b:1690:65",
        moments=[dict(id="cost", window=[30, 65],
                      expect="After 'nobody knows how much it will cost', gives a sourced estimate of what speech "
                             "recognition adds to a remote's cost, or offers to look it up; clearly an estimate.")]),
    "IS1001b-units": dict(
        source="ami:IS1001b:1200:85",
        moments=[dict(id="units", window=[8, 60],
                      expect="Answers the open question: 50 million euros at 12.50 euros per unit means selling "
                             "4 million units, before the room gets there at about 61 s.")]),
    "IS1000d-answered": dict(source="ami:IS1000d:1350:60", moments=[]),
    "ES2002b-design": dict(source="ami:ES2002b:300:180", moments=[]),
}

#: What every participant of the AMI design scenario was given (the project brief): the judge needs it to tell a
#: correction of a slip ("twenty and a half") from a wrong one.
AMI_BRIEF = ("AMI design scenario brief given to the team: a new TV remote control, selling price 25 euros, production "
             "cost at most 12.50 euros per unit, profit target 50 million euros, international market.")

JUDGE = """You judge the interventions of Kairos, an AI assistant, in a real meeting passage.
Background every participant knows: {brief}
Expected moments (where Kairos should say something, with a time window in seconds and what it should convey):
{moments}
Classify EVERY Kairos line (marked KAIROS) as one of:
- "hit": conveys an expected moment's content, inside its window (give the moment id);
- "useful": not expected, but a person in the room would welcome it (a correct, relevant fact or calculation);
- "redundant": repeats what someone just said, or comes after the room already solved it;
- "wrong": incorrect or misleading;
- "noise": generic advice, off topic, or an opinion or design preference Kairos should not give.
Return JSON {{"verdicts": [{{"t": seconds, "verdict": "...", "moment": "id or null", "why": "short"}}]}}."""


def replay(source: str) -> dict:
    httpx.post(B + "/api/start", json={"source": source, "mode": "closed", "speed": 1, "role": "discreet",
                                       "search": "exa", "prime": True}, timeout=240).raise_for_status()
    started = time.time()
    while time.time() - started < 900:
        s = httpx.get(B + "/api/state", timeout=10).json()
        if s.get("finished") or s.get("error"):
            break
        time.sleep(2)
    time.sleep(2)
    return httpx.get(B + "/api/state", timeout=10).json()


async def judge(case: dict, state: dict) -> list[dict]:
    llm = make_llm(Settings())  # one client per event loop
    kairos = [l for l in state["transcript"] if l["final"] and l["ai"]]
    if not kairos:
        return []
    lines = "\n".join(("KAIROS " if l["ai"] else "       ") + f'[{l["t"]:5.1f}s] {l["speaker"]}: {l["text"]}'
                      for l in state["transcript"] if l["final"])
    moments = "\n".join(f'- {m["id"]}: window {m["window"][0]}-{m["window"][1]} s: {m["expect"]}'
                        for m in case["moments"]) or "(none: Kairos should stay silent unless something is clearly useful)"
    out = await llm.json(JUDGE.format(moments=moments, brief=AMI_BRIEF), lines, purpose="bench judge", temperature=0.0)
    return out.get("verdicts") or []


def main() -> None:
    names = sys.argv[1:] or list(CASES)
    totals = {"moments": 0, "hits": 0, "interventions": 0, "good": 0}
    rows = []
    for name in names:
        case = CASES[name]
        print(f"\n=== {name} ({case['source']})", flush=True)
        state = replay(case["source"])
        if state.get("error"):
            print("ERROR", state["error"])
        for l in state["transcript"]:
            if l["final"] and (l["ai"] or len(l["text"].split()) > 2):
                print(("  >> " if l["ai"] else "     ") + f'{l["t"]:6.1f} {l["speaker"]}: {l["text"][:150]}')
        verdicts = asyncio.run(judge(case, state))
        for v in verdicts:
            print(f'  -> {v.get("t")} s {v.get("verdict")} {v.get("moment") or ""} | {v.get("why", "")[:100]}')
        hit_ids = {v.get("moment") for v in verdicts if v.get("verdict") == "hit"}
        hits = sum(1 for m in case["moments"] if m["id"] in hit_ids)
        good = sum(1 for v in verdicts if v.get("verdict") in ("hit", "useful"))
        n = len(verdicts)
        totals["moments"] += len(case["moments"])
        totals["hits"] += hits
        totals["interventions"] += n
        totals["good"] += good
        print(f"  recall {hits}/{len(case['moments'])} · precision {good}/{n} · decisions: "
              + "; ".join(f'{d["t"]} {d["why"][:50]}' for d in state["decisions"][-4:]), flush=True)
        (ROOT / "runs" / f"bench_{name}.json").write_text(json.dumps({"state": state, "verdicts": verdicts},
                                                                     ensure_ascii=False, indent=1), encoding="utf-8")
        rows.append((name, hits, len(case["moments"]), good, n))
    print("\n== summary")
    for name, hits, m, good, n in rows:
        print(f"  {name:18} recall {hits}/{m}  precision {good}/{n}")
    print(f"  TOTAL recall {totals['hits']}/{totals['moments']}  precision {totals['good']}/{totals['interventions']}")


if __name__ == "__main__":
    main()
