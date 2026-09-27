"""Replay a real meeting (AMI) in real time against the console and show how Kairos behaves in it.

    python scripts/ami_live.py ami:ES2002b:300:300 [--speed 1] [--role discreet]

Prints the meeting with Kairos's lines in place, each intervention with its reason and moment, what
Jev dispatched (and why), the searches, the reservoir's history, and a short summary: how often
Kairos spoke, how much of the talk it took, how many searches it started, and what it said.
The full state goes to runs/ami_<meeting>_<start>.json for a closer look.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

B = "http://127.0.0.1:8765"
RUNS = Path(__file__).resolve().parent.parent / "runs"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--role", default="discreet")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    httpx.post(B + "/api/start", json={"source": args.source, "mode": "closed", "speed": args.speed,
                                       "role": args.role, "search": "exa", "prime": True}, timeout=180).raise_for_status()
    started = time.time()
    while time.time() - started < 1200:
        s = httpx.get(B + "/api/state", timeout=10).json()
        if s.get("finished") or s.get("error"):
            break
        time.sleep(3)
    s = httpx.get(B + "/api/state", timeout=10).json()
    name = args.source.replace(":", "_") + (f"_{args.tag}" if args.tag else "")
    RUNS.mkdir(exist_ok=True)
    (RUNS / f"ami_{name}.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    if s.get("error"):
        print("ERROR", s["error"])
    print("== meeting (Kairos lines marked >>)")
    for l in s["transcript"]:
        if l["final"] and (l["ai"] or len(l["text"].split()) > 2):
            print(("  >> " if l["ai"] else "     ") + f'{l["t"]:6.1f} {l["speaker"]}: {l["text"][:170]}')
    print("\n== interventions")
    for r in s["interventions"]:
        print(f'  {r["t"]:6.1f} [{r["reason"]}] {r["moment"]} · after speech {r["after_speech_s"]} s'
              + (" · cut" if r["interrupted"] else "") + f' | {r["text"][:110]}')
    print("\n== Jev dispatch (lines that started work)")
    for d in s.get("dispatch", []):
        if d["actions"]:
            print(f'  {d["t"]:6.1f} {d["actions"]} {d["p"]} | {d["text"][:80]}')
    print("\n== context research (background)")
    for c in s.get("context", []):
        print(f'  {c["t"]:6.1f} {c["subject"][:50]} | {c["query"]}')
    print("\n== searches")
    for f in s["findings"]:
        print(f'  {f["status"]} {f["seconds"]} s | {f["query"][:60]} -> {f["answer"][:110]}')
    print("\n== reservoir history (additions and exits)")
    for r in s.get("reservoir_log", [])[-40:]:
        print(f'  {r["t"]:6.1f} {r["event"][:18]:18} {r["by"][:12]:12} {r["why"][:32]:32} | {r["thought"][:80]}')
    m = s["metrics"]
    print(f'\n== summary: {len(s["interventions"])} interventions, natural gaps {m["in_natural_gap"]}, '
          f'cut {m["interrupted"]}, searches {m["web_searches"]}, cost ${m["cost_usd"]}, '
          f'AI share {s["ai"]["share"]}')


if __name__ == "__main__":
    main()
