"""Fluidity evaluation: short conversations played live against the console, then scored.

    python scripts/flow_eval.py [scenario ...]      (the console must be running)

Each scenario is typed in Direct mode, one line at a time; the "person" waits for Kairos to
finish before speaking again, and pauses between lines like a real conversation. The score
looks at how information flows from the room to Jev, to the workers, back to the reservoir
and into the conversation:

- need picked up    Kairos reacted to the need (offer, question, "je regarde" or result)
- result said       every finished search reached the conversation
- promises kept     every "je regarde" / "je cherche" was followed by a result or a failure notice
- repetitions       pairs of Kairos lines saying nearly the same thing
- role              Kairos never talks as a participant ("je suis partant")
- silence           in small talk, Kairos stays mostly quiet and starts no search
- latency           end of the need's line -> Kairos's first reaction; last detail -> result said

Results go to runs/flow_<scenario>.json and a summary table is printed.
"""

from __future__ import annotations

import difflib
import json
import re
import sys
import time
from pathlib import Path

import httpx

B = "http://127.0.0.1:8765"
RUNS = Path(__file__).resolve().parent.parent / "runs"

SCENARIOS = {
    "restaurant": dict(
        need_line=2,
        lines=["Les gars, est-ce que quelqu'un serait chaud pour se faire un restaurant tout à l'heure ?",
               "En vrai, peu importe le restaurant, à Paris, pas très cher, ça me va.",
               "Kairos, tu peux nous proposer un truc ?",
               "Végétarien, dans le cinquième.",
               "Est-ce que tu as trouvé un truc ?"],
        expect=dict(search=True)),
    "cameroon": dict(
        need_line=1,
        lines=["je suis très fatigué, il faut que je me prépare un voyage",
               "j'ai bien envie de rentrer au Cameroun, mais je ne sais pas comment prendre le billet",
               "I would really like to travel this end of year",
               "Can you think what are the prices of the tickets around December?",
               "From Paris, maybe Yaoundé or Douala, I'm alone, let's say 22 December",
               "For the return date, let's say January 1st",
               "What did you find?"],
        expect=dict(search=True)),
    "vague_price": dict(
        need_line=0,
        lines=["C'est combien un billet d'avion pour Barcelone en ce moment ?",
               "Je sais pas encore, en mars peut-être.",
               "Alors, ça donne quoi ?",
               "Je pars de Lyon."],
        expect=dict(search=True)),
    "fact": dict(
        need_line=0,
        lines=["Il fait quel temps à Lisbonne en mai ?", "D'accord, merci."],
        expect=dict(search=True)),
    "smalltalk": dict(
        need_line=None,
        lines=["Je suis crevé aujourd'hui.", "Le boulot était long, beaucoup de réunions.",
               "Bon, rien de spécial à part ça.", "On verra demain."],
        expect=dict(search=False, max_kairos=1)),
    "declined": dict(
        need_line=0,
        lines=["J'aimerais bien partir à Rome un de ces jours.", "Non merci, pas maintenant, je verrai plus tard.",
               "Sinon, la semaine a été calme."],
        expect=dict(search=False, max_kairos=2)),
}

PROMISE = re.compile(r"\b(je regarde|je cherche|je lance|let me look|searching|i'll check|on it|je vérifie)\b", re.I)
PARTICIPANT = re.compile(r"\b(je suis partant|je n'ai pas de préférence|je n’ai pas de préférence|je viens|count me in)\b", re.I)


def state() -> dict:
    return httpx.get(B + "/api/state", timeout=10).json()


def wait_quiet(max_s: float = 25.0) -> None:
    """Wait until Kairos is not speaking and has been quiet for a moment."""
    quiet_since, start = None, time.time()
    while time.time() - start < max_s:
        s = state()
        if s.get("ai", {}).get("speaking"):
            quiet_since = None
        elif quiet_since is None:
            quiet_since = time.time()
        elif time.time() - quiet_since > 2.5:
            return
        time.sleep(0.25)


def run(name: str, spec: dict) -> dict:
    httpx.post(B + "/api/start", json={"source": "live", "mode": "closed", "speed": 1, "role": "discreet",
                                       "search": "exa", "context": "", "memory": ""}, timeout=60).raise_for_status()
    time.sleep(1.5)
    for text in spec["lines"]:
        httpx.post(B + "/api/say", json={"text": text}).raise_for_status()
        time.sleep(len(text.split()) / 2.75 + 1.0)  # the words arrive at speech rate
        time.sleep(6.0)                            # a real pause: room for Kairos, or for the next thought
        wait_quiet()
    time.sleep(4)
    wait_quiet()
    s = state()
    httpx.post(B + "/api/stop")
    RUNS.mkdir(exist_ok=True)
    (RUNS / f"flow_{name}.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    return s


def score(name: str, spec: dict, s: dict) -> dict:
    lines = [l for l in s["transcript"] if l["final"]]
    human = [l for l in lines if not l["ai"]]
    kairos = [l for l in lines if l["ai"]]
    log = s.get("reservoir_log", [])
    findings_ids = {r["id"] for r in log if r["event"].startswith("added (finding)")}
    finding_spoken = [r["t"] for r in log if r["event"] == "spoken" and r["id"] in findings_ids]
    promises = [l for l in kairos if PROMISE.search(l["text"]) and "?" not in l["text"]]  # an offer is not a promise
    kept = sum(1 for p in promises if any(p["t"] < t <= p["t"] + 30 for t in finding_spoken))
    repeats = []
    for i, a in enumerate(kairos):
        for b in kairos[i + 1:]:
            r = difflib.SequenceMatcher(None, a["text"].lower(), b["text"].lower()).ratio()
            if r >= 0.6 and len(a["text"].split()) > 3:
                repeats.append((round(r, 2), a["text"][:60], b["text"][:60]))
    need = spec.get("need_line")
    reaction = None
    if need is not None and need < len(human):
        need_end = human[need]["t"]
        after = [k for k in kairos if k["t"] >= need_end]
        reaction = round(after[0]["t"] - need_end, 1) if after else None
    addressed_segments = {d["segment"] for d in s.get("dispatch", []) if d["p"].get("addressed", 0) >= 0.65}
    addressed_lines = [l for l in human if l["id"] in addressed_segments]
    flows = s.get("flows", [])
    searches = [f for f in flows if any(e["kind"] == "demande" and e["text"] == "lancé" for e in f["events"])]
    done = [f for f in flows if f["status"] in ("done", "failed")]
    reused = {r["id"] for r in log if r["event"] == "stale" and "déjà dit par Kairos" in (r.get("why") or "")}
    spoken_ids = {r["id"] for r in log if r["event"] == "spoken"}
    said = [f for f in done if f["result_said"] or f"r{f['id']}" in reused
            or any(i.startswith(f"r{f['id']}a") for i in spoken_ids) or f["status"] == "failed" and not any(
        e["kind"] == "réservoir" and "added (finding)" in e["text"] for e in f["events"])]
    exp = spec["expect"]
    checks = {
        "need picked up": (reaction is not None) if need is not None else True,
        "search as expected": bool(searches) == exp.get("search", False),
        "results said": len(said) == len(done),
        "promises kept": kept == len(promises),
        "no repetition": not repeats,
        "assistant role": not any(PARTICIPANT.search(k["text"]) for k in kairos),
        "quiet enough": len(kairos) <= exp.get("max_kairos", 99),
        "questions answered": all(any(l["t"] < k["t"] <= l["t"] + 15 for k in kairos) for l in addressed_lines),
        "natural pauses": s["metrics"]["in_natural_gap"].split("/")[0] == s["metrics"]["in_natural_gap"].split("/")[1],
    }
    return {"scenario": name, "passed": sum(checks.values()), "of": len(checks), "checks": checks,
            "kairos_lines": len(kairos), "reaction_s": reaction, "promises": f"{kept}/{len(promises)}",
            "searches": len(searches), "results_said": f"{len(said)}/{len(done)}", "repeats": repeats,
            "interrupted": s["metrics"]["interrupted"], "cost_usd": s["metrics"]["cost_usd"]}


def main() -> None:
    names = sys.argv[1:] or list(SCENARIOS)
    results = []
    for name in names:
        print(f"\n=== {name}", flush=True)
        s = run(name, SCENARIOS[name])
        for l in s["transcript"]:
            if l["final"]:
                print(("  >> " if l["ai"] else "     ") + f'{l["t"]:6.1f} {l["speaker"]}: {l["text"][:150]}')
        r = score(name, SCENARIOS[name], s)
        results.append(r)
        failed = [k for k, v in r["checks"].items() if not v]
        print(f"  score {r['passed']}/{r['of']}" + (f"  FAILED: {', '.join(failed)}" if failed else "")
              + f" | reaction {r['reaction_s']} s | promises {r['promises']} | results said {r['results_said']}"
              + f" | kairos lines {r['kairos_lines']} | cut {r['interrupted']}", flush=True)
        for rep in r["repeats"]:
            print(f"    repeat {rep[0]}: « {rep[1]} » ~ « {rep[2]} »")
    total = sum(r["passed"] for r in results), sum(r["of"] for r in results)
    print(f"\nTOTAL {total[0]}/{total[1]}")
    (RUNS / "flow_summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
