"""Real-time test matrix against a running console (python -m kairos.server.app).

    python scripts/live_matrix.py [scenario ...]      # all scenarios by default

Each scenario starts a meeting through the console API, plays a participant's lines at
set times, waits for the end, and prints the conversation, the interventions with their
latency, the searches and the reservoir. A JSON copy of each final state goes to runs/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

B = "http://127.0.0.1:8765"
RUNS = Path(__file__).resolve().parent.parent / "runs"

SCENARIOS = {
    "direct": dict(
        start={"source": "live", "mode": "closed", "speed": 1, "role": "discreet", "search": "exa",
               "context": "", "memory": ""},
        lines=[(2, "Je veux me faire un voyage cet hiver"), (8, "les trajets pas loin de france"),
               (14, "je veux bien les avions, avec peu de budget"),
               (14, "j'ai environ 600€ pour toutes les activités, tout est inclus"),
               (14, "tu sais si il y a des pays autour de Paris"),
               (18, "et il fait combien en février à Lisbonne ?"), (20, None)],
        live=True),
    "flights": dict(
        start={"source": "live", "mode": "closed", "speed": 1, "role": "discreet", "search": "exa",
               "context": "Préparation du séminaire annuel", "memory": ""},
        lines=[(2, "Je prépare notre séminaire à Lisbonne pour 18 personnes, le 14 mai prochain."),
               (9, "Kairos, tu peux regarder les vols ?"),
               (12, "On part tous de Paris."),
               (16, "Et tu sais s'il y a un train direct pour Lisbonne ?"),
               (16, "Tu en es où pour les vols ?"), (16, None)],
        live=True),
    "cameroon": dict(  # the user's live test of 27 September, same lines in the same order
        start={"source": "live", "mode": "closed", "speed": 1, "role": "discreet", "search": "exa",
               "context": "", "memory": ""},
        lines=[(2, "je suis très fatigué"),
               (6, "peut-être il faut qu'on se prépare, je sais pas, un voyage"),
               (8, "j'ai bien envie de rentrer au Cameroun"),
               (10, "mais je ne sais pas comment prendre le billet, c'est chaud"),
               (12, "I don't know, I would really like to travel this end of year"),
               (12, "Can you think what are the prices of the tickets around December?"),
               (14, "Is there something cheaper than that? I don't want a direct flight, if you can find something cheaper I would be fine"),
               (14, "I would like to go from Paris to Cameroon, maybe Yaoundé or Douala, I'm alone, let's say 22 December"),
               (14, "For the return date, let's say January 1st"),
               (16, "What did you find?"), (16, None)],
        live=True),
    "seminar_discreet": dict(
        start={"source": "seminaire_lisbonne.txt", "mode": "closed", "speed": 1, "role": "discreet",
               "search": "exa"},
        lines=[(40, "tu sais s'il y a un train direct pour Lisbonne ?")]),
    "seminar_active": dict(
        start={"source": "seminaire_lisbonne.txt", "mode": "closed", "speed": 1, "role": "active",
               "search": "exa"},
        lines=[]),
    "product": dict(
        start={"source": "reunion_produit.txt", "mode": "closed", "speed": 1, "role": "discreet",
               "search": "exa"},
        lines=[]),
    "interrupt": dict(
        start={"source": "reunion_produit.txt", "mode": "open", "speed": 1, "role": "discreet",
               "search": "off"},
        lines=[(9.5, "Kairos, combien facturent nos concurrents ?")]),
}


def run(name: str, spec: dict) -> dict:
    print(f"\n=================== {name} ===================", flush=True)
    httpx.post(B + "/api/start", json=spec["start"], timeout=120).raise_for_status()
    for wait, text in spec["lines"]:
        time.sleep(wait)
        if spec.get("live"):
            # A person waits for Kairos to finish before speaking again (up to 15 s).
            for _ in range(60):
                if not httpx.get(B + "/api/state").json().get("ai", {}).get("speaking"):
                    break
                time.sleep(0.25)
        if text:
            httpx.post(B + "/api/say", json={"text": text}).raise_for_status()
    if spec.get("live"):
        state = httpx.get(B + "/api/state").json()
        httpx.post(B + "/api/stop")
    else:
        started = time.time()
        while time.time() - started < 400:
            state = httpx.get(B + "/api/state").json()
            if state.get("finished") or state.get("error"):
                break
            time.sleep(2)
    if state.get("error"):
        print("ERROR", state["error"])
    for line in state["transcript"]:
        if line["final"]:
            print(("  >> " if line["ai"] else "     ") + f'{line["t"]:6.1f} {line["speaker"]}: {line["text"]}')
    print("  -- interventions")
    for r in state["interventions"]:
        extra = f' · question→réponse prête {r["question_to_answer_s"]} s' if r["question_to_answer_s"] is not None else ""
        flags = (" · coupé" if r["interrupted"] else "") + (" · a cédé" if r["yielded"] else "") + \
                (" · a repris" if r["resumed"] else "")
        print(f'  {r["t"]:6.1f} [{r["reason"]}] {r["moment"]} · fin de parole→1er mot {r["after_speech_s"]} s{extra}{flags}')
    print("  -- recherches")
    for f in state["findings"]:
        print(f'  {f["status"]} {f["seconds"]} s | {f["question"][:70]} -> {f["answer"][:110]}')
    print("  -- aiguillage (Jev)")
    for d in state.get("dispatch", []):
        print(f'  {d["t"]:6.1f} L{d["segment"]} {d["actions"] or "-"} {d["p"]} | {d["text"][:60]}')
    print("  -- travaux")
    for b in state.get("briefs", []):
        print(f'  {b["id"]} {b["kind"]} {b["status"]} missing={b["missing"]} q={b["question"][:70]!r}')
        for t, h in b["history"]:
            print(f'       {t:6.1f} {h}')
    print("  -- réservoir")
    for t in state["thoughts"]:
        print(f'  {t["status"]:8} {t["note"][:55]:55} | {t["utterance"][:80]}')
    print("  --", state["metrics"], flush=True)
    RUNS.mkdir(exist_ok=True)
    (RUNS / f"live_{name}.json").write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    return state


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    for _ in range(40):
        try:
            httpx.get(B + "/api/sources")
            break
        except httpx.HTTPError:
            time.sleep(0.5)
    names = sys.argv[1:] or list(SCENARIOS)
    for name in names:
        run(name, SCENARIOS[name])


if __name__ == "__main__":
    main()
