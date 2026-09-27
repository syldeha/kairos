"""Compare the two judges on the same meeting moments: latency and agreement.

    python scripts/judge_compare.py

Each case is a meeting state and the statements the judge agent would ask about it,
with the answer a person would give (True/False). Both judges answer every case;
the script prints their probabilities, their latency, and how often each agrees
with the expected answer (threshold 0.5).
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kairos.config import Settings  # noqa: E402
from kairos.decide.judges import JevJudge, JudgeFailed, LlmJudge  # noqa: E402
from kairos.session import make_llm  # noqa: E402

MEMORY = ("Kairos's memory:\n- Budget validé par la direction : 12 000 €, transport compris.\n"
          "- 18 personnes inscrites au séminaire du 14 au 16 mai.\n- Karim se déplace en fauteuil roulant.\n"
          "- Léa et Tom ne prennent pas l'avion (raisons écologiques).")


def state(transcript: str) -> str:
    return ("Kairos is an AI assistant attending this meeting. A line marked 'still speaking' may be incomplete.\n\n"
            f"TRANSCRIPT, the only thing said out loud in the meeting (latest last):\n{transcript}\n\n"
            f"NOT SAID IN THE MEETING, only known to Kairos:\n{MEMORY}")


CASES = [
    (state("L1 Inès: Lisbonne, ça me va.\nL2 Inès: Côté budget, on a quinze mille euros si je me souviens bien."), {
        "fit budget": ('If Kairos said right after the last line "Attention, le budget validé est de 12 000 €, pas '
                       '15 000.", it would directly respond to that line and add specific new information.', True),
        "fit karim": ('If Kairos said right after the last line "Pour Karim, il faut un hôtel accessible.", it would '
                      "directly respond to that line and add specific new information.", False),
        "said budget": ('In the transcript lines, someone already said out loud the specific content of this idea: '
                        '"the validated budget is 12,000 € including transport".', False),
    }),
    (state("L1 Hugo: Ça fait combien de temps de vol depuis Paris ?\nL2 Kairos: Environ 2h30 en vol direct.\n"
           "L3 Inès: Bon, on part sur Lisbonne. Tout le monde prend l'avion le 14 au matin ?"), {
        "said flight": ('In the transcript lines, someone (or Kairos) already said out loud the specific content of '
                        'this idea: "a direct flight Paris-Lisbon takes about 2h30".', True),
        "fit lea": ('If Kairos said right after the last line "Attention, Léa et Tom ne prennent pas l\'avion : il '
                    'faut prévoir le train.", it would directly respond to that line and add specific new '
                    "information.", True),
        "outdated lea": ('A participant has explicitly rejected, contradicted or already answered this idea: '
                         '"Léa et Tom ne prennent pas l\'avion". The conversation simply moving on does NOT make it '
                         "outdated.", False),
    }),
    (state("L1 Vous: Bonjour Kairos, je prépare un voyage à Lisbonne.\nL2 Vous: Tu sais s'il fait froid en "
           "février là-bas ?"), {
        "addressed": ('The line "Tu sais s\'il fait froid en février là-bas ?" (said by Vous) speaks directly to '
                      "Kairos, the AI assistant, and asks it a question.", True),
    }),
    (state("L1 Camille: Kairos prend des notes aujourd'hui, c'est pratique.\nL2 Karim: Oui. Bon, le prix."), {
        "addressed": ('The line "Kairos prend des notes aujourd\'hui, c\'est pratique." (said by Camille) speaks '
                      "directly to Kairos and asks it a question or makes a request to it.", False),
        "fit generic": ('If Kairos said right after the last line "Il est important de bien réfléchir au prix.", it '
                        "would directly respond to that line and add specific new information. Generic remarks do "
                        "not count.", False),
    }),
]


async def run(judge, name: str) -> None:
    latencies, right, total = [], 0, 0
    for st, questions in CASES:
        statements = {k: text for k, (text, _) in questions.items()}
        t0 = time.perf_counter()
        try:
            p = await judge.probabilities(st, statements)
        except JudgeFailed as exc:
            print(f"  {name}: request failed: {exc}")
            return
        latencies.append(time.perf_counter() - t0)
        for k, (_, expected) in questions.items():
            total += 1
            right += (p[k] >= 0.5) == expected
            print(f"  {name:4} {k:14} p={p[k]:.2f}  expected {'yes' if expected else 'no '}"
                  f"{'' if (p[k] >= 0.5) == expected else '   <-- wrong'}")
    print(f"  {name}: {right}/{total} right, latency per request median {statistics.median(latencies):.2f} s "
          f"(max {max(latencies):.2f} s)\n")


async def main() -> None:
    settings = Settings()
    jev = (JevJudge(settings.typesafe_api_key, direct=True, model=settings.jev_model)
           if settings.typesafe_api_key else JevJudge(settings.gateway_api_key))
    await run(jev, "jev")
    if jev.tracer.calls and jev.tracer.calls[-1].ok:
        print(f"  jev cost for these requests: ${jev.tracer.cost_usd():.6f}\n")
    await run(LlmJudge(make_llm(settings)), "llm")


if __name__ == "__main__":
    asyncio.run(main())
