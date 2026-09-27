"""Compare writing models on Kairos's real tasks: latency, cost, and the outputs side by side.

    python scripts/model_compare.py gpt-5.4-nano gpt-5.6-luna

Tasks use Kairos's own prompts (answer lane, checker, thinkers) plus the brief filler
planned for version 2. Each task runs 3 times per model; the median latency is shown,
and the first output is printed for reading.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from openai import AsyncOpenAI  # noqa: E402

from kairos.agents.thinkers import ANSWER_SYSTEM, CHECK_SYSTEM, system_prompt  # noqa: E402
from kairos.config import Settings  # noqa: E402
from kairos.llm import PRICES, _sampling  # noqa: E402

PRICES.setdefault("gpt-5.6-luna", (1.0, 6.0))  # Sidecar's configuration

MEMORY = ("Kairos's memory:\nM0: Séminaire annuel produit du 14 au 16 mai, 18 personnes inscrites.\n"
          "M1: Budget validé par la direction : 12 000 € transport compris.\n"
          "M2: Karim se déplace en fauteuil roulant.\nM3: Léa et Tom ne prennent pas l'avion (raisons écologiques).\n"
          "M4: L'an dernier à Porto, note 3,2/5, trajets trop longs.")
TRANSCRIPT = ("L1 [0.5s] Inès: Bon, on doit caler le séminaire de mai. Hugo, tu avais une idée de destination ?\n"
              "L2 [7.4s] Hugo: Oui, je pensais à Lisbonne. Il fait beau, ce n'est pas trop cher.\n"
              "L3 [16.6s] Inès: Lisbonne, ça me va. On serait combien, une vingtaine ?\n"
              "L4 [60.3s] Inès: Côté budget, on a quinze mille euros si je me souviens bien.\n"
              "L5 [96.5s] Inès: Bon, on part sur Lisbonne. Tout le monde prend l'avion le 14 au matin ?")
CONTEXT = (f"{MEMORY}\n\nWhat Kairos looked up on the web during this meeting:\n(none)\n\nMeeting notes:\n(none yet)\n\n"
           f"Transcript (latest last; lines by Kairos are Kairos's own):\n{TRANSCRIPT}\n\nExisting thoughts:\n(none)")

BRIEF_SYSTEM = """A background worker of Kairos, an AI assistant in a meeting, must search flights. Fill its brief.
Required details: destination, departure_city, date, travellers, budget, who_flies.
Return JSON {"known": {detail: {"value": "...", "source": "L3 or M1"}}, "missing": ["..."], "question": "one short spoken French question asking for ALL missing details at once, or empty"}.
Use only the transcript and Kairos's memory; never guess a value. Use "vous"."""

TASKS = {
    "answer": (ANSWER_SYSTEM.replace("{language}", "French"),
               CONTEXT + "\n\nThe question, line L6: Vous: Kairos, on est combien d'inscrits déjà ?"),
    "check (mistake)": (CHECK_SYSTEM.replace("{language}", "French"),
                        f"What Kairos knows for sure:\n{MEMORY}\n\nTranscript (latest last):\n{TRANSCRIPT.rsplit(chr(10), 1)[0]}"
                        "\n\nThe last line, L4: Inès: Côté budget, on a quinze mille euros si je me souviens bien."),
    "check (fine)": (CHECK_SYSTEM.replace("{language}", "French"),
                     f"What Kairos knows for sure:\n{MEMORY}\n\nTranscript (latest last):\nL2 Hugo: Oui, je pensais à Lisbonne."
                     "\n\nThe last line, L2: Hugo: Oui, je pensais à Lisbonne. Il fait beau, ce n'est pas trop cher."),
    "thinkers": (system_prompt("discreet", "French"), CONTEXT),
    "brief filler": (BRIEF_SYSTEM, f"{MEMORY}\n\nTranscript:\n{TRANSCRIPT}"),
}


async def run(client: AsyncOpenAI, model: str, system: str, user: str) -> tuple[float, dict, int, int]:
    t = time.perf_counter()
    r = await client.chat.completions.create(model=model, response_format={"type": "json_object"},
                                             messages=[{"role": "system", "content": system},
                                                       {"role": "user", "content": user}],
                                             **_sampling(model, 0.4))
    return (time.perf_counter() - t, json.loads(r.choices[0].message.content or "{}"),
            r.usage.prompt_tokens, r.usage.completion_tokens)


async def main() -> None:
    models = sys.argv[1:] or ["gpt-5.4-nano", "gpt-5.6-luna"]
    client = AsyncOpenAI(api_key=Settings().openai_api_key)
    summary = {}
    for task, (system, user) in TASKS.items():
        print(f"\n===== {task}")
        for model in models:
            try:
                runs = [await run(client, model, system, user) for _ in range(3)]
            except Exception as exc:
                print(f"  {model}: failed: {type(exc).__name__}: {str(exc)[:200]}")
                continue
            latency = statistics.median(r[0] for r in runs)
            price_in, price_out = PRICES.get(model, (0.0, 0.0))
            cost = statistics.mean((r[2] * price_in + r[3] * price_out) / 1e6 for r in runs)
            summary.setdefault(model, []).append((latency, cost))
            print(f"  {model}: median {latency:.2f}s, ${cost:.5f} per call, out {runs[0][3]} tokens")
            print("    " + json.dumps(runs[0][1], ensure_ascii=False)[:600])
    print("\n===== summary (per call, averaged over tasks)")
    for model, rows in summary.items():
        print(f"  {model}: median latency {statistics.mean(r[0] for r in rows):.2f}s, "
              f"cost ${statistics.mean(r[1] for r in rows):.5f}")


if __name__ == "__main__":
    asyncio.run(main())
