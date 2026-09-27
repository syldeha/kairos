"""Live scenarios against Kairos behind the Atlas interface: open the page on the Chat view to watch them.

    python -m kairos.ui.server                      # KAIROS_PORT=8788 locally
    python scripts/ui_scenarios.py [scenario ...]   # all by default; KAIROS_URL to change the server

Each scenario starts a session, speaks its lines as typed participants ("Claude", "Léa") at speech rate
through /api/say, and records what Kairos says after each line and how long after the line ended. A line
expects silence, speech, or speech mentioning some words; any Kairos sentence said twice fails the scenario.
Results are printed and saved to runs/ui_scenarios/.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import httpx
import websockets

BASE = os.environ.get("KAIROS_URL", "http://127.0.0.1:8788")
PROTOCOL = 11
OUT = Path(__file__).resolve().parent.parent / "runs" / "ui_scenarios"


@dataclass
class Line:
    speaker: str
    text: str
    expect: str = "any"  # "silent" | "speak" | "any"
    words: tuple[str, ...] = ()  # speech after the line must mention one of these (case-insensitive)
    wait: float = 8.0  # seconds to listen after the line ends


@dataclass
class Heard:
    text: str
    delay: float  # seconds after the line ended
    cut: bool


@dataclass
class Result:
    line: Line
    heard: list[Heard] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        said = [h for h in self.heard if h.text]
        if self.line.expect == "silent":
            return not said
        if self.line.expect == "speak" and not said:
            return False
        if self.line.words:
            blob = " ".join(h.text for h in said).lower()
            return any(word.lower() in blob for word in self.line.words)
        return True


C, L, M = "Claude", "Léa", "Marc"

#: A long, realistic meeting: three friends plan a weekend among themselves; Kairos only speaks when called
#: (or to correct a wrong total). Every other line expects silence.
REUNION: list[Line] = [
    Line(C, "Bon, on s'était dit qu'on se ferait un week-end ensemble avant l'hiver, vous êtes toujours partants ?",
         "silent", wait=3),
    Line(L, "Carrément, j'ai besoin de souffler, le boulot en ce moment c'est intense.", "silent", wait=3),
    Line(M, "Moi aussi, par contre je ne peux pas avant la mi-octobre, j'ai un déménagement à gérer.", "silent", wait=3),
    Line(C, "Ok, alors disons le week-end du 16 octobre, du vendredi soir au lundi matin.", "silent", wait=3),
    Line(L, "Ça me va. Vous pensiez à quoi, plutôt la mer ou une ville ?", "silent", wait=3),
    Line(M, "Une ville avec un peu de soleil, j'aimerais bien Lisbonne, je n'y suis jamais allé.", "silent", wait=3),
    Line(C, "Lisbonne c'est top, j'y étais il y a trois ans, la nourriture est incroyable.", "silent", wait=3),
    Line(L, "Moi je suis partante pour Lisbonne. Il faudrait regarder les vols quand même.", "silent", wait=4),
    Line(M, "Kairos, tu peux nous trouver un vol Paris Lisbonne le vendredi 16 octobre ?", "speak", ("euro",), wait=18),
    Line(C, "Ah c'est pas mal, franchement ça passe.", "silent", wait=3),
    Line(L, "Et pour dormir, on prend un hôtel ou un appartement ?", "silent", wait=3),
    Line(M, "Un hôtel c'est plus simple, on ne va pas faire le ménage le dimanche soir.", "silent", wait=3),
    Line(C, "Kairos, regarde les hôtels à Lisbonne pour nous trois, du 16 au 19 octobre.", "speak",
         ("euro", "nuit"), wait=20),
    Line(L, "Parfait. Côté budget, je propose qu'on mette 300 euros chacun pour tout le week-end.", "silent", wait=3),
    Line(M, "Donc à trois ça nous fait 800 euros de budget total, c'est large.", "any", ("900", "neuf cents"), wait=6),
    Line(C, "Oui, bien vu. Bon, et qu'est-ce qu'on fait sur place ?", "silent", wait=3),
    Line(L, "Moi je veux absolument manger des pastéis de nata, c'est obligatoire.", "silent", wait=3),
    Line(M, "Et sortir un soir, il paraît que l'ambiance est géniale.", "silent", wait=3),
    Line(C, "Kairos, qu'est-ce que tu nous conseilles de faire le samedi soir à Lisbonne ?", "speak",
         ("alfama", "fado", "bairro", "bar", "cais", "miradouro"), wait=18),
    Line(L, "Le fado ça me tente bien, je n'en ai jamais écouté en vrai.", "silent", wait=3),
    Line(M, "Il faudra aussi prévoir comment aller de l'aéroport au centre.", "silent", wait=3),
    Line(C, "Kairos, comment on va de l'aéroport de Lisbonne au centre-ville ?", "speak",
         ("métro", "metro", "bus", "taxi", "uber", "minutes"), wait=18),
    Line(L, "Super, on prendra le métro alors, c'est simple.", "silent", wait=3),
    Line(M, "Bon, je crois qu'on a l'essentiel. Qui réserve les billets ?", "silent", wait=3),
    Line(C, "Je m'en occupe ce soir, et Léa tu t'occupes de l'hôtel ?", "silent", wait=3),
    Line(L, "Oui, je le fais demain matin. Merci Kairos, c'était utile.", "any", wait=5),
    Line(M, "Allez, on se refait un point la semaine prochaine.", "silent", wait=5),
]

N, T, S, A, J = "Nadia", "Tom", "Sarah", "Amine", "James"

#: Varied meetings: people talk among themselves, call Kairos two or three times. Kairos is an assistant: it
#: answers when called, corrects a wrong figure, and otherwise stays silent.
TRAVAIL: list[Line] = [
    Line(N, "Bon, on fait le point sur le lancement de la nouvelle gourde isotherme.", "silent", wait=3),
    Line(T, "Oui, côté production on est prêts, le fournisseur peut livrer 5000 unités début novembre.", "silent",
         wait=3),
    Line(C, "Et on part sur quel prix de vente ?", "silent", wait=3),
    Line(N, "On avait dit 12 euros, c'est cohérent avec la concurrence.", "silent", wait=3),
    Line(T, "Donc 5000 unités à 12 euros, ça fait 50 000 euros de chiffre d'affaires.", "any", ("60", "soixante"),
         wait=6),
    Line(C, "D'accord. Et pour la date de lancement, on vise toujours le 15 novembre ?", "silent", wait=3),
    Line(N, "Oui, le 15 novembre, pas plus tard, sinon on rate les commandes de Noël.", "silent", wait=3),
    Line(T, "Il faudra une campagne de pub, je pensais à LinkedIn et Instagram.", "silent", wait=3),
    Line(C, "Kairos, ça coûte combien en moyenne une campagne LinkedIn pour une petite marque ?", "speak",
         ("euro", "clic", "budget"), wait=18),
    Line(N, "Ok, on peut se le permettre si on reste raisonnables.", "silent", wait=3),
    Line(T, "Je prépare les visuels d'ici la fin de la semaine.", "silent", wait=3),
    Line(C, "Kairos, rappelle-nous la date de lancement qu'on a fixée ?", "speak", ("15 novembre", "quinze"), wait=10),
    Line(N, "Parfait, on se revoit jeudi pour valider les visuels.", "silent", wait=4),
]
DINER: list[Line] = [
    Line(S, "Pour l'anniversaire d'Inès samedi, on sera huit, il faut qu'on trouve un resto.", "silent", wait=3),
    Line(A, "Elle adore la cuisine italienne, on pourrait partir là-dessus.", "silent", wait=3),
    Line(S, "Oui, et plutôt dans le 11e, c'est à côté de chez elle.", "silent", wait=3),
    Line(A, "Il faudrait un endroit où on peut réserver pour un groupe.", "silent", wait=3),
    Line(S, "Kairos, trouve-nous un restaurant italien dans le 11e pour huit personnes samedi soir.", "speak",
         ("rue", "restaurant", "trattoria", "pizz"), wait=20),
    Line(A, "Ça a l'air bien. On garde celui-là en option.", "silent", wait=3),
    Line(S, "Et pour le gâteau, je peux le commander chez le pâtissier en bas de chez moi.", "silent", wait=3),
    Line(A, "Super. Et on se partage l'addition à huit ?", "silent", wait=3),
    Line(S, "Oui, sauf Inès évidemment. Si le repas fait 280 euros, à sept ça fait 40 euros chacun.", "silent",
         wait=5),
    Line(A, "Ça me va. Kairos, c'est à quelle heure d'habitude qu'on réserve un samedi soir à Paris ?", "speak",
         ("heure", "h", "20", "19", "21"), wait=15),
    Line(S, "Bon, je m'occupe de réserver demain.", "silent", wait=4),
]
ETUDE: list[Line] = [
    Line(A, "On révise l'histoire du vingtième siècle pour l'examen de vendredi.", "silent", wait=3),
    Line(S, "Commençons par la guerre froide, c'est ce qui tombe souvent.", "silent", wait=3),
    Line(A, "La construction du mur de Berlin, c'est en 1961 si je me souviens bien.", "silent", wait=3),
    Line(S, "Oui, et il tombe presque trente ans plus tard.", "silent", wait=3),
    Line(A, "Kairos, c'est en quelle année exactement la chute du mur de Berlin ?", "speak", ("1989",), wait=10),
    Line(S, "Ok merci. Ensuite il y a la crise des missiles de Cuba.", "silent", wait=3),
    Line(A, "Je crois que c'était en 1962, en octobre.", "silent", wait=3),
    Line(S, "On fait des fiches ou on se pose des questions à l'oral ?", "silent", wait=3),
    Line(A, "À l'oral, c'est plus efficace. Kairos, pose-nous une question sur la guerre froide.", "speak", wait=12),
    Line(S, "Bonne question, je pense que c'est le plan Marshall.", "any", wait=8),
    Line(A, "Allez, on fait une pause de dix minutes.", "silent", wait=4),
]
TECH: list[Line] = [
    Line(T, "Il faut que je change d'ordinateur, le mien a presque huit ans.", "silent", wait=3),
    Line(N, "Tu fais quoi dessus surtout, du montage vidéo ou juste du bureau ?", "silent", wait=3),
    Line(T, "Un peu de montage, et beaucoup de navigateur avec plein d'onglets.", "silent", wait=3),
    Line(N, "Tu préfères rester sur Windows ou passer sur Mac ?", "silent", wait=3),
    Line(T, "J'hésite, le Mac me tente pour l'autonomie.", "silent", wait=3),
    Line(N, "Kairos, quel est le prix d'un MacBook Air en ce moment en France ?", "speak", ("euro",), wait=18),
    Line(T, "C'est un peu cher mais ça tient longtemps.", "silent", wait=3),
    Line(N, "Et en Windows, tu regardais quelle marque ?", "silent", wait=3),
    Line(T, "Plutôt Lenovo ou Asus, pour le rapport qualité prix.", "silent", wait=3),
    Line(N, "Kairos, un bon ordinateur portable Windows pour du montage léger à moins de 1000 euros ?", "speak",
         ("asus", "lenovo", "acer", "dell", "hp", "euro"), wait=18),
    Line(T, "Merci, je vais comparer ce week-end.", "silent", wait=4),
]
ENGLISH: list[Line] = [
    Line(J, "Alright, let's plan the team offsite, we said Barcelona in early November.", "silent", wait=3),
    Line(C, "Yes, the sixth to the eighth of November works for everyone except Mark.", "silent", wait=3),
    Line(J, "Mark can join on the seventh, that's fine.", "silent", wait=3),
    Line(C, "Most of us fly from London, right?", "silent", wait=3),
    Line(J, "Kairos, can you find flights from London to Barcelona on November sixth?", "speak",
         ("pound", "euro", "£", "€"), wait=20),
    Line(C, "That's affordable. We should book this week before prices go up.", "silent", wait=3),
    Line(J, "Agreed. What about activities, I'd like something outdoors.", "silent", wait=3),
    Line(C, "Kairos, what's the weather usually like in Barcelona in early November?", "speak",
         ("degree", "°", "rain", "warm", "mild", "sun"), wait=18),
    Line(J, "Good, a boat trip could work then.", "silent", wait=3),
    Line(C, "Let's wrap up, I'll send a summary tonight.", "silent", wait=4),
]

#: The session language of a scenario (French unless said).
LANGUAGE = {"english": "en"}

SCENARIOS: dict[str, list[Line]] = {
    "reunion": REUNION,
    "travail": TRAVAIL,
    "diner": DINER,
    "etude": ETUDE,
    "tech": TECH,
    "english": ENGLISH,
    "appel": [
        Line(C, "Salut Léa, ça va depuis la dernière fois ?", "silent", wait=4),
        Line(L, "Oui très bien, et toi ?", "silent", wait=4),
        Line(C, "Kairos, tu es là ?", "speak", wait=6),
        Line(L, "Qui ose, tu peux nous aider à organiser un week-end ?", "speak", wait=8),
    ],
    "bavardage": [
        Line(C, "Tu as vu le match hier soir ?", "silent", wait=4),
        Line(L, "Non, j'étais chez ma sœur, on a regardé un film.", "silent", wait=4),
        Line(C, "Ah sympa, lequel ?", "silent", wait=4),
        Line(L, "Une comédie, rien de fou, mais on a bien ri.", "silent", wait=4),
        Line(C, "Bon, on se fait un café tout à l'heure ?", "silent", wait=5),
    ],
    "calcul": [
        Line(C, "Pour le cadeau de Marc, on est cinq.", "silent", wait=4),
        Line(L, "On met 40 euros chacun, ça fait 180 euros au total.", "speak", ("200", "deux cents"), wait=8),
    ],
    "vols": [
        Line(C, "Kairos, trouve-nous un vol Paris Lisbonne le 17 octobre.", "speak", ("euro",), wait=20),
    ],
    "hotel": [
        Line(C, "On part à Lisbonne du 17 au 20 octobre, on sera deux.", "any", wait=6),
        Line(L, "Kairos, tu peux regarder les hôtels là-bas ?", "speak", ("euro", "nuit", "hôtel"), wait=25),
    ],
    "train": [
        Line(C, "Kairos, il y a un train Paris Bruxelles demain matin ?", "speak", ("train", "heure", "h"), wait=20),
    ],
    "activites": [
        Line(C, "On sera à Lisbonne un samedi soir.", "any", wait=5),
        Line(L, "Kairos, qu'est-ce qu'on peut faire là-bas le soir ?", "speak", ("alfama", "fado", "bairro", "bar",
                                                                            "restaurant", "concert"), wait=20),
    ],
    "question_ouverte": [
        Line(C, "Au fait, qui a gagné la Coupe du monde cette année ?", "speak", ("espagne",), wait=20),
    ],
    "repetition": [
        Line(C, "Kairos, c'est quoi la capitale de l'Australie ?", "speak", ("canberra",), wait=8),
        Line(L, "Ah ok, je vois.", "silent", wait=6),
        Line(C, "D'accord.", "silent", wait=5),
    ],
}


def iso(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


async def run(name: str, lines: list[Line]) -> list[Result]:
    language = LANGUAGE.get(name, "fr")
    state: dict = {}
    async with websockets.connect(BASE.replace("http", "ws", 1) + "/v1/live", max_size=2**24) as ws, \
            httpx.AsyncClient(base_url=BASE, timeout=20) as http:

        async def pump(seconds: float) -> None:
            nonlocal state
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                try:
                    message = json.loads(await asyncio.wait_for(ws.recv(), 0.2))
                except TimeoutError:
                    continue
                if message.get("type") == "state.snapshot":
                    state = message["state"]

        await ws.send(json.dumps({"type": "client.hello", "protocol_version": PROTOCOL, "client_id": "scenarios"}))
        await ws.send(json.dumps({"type": "session.start", "language": language, "capture_mode": "microphone"}))
        await pump(4)  # speech recognition and voice connect
        results: list[Result] = []
        for line in lines:
            before = {u["id"] for u in state.get("transcript", [])}
            await http.post("/api/say", json={"speaker": line.speaker, "text": line.text})
            # The line is typed at speech rate: wait until it is committed.
            deadline = time.monotonic() + len(line.text.split()) / 2.0 + 6
            committed = None
            while time.monotonic() < deadline and committed is None:
                await pump(0.3)
                committed = next((u for u in state.get("transcript", []) if u["id"] not in before
                                  and u["text"][:20].lower() in line.text.lower() + " "), None) or next(
                    (u for u in state.get("transcript", []) if u["id"] not in before), None)
            await pump(line.wait)
            end = iso(committed["committed_at"]) if committed else time.time()
            result = Result(line)
            for speech in state.get("speeches", []):
                at = iso(speech["created_at"])
                if at >= end - 0.2 and at <= end + line.wait + 1:
                    result.heard.append(Heard(speech["text"], round(at - end, 1), speech["status"] == "interrupted"))
            results.append(result)
            mark = "OK  " if result.ok else "FAIL"
            print(f"{mark} [{line.expect}{'/' + ','.join(line.words) if line.words else ''}] {line.speaker}: {line.text}")
            for heard in result.heard:
                print(f"       +{heard.delay:4.1f}s {'(cut) ' if heard.cut else ''}Kairos: {heard.text}")
        await ws.send(json.dumps({"type": "session.command", "command": "stop"}))
        await pump(2)
    said = [h.text for r in results for h in r.heard if h.text and not h.cut]
    repeats = [text for i, text in enumerate(said) if any(_same(text, other) for other in said[:i])]
    if repeats:
        print(f"FAIL repetition: {repeats}")
    return results + ([Result(Line("-", "no repetition", "silent"), [Heard(t, 0, False) for t in repeats])]
                      if repeats else [])


def _same(a: str, b: str) -> bool:
    words_a, words_b = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    return bool(words_a and words_b) and len(words_a & words_b) / len(words_a | words_b) >= 0.7


CUES = {"alors…", "hmm, voyons…", "oui…", "attends…", "so…", "hmm, let's see…", "right…", "one sec…"}
CALL = re.compile(r"\b(kairos|qui ose)\b", re.IGNORECASE)


def monitor(results: list[Result]) -> dict:
    """Calls served, unprompted interventions, reaction and answer delays, repetitions."""
    calls = [r for r in results if CALL.search(r.line.text)]
    served = [r for r in calls if any(h.text and h.text.lower() not in CUES for h in r.heard)]
    reactions = [min(h.delay for h in r.heard if h.text) for r in calls if any(h.text for h in r.heard)]
    answers = [min(h.delay for h in r.heard if h.text and h.text.lower() not in CUES and len(h.text.split()) > 3)
               for r in calls if any(h.text and len(h.text.split()) > 3 for h in r.heard)]
    unprompted = [h.text for r in results if not CALL.search(r.line.text) and r.line.speaker != "-"
                  for h in r.heard if h.text]
    return {"calls": len(calls), "served": len(served), "unprompted": unprompted,
            "reaction_s": round(sum(reactions) / len(reactions), 1) if reactions else None,
            "answer_s": round(sum(answers) / len(answers), 1) if answers else None}


JUDGE = ("You review what Kairos, an AI assistant attending a meeting, said. Kairos should behave as an assistant: "
         "answer when someone calls it (by name), give the information asked for, correct a clearly wrong figure, "
         "and otherwise stay silent. Given the conversation before it and whether it was called, rate the "
         'intervention. Return JSON {"appropriate": true|false, "score": 1-5, "why": "one short sentence"}. '
         "5 = exactly what an excellent assistant would say then (right, useful, concise, well timed); 1 = wrong, "
         "useless, repetitive or an interruption nobody asked for.")


async def judge(results: list[Result]) -> list[dict]:
    """Each intervention rated in its context by a model: was it the right thing to say?"""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from kairos.config import Settings
    from kairos.session import make_llm

    llm = make_llm(Settings())
    history: list[str] = []
    verdicts = []
    for r in results:
        if r.line.speaker == "-":
            continue
        history.append(f"{r.line.speaker}: {r.line.text}")
        for h in r.heard:
            if not h.text or h.text.lower() in CUES:
                continue
            called = bool(CALL.search(r.line.text))
            user = ("Conversation (latest last):\n" + "\n".join(history[-8:])
                    + f"\n\nKairos was {'called on the last line' if called else 'NOT called'}; it said, "
                    f"{h.delay:.1f} s after the last line: {h.text}")
            try:
                verdict = await llm.json(JUDGE, user, purpose="scenario judge", temperature=0.0)
            except Exception as error:
                verdict = {"appropriate": None, "score": None, "why": f"judge failed: {type(error).__name__}"}
            verdicts.append({"said": h.text, "called": called, **verdict})
            history.append(f"Kairos: {h.text}")
    return verdicts


async def main() -> None:
    chosen = [name for name in sys.argv[1:] if name in SCENARIOS] or list(SCENARIOS)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    report = {}
    for name in chosen:
        print(f"\n=== {name}")
        results = await run(name, SCENARIOS[name])
        passed = sum(r.ok for r in results)
        delays = [h.delay for r in results for h in r.heard if h.text]
        first = min(delays) if delays else None
        watch = monitor(results)
        verdicts = await judge(results)
        scores = [v["score"] for v in verdicts if isinstance(v.get("score"), int | float)]
        line = (f"{name}: {passed}/{len(results)} lines, calls served {watch['served']}/{watch['calls']}, "
                f"unprompted {len(watch['unprompted'])}, reaction {watch['reaction_s']}s, answer {watch['answer_s']}s")
        if scores:
            line += f", judge {sum(scores) / len(scores):.1f}/5"
        elif first is not None:
            line += f", first reply +{first:.1f}s"
        summary.append(line)
        for v in verdicts:
            if v.get("appropriate") is False:
                print(f"   judge: NOT appropriate ({v.get('score')}/5, {'called' if v['called'] else 'not called'}): "
                      f"{v['said'][:90]} -- {v.get('why')}")
        report[name] = {"lines": [{"line": asdict(r.line), "ok": r.ok, "heard": [asdict(h) for h in r.heard]}
                                  for r in results], "monitor": watch, "judge": verdicts}
    print("\n" + "\n".join(summary))
    path = OUT / time.strftime("%Y%m%d-%H%M%S.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {path}")


if __name__ == "__main__":
    asyncio.run(main())
