"""The dispatcher: Jev reads each sentence and decides which background work starts.

One request per sentence (and early, on a partial ending with "?"), yes/no questions:
is Kairos addressed, which search helps (flights through Jinko for a trip, tickets or their price, the web
for anything else), did the subject change
(the topic tracker then names the new one). The dispatcher never writes the reservoir: it opens the answer lane or a worker
brief, and the workers write what they find, or offer to search.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from ..board import Board
from .common import render_transcript

QUESTIONS = {
    "addressed": ("In the LAST line, is the speaker talking to Kairos, the AI assistant: naming it, or asking "
                  "'tu'/'vous' for something only the assistant could answer or do (look up, calculate, recall)? "
                  "Talking to another participant does not count, nor a question asking the group for its opinion "
                  "or judgment ('do you think...', 'what do you guys reckon'), which is for the people in the room, "
                  "nor a 'tu' question about a person's own plans, trip or life ('toi tu pars de Paris aussi ?'): "
                  "Kairos, the assistant, is not part of the plans."),
    "fact": ("Does the LAST line ask a factual question about the world that the people in the room do not seem to "
             "know (a result, a winner, a date, a price, how something works: 'qui a gagné la Coupe du monde ?'), "
             "or say they cannot remember a fact ('il y a eu une catastrophe dernièrement, je ne me rappelle plus "
             "c'était quoi', 'c'était quoi déjà le nom de ce film ?'), rather than a question for the people themselves (their plans, choices or preferences: 'on prend un "
             "hôtel ou un appartement ?', 'qu'est-ce qu'on fait sur place ?')?"),
    "new_topic": ("Do the five latest lines, taken together, talk about a different subject than the CURRENT TOPIC "
                  "(another trip, another problem, another plan)? No if they continue it, add details to it or come "
                  "back to it, and no when there is no current topic yet."),
}
#: Which background work helps now: one choice, so that the options are weighed against each other. Two separate
#: yes/no questions let "web" (0.96) always beat "flights" (0.61-0.70) on every travel request, and Jinko never ran.
WORK_QUESTION = {
    "type": "choice",
    "instructions": ("Which background search would help the speaker right now, given the LAST line and the "
                     "conversation? Pick one."),
    "criteria": {
        "flights": ("travel between places: the LAST line asks for, or needs, tickets, a way to get somewhere, a "
                    "trip's price, schedule or duration, or expresses a wish or plan to travel ('trouve-nous des "
                    "billets Paris Groningen', 'ça coûte combien le trajet ?' while a trip is discussed, 'I'd like "
                    "to go home to Cameroon'), unless the person only wants a train, a bus or a car"),
        "hotels": ("a place to stay: hotels, accommodation, rooms or nights somewhere ('tu peux regarder les hôtels "
                   "là-bas ?', 'où dormir à Lisbonne', 'a hotel near the station')"),
        "web": ("any other thing to find, check or look up: a place, a restaurant, an address, opening hours, a "
                "price of something else, a fact, a result, a trip explicitly by train, bus or car (SNCF, "
                "BlaBlaCar), a need or factual question nobody answered, even raised in passing, or someone who "
                "cannot remember a fact ('je ne me rappelle plus c'était quoi', 'c'était qui déjà ?')"),
        "none": ("no search helps: opinions, small talk, what only the team can know (its own decisions, budget, "
                 "people), travel mentioned only in passing, a line only reacting to what was said, or someone "
                 "saying what they will do themselves ('parfait, j'envoie un mail à l'équipe', 'je m'en occupe', "
                 "'on se rappelle demain')"),
    },
}
ADDRESSED = 0.55  # live sessions: every line Jev scored 0.55-0.65 was for Kairos ("Tu disais quoi ?", "Merci beaucoup")
WORK = 0.5  # the chosen search must be the likely one, and more likely than "none"
TOPIC = 0.5
FACT = 0.6  # a factual question to the room: its search result is owed to that line  # a cheap filter: the writing model confirms every change (live, Jev scores shifts 0.57-0.76)


@dataclass(slots=True)
class Dispatch:
    t: float
    segment: int
    text: str
    p: dict[str, float] = field(default_factory=dict)
    actions: list[str] = field(default_factory=list)
    seconds: float = 0.0


class Dispatcher:
    def __init__(self, board: Board, judge,
                 on_addressed: Callable[[int], None],
                 open_work: Callable[[str, int, str, bool], Awaitable[None]],
                 on_topic: Callable[[float], None] | None = None) -> None:
        self.board = board
        self.on_topic = on_topic  # called with Jev's probability that the subject changed
        self.judge = judge  # must offer ask() (Jev)
        self.on_addressed = on_addressed
        self.open_work = open_work
        self.log: list[Dispatch] = []
        #: segment -> work handed out, known before the worker has finished taking it
        self.routing: dict[int, list[str]] = {}
        self._done: dict[int, int] = {}  # segment -> number of words already dispatched
        self.facts: set[int] = set()  # lines asking the room a factual question: their result is owed

    def wants(self, segment: int, text: str, final: bool) -> bool:
        """Dispatch a line once when it ends with "?", and again when committed if it grew."""
        words = len(text.split())
        seen = self._done.get(segment)
        if seen is None:
            return final or text.rstrip().endswith("?")
        return final and words > seen + 3

    def claim(self, segment: int, text: str) -> None:
        """Mark the line as dispatched before the request leaves, so the committed line is not sent twice."""
        self._done[segment] = len(text.split())

    async def run(self, segment: int, text: str) -> Dispatch | None:
        self.claim(segment, text)
        snap = self.board.snapshot()
        memory = "\n".join(f"- {m}" for m in snap.long_term) or "(none)"
        topic = snap.signals.topic or "(none yet)"
        state = (f"Kairos is an AI assistant attending this meeting.\n\nCURRENT TOPIC: {topic}\n\n"
                 f"TRANSCRIPT (latest last):\n{render_transcript(snap, last=8)}\n\nLAST line: {text}\n\n"
                 f"Kairos's memory:\n{memory}")
        start = time.perf_counter()
        try:
            questions: dict[str, dict] = {k: {"type": "noul", "instructions": v} for k, v in QUESTIONS.items()}
            answers = await self.judge.ask(state, questions | {"work": WORK_QUESTION}, purpose="dispatcher")
        except Exception:
            return None  # Jev unavailable: the regex fast lane and the thinkers still work
        p = {k: _p(answers.get(k)) for k in QUESTIONS} | _choices(answers.get("work"), WORK_QUESTION["criteria"])
        d = Dispatch(t=snap.room.t, segment=segment, text=text, p=p, seconds=round(time.perf_counter() - start, 2))
        addressed = p["addressed"] >= ADDRESSED
        self.routing[segment] = d.actions
        if self.on_topic is not None:
            self.on_topic(p["new_topic"])
        if addressed:
            d.actions.append("réponse")
            self.on_addressed(segment)
        if p.get("fact", 0.0) >= FACT:
            self.facts.add(segment)
        work = max(("flights", "hotels", "web", "none"), key=lambda option: p.get(option, 0.0))
        if work != "none" and p[work] >= WORK and p[work] > p.get("none", 0.0):
            if work == "flights":
                # Tickets, a trip, its price: the flight worker (Jinko), asked for or not. It asks only what it
                # cannot assume.
                d.actions.append("vols")
            elif work == "hotels":
                d.actions.append("hôtels")  # Jinko's live rates, for the trip being discussed
            else:
                # A restaurant, an address, a schedule, a fact, a train: the web worker, asked for or not. Asked
                # for, it owes the reply ("je regarde", then the result); otherwise it is offered at a good pause.
                d.actions.append("recherche web")
            await self.open_work(work, segment, text, addressed)
        self.log.append(d)
        del self.log[:-100]
        return d


def _choices(answer, options) -> dict[str, float]:
    """A choice's probability per option (0 for an option Jev did not score)."""
    probabilities = answer.get("probabilities") if isinstance(answer, dict) else None
    if not isinstance(probabilities, dict):
        chosen = answer.get("choice") if isinstance(answer, dict) else None
        return {option: 1.0 if option == chosen else 0.0 for option in options}
    return {option: float(probabilities.get(option) or 0.0) for option in options}


def _p(answer) -> float:
    if isinstance(answer, dict):
        for key in ("noul", "probability"):
            if isinstance(answer.get(key), (int, float)):
                return float(answer[key])
    return 0.0
