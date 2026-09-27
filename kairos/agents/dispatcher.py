"""The dispatcher: Jev reads each sentence and decides which background work starts.

One request per sentence (and early, on a partial ending with "?"), yes/no questions:
is Kairos addressed, would a web search help (a restaurant, a place, a schedule, a fact), would
a flight search help (asked for, or a need: "j'ai envie de rentrer au Cameroun"), did the subject change
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
    "search": ("Would a web search help the speaker right now? Yes if the LAST line asks Kairos or the room to "
               "find, recommend, look up or check something (a restaurant, a place, an address, opening hours, a "
               "price, a schedule, a fact), or raises such a need or factual question nobody answered, even in "
               "passing ('nobody knows how much that would cost', 'I wonder how common that is'), or says they will "
               "have to find, book or compare something themselves ('je dois trouver des billets pour Groningen le "
               "11'): what a product, "
               "a technology or a component costs, market figures, how others do it. No for opinions, small talk, "
               "or what only the team can know (its own decisions, budget or people)."),
    "flights": ("Would a flight search (prices, schedules, options) help the speaker right now? Yes if the LAST line "
                "asks about flights or tickets, or expresses a wish, need or plan to travel somewhere by plane (even "
                "without asking Kairos, e.g. 'I'd like to go home to Cameroon', 'I don't know how to get a ticket'). "
                "No if travel is only mentioned in passing."),
    "new_topic": ("Do the five latest lines, taken together, talk about a different subject than the CURRENT TOPIC "
                  "(another trip, another problem, another plan)? No if they continue it, add details to it or come "
                  "back to it, and no when there is no current topic yet."),
}
ADDRESSED = 0.55  # live sessions: every line Jev scored 0.55-0.65 was for Kairos ("Tu disais quoi ?", "Merci beaucoup")
WORK = 0.6  # AMI: real factual gaps score 0.63-0.83, group talk and controls stay under 0.4; Jev still rates the offer
FLIGHTS = 0.8  # Jev leans towards flights whenever travel is the topic: a higher bar
TOPIC = 0.5  # a cheap filter: the writing model confirms every change (live, Jev scores shifts 0.57-0.76)


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
            answers = await self.judge.ask(state, {k: {"type": "noul", "instructions": v} for k, v in QUESTIONS.items()},
                                           purpose="dispatcher")
        except Exception:
            return None  # Jev unavailable: the regex fast lane and the thinkers still work
        p = {k: _p(answers.get(k)) for k in QUESTIONS}
        d = Dispatch(t=snap.room.t, segment=segment, text=text, p=p, seconds=round(time.perf_counter() - start, 2))
        addressed = p["addressed"] >= ADDRESSED
        self.routing[segment] = d.actions
        if self.on_topic is not None:
            self.on_topic(p["new_topic"])
        if addressed:
            d.actions.append("réponse")
            self.on_addressed(segment)
        # Prices, schedules, a trip to plan: the flight worker (Jinko), asked for or not. It asks what it misses.
        if p["flights"] >= FLIGHTS:
            d.actions.append("vols")
            await self.open_work("flights", segment, text, addressed)
        elif p["search"] >= WORK:
            # A restaurant, an address, a schedule, a fact: the web worker, asked for or not. Asked for, it owes
            # the reply ("je regarde", then the result); otherwise it is offered at a good pause.
            d.actions.append("recherche web")
            await self.open_work("web", segment, text, addressed)
        self.log.append(d)
        del self.log[:-100]
        return d


def _p(answer) -> float:
    if isinstance(answer, dict):
        for key in ("noul", "probability"):
            if isinstance(answer.get(key), (int, float)):
                return float(answer[key])
    return 0.0
