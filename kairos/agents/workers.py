"""Background workers and their briefs.

A brief is what a worker needs to do its job. It is opened by the dispatcher, when someone
asks for it or simply expresses the need ("j'ai envie de rentrer au Cameroun"). It is filled
from Kairos's memory and what was said (every value with its source); what is missing
becomes one thought in the reservoir: an offer when Kairos takes the initiative ("Tu veux
que je regarde les vols ? Tu partirais d'où, et quand ?"), a question when it was asked.
While a brief waits, the next sentences belong to it: each one can complete it, and a
follow-up question asks only for what is still missing. Once complete, the worker runs in
the background and its result comes back as a thought owed to the person who answered.

    needs_details ─ question said ─► asked ─ answers ─► running ─► done
                                       └─ no progress for 4 lines, or declined ─► expired

Workers: "flights" (Jinko) and "web" (Exa: places, restaurants, schedules, prices, facts).
Every search Kairos announces is a brief here: what it says it does, it does, and a failure is said too.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import itertools
import re
from dataclasses import dataclass, field, replace

from ..board import Board
from ..contracts import Finding, Thought, ThoughtStatus
from ..llm import LLM
from ..travel import JinkoFlights, JinkoHotels
from .researcher import NOT_FOUND, spoken
from .common import enforce_cap, language_of, recent_ack, render_transcript
from .thinkers import _speak_in

SPECS: dict[str, dict] = {
    "flights": {
        "goal": "search flights for the trip the person needs",
        "details": {
            "origin": "departure city with its IATA code, e.g. \"Paris (PAR)\"",
            "destination": ("destination with IATA codes; for a country or several cities, list the main airports, "
                            "e.g. \"Cameroun: Douala (DLA), Yaoundé (NSI)\""),
            "date": ("departure date as YYYY-MM-DD, or up to 7 comma-separated dates when a period around a day was "
                     "given; the next occurrence, never in the past; a month alone is not enough"),
        },
        "optional": {"return_date": "return date as YYYY-MM-DD, only if one was given",
                     "travellers": "number of people flying, a whole number (prices are per person: never ask)"},
    },
    "hotels": {
        "goal": "find hotels for the stay the person needs",
        "details": {
            "city": ("the city to stay in, its English name with its two-letter ISO country code, e.g. \"Lisbon (PT)\" "
                     "for Lisbonne; \"là-bas\" "
                     "or \"sur place\" is the destination of the trip being discussed"),
            "checkin": "arrival date as YYYY-MM-DD: the trip's arrival day when a trip is being discussed",
            "checkout": ("departure date as YYYY-MM-DD: the trip's return day, or arrival plus the number of nights "
                         "said; if nobody said how long, assume 2 nights"),
            "guests": "number of people staying, a whole number",
        },
        "optional": {"stars": "minimum star rating asked for (\"un cinq étoiles\" -> 5), only if said"},
    },
    "web": {
        "goal": "find on the web what the person needs (a place, a restaurant, an address, a schedule, a price, a fact)",
        "details": {
            "query": ("a specific web search query: what is sought and, when it matters, where or when (e.g. "
                      "\"restaurant végétarien pas cher Paris 5e\"); a broad but real need is searchable as it is "
                      "(\"des villes avec de belles plages en Afrique\" -> \"meilleures villes balnéaires Afrique\"); "
                      "a vague question on recent news is searchable with today's month (\"la catastrophe de "
                      "ces derniers jours\" -> \"catastrophe naturelle actualité septembre 2026\"); "
                      "missing only when there is nothing to search at all (\"un resto\" with no place); "
                      "\"là-bas\", \"there\", \"sur place\" is the place mentioned just before or in the notes "
                      "(\"On sera à Lisbonne\" then \"quoi faire là-bas le soir ?\" -> \"que faire le soir à "
                      "Lisbonne\"); never the assistant's name (Kairos); no names of people, nothing about this team"),
        },
        "optional": {},
    },
}

FILL_SYSTEM = """You prepare the brief of a background worker for Kairos, an AI assistant attending a conversation.
Goal of the worker: {goal}.
Required details:
{details}
Optional details:
{optional}
Kairos is an AI assistant: it has no preferences of its own ("mars me convient" is wrong; "d'accord pour mars" is right).
The transcript comes from live speech recognition: a name that sounds like "Kairos" ("Kéros", "Kiros", "Caïros") is people talking to the assistant, never a place, a restaurant or a name to search. Other misheard words: understand them by the conversation, and take names of places from what was said or found before (a restaurant Kairos proposed).
Fill each detail from the transcript, the meeting notes, the other searches and Kairos's memory.
The latest line wins: when it changes something already known (other dates, "deux nuits", "le week-end prochain", a star rating, another city), replace the old value with it.
Link the request to the plan being discussed: "là-bas", "the hotels there", "pour le voyage", "on the same day" take the destination, the dates and the number of people of the trip in the notes or in the other searches; never ask again for what the conversation already settled. Take what the conversation makes clear, even if said in other words ("rentrer au Cameroun" gives the destination; "I'm alone" gives 1 traveller); never invent what nobody said.
Return one JSON object: {{"details": {{"name": {{"value": "...", "source": "L12 or M3"}}}}, "missing": ["name"], "assumed": {{"name": {{"value": "...", "why": "..."}}}}, "declined": false, "question": "..."}}.
"assumed" proposes a sensible default for every missing detail that has one: 1 traveller when the person speaks only of themselves, the number of people mentioned otherwise ("on sera six", "with my two boys" -> 3); when no date at all was given, the next 7 days from tomorrow; when only a month or a period was given ("en mars", "around December", "end of year"), up to 7 comma-separated dates in the middle of it (e.g. March -> 2027-03-12,...,2027-03-18). Never for a place nobody mentioned.
"declined" is true only if the person clearly refused the search.
"question" is empty when nothing required is missing; otherwise ONE short spoken sentence asking for every missing detail at once. Write it in {language}: the language of the person's latest line, even if the conversation started in another language. {style} Use "tu" if the person says "tu" or speaks casually alone with Kairos."""

NO_REPEAT = ('Your previous question was: "{previous}". Do not repeat it: ask only what is still missing, in fewer and '
             'different words.')
OFFER = ('Kairos was not asked: make it an offer, e.g. "Tu veux que je regarde les vols pour le Cameroun ? Tu partirais '
         'd\'où, et vers quelle date ?"')
ASK = 'Kairos was asked: say what you need, e.g. "Pour chercher les vols, tu partirais quand ?"'
FOLLOW_UP = 'The person already answered part of it: acknowledge briefly and ask only for what is still missing.'

#: What Kairos says when a search starts (varied: the same sentence every time sounds like a machine).
LAUNCH = {"French": ["Je regarde.", "C'est noté, je cherche.", "D'accord, je regarde ça.", "Je lance la recherche."],
          "English": ["Let me look.", "Got it, searching now.", "Okay, I'll check.", "On it."]}
#: What Kairos says when the search itself could not run (a provider down).
UNAVAILABLE = {"French": "Je n'arrive pas à obtenir les prix pour l'instant. Je peux réessayer dans un moment.",
               "English": "I can't get the prices right now. I can try again in a moment."}
#: What Kairos says when a search it owed someone found nothing.
FAILED = {"French": "Je n'ai rien trouvé de fiable sur « {q} ». Tu peux préciser un peu ?",
          "English": "I couldn't find anything reliable on \"{q}\". Can you tell me a bit more?"}

HOTEL_SAY_SYSTEM = """Kairos, an AI assistant in a conversation, searched hotels. Say the result in two short spoken sentences in {language}: the best value option (name, rating out of 10, price per night and in total), and one alternative. Prices are live rates for these dates. Plain words, no symbols, no codes. Return JSON {{"say": "..."}}."""

SAY_SYSTEM = """Kairos, an AI assistant in a conversation, searched flights. Say the result in two short spoken sentences in {language}, answering what the person wanted (cheapest option, direct or not, duration, dates): destination, stops, time, price per person, and that prices are indicative. If Kairos already gave a result for this trip, say only what changed ("with the return on January 1st, it's 1085 euros"), not the whole result again. Plain words, no symbols, no codes. Return JSON {{"say": "..."}}."""


@dataclass(slots=True)
class Brief:
    id: str
    kind: str
    segment: int
    line: str
    addressed: bool
    created_at: float
    status: str = "needs_details"   # needs_details · asked · running · done · failed · expired
    details: dict[str, dict] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    question: str = ""
    question_id: str | None = None
    questions_asked: int = 0
    asked_at: float | None = None
    stalled: int = 0                  # lines since the last progress, while waiting for answers
    last_line: int | None = None      # the latest line that completed part of it (Kairos owes it a reply)
    language: str = "French"
    result: str = ""
    sources: tuple[str, ...] = ()
    error: str = ""
    history: list[tuple[float, str]] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)  # defaults used because the person did not say
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)  # one fill at a time: never two questions


class WorkerSupervisor:
    def __init__(self, board: Board, llm: LLM, language: str = "French", flights: JinkoFlights | None = None,
                 researcher=None, today: dt.date | None = None, search=None,
                 hotels: JinkoHotels | None = None) -> None:
        self.board = board
        self.llm = llm
        self.language = language
        self.flights = flights
        self.hotels = hotels
        self.researcher = researcher
        self.search = search if search is not None else getattr(researcher, "provider", None)
        self._launches = 0
        self.today = today or dt.date.today()
        self.briefs: list[Brief] = []
        self._ids = itertools.count(1)
        self._tasks: set[asyncio.Task] = set()
        self._owned: set[int] = set()  # lines a brief took as answers: the answer lane leaves them alone
        self.room_questions: set[int] = set()  # factual questions to the room (the dispatcher's reading)

    # -- the dispatcher opens work ----------------------------------------------------

    def can(self, kind: str) -> bool:
        return ((kind == "flights" and self.flights is not None) or (kind == "web" and self.search is not None)
                or (kind == "hotels" and self.hotels is not None))

    def waiting(self) -> list[Brief]:
        return [b for b in self.briefs if b.status in ("needs_details", "asked")]

    def handles(self, segment: int) -> bool:
        """A brief answers this line: the request itself, or an answer to the brief's question."""
        return segment in self._owned or any(b.segment == segment and b.addressed and b.status not in ("expired", "failed")
                                             for b in self.briefs)

    async def open(self, kind: str, segment: int, line: str, addressed: bool) -> Brief | None:
        if not self.can(kind):
            return None
        active = next((b for b in self.waiting() if b.kind == kind), None)
        if active is not None:
            # Same need raised again ("what are the prices?"): the line belongs to the waiting brief, which
            # takes what it brings and asks again for what is still missing.
            active.addressed = active.addressed or addressed
            await self.absorb(segment, reask=addressed)
            return active
        if kind == "flights" and any(b.kind == "flights" and b.status == "running" for b in self.briefs):
            return None
        now = self.board.snapshot().room.t
        recent = next((b for b in reversed(self.briefs) if b.kind == kind and b.status == "done"
                       and now - b.created_at < 180), None)
        brief = Brief(id=f"B{next(self._ids)}", kind=kind, segment=segment, line=line, addressed=addressed,
                      created_at=now, language=language_of(line, self.language))
        if recent is not None:
            # A search just ran: this line may refine it (a return date, another day), or only ask about it.
            brief.details = {k: dict(v) for k, v in recent.details.items()}
        brief.history.append((now, f"ouvert ({'demandé' if addressed else 'besoin détecté'}) : {line[:80]}"))
        if recent is not None:
            # A trial brief, kept out of the list until it proves new: meanwhile the answer lane must not think
            # a worker handles this line ("tu as trouvé ?" would get no reply at all).
            before = _request_key(brief.details)
            async with brief.lock:
                await self._fill(brief, segment, run=False)
            if _request_key(brief.details) == before:
                if addressed:
                    self._repeat_unsaid_result(recent, segment)  # nothing new: answered from the last result
                return None
        self.briefs.append(brief)
        if recent is not None:
            brief.history.append((self.board.snapshot().room.t, f"affine la recherche {recent.id}"))
            if not brief.missing:
                if addressed:
                    # Asked: "je regarde", and the refined result is owed to this line.
                    brief.last_line = segment
                    self._acknowledge(brief)
                    self._owned.add(segment)
                # Not asked ("le fado ça me tente bien"): the search runs silently, its result waits.
                self._spawn(self.run(brief))
                return brief
            brief.last_line = segment
        if addressed:
            self._owned.add(segment)
            brief.last_line = segment  # Kairos owes this person a reply: "je regarde", then the result
        await self.fill(brief)
        return brief

    async def request_web(self, question: str, query: str, segment: int | None) -> Brief | None:
        """The answer lane does not know: a web brief, already complete, that owes the result to that line."""
        if not self.can("web"):
            return None
        now = self.board.snapshot().room.t
        brief = Brief(id=f"B{next(self._ids)}", kind="web", segment=segment or 0, line=question, addressed=True,
                      created_at=now, language=language_of(question, self.language), last_line=segment,
                      details={"query": {"value": query, "source": f"L{segment}"}})
        brief.history.append((now, f"ouvert (réponse inconnue) : {question[:80]}"))
        self.briefs.append(brief)
        if segment is not None:
            self._owned.add(segment)
        self._spawn(self.run(brief))  # the answer lane already said "je regarde"
        return brief

    # -- filling ---------------------------------------------------------------------

    async def absorb(self, segment: int, reask: bool = False) -> bool:
        """A new line while briefs wait: does it complete one? If so, the brief owns the line (one voice).
        With `reask` (the line is about the brief's subject), the brief owns it anyway and asks again."""
        took = False
        for brief in self.waiting():
            if segment == brief.segment:
                continue
            async with brief.lock:
                if brief.status in ("needs_details", "asked") and await self._fill(brief, segment, reask=reask):
                    took = True
        if took:
            self._owned.add(segment)
        return took

    async def fill(self, brief: Brief, segment: int | None = None) -> bool:
        """Fill the brief from memory and the transcript. True if this call made progress."""
        async with brief.lock:
            if brief.status not in ("needs_details", "asked"):
                return False
            return await self._fill(brief, segment)

    async def _fill(self, brief: Brief, segment: int | None, reask: bool = False, run: bool = True) -> bool:
        spec = SPECS[brief.kind]
        snap = self.board.snapshot()
        line = next((s for s in snap.transcript if s.id == segment), None)
        if line is not None:
            brief.language = language_of(line.text, brief.language)
        memory = "\n".join(f"M{i}: {m}" for i, m in enumerate(snap.long_term)) or "(none)"
        known = "\n".join(f"- {k}: {v.get('value')} ({v.get('source')})" for k, v in brief.details.items()) or "(none)"
        style = FOLLOW_UP if brief.questions_asked else (ASK if brief.addressed else OFFER)
        if brief.question:
            style += " " + NO_REPEAT.format(previous=brief.question)
        system = FILL_SYSTEM.format(goal=spec["goal"], language=brief.language, style=style,
                                    details="\n".join(f"- {k}: {v}" for k, v in spec["details"].items()),
                                    optional="\n".join(f"- {k}: {v}" for k, v in spec["optional"].items()) or "(none)")
        others = "\n".join(
            f"- {b.id} ({b.kind}, {b.status}): " + ", ".join(f"{k} = {v.get('value')}" for k, v in b.details.items())
            for b in self.briefs[-6:] if b is not brief and b.details) or "(none)"
        user = (f"Today is {self.today.isoformat()}.\n\nKairos's memory:\n{memory}\n\n"
                f"Meeting notes (short-term memory: what the room said and settled so far):\n{snap.notes or '(none yet)'}"
                f"\n\nOther searches in this conversation, with their details:\n{others}\n\n"
                f"Transcript (latest last):\n{render_transcript(snap, last=14)}\n\n"
                f"The line that opened this work: L{brief.segment}: {brief.line}\n\nAlready known:\n{known}")
        try:
            raw = await self.llm.json(_speak_in(brief.language) + system, user, purpose="brief filler",
                                      temperature=0.2)
        except Exception as exc:
            brief.history.append((snap.room.t, f"remplissage échoué : {type(exc).__name__}"))
            return False
        before = {k for k, v in brief.details.items() if _valid(k, v)}
        values_before = {k: v.get("value") for k, v in brief.details.items()}
        names = set(spec["details"]) | set(spec["optional"])
        for name, item in (raw.get("details") or {}).items():
            if name in names and isinstance(item, dict) and str(item.get("value") or "").strip():
                brief.details[name] = {"value": str(item["value"]).strip(), "source": str(item.get("source") or "")}
        now = self.board.snapshot().room.t
        if raw.get("declined") is True:
            brief.status = "expired"
            brief.history.append((now, "refusé par la personne"))
            self._retire_question(brief, "la personne a décliné")
            return True
        brief.missing = [n for n in spec["details"] if n not in brief.details or not _valid(n, brief.details[n])]
        assumed = {k: v for k, v in (raw.get("assumed") or {}).items()
                   if k in brief.missing and isinstance(v, dict) and _valid(k, v)}
        if (brief.questions_asked or brief.addressed) and assumed and not snap.room.speaking:
            # Asked for, or already asked once: what can be assumed (the next days, one traveller, a week in March)
            # is assumed now and said with the result; only what cannot be guessed (where from, where to) is asked.
            # Too many questions before a search felt like a form. Never while the person is still speaking: the
            # date may be the next words ("je pars de Paris le..." searched the next 7 days before "27 août").
            for k, v in assumed.items():
                brief.details[k] = {"value": str(v["value"]), "source": "supposé"}
                brief.assumptions.append(f"{k} = {v['value']} ({v.get('why', '')})")
            brief.missing = [n for n in brief.missing if n not in assumed]
            if not brief.missing and segment is not None:
                brief.last_line = segment
        progress = {k for k, v in brief.details.items() if _valid(k, v)} - before
        if not progress and segment is not None and any(v.get("value") != values_before.get(k)
                                                        for k, v in brief.details.items()):
            progress = {"partial"}  # "en mars peut-être": not a date yet, but an answer; ask only what is left
        if progress and segment is not None:
            brief.last_line = segment
            brief.stalled = 0
        brief.history.append((now, "complet" if not brief.missing else f"manque : {', '.join(brief.missing)}"))
        if not run:
            return bool(progress)
        if not brief.missing:
            self._retire_question(brief, "réponses reçues")
            if brief.last_line is not None and (brief.asked_at is not None or brief.addressed):
                self._acknowledge(brief)  # someone is waiting: "je regarde" at once, the result follows
            self._spawn(self.run(brief))
            return bool(progress) or segment is None
        question = str(raw.get("question") or "").strip()
        reply = segment if (progress or reask) and segment is not None else None
        if question and (brief.question_id is None or (reply is not None and brief.status in ("asked", "needs_details"))):
            # First question, or a follow-up asking only for what is still missing (to the person who spoke).
            if brief.question_id is not None and not brief.addressed and brief.asked_at is None:
                return bool(progress)  # an offer nobody took up is not made again, in other words or not
            if not progress and not reask and _normalized(question) == _normalized(brief.question):
                return False  # the same question again, nothing new said, nobody asked about it: once was enough
            self._retire_question(brief, "remplacée par une question plus précise")
            brief.question = question
            self._ask(brief, reply_to=reply)
        return bool(progress) or reply is not None

    def on_line(self, segment: int, speaker: str | None) -> None:
        """A committed sentence: it may complete a waiting brief, or show the conversation moved on."""
        for brief in self.waiting():
            if segment == brief.segment:
                continue
            if brief.status == "asked":
                brief.stalled += 1
                if brief.stalled > 4:
                    brief.status = "expired"
                    brief.history.append((self.board.snapshot().room.t, "expiré : plus de réponse"))
                    self._retire_question(brief, "personne n'a répondu")
                    continue
            self._spawn(self.absorb(segment))
            break

    def on_spoken(self, thought_ids: list[str]) -> None:
        """The speaker said a question for the room: from now on the brief waits for the answers."""
        for brief in self.briefs:
            if brief.question_id in thought_ids and brief.status in ("needs_details", "asked"):
                brief.status = "asked"
                brief.asked_at = self.board.snapshot().room.t
                brief.stalled = 0
                brief.history.append((brief.asked_at, "question posée"))

    def _ask(self, brief: Brief, reply_to: int | None = None) -> None:
        snap = self.board.snapshot()
        brief.questions_asked += 1
        owed = reply_to if reply_to is not None else (brief.segment if brief.addressed else None)
        if owed is not None:
            self._open_reply(owed)
        thought = Thought(
            id=f"q{brief.id}n{brief.questions_asked}", topic=SPECS[brief.kind]["goal"],
            content=f"Missing for {brief.kind}: {', '.join(brief.missing)}", utterance=brief.question, transition=None,
            # A reply to the person, or an offer answering a need just expressed: decisive (it passes the talk
            # budget), and there is one per brief. An offer still has to be picked by Jev at a good pause.
            importance=5.0, relevance=0.0,
            fit_now=1.0 if owed is not None else 0.0,
            already_said=0.0, status=ThoughtStatus.READY, stimuli=(f"L{brief.segment}", brief.id),
            version=snap.version, created_at=snap.room.t, answers=owed, kind="question", brief=brief.id,
            note=("offre" if owed is None else "il manque") + f" : {', '.join(brief.missing)}")
        brief.question_id = thought.id
        self.board.update_thoughts({}, (thought,), by=f"travaux {brief.id}")
        enforce_cap(self.board)

    def _open_reply(self, segment: int) -> None:
        """Kairos owes a reply to this line: the policy's answer lane (low bar, first pause) applies."""
        signals = self.board.snapshot().signals
        self.board.publish("signals", replace(signals, addressed=0.9, addressed_segment=segment, answered=False))

    def _acknowledge(self, brief: Brief) -> None:
        snap = self.board.snapshot()
        line = brief.last_line
        brief.addressed = True  # Kairos said "je regarde": the result is now owed to this person
        self._open_reply(line)
        if recent_ack(snap):
            return  # the answer lane already said "je regarde" for this request: one is enough
        self.board.update_thoughts({}, (Thought(
            id=f"a{brief.id}", topic=SPECS[brief.kind]["goal"], content="search launched",
            utterance=self._launch(brief.language), transition=None, importance=5.0, relevance=0.0,
            fit_now=1.0, already_said=0.0, status=ThoughtStatus.READY, stimuli=(f"L{line}", brief.id),
            version=snap.version, created_at=snap.room.t, answers=line, ack=True, kind="answer", brief=brief.id,
            note="la personne a répondu : recherche lancée"),), by=f"travaux {brief.id}")

    def _repeat_unsaid_result(self, recent: Brief, segment: int) -> None:
        """The person asks for what a search found: if its result never reached the conversation, it is owed now."""
        snap = self.board.snapshot()
        result = next((t for t in snap.thoughts if t.id == f"r{recent.id}"), None)
        said = any(t.id.startswith(f"r{recent.id}") and t.status == ThoughtStatus.SPOKEN for t in snap.thoughts)
        if result is None or said:
            return  # already said, in one copy or another (the answer lane confirms briefly), or nothing to give
        self._owned.add(segment)
        self._open_reply(segment)
        changes = {result.id: {"status": ThoughtStatus.STALE, "note": "redonnée en réponse"}} \
            if result.status in (ThoughtStatus.READY, ThoughtStatus.PENDING) else {}
        self.board.update_thoughts(changes, (Thought(
            id=f"r{recent.id}a{segment}", topic=result.topic, content=result.content, utterance=result.utterance,
            transition=None, importance=5.0, relevance=0.0, fit_now=1.0, already_said=0.0,
            status=ThoughtStatus.READY, stimuli=(f"L{segment}", recent.id), version=snap.version,
            created_at=snap.room.t, answers=segment, kind="finding", brief=recent.id,
            note=f"résultat de {recent.id}, demandé à nouveau"),), by=f"travaux {recent.id}")

    def _launch(self, language: str) -> str:
        options = LAUNCH.get(language, LAUNCH["English"])
        self._launches += 1
        return options[(self._launches - 1) % len(options)]

    def _retire_question(self, brief: Brief, why: str) -> None:
        if brief.question_id is None:
            return
        current = next((t for t in self.board.snapshot().thoughts if t.id == brief.question_id), None)
        if current is not None and current.status in (ThoughtStatus.READY, ThoughtStatus.PENDING):
            self.board.update_thoughts({current.id: {"status": ThoughtStatus.STALE, "note": why}},
                                       by=f"travaux {brief.id}")

    def _owed_line(self, brief: Brief) -> int | None:
        """The line Kairos owes the result to: the latest answer, else the request itself."""
        if brief.last_line is not None and (brief.asked_at is not None or brief.addressed):
            return brief.last_line
        return brief.segment if brief.addressed else None

    def _result_line(self, brief: Brief) -> int | None:
        """The line a result answers: the one it is owed to, or a question put to the room ("qui a gagné la Coupe
        du monde ?"): the room waits for that answer, so it is said at the next pause, not when Jev happens to
        pick it (15 s later in a live test). Offers and questions for details do not get this."""
        owed = self._owed_line(brief)
        line = owed if owed is not None else (brief.segment if brief.segment in self.room_questions else None)
        if line is not None and any(t.answers == line and t.kind == "finding" and t.status == ThoughtStatus.SPOKEN
                                    for t in self.board.snapshot().thoughts):
            return None  # another search already answered this line: a second answer would repeat it
        return line

    # -- running ---------------------------------------------------------------------

    async def run(self, brief: Brief) -> None:
        if brief.status in ("running", "done"):
            return
        brief.status = "running"
        brief.history.append((self.board.snapshot().room.t, "lancé"))
        try:
            if brief.kind == "flights":
                await self._run_flights(brief)
            elif brief.kind == "hotels":
                await self._run_hotels(brief)
            else:
                await self._run_web(brief)
        except Exception as exc:
            brief.status, brief.error = "failed", f"{type(exc).__name__}: {str(exc)[:120]}"
            brief.history.append((self.board.snapshot().room.t, f"échec : {brief.error[:80]}"))
            await self._after_failure(brief)
            return
        brief.history.append((self.board.snapshot().room.t, brief.status))

    async def _after_failure(self, brief: Brief) -> None:
        """A search failed. Flights: try the web on the same trip. Someone waiting always hears something."""
        if brief.kind == "flights" and self.search is not None:
            d = brief.details
            query = " ".join(str(d.get(k, {}).get("value", "")) for k in ("origin", "destination", "date"))
            brief.details["query"] = {"value": f"prix billet avion {query}", "source": "vols indisponibles"}
            brief.status = "running"
            brief.history.append((self.board.snapshot().room.t, "repli sur la recherche web"))
            try:
                await self._run_web(brief)
                return
            except Exception as exc:
                brief.status, brief.error = "failed", f"{type(exc).__name__}: {str(exc)[:120]}"
        owed = self._owed_line(brief)
        if owed is None:
            return
        snap = self.board.snapshot()
        self._open_reply(owed)
        self.board.update_thoughts({}, (Thought(
            id=f"r{brief.id}", topic="recherche", content=f"search failed: {brief.error[:80]}",
            utterance=UNAVAILABLE.get(brief.language, UNAVAILABLE["English"]), transition=None, importance=5.0,
            relevance=0.0, fit_now=1.0, already_said=0.0, status=ThoughtStatus.READY,
            stimuli=(f"L{brief.segment}", brief.id), version=snap.version, created_at=snap.room.t, answers=owed,
            kind="finding", brief=brief.id, note=f"échec de {brief.id}"),), by=f"travaux {brief.id}")

    async def _run_web(self, brief: Brief) -> None:
        query = brief.details["query"]["value"]
        owed = self._result_line(brief)
        snap = self.board.snapshot()
        owed_line = next((s for s in snap.transcript if s.id == owed), None)
        if owed_line is not None:
            brief.language = language_of(owed_line.text, brief.language)
        finding = Finding(id=f"f{brief.id}", question=brief.line, query=query, segment=brief.segment,
                          status="searching", started_at=snap.room.t)
        self.board.publish("findings", tuple(f for f in snap.findings if f.id != finding.id) + (finding,))
        result = await self.search.search(query, owed_line.text if owed_line else brief.line, brief.language)
        found = bool(result.answer) and not NOT_FOUND.search(result.answer)
        brief.result, brief.sources = result.answer, tuple(result.sources)
        snap = self.board.snapshot()
        done = Finding(id=finding.id, question=brief.line, query=query, segment=brief.segment,
                       status="done" if found else "failed", started_at=finding.started_at, answer=result.answer,
                       sources=tuple(result.sources), seconds=round(result.seconds, 2))
        self.board.publish("findings", tuple(f for f in snap.findings if f.id != finding.id) + (done,))
        brief.status = "done" if found else "failed"
        if found:
            utterance = spoken(result.answer)
        elif owed is not None:
            utterance = FAILED.get(brief.language, FAILED["English"]).format(q=query)  # a promise gets an answer
        else:
            return  # a background search that found nothing: no need to bother the room
        if found and self._supersede(brief) and owed is None:
            owed = brief.segment  # the line waited for an answer: the finding keeps that promise
        self._retire_question(brief, "remplacée par le résultat")
        if owed is not None:
            self._open_reply(owed)
        self.board.update_thoughts({}, (Thought(
            id=f"r{brief.id}", topic="recherche", content=result.answer[:200] or query, utterance=utterance,
            transition=None, importance=5.0 if owed is not None or found else 4.0, relevance=0.0,
            fit_now=1.0 if owed is not None else 0.0, already_said=0.0, status=ThoughtStatus.READY,
            stimuli=(f"L{brief.segment}", brief.id), version=snap.version, created_at=snap.room.t, answers=owed,
            kind="finding", brief=brief.id, note=f"résultat de {brief.id}" if found else f"rien trouvé ({brief.id})"),),
            by=f"travaux {brief.id}")
        enforce_cap(self.board)

    async def _run_flights(self, brief: Brief) -> None:
        d = brief.details
        dates = re.findall(r"\d{4}-\d{2}-\d{2}", d["date"]["value"])[:7]
        back = re.findall(r"\d{4}-\d{2}-\d{2}", d.get("return_date", {}).get("value", ""))
        summary = await self.flights.search(_codes(d["origin"]["value"])[:2], _codes(d["destination"]["value"])[:3],
                                            dates, int(re.sub(r"\D", "", d.get("travellers", {}).get("value", "1")) or 1),
                                            return_date=back[0] if back else None)
        brief.result = summary.as_text()
        owed_line = next((s for s in self.board.snapshot().transcript if s.id == self._owed_line(brief)), None)
        if owed_line is not None:
            brief.language = language_of(owed_line.text, brief.language)  # the language of who it answers
        recent = [s.text for s in self.board.snapshot().transcript if s.speaker == "Kairos"][-3:]
        said = await self.llm.json(_speak_in(brief.language) + SAY_SYSTEM.format(language=brief.language),
                                   f"What the person asked: {brief.line}"
                                   + (f"\nTheir latest words: {owed_line.text}" if owed_line is not None else "")
                                   + (f"\nAssumptions you made (say them briefly and invite a correction): "
                                      f"{'; '.join(brief.assumptions)}" if brief.assumptions else "")
                                   + (f"\nWhat Kairos already said (do not repeat it, e.g. that prices are "
                                      f"indicative): {' | '.join(recent)}" if recent else "")
                                   + f"\n\nResults:\n{brief.result}",
                                   purpose="brief result", temperature=0.2)
        utterance = str(said.get("say") or "").strip()
        snap = self.board.snapshot()
        finding = Finding(id=f"f{brief.id}", question=brief.line, query=f"vols {summary.route} {summary.date}",
                          segment=brief.segment, status="done", started_at=brief.created_at, answer=brief.result,
                          sources=("Jinko (prix indicatifs)",), seconds=round(summary.seconds, 2))
        self.board.publish("findings", tuple(f for f in snap.findings if f.id != finding.id) + (finding,))
        brief.status = "done"
        if not utterance:
            return
        self._retire_question(brief, "remplacée par le résultat")
        owed = self._result_line(brief)
        if self._supersede(brief) and owed is None:
            owed = brief.segment  # the line waited for an answer: the result keeps that promise
        if owed is not None:
            self._open_reply(owed)
        self.board.update_thoughts({}, (Thought(
            id=f"r{brief.id}", topic="vols", content=brief.result.splitlines()[0], utterance=utterance, transition=None,
            importance=5.0 if owed is not None else 4.0, relevance=0.0, fit_now=1.0 if owed is not None else 0.0,
            already_said=0.0, status=ThoughtStatus.READY, stimuli=(f"L{brief.segment}", brief.id), version=snap.version,
            created_at=snap.room.t, answers=owed, kind="finding",
            brief=brief.id, note=f"résultat de {brief.id}"),), by=f"travaux {brief.id}")
        enforce_cap(self.board)

    async def _run_hotels(self, brief: Brief) -> None:
        d = brief.details
        city, country = _city(d["city"]["value"])
        checkin = re.findall(r"\d{4}-\d{2}-\d{2}", d["checkin"]["value"])[0]
        checkout = re.findall(r"\d{4}-\d{2}-\d{2}", d["checkout"]["value"])[0]
        guests = int(re.sub(r"\D", "", d["guests"]["value"]) or 2)
        stars = int(re.sub(r"\D", "", d.get("stars", {}).get("value", "")) or 0)
        summary = await self.hotels.search(city, country, checkin, checkout, guests, min_stars=stars)
        brief.result = summary.as_text()
        owed_line = next((s for s in self.board.snapshot().transcript if s.id == self._owed_line(brief)), None)
        if owed_line is not None:
            brief.language = language_of(owed_line.text, brief.language)
        said = await self.llm.json(_speak_in(brief.language) + HOTEL_SAY_SYSTEM.format(language=brief.language),
                                   f"What the person asked: {brief.line}"
                                   + (f"\nTheir latest words: {owed_line.text}" if owed_line is not None else "")
                                   + (f"\nAssumptions you made (say them briefly): {'; '.join(brief.assumptions)}"
                                      if brief.assumptions else "")
                                   + f"\n\nResults:\n{brief.result}", purpose="brief result", temperature=0.2)
        snap = self.board.snapshot()
        finding = Finding(id=f"f{brief.id}", question=brief.line, query=f"hôtels {city} {checkin} {checkout}",
                          segment=brief.segment, status="done", started_at=brief.created_at, answer=brief.result,
                          sources=("Jinko (tarifs en direct)",), seconds=round(summary.seconds, 2))
        self.board.publish("findings", tuple(f for f in snap.findings if f.id != finding.id) + (finding,))
        brief.status = "done"
        utterance = str(said.get("say") or "").strip()
        if utterance:
            self._publish_result(brief, utterance, "hôtels", brief.result.splitlines()[0])

    def _publish_result(self, brief: Brief, utterance: str, topic: str, content: str) -> None:
        """A travel result as a thought, owed to the line that asked (or that waited for an answer)."""
        self._retire_question(brief, "remplacée par le résultat")
        owed = self._result_line(brief)
        if self._supersede(brief) and owed is None:
            owed = brief.segment
        if owed is not None:
            self._open_reply(owed)
        snap = self.board.snapshot()
        self.board.update_thoughts({}, (Thought(
            id=f"r{brief.id}", topic=topic, content=content, utterance=utterance, transition=None,
            importance=5.0 if owed is not None else 4.0, relevance=0.0, fit_now=1.0 if owed is not None else 0.0,
            already_said=0.0, status=ThoughtStatus.READY, stimuli=(f"L{brief.segment}", brief.id), version=snap.version,
            created_at=snap.room.t, answers=owed, kind="finding",
            brief=brief.id, note=f"résultat de {brief.id}"),), by=f"travaux {brief.id}")
        enforce_cap(self.board)

    def _supersede(self, brief: Brief) -> bool:
        """What a search found replaces an answer to the same line written from the model's own knowledge
        ("la Coupe du monde 2026 n'a pas encore eu lieu", when the search found who won): that answer is not
        said. True if one was waiting, so the result inherits its promise."""
        lines = {brief.segment, brief.last_line}
        replaced = {t.id: {"status": ThoughtStatus.STALE, "note": f"remplacée par ce que {brief.id} a trouvé"}
                    for t in self.board.snapshot().thoughts
                    if t.kind == "answer" and not t.ack and t.answers in lines
                    and t.status in (ThoughtStatus.READY, ThoughtStatus.PENDING)}
        if replaced:
            self.board.update_thoughts(replaced, by=f"travaux {brief.id}")
        return bool(replaced)

    # -- for the other agents and the console ------------------------------------------

    def status_text(self) -> str:
        """What is running, for the answer lane ("tu en es où ?") and the thinkers."""
        lines = []
        for b in self.briefs[-5:]:
            known = ", ".join(f"{k}={v['value']}" for k, v in b.details.items())
            state = {"needs_details": f"waiting for: {', '.join(b.missing)}", "asked": f"asked for: {', '.join(b.missing)}",
                     "running": "searching now", "done": f"done: {b.result.splitlines()[0] if b.result else ''}",
                     "failed": f"failed: {b.error}", "expired": "dropped"}.get(b.status, b.status)
            lines.append(f"- {b.id} {SPECS[b.kind]['goal']} (from L{b.segment}): {state}; known: {known or 'nothing'}")
        return "\n".join(lines) or "(none)"

    def _spawn(self, coro) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def drain(self) -> None:
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)


def _request_key(details: dict[str, dict]) -> tuple:
    """What the search depends on: codes, dates, numbers. "Cameroun :" and "Cameroun:" are the same request."""
    return tuple(sorted((k, tuple(sorted(re.findall(r"\b[A-Z]{3}\b|\d{4}-\d{2}-\d{2}|\d+", v.get("value", "")))))
                        for k, v in details.items()))


def _codes(value: str) -> list[str]:
    codes = re.findall(r"\b([A-Z]{3})\b", value)
    return list(dict.fromkeys(codes)) or [value.strip()[:3].upper()]


def _valid(name: str, item: dict) -> bool:
    value = str(item.get("value") or "")
    if name in ("origin", "destination"):
        return bool(re.search(r"\b[A-Z]{3}\b", value))
    if name in ("date", "return_date"):
        return bool(re.search(r"\d{4}-\d{2}-\d{2}", value))
    if name == "travellers":
        return bool(re.search(r"\d", value))
    return bool(value.strip())


def _city(value: str) -> tuple[str, str]:
    """"Lisbonne (PT)" -> ("Lisbonne", "PT"); without a code, France."""
    match = re.search(r"\(([A-Za-z]{2})\)", value)
    return re.sub(r"\s*\(.*?\)", "", value).strip(), (match.group(1).upper() if match else "FR")


def _normalized(text: str) -> str:
    return " ".join(re.findall(r"\w+", (text or "").lower()))
