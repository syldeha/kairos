"""Version 2: Jev dispatcher, worker briefs that ask the room, Jev rater, the reservoir's log. No network."""

import asyncio
from dataclasses import replace
import datetime as dt
import json

import httpx

from kairos.agents.dispatcher import Dispatcher
from kairos.agents.workers import WorkerSupervisor
from kairos.board import Board
from kairos.contracts import RoomState, Segment, Signals, Thought, ThoughtStatus
from kairos.decide.policy import decide
from kairos.travel import FlightSummary


def thought(id, chosen=None, status=ThoughtStatus.READY, importance=4.0, judged_line=1, **kw):
    return Thought(id=id, topic=f"topic {id}", content=f"content {id}", utterance=f"utterance {id}", transition=None,
                   importance=importance, relevance=0.5, fit_now=0.9, already_said=0.0, status=status, stimuli=(),
                   version=0, created_at=0.0, judged_line=judged_line, chosen=chosen, **kw)


def board_with(thoughts=(), transcript=(), signals=Signals(), t=10.0) -> Board:
    board = Board()
    board.publish("room", RoomState(t=t, speaking=False, silence_since=t - 1.0, p_silence=(0.9,) * 4))
    board.publish("transcript", tuple(transcript))
    board.publish("thoughts", tuple(thoughts))
    board.publish("signals", signals)
    return board


# -- dispatcher --------------------------------------------------------------------------

class FakeJev:
    def __init__(self, **p):
        self.p = p

    async def ask(self, state, questions, purpose="judge"):
        answers = {}
        for k, question in questions.items():
            if question.get("type") == "choice":
                # flights / web / none: what each test gives, the rest to "none"
                flights, web = self.p.get("flights", 0.0), self.p.get("search", 0.0)
                answers[k] = {"choice": "", "probabilities": {"flights": flights, "web": web,
                                                              "none": max(0.0, 1.0 - flights - web)}}
            else:
                answers[k] = {"type": "noul", "noul": self.p.get(k, 0.0)}
        return answers


def test_dispatcher_opens_the_answer_lane_and_a_flight_brief():
    addressed, work = [], []

    async def open_work(kind, segment, text, is_addressed):
        work.append((kind, segment, is_addressed))

    board = board_with()
    d = Dispatcher(board, FakeJev(addressed=0.9, flights=0.85, search=0.1), addressed.append, open_work)
    result = asyncio.run(d.run(4, "Kairos, tu peux regarder les vols pour Lisbonne ?"))
    assert addressed == [4] and work == [("flights", 4, True)] and result.actions == ["réponse", "vols"]


def test_a_search_starts_the_web_worker_asked_for_or_not():
    work = []

    async def open_work(kind, segment, text, is_addressed):
        work.append((kind, is_addressed))

    asyncio.run(Dispatcher(board_with(), FakeJev(search=0.9), lambda s: None, open_work).run(2, "Il y a un bon resto par là ?"))
    asyncio.run(Dispatcher(board_with(), FakeJev(addressed=0.9, search=0.9), lambda s: None, open_work)
                .run(3, "Kairos, tu peux nous trouver un restaurant végétarien dans le 5e ?"))
    asyncio.run(Dispatcher(board_with(), FakeJev(addressed=0.9, search=0.2), lambda s: None, open_work)
                .run(4, "Kairos, tu es là ?"))
    assert work == [("web", False), ("web", True)]  # the third is for the answer lane alone


def test_dispatcher_runs_once_on_the_question_mark_and_again_only_if_the_line_grew():
    d = Dispatcher(board_with(), FakeJev(), lambda s: None, None)
    assert d.wants(1, "Tu sais combien ?", final=False)
    asyncio.run(d.run(1, "Tu sais combien ?"))
    assert not d.wants(1, "Tu sais combien ?", final=True)
    assert d.wants(1, "Tu sais combien ? Et aussi pour le retour en train de nuit", final=True)


# -- worker briefs -----------------------------------------------------------------------

class BriefLLM:
    """First fill: the departure city is missing. Once someone says "Paris", everything is known."""

    def __init__(self):
        self.calls = []

    async def json(self, system, user, *, purpose, temperature=0.4):
        self.calls.append(purpose)
        if purpose == "brief filler":
            details = {"destination": {"value": "Lisbonne (LIS)", "source": "L1"},
                       "date": {"value": "2027-05-14", "source": "L1"},
                       "travellers": {"value": "18", "source": "M0"}}
            if "Paris" in user:
                details["origin"] = {"value": "Paris (PAR)", "source": "L3"}
                return {"details": details, "missing": [], "question": ""}
            return {"details": details, "missing": ["origin"], "question": "Vous partez tous de Paris ?"}
        if purpose == "brief result":
            return {"say": "Il y a un vol direct à 6 h, environ 157 euros, prix indicatifs."}
        return {}


class FakeFlights:
    tracer = None

    def __init__(self):
        self.searched = []

    async def search(self, origin, destination, date, adults=1, return_date=None):
        self.searched.append((origin, destination, date, adults))
        return FlightSummary(f"{'/'.join(origin)}-{'/'.join(destination)}", ", ".join(date), 20, 144.0, 1,
                             ("- TAP Portugal, direct, departs 06:00 from ORY, 157 EUR",), "2026-09-22", 1.2)


def test_a_brief_asks_the_room_once_then_runs_when_answered():
    board = board_with(transcript=[Segment(1, "Inès", "On part à Lisbonne le 14 mai en avion.", 0, 3, True)])
    board.publish("long_term", ("M0: 18 personnes inscrites.",))
    flights, llm = FakeFlights(), BriefLLM()
    sup = WorkerSupervisor(board, llm, "French", flights=flights, today=dt.date(2026, 9, 27))

    async def scenario():
        brief = await sup.open("flights", 1, "Kairos, tu peux regarder les vols ?", addressed=True)
        assert brief.status == "needs_details" and brief.missing == ["origin"]
        [q] = [t for t in board.snapshot().thoughts if t.kind == "question"]
        assert q.utterance == "Vous partez tous de Paris ?" and q.answers == 1 and q.importance == 5.0
        assert sup.handles(1)  # the answer lane leaves this line to the worker
        sup.on_spoken([q.id])
        assert brief.status == "asked"
        board.publish("transcript", board.snapshot().transcript + (Segment(3, "Hugo", "Oui, on part de Paris.", 5, 7, True),))
        sup.on_line(3, "Hugo")
        await sup.drain()
        return brief

    brief = asyncio.run(scenario())
    assert brief.status == "done" and flights.searched == [(["PAR"], ["LIS"], ["2027-05-14"], 18)]
    thoughts = {t.id: t for t in board.snapshot().thoughts}
    assert thoughts["qB1n1"].status == ThoughtStatus.STALE and thoughts["qB1n1"].note == "réponses reçues"
    result = thoughts["rB1"]
    # Kairos owes a reply to the room's answer (L3): "c'est noté", then the result, at the first pauses.
    assert result.kind == "finding" and result.answers == 3 and "vol direct" in result.utterance
    ack = thoughts["aB1"]
    from kairos.agents.workers import LAUNCH
    assert ack.ack and ack.answers == 3 and ack.utterance in LAUNCH["French"]
    signals = board.snapshot().signals
    assert signals.addressed_segment == 3 and not signals.answered and sup.handles(3)
    assert board.snapshot().findings[-1].answer.startswith("PAR-LIS")
    assert any(e[2] == "added (question)" and e[3] == "travaux B1" for e in board.log)


def test_an_unanswered_question_expires_and_leaves_the_reservoir():
    board = board_with(transcript=[Segment(1, "Inès", "On part à Lisbonne.", 0, 3, True)])
    sup = WorkerSupervisor(board, BriefLLM(), "French", flights=FakeFlights())

    async def scenario():
        brief = await sup.open("flights", 1, "On prend l'avion ?", addressed=False)
        sup.on_spoken([brief.question_id])
        for seg in range(2, 8):
            sup.on_line(seg, "Hugo")  # the room talks about something else
        await sup.drain()
        return brief

    brief = asyncio.run(scenario())
    assert brief.status == "expired"
    assert {t.id: t for t in board.snapshot().thoughts}["qB1n1"].status == ThoughtStatus.STALE


# -- rater and trigger -------------------------------------------------------------------

def _line(id=1):
    return Segment(id, "Inès", "On a quinze mille euros.", 0, 3, True)


def test_the_trigger_says_what_jev_picks():
    board = board_with([thought("a", chosen=0.2), thought("b", chosen=0.7)], [_line()],
                       Signals(judged_segment=1, judged_words=5, rater_line=1, rater_none=0.1))
    d = decide(board.snapshot())
    assert d.speak and d.primary == "b" and "Jev" in d.why


def test_the_trigger_stays_silent_when_jev_says_none():
    board = board_with([thought("a", chosen=0.3)], [_line()],
                       Signals(judged_segment=1, judged_words=5, rater_line=1, rater_none=0.65))
    d = decide(board.snapshot())
    assert not d.speak and "nothing to say" in d.why


def test_without_a_rating_on_this_line_the_old_score_decides():
    board = board_with([thought("a", chosen=0.9, importance=5.0)], [_line(2)],
                       Signals(judged_segment=2, judged_words=5, rater_line=1, rater_none=0.0))
    board.update_thoughts({"a": {"judged_line": 2}})
    d = decide(board.snapshot())
    assert "Jev" not in d.why


# -- the reservoir's log -----------------------------------------------------------------

def test_every_change_is_logged_with_who_and_why():
    board = board_with([thought("a")])
    board.update_thoughts({"a": {"status": ThoughtStatus.STALE, "note": "déjà dit"}}, by="juge Jev")
    board.update_thoughts({}, (thought("b", kind="correction"),), by="vérificateur")
    events = [(e[1], e[2], e[3], e[4]) for e in board.log]
    assert ("a", "stale", "juge Jev", "déjà dit") in events and ("b", "added (correction)", "vérificateur", "") in events


# -- Exa ---------------------------------------------------------------------------------

def test_exa_results_become_a_spoken_answer_with_sources():
    from kairos.search import ExaSearch

    class Summarizer:
        async def json(self, system, user, *, purpose, temperature=0.4):
            assert "portugal.fr" in user
            return {"answer": "Non, pas de train direct, selon Portugal.fr.", "used": [1]}

    def handler(request):
        assert json.loads(request.content)["contents"] == {"highlights": True}
        return httpx.Response(200, json={"results": [
            {"title": "Voyager en train", "url": "https://www.portugal.fr/train", "highlights": ["pas de train direct"]}],
            "costDollars": {"total": 0.007}})

    exa = ExaSearch("key", Summarizer())
    exa._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    result = asyncio.run(exa.search("train direct Paris Lisbonne", "Il y a un train direct ?", "French"))
    assert result.answer.startswith("Non") and result.sources == ("https://www.portugal.fr/train",)
    assert abs(exa.tracer.cost_usd() - 0.007) < 1e-9


def test_flights_win_when_likely_and_travel_mentioned_in_passing_does_not_count():
    work = []

    async def open_work(kind, segment, text, is_addressed):
        work.append(kind)

    asyncio.run(Dispatcher(board_with(), FakeJev(flights=0.83, search=0.1), lambda s: None, open_work)
                .run(3, "Ça fait combien de temps de vol depuis Paris, déjà ?"))
    asyncio.run(Dispatcher(board_with(), FakeJev(flights=0.3, search=0.1), lambda s: None, open_work)
                .run(11, "Parfait, j'envoie un mail à l'équipe."))
    asyncio.run(Dispatcher(board_with(), FakeJev(flights=0.61, search=0.3), lambda s: None, open_work)
                .run(10, "Tu peux nous trouver un trajet Paris Groningen ?"))
    assert work == ["flights", "flights"]  # a trip goes to Jinko; a line Jev gives mostly to "none" opens nothing


def test_a_train_or_a_fact_goes_to_the_web_worker_even_when_travel_is_the_topic():
    work = []

    async def open_work(kind, segment, text, is_addressed):
        work.append(kind)

    asyncio.run(Dispatcher(board_with(), FakeJev(flights=0.2, search=0.75), lambda s: None, open_work)
                .run(5, "Non, regarde-moi un trajet en train avec la SNCF."))
    assert work == ["web"]


# -- fixes after the live test (Cameroon conversation) ------------------------------------

class CameroonLLM:
    """Destination known from "rentrer au Cameroun"; origin and date come later, one answer at a time."""

    def __init__(self):
        self.systems = []

    async def json(self, system, user, *, purpose, temperature=0.4):
        if purpose != "brief filler":
            return {"say": "Il y a un vol avec escale à 1085 euros, prix indicatifs."}
        self.systems.append(system)
        details = {"destination": {"value": "Cameroun: Douala (DLA), Yaoundé (NSI)", "source": "L1"},
                   "travellers": {"value": "1", "source": "L1"}}
        if "Paris" in user:
            details["origin"] = {"value": "Paris (PAR)", "source": "L2"}
        if "22" in user:
            details["date"] = {"value": "2026-12-22", "source": "L3"}
        missing = [k for k in ("origin", "date") if k not in details]
        return {"details": details, "missing": missing,
                "question": "Tu veux que je regarde les vols ? Tu partirais d'où, et quand ?" if missing else ""}


def test_a_need_becomes_an_offer_then_follow_ups_ask_only_what_is_missing():
    lines = [Segment(1, "Vous", "J'ai bien envie de rentrer au Cameroun, je suis seul.", 0, 3, True)]
    board = board_with(transcript=lines)
    llm, flights = CameroonLLM(), FakeFlights()
    sup = WorkerSupervisor(board, llm, "French", flights=flights, today=dt.date(2026, 9, 27))

    async def scenario():
        brief = await sup.open("flights", 1, lines[0].text, addressed=False)
        [offer] = [t for t in board.snapshot().thoughts if t.kind == "question"]
        assert "make it an offer" in llm.systems[0] and offer.answers is None and offer.importance == 5.0
        sup.on_spoken([offer.id])
        board.publish("transcript", board.snapshot().transcript + (Segment(2, "Vous", "Oui, de Paris.", 5, 6, True),))
        assert await sup.absorb(2)  # the answer belongs to the brief: the answer lane stays quiet
        assert sup.handles(2) and brief.missing == ["date"] and brief.questions_asked == 2
        assert "answered part of it" in llm.systems[-1]
        follow_up = next(t for t in board.snapshot().thoughts if t.id == brief.question_id)
        assert follow_up.answers == 2  # Kairos owes this reply to the person who answered
        board.publish("transcript", board.snapshot().transcript + (Segment(3, "Vous", "Le 22 décembre.", 8, 9, True),))
        assert await sup.absorb(3)
        await sup.drain()
        return brief

    brief = asyncio.run(scenario())
    assert brief.status == "done" and flights.searched == [(["PAR"], ["DLA", "NSI"], ["2026-12-22"], 1)]


def test_a_line_still_being_spoken_does_not_open_the_floor_too_soon():
    partial = Segment(1, "Vous", "Is there something cheaper than that? I mean", 0, 3, False)
    board = board_with([thought("a", chosen=0.9)], [partial],
                       Signals(judged_segment=1, judged_words=9, rater_line=1, rater_none=0.0))
    board.publish("room", RoomState(t=10.0, speaking=False, silence_since=9.6, p_silence=(0.9,) * 4))
    assert "not finished" in decide(board.snapshot()).why
    board.publish("room", RoomState(t=10.0, speaking=False, silence_since=8.8, p_silence=(0.9,) * 4))
    assert decide(board.snapshot()).speak  # a real pause (1.2 s): the floor is open


def test_the_speech_to_text_session_is_renewed_before_gradium_closes_it(monkeypatch):
    import kairos.sources.gradium as g

    class FakeWs:
        def __init__(self):
            self.sent = []

        async def send(self, text):
            self.sent.append(json.loads(text)["type"])

    src = g.GradiumSource("key", "wss://example/api/speech")
    src.attach(None, lambda: 0.0)
    src.push_audio(b"\x00" * 3840)
    ws = FakeWs()
    monkeypatch.setattr(g, "ROTATE_S", -1.0)  # the session is already too old
    asyncio.run(src._send(ws, 0.0))
    assert ws.sent == ["end_of_stream"] and src._carry == b"\x00" * 3840  # the chunk goes to the next session


def test_kairos_answers_in_the_language_just_used():
    from kairos.agents.common import language_of
    assert language_of("Can you think what are the prices of the tickets around December?") == "English"
    assert language_of("Tu sais combien de temps dure le vol ?") == "French"
    assert language_of("OK", "French") == "French"


def test_a_question_about_the_waiting_search_is_answered_by_the_brief_itself():
    lines = [Segment(1, "Vous", "J'ai bien envie de rentrer au Cameroun, je suis seul.", 0, 3, True),
             Segment(2, "Vous", "Can you think what are the prices of the tickets?", 5, 7, True)]
    board = board_with(transcript=lines)
    sup = WorkerSupervisor(board, CameroonLLM(), "French", flights=FakeFlights())

    async def scenario():
        brief = await sup.open("flights", 1, lines[0].text, addressed=False)
        sup.on_spoken([brief.question_id])
        again = await sup.open("flights", 2, lines[1].text, addressed=True)  # nothing new, but about the search
        return brief, again

    brief, again = asyncio.run(scenario())
    assert again is brief and sup.handles(2) and brief.questions_asked == 2
    follow_up = next(t for t in board.snapshot().thoughts if t.id == brief.question_id)
    assert follow_up.answers == 2  # it asks again, to the person who just spoke, what is still missing


def test_what_did_you_find_does_not_search_again_but_a_new_date_refines_the_search():
    lines = [Segment(1, "Vous", "From Paris to Cameroon, alone, 22 December.", 0, 3, True)]
    board = board_with(transcript=lines)
    flights = FakeFlights()
    sup = WorkerSupervisor(board, CameroonLLM(), "French", flights=flights)

    async def scenario():
        await sup.open("flights", 1, lines[0].text, addressed=True)
        await sup.drain()
        board.publish("transcript", board.snapshot().transcript + (Segment(2, "Vous", "What did you find?", 5, 6, True),))
        repeat = await sup.open("flights", 2, "What did you find?", addressed=True)
        return repeat

    assert asyncio.run(scenario()) is None and len(flights.searched) == 1


# -- round 1 of the fluidity work: promises kept ------------------------------------------

class FakeSearch:
    def __init__(self, answer):
        self.answer, self.queries = answer, []

    async def search(self, query, question, language):
        from kairos.search import SearchResult
        self.queries.append(query)
        return SearchResult(self.answer, ("https://example.org",), 1.0)


class QueryLLM:
    async def json(self, system, user, *, purpose, temperature=0.4):
        return {"details": {"query": {"value": "restaurant végétarien pas cher Paris 5e", "source": "L1"}},
                "missing": [], "question": ""}


def test_a_restaurant_request_is_searched_and_the_result_is_owed_to_the_person():
    lines = [Segment(1, "Vous", "Kairos, tu peux nous trouver un resto végétarien pas cher dans le 5e ?", 0, 4, True)]
    board = board_with(transcript=lines)
    search = FakeSearch("Selon Le Fooding, Le Grenier de Notre-Dame, végétarien, environ 20 euros.")
    sup = WorkerSupervisor(board, QueryLLM(), "French", search=search)

    async def scenario():
        await sup.open("web", 1, lines[0].text, addressed=True)
        await sup.drain()

    asyncio.run(scenario())
    thoughts = {t.id: t for t in board.snapshot().thoughts}
    assert search.queries == ["restaurant végétarien pas cher Paris 5e"]
    assert thoughts["aB1"].ack and thoughts["aB1"].answers == 1  # "je regarde" backed by a real search
    assert thoughts["rB1"].answers == 1 and "Grenier" in thoughts["rB1"].utterance


def test_a_search_that_finds_nothing_is_still_answered_when_someone_waits():
    board = board_with(transcript=[Segment(1, "Vous", "Tu penses que Wokflou ?", 0, 2, True)])
    sup = WorkerSupervisor(board, QueryLLM(), "French", search=FakeSearch("Je n'ai pas trouvé d'informations sur Wokflou."))

    async def scenario():
        await sup.request_web("Tu penses que Wokflou ?", "Wokflou restaurant Paris", 1)
        await sup.drain()

    asyncio.run(scenario())
    result = {t.id: t for t in board.snapshot().thoughts}["rB1"]
    assert result.answers == 1 and result.utterance.startswith("Je n'ai rien trouvé")
    assert sup.briefs[0].status == "failed" and "Wokflou" in sup.status_text()


class AssumingLLM:
    async def json(self, system, user, *, purpose, temperature=0.4):
        if purpose == "brief result":
            return {"say": "En supposant un départ vers le 15 décembre, c'est à partir de 423 euros."}
        return {"details": {"origin": {"value": "Paris (PAR)", "source": "L1"},
                            "destination": {"value": "Douala (DLA)", "source": "L1"},
                            "travellers": {"value": "1", "source": "L1"}},
                "missing": ["date"], "assumed": {"date": {"value": "2026-12-14,2026-12-15,2026-12-16", "why": "mi-décembre"}},
                "question": "Tu partirais quand ?"}


def test_asked_without_a_precise_date_kairos_searches_at_once_on_stated_assumptions():
    lines = [Segment(1, "Vous", "Je veux aller de Paris à Douala en décembre, seul.", 0, 3, True)]
    board = board_with(transcript=lines)
    flights = FakeFlights()
    sup = WorkerSupervisor(board, AssumingLLM(), "French", flights=flights)

    async def scenario():
        brief = await sup.open("flights", 1, lines[0].text, addressed=True)
        await sup.drain()
        return brief

    brief = asyncio.run(scenario())
    # Asked for: no form to fill first ("quelle date ?"); a week in December is assumed and said with the result.
    assert brief.questions_asked == 0 and brief.status == "done"
    assert brief.assumptions and flights.searched[0][2][0] == "2026-12-14"
    assert "supposant" in {t.id: t for t in board.snapshot().thoughts}["rB1"].utterance


def test_a_question_left_without_answer_is_replied_to_by_what_jev_picks():
    question = Segment(4, "Vous", "Est-ce que tu as trouvé un truc ?", 0, 2.0, True)
    idea = thought("green", chosen=0.95, judged_line=4)
    signals = Signals(addressed=0.9, addressed_segment=4, answered=False, judged_segment=4, judged_words=7,
                      rater_line=4, rater_none=0.05)
    early = board_with([idea], [question], signals, t=4.0)
    assert not decide(early.snapshot()).speak  # the answer lane still has time
    late = board_with([idea], [question], signals, t=8.0)
    d = decide(late.snapshot())
    assert d.speak and d.primary == "green" and d.reason == "asked"


def test_a_thought_in_the_old_language_is_not_said_after_the_conversation_switched():
    english = Segment(3, "Vous", "I would really like to travel this end of year", 0, 3, True)
    french = replace(thought("fr", chosen=0.9, judged_line=3),
                     utterance="Pour t'aider à trouver un billet, de quelle ville pars-tu et à quelle date ?")
    board = board_with([french], [english], Signals(judged_segment=3, judged_words=10, rater_line=3, rater_none=0.05))
    assert not decide(board.snapshot()).speak
    same = replace(french, utterance="Which city would you fly from, and around which date?")
    board = board_with([same], [english], Signals(judged_segment=3, judged_words=10, rater_line=3, rater_none=0.05))
    assert decide(board.snapshot()).speak


class DownFlights:
    tracer = None

    async def search(self, *args, **kwargs):
        raise RuntimeError("502 Bad Gateway")


def test_a_flight_search_that_fails_falls_back_to_the_web_and_someone_waiting_hears_it():
    lines = [Segment(1, "Vous", "From Paris to Douala, alone, 22 December.", 0, 3, True)]
    board = board_with(transcript=lines)
    search = FakeSearch("Selon Kayak, un aller Paris-Douala en décembre coûte environ 450 euros.")
    sup = WorkerSupervisor(board, CameroonLLM(), "French", flights=DownFlights(), search=search)

    async def scenario():
        await sup.open("flights", 1, lines[0].text, addressed=True)
        await sup.drain()

    asyncio.run(scenario())
    assert search.queries and "Douala" in search.queries[0]
    assert "Kayak" in {t.id: t for t in board.snapshot().thoughts}["rB1"].utterance

    board = board_with(transcript=lines)
    nothing = WorkerSupervisor(board, CameroonLLM(), "French", flights=DownFlights())  # no web either

    async def scenario2():
        await nothing.open("flights", 1, lines[0].text, addressed=True)
        await nothing.drain()

    asyncio.run(scenario2())
    said = {t.id: t for t in board.snapshot().thoughts}["rB1"]
    assert said.answers == 1 and said.utterance.startswith("Je n'arrive pas")


# -- context research ----------------------------------------------------------------------

def test_context_research_reads_up_on_the_subject_and_stores_facts_not_thoughts():
    from kairos.agents.background import ContextResearch

    class Planner:
        async def json(self, system, user, *, purpose, temperature=0.4):
            return {"subject": "choosing remote control buttons", "queries": ["share of remote buttons people use"]}

        async def embed(self, texts, *, purpose):
            return [[1.0, 0.0] for _ in texts]

    lines = [Segment(i, "B", f"line {i} about buttons", i, i + 1, True) for i in range(1, 6)]
    board = board_with(transcript=lines)
    search = FakeSearch("According to a survey, people use about 10 percent of the buttons.")
    ctx = ContextResearch(board, Planner(), search, min_interval_s=90)
    asyncio.run(ctx.run_once())
    [f] = board.snapshot().findings
    assert f.status == "done" and "10 percent" in f.answer and f.question.startswith("contexte")
    assert not board.snapshot().thoughts  # facts for the thinkers, not something to say
    asyncio.run(ctx.run_once())  # too soon, and the same query anyway
    assert search.queries == ["share of remote buttons people use"]


# -- the checker computes numbers in code ---------------------------------------------------

def test_the_checker_corrects_a_wrong_calculation_with_a_number_computed_in_code():
    from kairos.agents.relevance import Relevance
    from kairos.agents.thinkers import Thinkers

    class CalcLLM:
        def __init__(self, reply):
            self.reply = reply

        async def json(self, system, user, *, purpose, temperature=0.4):
            return self.reply

        async def embed(self, texts, *, purpose):
            return [[float(len(t) % 7), 1.0] for t in texts]

    lines = (Segment(1, "A", "The selling price is 25 euros and the production cost 20.50 euros.", 0, 5, True),
             Segment(2, "A", "We need 50 million profit, so we sell two million remotes.", 6, 10, True))

    def run(reply):
        board = board_with(transcript=lines)
        llm = CalcLLM(reply)
        asyncio.run(Thinkers(board, llm, "English", relevance=Relevance(board, llm)).check(2))
        return board.snapshot().thoughts

    [fix] = run({"correction": {"topic": "units", "kind": "calculation", "expression": "50000000 / (25 - 20.5)",
                                "utterance": "With a 4.50 euro margin, that means about {result} remotes, not 2 million."}})
    assert "11.1 million" in fix.utterance and fix.importance == 5.0 and fix.kind == "correction"
    assert run({"correction": {"topic": "x", "expression": "__import__('os')", "utterance": "about {result}"}}) == ()
    assert run({"correction": {"topic": "x", "expression": "50000000 / 4.5", "utterance": "about 11 million"}}) == ()
    # An unclear base: the figure and its share of each plausible base, every value computed in code.
    [share] = run({"correction": {"topic": "share", "kind": "calculation",
                                  "expression": ["18 * 0.5", "18 * 0.5 / 25 * 100", "18 * 0.5 / 12.5 * 100"],
                                  "utterance": "That is {result} euros: {result2} percent of the price, {result3} "
                                               "percent of the cost limit."}})
    assert share.utterance == "That is 9 euros: 36 percent of the price, 72 percent of the cost limit."
    assert run({"correction": {"topic": "x", "expression": ["9"], "utterance": "{result} and {result2}"}}) == ()


def test_correction_needs_a_lower_fit_but_still_jev_pick():
    """A correction is judged on a compound statement (right, unsettled, useful now): 0.6 is a good score for it."""
    from kairos.decide.policy import PolicyParams, fit_bar
    p = PolicyParams()
    assert fit_bar(thought("c1", kind="correction"), p) < fit_bar(thought("t1"), p)
    assert fit_bar(thought("c1", kind="correction"), p) <= 0.5 < p.fit_min


# -- topic tracker and transcript cleaner -------------------------------------------------

class ScriptedLLM:
    """Returns the given replies in order, and records what it was asked."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.users: list[str] = []

    async def json(self, system, user, *, purpose, temperature=0.4):
        self.users.append(user)
        return self.replies.pop(0)


def test_topic_change_retires_ideas_of_the_old_subject_but_keeps_corrections_and_answers():
    from kairos.agents.topic import TopicTracker
    lines = tuple(Segment(i, "A", text, i * 2.0, i * 2.0 + 1, True) for i, text in enumerate([
        "On teste l'agent en réunion.", "Il faut voir ses faiblesses.", "On regarde les métriques.",
        "Bon, les billets pour Rotterdam, c'est 50 euros pour 5.", "Aller-retour ou aller simple ?"], 1))
    old_idea = thought("t1", kind="idea")
    correction = thought("c1", kind="correction", importance=5.0)
    owed = thought("t2", answers=5)
    board = board_with(thoughts=(old_idea, correction, owed), transcript=lines[:3], t=6.0)
    llm = ScriptedLLM({"topic": "tester l'agent", "changed": True}, {"topic": "billets pour Rotterdam", "changed": True})
    tracker = TopicTracker(board, llm, "French", min_gap_s=0.0)

    assert tracker.wants_name()
    assert asyncio.run(tracker.check()) is False  # the first naming changes nothing
    assert board.snapshot().signals.topic == "tester l'agent"
    assert {t.id for t in board.snapshot().thoughts if t.status == ThoughtStatus.READY} == {"t1", "c1", "t2"}

    board.publish("transcript", lines)
    board.publish("room", RoomState(t=20.0, speaking=False, silence_since=19.0, p_silence=(0.9,) * 4))
    assert asyncio.run(tracker.check()) is True
    signals = board.snapshot().signals
    assert (signals.topic, signals.previous_topic) == ("billets pour Rotterdam", "tester l'agent")
    status = {t.id: t.status for t in board.snapshot().thoughts}
    assert status == {"t1": ThoughtStatus.STALE, "c1": ThoughtStatus.READY, "t2": ThoughtStatus.READY}
    assert "Previous topic: tester l'agent" in llm.users[-1]


def test_topic_tracker_keeps_the_subject_when_the_model_sees_no_change():
    from kairos.agents.topic import TopicTracker
    lines = tuple(Segment(i, "A", f"ligne {i} sur le budget", i, i + 0.5, True) for i in range(1, 5))
    board = board_with(thoughts=(thought("t1"),), transcript=lines, t=30.0)
    tracker = TopicTracker(board, ScriptedLLM({"topic": "budget", "changed": True},
                                              {"topic": "budget du salon", "changed": False}), min_gap_s=0.0)
    asyncio.run(tracker.check())
    assert asyncio.run(tracker.check()) is False
    assert board.snapshot().signals.topic == "budget"
    assert board.snapshot().thoughts[0].status == ThoughtStatus.READY


def test_dispatcher_reports_a_change_of_subject_to_the_topic_tracker():
    seen = []
    board = board_with(transcript=(Segment(1, "A", "Bon, parlons des billets de train.", 0, 1, True),))
    dispatcher = Dispatcher(board, FakeJev(new_topic=0.9), lambda s: None, _no_work, on_topic=seen.append)
    asyncio.run(dispatcher.run(1, "Bon, parlons des billets de train."))
    assert seen == [0.9]


async def _no_work(kind, segment, text, addressed):
    return None


def test_the_cleaner_corrects_a_live_line_and_keeps_what_was_heard():
    from kairos.agents.cleaner import TranscriptCleaner
    from kairos.agents.scribe import Scribe
    from kairos.contracts import SpeechFinal

    board = Board()
    scribe = Scribe(board)
    scribe.on_event(SpeechFinal(segment=1, speaker="Vous", text="En gros, en gros, on a arrêté de parler de lard gens.",
                                t_start=0.0, t_end=2.0, t=2.0))
    llm = ScriptedLLM({"text": "En gros, on a arrêté de parler de l'argent."})
    cleaned = asyncio.run(TranscriptCleaner(board, llm, scribe.revise).clean(1))
    assert cleaned == "En gros, on a arrêté de parler de l'argent."
    assert board.snapshot().transcript[0].text == cleaned
    assert scribe.raw[1] == "En gros, en gros, on a arrêté de parler de lard gens."

    # A "correction" that adds a whole sentence is an invention: the line stays as heard.
    scribe.on_event(SpeechFinal(segment=2, speaker="Vous", text="Ça fait 20 euros.", t_start=3.0, t_end=4.0, t=4.0))
    llm = ScriptedLLM({"text": "Ça fait 20 euros par personne pour l'aller-retour à Rotterdam, soit 100 euros au total."})
    assert asyncio.run(TranscriptCleaner(board, llm, scribe.revise).clean(2)) is None
    assert board.snapshot().transcript[1].text == "Ça fait 20 euros."


def test_a_new_subject_is_renamed_once_it_has_lines_of_its_own_without_retiring_anything():
    from kairos.agents.topic import TopicTracker
    say = lambda n: tuple(Segment(i, "A", f"ligne {i}", i, i + 0.5, True) for i in range(1, n + 1))  # noqa: E731
    board = board_with(transcript=say(3), t=10.0)
    tracker = TopicTracker(board, ScriptedLLM({"topic": "tests", "changed": True},
                                              {"topic": "la photo des groupes tickets", "changed": True},
                                              {"topic": "billets de train pour Rotterdam", "changed": True}),
                           min_gap_s=0.0)
    asyncio.run(tracker.check())
    board.publish("transcript", say(5))
    assert asyncio.run(tracker.check()) is True
    board.publish("thoughts", (thought("t9", kind="idea"),))  # an idea about the new subject
    board.publish("transcript", say(7))
    assert not tracker.wants_name()
    board.publish("transcript", say(8))
    assert tracker.wants_name()
    assert asyncio.run(tracker.check()) is False  # a rename, not a change
    assert board.snapshot().signals.topic == "billets de train pour Rotterdam"
    assert board.snapshot().signals.previous_topic == "tests"
    assert board.snapshot().thoughts[0].status == ThoughtStatus.READY
    assert not tracker.wants_name()


def test_one_acknowledgement_per_search():
    from kairos.agents.common import recent_ack
    said = thought("a1", status=ThoughtStatus.SPOKEN, ack=True, answers=3)
    assert recent_ack(board_with(thoughts=(said,), t=5.0).snapshot())       # said 5 s ago: no second one
    assert not recent_ack(board_with(thoughts=(said,), t=30.0).snapshot())  # a new search later may be announced
    assert recent_ack(board_with(thoughts=(thought("a2", ack=True, answers=3),), t=30.0).snapshot())  # one waiting


def test_after_a_false_start_kairos_restarts_the_sentence_that_was_cut():
    from kairos.speaker import _lower_first, _sentence_start
    words = "Pour Groningen, je vérifie. Le moins cher connu est Paris-Amsterdam.".split()
    cut = words.index("connu")
    assert " ".join(_lower_first(words[_sentence_start(words, cut):])) == "le moins cher connu est Paris-Amsterdam."
    assert _lower_first(["Paris", "est"]) == ["Paris", "est"]  # a name keeps its capital


def test_kairos_waits_for_the_checker_reading_a_figure_before_saying_something_else():
    signals = Signals(judged_segment=1, judged_words=5, rater_line=1, rater_none=0.1, checking=1)
    board = board_with([thought("lodging", chosen=0.7)], [_line()], signals)  # silence: 1 s
    d = decide(board.snapshot())
    assert not d.speak and "checker" in d.why
    board.publish("room", RoomState(t=10.0, speaking=False, silence_since=8.0, p_silence=(0.9,) * 4))
    assert decide(board.snapshot()).speak  # 2 s of silence: the checker found nothing in time, go on
    board.publish("signals", Signals(judged_segment=1, judged_words=5, rater_line=1, rater_none=0.1))
    board.publish("room", RoomState(t=10.0, speaking=False, silence_since=9.0, p_silence=(0.9,) * 4))
    assert decide(board.snapshot()).speak  # checked: nothing holds the floor


def test_the_cleaner_corrects_a_turn_the_recognizer_cut_in_two():
    from kairos.agents.cleaner import TranscriptCleaner
    from kairos.agents.scribe import Scribe
    from kairos.contracts import SpeechFinal

    board = Board()
    scribe = Scribe(board)
    scribe.on_event(SpeechFinal(t=1.5, segment=1, speaker="Vous", text="Ah oui, oui, ça pourrait dire",
                                t_start=0.0, t_end=1.5))
    scribe.on_event(SpeechFinal(t=4.0, segment=2, speaker="Vous", text="ça aurait déjà raison, c'était prévu le 18.",
                                t_start=2.3, t_end=4.0))
    llm = ScriptedLLM({"previous": "Ah oui, oui,", "text": "t'as raison, c'était prévu le 18."})
    asyncio.run(TranscriptCleaner(board, llm, scribe.revise).clean(2))
    assert "Previous part of the same turn" in llm.users[0]
    assert [s.text for s in board.snapshot().transcript] == ["Ah oui, oui,", "t'as raison, c'était prévu le 18."]
    assert scribe.raw == {1: "Ah oui, oui, ça pourrait dire", 2: "ça aurait déjà raison, c'était prévu le 18."}


def test_a_second_resume_does_not_say_donc_je_disais_twice():
    from kairos.speaker import _lower_first, _sentence_start, _without_resume
    once = "donc je disais, c’est surtout compliqué à organiser.".split()
    again = _without_resume(once, "French")
    assert again == "c’est surtout compliqué à organiser.".split()
    assert _lower_first("C’est surtout compliqué.".split())[0] == "c’est"
    assert _sentence_start(again, 3) == 0


def test_a_question_to_kairos_ending_with_a_question_mark_is_answered_sooner():
    question = Segment(1, "Vous", "Kairos, tu peux regarder les prix ?", 0, 3, False)  # not committed yet
    answer = thought("t1", answers=1)
    signals = Signals(addressed_segment=1, answered=False, judged_segment=1, judged_words=7)
    board = board_with([answer], [question], signals)
    board.publish("room", RoomState(t=10.0, speaking=False, silence_since=9.4, p_silence=(0.9,) * 4))  # 0.6 s
    assert decide(board.snapshot()).speak
    board.publish("transcript", (replace(question, text="On regarde les prix, et puis"),))
    board.publish("signals", Signals(judged_segment=1, judged_words=6))
    assert "not finished" in decide(board.snapshot()).why  # an ordinary unfinished line still waits 1 s


def test_no_markdown_enters_the_reservoir():
    board = board_with()
    result = replace(thought("rB1", kind="finding"),
                     utterance="## [Pharmacie de la Gare](https://maps.example/x) est **ouverte** jusqu'à 20 h.")
    board.update_thoughts({}, (result,), by="travaux B1")
    assert board.snapshot().thoughts[0].utterance == "Pharmacie de la Gare est ouverte jusqu'à 20 h."


class WorldCupLLM:
    async def json(self, system, user, *, purpose, temperature=0.4):
        return {"details": {"query": {"value": "vainqueur Coupe du monde 2026", "source": "L31"}},
                "missing": [], "question": ""}


def test_what_the_search_found_replaces_an_answer_written_from_memory_and_keeps_its_promise():
    line = Segment(31, "Vous", "Alors, est-ce que tu sais qui a gagné la Coupe du Monde ?", 0, 4, True)
    stale_knowledge = replace(thought("t9", importance=5.0, judged_line=31), kind="answer", answers=31,
                              utterance="La Coupe du Monde 2026 n'a pas encore eu lieu.")
    board = board_with([stale_knowledge], [line])
    search = FakeSearch("L'Espagne a gagné la Coupe du monde 2026 contre l'Argentine, 1 à 0.")
    sup = WorkerSupervisor(board, WorldCupLLM(), "French", search=search)

    async def scenario():
        await sup.open("web", 31, line.text, addressed=False)  # Jev did not read it as addressed
        await sup.drain()

    asyncio.run(scenario())
    thoughts = {t.id: t for t in board.snapshot().thoughts}
    assert thoughts["t9"].status == ThoughtStatus.STALE  # never said: the finding replaces it
    assert thoughts["rB1"].answers == 31 and "Espagne" in thoughts["rB1"].utterance


# -- coherence: a request is linked to the trip being discussed ------------------------------------------

class RecordingLLM:
    def __init__(self):
        self.prompts = []

    async def json(self, system, user, *, purpose, temperature=0.4):
        self.prompts.append(user)
        return {"details": {"query": {"value": "hôtels Zanzibar 28 octobre 2026", "source": "notes"}},
                "missing": [], "question": ""}


def test_a_hotel_request_sees_the_trip_in_the_notes_and_in_the_flight_search():
    line = Segment(40, "Vous", "Et pour les hôtels qui se trouvent là-bas, comment on fait ?", 0, 3, True)
    board = board_with(transcript=[line])
    board.publish("notes", "- Voyage à Zanzibar depuis Paris, départ le 28 octobre.")
    llm = RecordingLLM()
    sup = WorkerSupervisor(board, llm, "French", search=FakeSearch("Hôtels à Stone Town dès 45 euros la nuit."))

    async def scenario():
        from kairos.agents.workers import Brief
        sup.briefs.append(Brief(id="B3", kind="flights", segment=20, line="Paris Zanzibar", addressed=True,
                                created_at=0.0, status="done",
                                details={"destination": {"value": "Zanzibar (ZNZ)"}, "date": {"value": "2026-10-28"}}))
        await sup.open("web", 40, line.text, addressed=True)
        await sup.drain()

    asyncio.run(scenario())
    prompt = llm.prompts[0]
    assert "Voyage à Zanzibar depuis Paris" in prompt  # the short-term memory
    assert "B3 (flights, done): destination = Zanzibar (ZNZ), date = 2026-10-28" in prompt


def test_nothing_is_assumed_while_the_person_is_still_giving_the_details():
    lines = [Segment(1, "Vous", "Je veux aller de Paris à Douala, disons que je pars le", 0, 3, False)]
    board = board_with(transcript=lines)
    board.publish("room", RoomState(t=10.0, speaking=True, silence_since=None, p_silence=(0.1,) * 4))
    flights = FakeFlights()
    sup = WorkerSupervisor(board, AssumingLLM(), "French", flights=flights)

    async def scenario():
        brief = await sup.open("flights", 1, lines[0].text, addressed=True)
        await sup.drain()
        return brief

    brief = asyncio.run(scenario())
    assert brief.missing == ["date"] and not flights.searched  # "le 27 août" may be the next words


def test_an_acknowledgment_is_not_a_reason_to_write_the_cut_answer_again():
    from kairos.runtime import _acknowledges
    assert _acknowledges("Ah, ok. Je vois.") and _acknowledges("D'accord.")
    assert not _acknowledges("Comment tu peux m'aider à réserver, alors ?")
    assert not _acknowledges("Non, je pars plutôt de Lyon le 28 octobre avec mes deux fils.")


def test_a_thought_jev_clearly_picks_is_said_even_with_a_middling_coherence_score():
    line = Segment(3, "Vous", "Est-ce que tu aurais des idées de villes avec de la plage, en Afrique ?", 0, 4.0, True)
    offer = replace(thought("qB1n1", chosen=0.91, judged_line=3), kind="question", fit_now=0.52,
                    utterance="Tu veux que je cherche des destinations en Afrique avec de belles plages ?")
    signals = Signals(judged_segment=3, judged_words=13, rater_line=3, rater_none=0.05)
    assert decide(board_with([offer], [line], signals).snapshot()).speak
    weak = replace(offer, chosen=0.6)  # not a clear pick: the coherence bar stays
    assert not decide(board_with([weak], [line], signals).snapshot()).speak


def test_when_jev_says_to_speak_the_best_of_several_good_thoughts_is_said():
    line = Segment(7, "Vous", "Qui ose ? Est-ce que tu aurais des idées ?", 0, 3.0, True)
    ideas = [replace(thought(f"t{i}", chosen=c, judged_line=7), fit_now=f)
             for i, (c, f) in enumerate([(0.31, 0.78), (0.28, 0.71), (0.25, 0.60)])]
    speak = Signals(judged_segment=7, judged_words=9, rater_line=7, rater_none=0.09)
    d = decide(board_with(ideas, [line], speak).snapshot())
    assert d.speak and d.primary == "t0"  # "none" at 0.09: something should be said, the best one is
    quiet = Signals(judged_segment=7, judged_words=9, rater_line=7, rater_none=0.55)
    assert not decide(board_with(ideas, [line], quiet).snapshot()).speak


def test_a_search_result_someone_waits_for_follows_kairos_at_once_but_an_idea_waits():
    from kairos.contracts import AiState
    line = Segment(8, "Vous", "Ça coûterait combien, les billets de Paris jusqu'aux Antilles ?", 0, 4.0, True)
    result = replace(thought("rB1", chosen=0.8, judged_line=8, importance=5.0), kind="finding", answers=8,
                     fit_now=0.87, utterance="Le moins cher est un vol direct Paris Pointe-à-Pitre à 402 euros.")
    signals = Signals(judged_segment=8, judged_words=12, rater_line=8, rater_none=0.1, answered=True)

    def after_kairos(thoughts, silence_s=1.0):
        board = board_with(thoughts, [line], signals, t=10.0)
        board.publish("room", RoomState(t=10.0, speaking=False, silence_since=10.0 - silence_s, p_silence=(0.9,) * 4))
        board.publish("ai", AiState(last_spoke_at=9.5))  # "la recherche est en cours…" just ended
        return decide(board.snapshot())

    assert after_kairos([result]).speak
    idea = replace(thought("t7", chosen=0.4, judged_line=8), fit_now=0.8)
    assert not after_kairos([idea], silence_s=2.0).speak  # a weak pick waits for a person
    clear = replace(idea, chosen=0.71)
    assert not after_kairos([clear], silence_s=1.0).speak and after_kairos([clear], silence_s=2.0).speak


class FakeHotels:
    def __init__(self):
        self.searched = []

    async def search(self, city, country, checkin, checkout, guests, min_stars=0):
        from kairos.travel import HotelSummary
        self.searched.append((city, country, checkin, checkout, guests))
        return HotelSummary(city, checkin, checkout, 27, 279.0,
                            ("- VIP Inn Berna Hotel: 3 stars, rated 8.3/10, 279 EUR in total (93 EUR a night)",), 3.8)


class HotelLLM:
    async def json(self, system, user, *, purpose, temperature=0.4):
        if purpose == "brief result":
            return {"say": "Le VIP Inn Berna, noté 8,3 sur 10, revient à 93 euros la nuit, 279 euros pour les trois nuits."}
        return {"details": {"city": {"value": "Lisbonne (PT)", "source": "L1"},
                            "checkin": {"value": "2026-10-17", "source": "L1"},
                            "checkout": {"value": "2026-10-20", "source": "L1"},
                            "guests": {"value": "2", "source": "L1"}}, "missing": [], "question": ""}


def test_a_hotel_request_for_the_trip_is_searched_on_jinko_and_owed_to_the_person():
    lines = [Segment(1, "Claude", "On part à Lisbonne du 17 au 20 octobre, on sera deux.", 0, 3, True),
             Segment(2, "Léa", "Kairos, tu peux regarder les hôtels là-bas ?", 4, 6, True)]
    board = board_with(transcript=lines)
    hotels = FakeHotels()
    sup = WorkerSupervisor(board, HotelLLM(), "French", hotels=hotels)

    async def scenario():
        await sup.open("hotels", 2, lines[1].text, addressed=True)
        await sup.drain()

    asyncio.run(scenario())
    assert hotels.searched == [("Lisbonne", "PT", "2026-10-17", "2026-10-20", 2)]
    result = {t.id: t for t in board.snapshot().thoughts}["rB1"]
    assert result.answers == 2 and "93 euros la nuit" in result.utterance


def test_a_city_jinko_does_not_know_by_that_name_is_retried_with_its_suggestion():
    from kairos.travel import _suggested_city
    low = httpx.Response(422, json={"error": {"code": "DESTINATION_LOW_CONFIDENCE", "suggested_retry": {
        "destination": {"city_name": "Lisbon", "country_code": "PT"}}}})
    assert _suggested_city(low) == {"city_name": "Lisbon", "country_code": "PT"}
    assert _suggested_city(httpx.Response(422, json={"error": {"code": "BAD_REQUEST"}})) is None
    assert _suggested_city(httpx.Response(200, json={"hotels": []})) is None


def test_in_a_meeting_kairos_takes_no_initiative_until_someone_speaks_to_it():
    from kairos.decide.policy import PolicyParams
    meeting = PolicyParams(initiative=False)
    line = Segment(6, "Marc", "Une ville avec un peu de soleil, j'aimerais bien Lisbonne.", 0, 3.0, True)
    offer = replace(thought("q1", chosen=0.9, judged_line=6), kind="question", fit_now=0.9,
                    utterance="Tu veux que je regarde les vols pour Lisbonne ?")
    among_themselves = Signals(judged_segment=6, judged_words=10, rater_line=6, rater_none=0.05)
    assert not decide(board_with([offer], [line], among_themselves).snapshot(), meeting).speak
    fix = replace(thought("c1", chosen=0.9, judged_line=6, importance=5.0), kind="correction", fit_now=0.9)
    assert decide(board_with([fix], [line], among_themselves).snapshot(), meeting).speak  # corrections pass
    to_kairos = replace(among_themselves, addressed=0.9, addressed_segment=6, answered=True)
    assert decide(board_with([offer], [line], to_kairos).snapshot(), meeting).speak  # in the conversation
    assert decide(board_with([offer], [line], among_themselves).snapshot()).speak  # the console keeps initiative


def test_a_factual_question_to_the_room_is_marked_but_a_question_among_friends_is_not():
    async def open_work(kind, segment, text, is_addressed):
        pass

    d = Dispatcher(board_with(), FakeJev(search=0.9, fact=0.9), lambda s: None, open_work)
    asyncio.run(d.run(4, "Au fait, qui a gagné la Coupe du monde cette année ?"))
    friends = Dispatcher(board_with(), FakeJev(search=0.6, fact=0.2), lambda s: None, open_work)
    asyncio.run(friends.run(5, "On prend un hôtel ou un appartement ?"))
    assert d.facts == {4} and friends.facts == set()


def test_a_result_already_said_in_one_copy_is_not_given_again():
    from kairos.agents.workers import Brief
    board = board_with(transcript=[Segment(9, "Vous", "Tu disais quoi pour les hôtels ?", 0, 2, True)])
    first = replace(thought("rB3"), status=ThoughtStatus.STALE, kind="finding")  # set aside before being said
    said = replace(thought("rB3a226"), status=ThoughtStatus.SPOKEN, kind="finding")  # its copy was said
    board.publish("thoughts", (first, said))
    sup = WorkerSupervisor(board, HotelLLM(), "French", hotels=FakeHotels())
    recent = Brief(id="B3", kind="hotels", segment=5, line="les hôtels là-bas", addressed=True, created_at=0.0,
                   status="done")
    sup._repeat_unsaid_result(recent, 9)
    assert {t.id for t in board.snapshot().thoughts} == {"rB3", "rB3a226"}  # nothing new to say again


def test_a_five_star_request_keeps_only_five_star_hotels():
    import asyncio as aio
    from kairos.travel import JinkoHotels

    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            def hotel(name, stars, price):
                return {"name": name, "star_rating": stars, "rating": 8.5,
                        "rooms": [{"rates": [{"total_amount": price, "board_name": "Room Only"}]}]}
            return {"hotels": [hotel("Ibis", 3, 256), hotel("Tivoli", 5, 610), hotel("Pestana", 5, 540)]}

    class Client:
        async def post(self, url, headers, json):
            return Response()

    hotels = JinkoHotels("key")
    hotels._client = Client()
    summary = aio.run(hotels.search("Lisbon", "PT", "2026-10-02", "2026-10-04", 2, min_stars=5))
    assert summary.cheapest_eur == 540 and all("Ibis" not in line for line in summary.lines)
