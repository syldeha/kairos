"""Milestones J2-J4 with a scripted fake model: no network, deterministic."""

import asyncio
import hashlib
import math
from dataclasses import replace
from pathlib import Path

import pytest

from kairos.agents.common import is_ai
from kairos.agents.scribe import Scribe
from kairos.backchannel import is_backchannel
from kairos.board import Board
from kairos.clock import SimClock
from kairos.contracts import (AiState, RoomState, Signals, SpeechPartial, Thought, ThoughtStatus,
                              UtterancePlan)
from kairos.decide.judges import LlmJudge
from kairos.decide.policy import PolicyParams, decide, floor_open
from kairos.harness import Harness, HarnessParams
from kairos.llm import Tracer
from kairos.report import metrics, rows
from kairos.runtime import RunConfig, Runtime
from kairos.session import load_memory
from kairos.sources.replay import build_timeline, parse_script
from kairos.speaker import Speaker, SpeakerParams

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


# -- fake model ----------------------------------------------------------------------

class FakeLLM:
    """Answers questions addressed to Kairos, has no other idea, and judges generously."""

    def __init__(self) -> None:
        self.tracer = Tracer()

    async def json(self, system, user, *, purpose, temperature=0.4):
        if purpose == "answer":
            return {"topic": "retours du test", "content": "Les retours arrivent le 12 février",
                    "utterance": "Les retours du test arrivent le 12 février."}
        if purpose == "thinker":
            return {"current_topic": "réunion", "new_thoughts": []}
        return {"notes": "Discussion sur le prix et la date de la bêta."}

    async def yes_probability(self, system, user, *, purpose):
        statement = user.split("Statement:", 1)[1]
        if "speaks directly to Kairos" in statement:
            return 0.95 if "Kairos," in statement else 0.05
        if "already said" in statement or "outdated" in statement:
            return 0.05
        return 0.9

    async def embed(self, texts, *, purpose):
        return [_vector(t) for t in texts]


def _vector(text: str) -> list[float]:
    v = [0.0] * 32
    for word in text.lower().split():
        v[int(hashlib.md5(word.encode()).hexdigest(), 16) % 32] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def thought(id, status=ThoughtStatus.READY, importance=4.0, relevance=0.5, fit=0.9, said=0.0,
            topic="t", created=0.0, answers=None, transition=None, utterance=None, judged_line=None) -> Thought:
    return Thought(id=id, topic=topic, content=f"content {id}", utterance=utterance or f"utterance {id}",
                   transition=transition, importance=importance, relevance=relevance, fit_now=fit,
                   already_said=said, status=status, stimuli=(), version=0, created_at=created, answers=answers,
                   judged_line=judged_line)


def board_with(thoughts=(), silence=1.0, p1=0.9, speaking=False, signals=Signals(), ai=AiState(), t=10.0) -> Board:
    board = Board()
    board.publish("room", RoomState(t=t, speaking=speaking, silence_since=None if speaking else t - silence,
                                    p_silence=(p1, p1, p1, p1)))
    board.publish("thoughts", tuple(thoughts))
    board.publish("signals", signals)
    board.publish("ai", ai)
    return board


# -- policy ---------------------------------------------------------------------------

def test_floor_is_closed_while_someone_speaks_or_too_soon():
    p = PolicyParams()
    assert not floor_open(board_with(speaking=True).snapshot(), p)
    assert not floor_open(board_with(silence=0.1).snapshot(), p)
    assert floor_open(board_with(silence=0.4, p1=0.8).snapshot(), p)
    assert not floor_open(board_with(silence=0.4, p1=0.2).snapshot(), p)
    assert floor_open(board_with(silence=1.0, p1=0.2).snapshot(), p)  # long silence


def test_policy_needs_the_judge_before_speaking():
    unjudged = board_with([thought("a", fit=0.0)]).snapshot()
    assert not decide(unjudged).speak
    assert decide(board_with([thought("a", fit=0.9)]).snapshot()).speak


def asked_board(thoughts):
    from kairos.contracts import Segment
    board = board_with(thoughts, signals=Signals(addressed=0.9, addressed_segment=7, answered=False, last_final=7,
                                                 judged_segment=7, judged_words=4))
    board.publish("transcript", (Segment(7, "A", "Kairos, tu te souviens ?", 7.0, 9.0, final=True),))
    return board


def test_policy_answers_a_direct_question_first():
    board = asked_board([thought("idea", importance=5), thought("ans", importance=3, answers=7)])
    d = decide(board.snapshot())
    assert d.speak and d.reason == "asked" and d.primary == "ans"


def test_policy_waits_when_asked_but_answer_not_ready():
    assert not decide(asked_board([thought("idea", importance=5)]).snapshot()).speak


def test_an_unanswered_question_expires():
    board = asked_board([thought("idea", importance=5, judged_line=7)])
    board.publish("room", replace(board.snapshot().room, t=40.0, silence_since=39.0))
    assert decide(board.snapshot()).speak


def test_policy_chains_a_pending_thought_on_another_topic():
    board = board_with([thought("now", topic="beta"),
                        thought("later", status=ThoughtStatus.PENDING, topic="price", importance=5)])
    d = decide(board.snapshot())
    assert d.primary == "now" and d.chain == "later"


def test_policy_raises_the_bar_when_kairos_talks_too_much():
    t = [thought("a", importance=3.5, relevance=0.55)]
    assert decide(board_with(t).snapshot()).speak
    chatty = AiState(total_speech_s=5.0)  # 50 % of a 10 s meeting
    assert not decide(board_with(t, ai=chatty).snapshot()).speak


def test_policy_does_not_speak_twice_in_a_row():
    board = board_with([thought("a")], ai=AiState(last_spoke_at=9.5), silence=1.0)  # silence began at 9.0
    assert "Kairos spoke last" in decide(board.snapshot()).why


def test_policy_waits_for_the_last_line_to_be_committed_and_judged():
    from kairos.contracts import Segment
    board = board_with([thought("a", judged_line=3)])
    board.publish("transcript", (Segment(3, "A", "so what do", 5.0, 9.0, final=False),))
    assert "has not seen" in decide(board.snapshot()).why
    board.publish("signals", Signals(judged_segment=3, judged_words=2))  # judge lags 2 words: fine
    assert decide(board.snapshot()).speak
    board.publish("transcript", (Segment(3, "A", "so what do we do about it", 5.0, 9.0, final=False),))
    assert "has not seen" in decide(board.snapshot()).why  # 4 words behind: wait


def test_judge_only_considers_lines_that_name_kairos():
    from kairos.agents.judge import NAME
    assert NAME.search("Kairos, tu te souviens ?") and NAME.search("what does cairos think")
    assert not NAME.search("Et on reparle du prix la semaine prochaine")


# -- harness --------------------------------------------------------------------------

def test_harness_budget_and_length():
    board = board_with([thought("a", utterance="word " * 60)])
    harness = Harness(HarnessParams(unsolicited_gap_s=30, urgent_score=0.85, max_words=45))
    d = decide(board.snapshot())
    plan = harness.check(d, board.snapshot())
    assert plan and len(plan.parts[0][1].split()) == 45
    later = replace(d, t=d.t + 10)
    assert harness.check(later, board.snapshot()) is None
    assert "budget" in harness.refusals[-1][1]


def test_budget_lets_a_correction_through_when_it_holds_back_the_best_idea():
    from kairos.harness import decisive_only
    board = board_with([thought("idea", importance=4, relevance=0.6), thought("fix", importance=5, relevance=0.3)])
    harness = Harness(HarnessParams(unsolicited_gap_s=30, urgent_score=0.85))
    harness._last_unsolicited = 5.0  # Kairos spoke unprompted 5 s ago
    d = decide(board.snapshot())
    assert d.primary == "idea" and harness.check(d, board.snapshot()) is None
    fallback = decide(decisive_only(board.snapshot()))
    assert fallback.primary == "fix" and harness.check(fallback, board.snapshot()) is not None


def test_a_fresh_correction_holds_the_floor_until_it_is_judged():
    remark = thought("remark", importance=4, relevance=0.6, created=5.0, judged_line=None)
    fix = thought("fix", importance=5, relevance=0.5, created=9.0, fit=0.0, judged_line=-1)  # written 1 s ago
    d = decide(board_with([remark, fix], t=10.0).snapshot())
    assert not d.speak and "correction" in d.why
    judged = replace(fix, fit_now=0.9, judged_line=None)
    assert decide(board_with([remark, judged], t=10.0).snapshot()).primary == "fix"
    old = replace(fix, created_at=2.0)  # never judged fit in 3 s: the rest may speak
    assert decide(board_with([remark, old], t=10.0).snapshot()).primary == "remark"


def test_harness_prefixes_pending_thoughts_with_their_lead_in():
    board = board_with([thought("p", status=ThoughtStatus.PENDING, importance=5, transition="Back to the price,")])
    plan = Harness().check(decide(board.snapshot()), board.snapshot())
    assert plan.parts[0][1].startswith("Back to the price,")


# -- speaker --------------------------------------------------------------------------

def test_backchannels():
    assert is_backchannel("Mm.") and is_backchannel("yeah okay") and is_backchannel("d'accord")
    assert not is_backchannel("well actually") and not is_backchannel("yeah but no, wait")


def run_speaker(interrupt_at_word=None, heard=()):
    board = Board()
    board.publish("thoughts", (thought("a"), thought("b")))
    scribe = Scribe(board)
    clock = SimClock()
    words = {"n": 0}
    speaker = None

    def on_word(_):
        words["n"] += 1
        if words["n"] == interrupt_at_word:
            for text in heard:
                speaker.hear(SpeechPartial(t=0, segment=1, speaker="A", text=text, t_start=0))

    speaker = Speaker(board, scribe, clock, clock.now, SpeakerParams(), on_word=on_word)
    plan = UtterancePlan(0.0, "important", (("a", "one two three four five"), ("b", "six seven")))
    outcome = asyncio.run(speaker.say(plan))
    return outcome, board


def test_speaker_says_everything_when_undisturbed():
    outcome, board = run_speaker()
    assert outcome.said == ["a", "b"] and outcome.text == "one two three four five six seven"
    statuses = {t.id: t.status for t in board.snapshot().thoughts}
    assert statuses == {"a": ThoughtStatus.SPOKEN, "b": ThoughtStatus.SPOKEN}
    assert any(is_ai(s) and s.final for s in board.snapshot().transcript)


def test_speaker_ignores_a_backchannel():
    outcome, _ = run_speaker(2, heard=["mm-hmm"])
    assert outcome.interruptions == 0 and outcome.said == ["a", "b"]


def test_speaker_resumes_after_a_false_start():
    outcome, _ = run_speaker(4, heard=["so"])
    assert outcome.interruptions == 1 and outcome.resumed == 1
    assert outcome.text.count("as I was saying,") == 1 and outcome.said == ["a", "b"]


def test_speaker_barely_started_simply_starts_again_without_a_resume_phrase():
    outcome, _ = run_speaker(2, heard=["so"])
    assert outcome.resumed == 1 and "as I was saying" not in outcome.text and outcome.said == ["a", "b"]


def test_speaker_yields_to_a_real_interruption_and_keeps_the_rest_for_later():
    outcome, board = run_speaker(2, heard=["so I", "so I think", "so I think we"])
    assert outcome.yielded and outcome.cut == ["a", "b"] and outcome.text == "one two"
    statuses = {t.id: t.status for t in board.snapshot().thoughts}
    assert statuses == {"a": ThoughtStatus.PENDING, "b": ThoughtStatus.PENDING}


# -- end to end ---------------------------------------------------------------------------

def run_meeting(mode):
    timeline = build_timeline(parse_script((FIXTURES / "reunion_produit.txt").read_text(encoding="utf-8")))
    llm = FakeLLM()
    runtime = Runtime(timeline, load_memory(FIXTURES / "kairos_fr.txt"), llm, LlmJudge(llm),
                      RunConfig(mode=mode, speed=0, language="French"))
    asyncio.run(runtime.run())
    return runtime


def test_closed_loop_kairos_answers_the_question_in_the_gap_and_the_room_waits():
    runtime = run_meeting("closed")
    table = rows(runtime)
    assert len(table) == 1
    answer = table[0]
    assert answer.reason == "asked" and answer.moment == "turn change"
    assert answer.text == "Les retours du test arrivent le 12 février."
    transcript = runtime.board.snapshot().transcript
    kairos = next(s for s in transcript if is_ai(s))
    after = [s for s in transcript if not is_ai(s) and s.t_start > kairos.t_start]
    assert after and after[0].t_start >= kairos.t_end  # Karim waited for Kairos to finish
    assert runtime.board.snapshot().signals.answered
    m = metrics(runtime)
    assert m["interventions"] == 1 and m["in_natural_gap"] == "1/1"


def test_offline_mode_records_the_decision_without_speaking():
    runtime = run_meeting("offline")
    assert [r.reason for r in rows(runtime)] == ["asked"]
    assert not any(is_ai(s) for s in runtime.board.snapshot().transcript)


def test_open_loop_requires_a_real_clock():
    timeline = build_timeline(parse_script("+0.5 A: bonjour"))
    llm = FakeLLM()
    with pytest.raises(ValueError, match="real clock"):
        Runtime(timeline, [], llm, LlmJudge(llm), RunConfig(mode="open", speed=0))


# -- researcher -------------------------------------------------------------------------

class FakeSearch:
    name = "fake"

    def __init__(self):
        self.tracer = Tracer()
        self.queries = []

    async def search(self, query, question, language):
        from kairos.search import SearchResult
        self.queries.append(query)
        return SearchResult("People use about 10 to 20 percent of the buttons, according to a survey.",
                            ("https://example.org/survey",), 0.8)


class PlanningLLM(FakeLLM):
    async def json(self, system, user, *, purpose, temperature=0.4):
        if purpose == "research plan":
            return {"search": {"question": "How many buttons do people use?", "query": "share of remote buttons used",
                               "topic": "button usage", "segment": 3}}
        return await super().json(system, user, purpose=purpose, temperature=temperature)


def test_researcher_prepares_a_finding_and_a_ready_thought():
    from kairos.agents.researcher import Researcher
    board = board_with()
    search = FakeSearch()
    researcher = Researcher(board, PlanningLLM(), search, language="English", min_interval_s=20)
    asyncio.run(researcher.run_once())
    snap = board.snapshot()
    assert [f.status for f in snap.findings] == ["done"]
    assert snap.findings[0].sources == ("https://example.org/survey",)
    thought = next(t for t in snap.thoughts if t.id.startswith("r"))
    assert thought.status == ThoughtStatus.READY and thought.importance == 4.0
    assert thought.transition == "I looked up button usage:" and "10 to 20 percent" in thought.utterance
    asyncio.run(researcher.run_once())  # too soon: budget keeps searches apart
    assert len(search.queries) == 1


def test_researcher_answers_an_open_question_to_kairos():
    from kairos.agents.researcher import Researcher
    board = board_with(signals=Signals(addressed=0.9, addressed_segment=3, answered=False))
    researcher = Researcher(board, PlanningLLM(), FakeSearch(), language="English")
    asyncio.run(researcher.run_once())
    thought = next(t for t in board.snapshot().thoughts if t.id.startswith("r"))
    assert thought.answers == 3 and thought.importance == 5.0


def test_a_finding_answering_a_question_the_room_left_open_is_decisive():
    from kairos.agents.researcher import Researcher

    class OpenQuestionLLM(PlanningLLM):
        async def json(self, system, user, *, purpose, temperature=0.4):
            result = await super().json(system, user, purpose=purpose, temperature=temperature)
            if purpose == "research plan":
                result["search"]["unanswered"] = True
            return result

    board = board_with()
    asyncio.run(Researcher(board, OpenQuestionLLM(), FakeSearch(), language="English").run_once())
    thought = next(t for t in board.snapshot().thoughts if t.id.startswith("r"))
    assert thought.importance == 5.0 and thought.answers is None  # decisive, but not a reply owed to Kairos


def test_researcher_output_is_speakable():
    from kairos.agents.researcher import spoken
    text = spoken("About 10% of buttons are used ([example.org](https://example.org/a)). See https://x.y/z too.")
    assert "http" not in text and "[" not in text and text.startswith("About 10% of buttons are used")
    assert len(spoken("word " * 80).split()) == 35


def test_researcher_gives_nothing_to_say_when_nothing_was_found():
    from kairos.agents.researcher import Researcher
    from kairos.search import SearchResult

    class Empty(FakeSearch):
        async def search(self, query, question, language):
            return SearchResult("I couldn't find the exact figure.", (), 0.5)

    board = board_with()
    asyncio.run(Researcher(board, PlanningLLM(), Empty(), language="English").run_once())
    assert [f.status for f in board.snapshot().findings] == ["failed"]
    assert not any(t.id.startswith("r") for t in board.snapshot().thoughts)


def test_researcher_does_not_repeat_a_search():
    from kairos.agents.relevance import Relevance
    from kairos.agents.researcher import Researcher
    board = board_with()
    llm, search = PlanningLLM(), FakeSearch()
    researcher = Researcher(board, llm, search, language="English", min_interval_s=0,
                            relevance=Relevance(board, llm))
    asyncio.run(researcher.run_once())
    asyncio.run(researcher.run_once())  # same plan again: same query
    assert len(search.queries) == 1 and researcher.skipped_repeats == 1


def test_active_role_speaks_up_more_readily():
    active, discreet = RunConfig(role="active"), RunConfig()
    assert active.policy.open_threshold < discreet.policy.open_threshold
    assert active.harness.unsolicited_gap_s < discreet.harness.unsolicited_gap_s


def test_urgent_thought_overrides_the_budget():
    board = board_with([thought("a", importance=4, relevance=0.6)])
    harness = Harness(HarnessParams(unsolicited_gap_s=30, urgent_score=0.85))
    d = decide(board.snapshot())
    assert harness.check(d, board.snapshot())
    assert harness.check(replace(d, t=d.t + 5, score=0.9), board.snapshot())  # urgent: allowed
    assert harness.check(replace(d, t=d.t + 6, score=0.6), board.snapshot()) is None


def test_live_meeting_hears_a_participant_and_ends_on_stop():
    llm = FakeLLM()
    runtime = Runtime(None, [], llm, LlmJudge(llm), RunConfig(speed=20, language="French"))

    async def scenario():
        task = asyncio.create_task(runtime.run())
        await asyncio.sleep(0.05)
        await runtime.say_live("Vous", "Kairos, quand arrivent les retours ?", 900)
        await asyncio.sleep(0.3)
        runtime.stop()
        await asyncio.wait_for(task, 5)

    asyncio.run(scenario())
    texts = [s.text for s in runtime.board.snapshot().transcript]
    assert "Kairos, quand arrivent les retours ?" in texts
    assert "Les retours du test arrivent le 12 février." in texts  # answered live


def test_old_judgments_do_not_count():
    from kairos.contracts import Segment
    board = board_with([thought("a", judged_line=2)], signals=Signals(judged_segment=3, judged_words=5))
    board.publish("transcript", (Segment(3, "A", "so what do we do now", 5.0, 9.0, final=True),))
    assert "no thought judged coherent" in decide(board.snapshot()).why


def test_what_was_just_said_leaves_the_reservoir():
    from kairos.agents.relevance import Relevance
    board = board_with([thought("dup", utterance="les retours arrivent le 12 février"),
                        thought("other", utterance="le salon de Lyon a lieu fin mars")])
    relevance = Relevance(board, FakeLLM())
    asyncio.run(relevance.run_once())  # embeds the thoughts
    n = asyncio.run(relevance.retire_similar("les retours arrivent le 12 février", 0.75, "said by Kairos"))
    statuses = {t.id: (t.status, t.note) for t in board.snapshot().thoughts}
    assert n == 1 and statuses["dup"] == (ThoughtStatus.STALE, "said by Kairos")
    assert statuses["other"][0] == ThoughtStatus.READY


def test_lead_in_is_not_doubled():
    from kairos.agents.common import with_lead_in
    assert with_lead_in("Du côté de la bêta,", "Sur la bêta, je viserais fin février.") == \
        "Sur la bêta, je viserais fin février."
    assert with_lead_in("Pour revenir au prix,", "15 € c'est cher.") == "Pour revenir au prix, 15 € c'est cher."
    assert with_lead_in("Et sur la bêta,", "Pour la date de bêta, on verrouille.") == "Pour la date de bêta, on verrouille."


def test_reservoir_never_holds_more_than_five():
    from kairos.agents.common import enforce_cap
    board = board_with([thought(f"t{i}", importance=i % 5 + 1, created=i) for i in range(8)])
    enforce_cap(board)
    active = [t for t in board.snapshot().thoughts if t.status == ThoughtStatus.READY]
    assert len(active) == 5 and min(t.importance for t in active) >= 2


def test_cap_evicts_by_current_worth_and_keeps_few_pending():
    from kairos.agents.common import enforce_cap
    judged = [thought(f"old{i}", importance=3, relevance=0.2, judged_line=1) for i in range(3)]
    pending = [thought(f"p{i}", status=ThoughtStatus.PENDING, importance=3 + i % 2, judged_line=1) for i in range(3)]
    new = thought("new", importance=4, relevance=0.0, created=10.0)  # not measured yet: not evicted for being new
    board = board_with(judged + pending + [new])
    enforce_cap(board)
    by_id = {t.id: t for t in board.snapshot().thoughts}
    active = [t for t in by_id.values() if t.status in (ThoughtStatus.READY, ThoughtStatus.PENDING)]
    assert len(active) == 5
    assert sum(t.status == ThoughtStatus.PENDING for t in active) <= 2
    assert by_id["new"].status == ThoughtStatus.READY
    assert by_id["p0"].status == ThoughtStatus.STALE and by_id["p0"].note.startswith("trop de points en attente")


def test_cap_keeps_an_old_correction_over_fresher_remarks():
    from kairos.agents.common import enforce_cap
    fix = thought("fix", importance=5, relevance=0.2, created=-200.0, judged_line=1)  # old, off the current topic
    board = board_with([fix] + [thought(f"r{i}", importance=3, relevance=0.6, judged_line=1) for i in range(5)])
    enforce_cap(board)
    by_id = {t.id: t for t in board.snapshot().thoughts}
    assert by_id["fix"].status == ThoughtStatus.READY
    assert sum(t.status == ThoughtStatus.STALE for t in by_id.values()) == 1


def test_checker_turns_a_contradiction_with_memory_into_a_decisive_thought():
    from kairos.agents.relevance import Relevance
    from kairos.agents.thinkers import Thinkers
    from kairos.contracts import Segment

    class CheckingLLM(FakeLLM):
        async def json(self, system, user, *, purpose, temperature=0.4):
            if purpose == "checker":
                wrong = "quinze mille" in user.rsplit("The last line", 1)[1]
                return {"correction": {"topic": "budget", "utterance": "Attention, le budget validé est de 12 000 €, "
                                                                       "pas 15 000."} if wrong else None}
            return await super().json(system, user, purpose=purpose, temperature=temperature)

    board = board_with()
    board.publish("long_term", ("Budget validé : 12 000 €, transport inclus.",))
    board.publish("transcript", (Segment(1, "Inès", "Lisbonne, ça me va.", 1.0, 3.0, final=True),
                                 Segment(2, "Inès", "Côté budget, on a quinze mille euros.", 4.0, 7.0, final=True)))
    llm = CheckingLLM()
    thinkers = Thinkers(board, llm, "French", relevance=Relevance(board, llm))
    asyncio.run(thinkers.check(1))
    assert not board.snapshot().thoughts  # nothing contradicts memory
    asyncio.run(thinkers.check(2))
    asyncio.run(thinkers.check(2))  # the same correction twice: one thought
    [fix] = board.snapshot().thoughts
    assert fix.importance == 5.0 and fix.id.startswith("c") and "15 000" in fix.utterance


def test_a_new_correction_replaces_an_older_one_still_waiting():
    """Cut off by "900 euros for four", "250 euros is not enough" came back in place of the correction of 900."""
    from kairos.agents.relevance import Relevance
    from kairos.agents.thinkers import Thinkers
    from kairos.contracts import Segment

    class CheckingLLM(FakeLLM):
        async def json(self, system, user, *, purpose, temperature=0.4):
            if purpose == "checker":
                last = user.rsplit("The last line", 1)[1]
                if "250" in last:
                    return {"correction": {"topic": "budget", "utterance": "The cheapest flight is 282 euros, so 250 "
                                                                            "per person is not enough."}}
                if "900" in last:
                    return {"correction": {"topic": "total", "utterance": "For four people at 282 euros each, the "
                                                                           "total is {result} euros, not 900.",
                                           "expression": "4 * 282"}}
                return {"correction": None}
            return await super().json(system, user, purpose=purpose, temperature=temperature)

    board = board_with()
    board.publish("transcript", (Segment(1, "Sarah", "Let's keep 250 euros per person.", 1.0, 3.0, final=True),
                                 Segment(2, "Mehdi", "So for four of us that's 900 euros.", 4.0, 7.0, final=True)))
    llm = CheckingLLM()
    thinkers = Thinkers(board, llm, "English", relevance=Relevance(board, llm))
    asyncio.run(thinkers.check(1))
    [budget] = board.snapshot().thoughts
    board.update_thoughts({budget.id: {"status": ThoughtStatus.PENDING, "note": "coupée : en attente"}})  # cut off
    asyncio.run(thinkers.check(2))
    by_id = {t.id: t for t in board.snapshot().thoughts}
    assert by_id[budget.id].status == ThoughtStatus.STALE
    [total] = [t for t in by_id.values() if t.id != budget.id]
    assert total.status == ThoughtStatus.READY and "1128" in total.utterance.replace(",", "").replace(" ", "")


def test_a_correction_may_repeat_what_kairos_said_but_a_remark_may_not():
    from kairos.agents.relevance import Relevance
    board = board_with([thought("said", status=ThoughtStatus.SPOKEN)])
    relevance = Relevance(board, FakeLLM())
    relevance.remember("said", _vector("le budget validé est de 12 000 euros transport inclus"))
    text = "attention 15 000 dépasse le budget validé de 12 000 euros"  # cosine 0.70 with what was said
    assert asyncio.run(relevance.novel([text], decisive=[False])) == [None]
    assert asyncio.run(relevance.novel([text], decisive=[True]))[0] is not None


def test_paraphrase_of_an_active_thought_is_refused():
    from kairos.agents.relevance import Relevance
    board = board_with([thought("a", utterance="les retours du test arrivent le 12 février")])
    relevance = Relevance(board, FakeLLM())
    asyncio.run(relevance.run_once())
    kept = asyncio.run(relevance.novel(["les retours du test arrivent le 12 février",
                                        "le salon de Lyon est fin mars"]))
    assert kept[0] is None and kept[1] is not None


def test_thinker_asks_for_research_when_a_fact_is_missing():
    from kairos.agents.thinkers import Thinkers
    asked = []

    class NeedsFact(FakeLLM):
        async def json(self, system, user, *, purpose, temperature=0.4):
            if purpose == "thinker":
                return {"current_topic": "voyage", "new_thoughts": [],
                        "research": {"question": "Durée du vol Paris-Lisbonne ?", "query": "flight time Paris Lisbon",
                                     "topic": "vol"}}
            return await super().json(system, user, purpose=purpose, temperature=temperature)

    async def request(question, query, topic, segment, urgent):
        asked.append((query, urgent))

    thinkers = Thinkers(board_with(), NeedsFact(), "French", request_research=request)
    asyncio.run(thinkers.run_once())
    assert asked == [("flight time Paris Lisbon", False)]


def test_unknown_answer_says_it_checks_and_researches_urgently():
    from kairos.agents.thinkers import Thinkers
    from kairos.contracts import Segment
    asked = []

    class DoesNotKnow(FakeLLM):
        async def json(self, system, user, *, purpose, temperature=0.4):
            if purpose == "answer":
                return {"known": False, "topic": "vol", "utterance": "Je ne sais pas.", "query": "flight time Paris Lisbon"}
            return await super().json(system, user, purpose=purpose, temperature=temperature)

    async def request(question, query, topic, segment, urgent):
        asked.append((segment, urgent))

    board = board_with()
    board.publish("transcript", (Segment(4, "Hugo", "Kairos, combien de temps de vol ?", 1.0, 3.0, final=True),))
    asyncio.run(Thinkers(board, DoesNotKnow(), "French", request_research=request).answer(4))
    reply = next(t for t in board.snapshot().thoughts if t.answers == 4)
    assert reply.utterance == "Je regarde ça tout de suite." and asked == [(4, True)]



def test_who_is_kairos_talking_to():
    from kairos.addressing import address_strength
    assert address_strength("Kairos, on part quand ?", humans=3) == "strong"
    assert address_strength("tu sais si il y a des pays autour de Paris", humans=1) == "strong"
    assert address_strength("tu sais si il y a des pays autour de Paris", humans=3) == "weak"
    assert address_strength("Combien de temps de vol ?", humans=3) is None  # to the room, not to Kairos
    assert address_strength("On part sur Lisbonne.", humans=1) is None


def test_live_meeting_starts_without_memory():
    from kairos.session import default_memory, load_memory
    assert default_memory("live") is None and load_memory(default_memory("live")) == []
    assert default_memory("seminaire_lisbonne.txt") == "kairos_seminaire.txt"


def test_outdated_thoughts_leave_the_reservoir_with_a_reason():
    from kairos.agents.judge import JudgeAgent
    from kairos.contracts import Segment

    class Outdating(FakeLLM):
        async def yes_probability(self, system, user, *, purpose):
            return 0.95 if "outdated" in user.split("Statement:", 1)[1] else 0.1

    board = board_with([thought("train", utterance="On privilégie le train.")])
    board.publish("transcript", (Segment(5, "Vous", "je veux bien les avions", 1.0, 3.0, final=True),))
    agent = JudgeAgent(board, LlmJudge(Outdating()))
    asyncio.run(agent.run_once())
    t = board.snapshot().thoughts[0]
    assert t.status == ThoughtStatus.PENDING and t.note.startswith("mise en attente")  # first alert: set aside
    board.publish("transcript", board.snapshot().transcript +
                  (Segment(6, "Vous", "vraiment, l'avion me va très bien", 4.0, 6.0, final=True),))
    asyncio.run(agent.run_once())
    t = board.snapshot().thoughts[0]
    assert t.status == ThoughtStatus.STALE and t.note.startswith("dépassée après « vraiment, l'avion")


def test_judge_is_told_that_findings_are_not_said():
    from kairos.agents.judge import JudgeAgent
    from kairos.contracts import Segment
    seen = []

    class Recording(FakeLLM):
        async def yes_probability(self, system, user, *, purpose):
            seen.append(user)
            return 0.1

    board = board_with([thought("f")])
    board.publish("transcript", (Segment(5, "Vous", "des pays autour de Paris ?", 1.0, 3.0, final=True),))
    asyncio.run(JudgeAgent(board, LlmJudge(Recording())).run_once())
    said = next(u for u in seen if "already said out loud" in u)
    assert "NOT SAID IN THE MEETING" in said and "Only the spoken transcript counts" in said


def test_judge_confirms_a_you_question_and_opens_the_fast_lane():
    from kairos.agents.judge import JudgeAgent
    from kairos.contracts import Segment
    opened = []

    class Addressed(FakeLLM):
        async def yes_probability(self, system, user, *, purpose):
            return 0.9 if "speaks directly to Kairos" in user.split("Statement:", 1)[1] else 0.1

    board = board_with()
    board.publish("transcript", (Segment(6, "Hugo", "tu sais combien ça coûte ?", 1.0, 3.0, final=True),))
    agent = JudgeAgent(board, LlmJudge(Addressed()))
    agent.address_candidates.add(6)
    agent.on_addressed = opened.append
    asyncio.run(agent.run_once())
    assert opened == [6]


def test_after_let_me_check_the_finding_follows_at_once():
    from kairos.contracts import Segment
    ack = replace(thought("ack", answers=7, utterance="Je regarde ça."), ack=True, status=ThoughtStatus.SPOKEN)
    found = thought("found", importance=5, answers=7, utterance="9 à 16 °C en février.")
    board = board_with([ack, found], ai=AiState(last_spoke_at=9.8),  # Kairos just said "je regarde"
                       signals=Signals(addressed=0.9, addressed_segment=7, answered=False,
                                       judged_segment=7, judged_words=5))
    board.publish("transcript", (Segment(7, "Vous", "il fait combien à Lisbonne ?", 5.0, 9.0, final=True),))
    d = decide(board.snapshot())
    assert d.speak and d.reason == "asked" and d.primary == "found"



def test_a_question_to_a_named_colleague_is_not_for_kairos():
    from kairos.addressing import address_strength
    others = {"Hugo", "Inès"}
    assert address_strength("Hugo, tu avais une idée de destination ?", 3, others) is None
    assert address_strength("tu sais s'il y a un train direct ?", 3, others) == "weak"
    assert address_strength("Kairos, et toi Hugo, vous en pensez quoi ?", 3, others) == "strong"


def test_the_roster_is_known_before_people_speak():
    timeline = build_timeline(parse_script("+0.5 Inès: Hugo, tu avais une idée ?\n+0.5 Hugo: Oui, Lisbonne."))
    llm = FakeLLM()
    runtime = Runtime(timeline, [], llm, LlmJudge(llm), RunConfig())
    assert runtime._human_count() == 2 and runtime._others("Inès") == {"Hugo"}



def test_a_correction_is_never_held_back_by_the_budget():
    board = board_with([thought("fix", importance=5, relevance=0.3)])
    harness = Harness(HarnessParams(unsolicited_gap_s=30, urgent_score=0.85))
    d = decide(board.snapshot())
    assert harness.check(d, board.snapshot())
    assert harness.check(replace(d, t=d.t + 5, score=0.6), board.snapshot())  # importance 5: allowed


def test_a_thought_cut_off_twice_is_dropped():
    board = Board()
    board.publish("thoughts", (thought("a"),))
    clock = SimClock()
    speaker = None

    def on_word(_):
        speaker.hear(SpeechPartial(t=0, segment=1, speaker="A", text="non attends", t_start=0))
        for text in ("non attends je", "non attends je voulais"):
            speaker.hear(SpeechPartial(t=0, segment=1, speaker="A", text=text, t_start=0))

    speaker = Speaker(board, Scribe(board), clock, clock.now, SpeakerParams(), on_word=on_word)
    plan = UtterancePlan(0.0, "important", (("a", "un deux trois"),))
    asyncio.run(speaker.say(plan))
    assert board.snapshot().thoughts[0].status == ThoughtStatus.PENDING
    asyncio.run(speaker.say(plan))
    t = board.snapshot().thoughts[0]
    assert t.status == ThoughtStatus.STALE and "deux fois" in t.note


def test_an_urgent_duplicate_request_still_gets_its_answer():
    from kairos.agents.relevance import Relevance
    from kairos.agents.researcher import Researcher
    board = board_with(signals=Signals(addressed=0.9, addressed_segment=9, answered=False))
    llm, search = PlanningLLM(), FakeSearch()
    researcher = Researcher(board, llm, search, language="English", relevance=Relevance(board, llm))
    asyncio.run(researcher.run_once())  # the planner looked it up first (its segment is 3, not 9)
    asyncio.run(researcher.request("How many buttons?", "share of remote buttons used", "buttons", 9, True))
    assert len(search.queries) == 1  # no second search
    reply = [t for t in board.snapshot().thoughts if t.answers == 9]
    assert reply and "10 to 20 percent" in reply[0].utterance and reply[0].importance == 5.0


def test_repeating_a_mistake_does_not_outdate_its_correction():
    from kairos.agents.judge import JudgeAgent
    from kairos.contracts import Segment
    asked = []

    class Recording(FakeLLM):
        async def yes_probability(self, system, user, *, purpose):
            asked.append(user.split("Statement:", 1)[1])
            return 0.95 if "outdated" in asked[-1] else 0.1

    board = board_with([thought("fix", importance=5, utterance="Le budget est de 12 000 €, pas 15 000.")])
    board.publish("transcript", (Segment(5, "Hugo", "Avec quinze mille on peut se faire plaisir", 1.0, 3.0, final=True),))
    asyncio.run(JudgeAgent(board, LlmJudge(Recording())).run_once())
    assert board.snapshot().thoughts[0].status == ThoughtStatus.READY
    assert not any("outdated" in s for s in asked)


def test_a_pending_point_is_judged_as_a_comeback_not_as_a_reply():
    from kairos.agents.judge import JudgeAgent
    from kairos.contracts import Segment
    seen = []

    class Recording(FakeLLM):
        async def yes_probability(self, system, user, *, purpose):
            seen.append(user.split("Statement:", 1)[1])
            return 0.1

    board = board_with([thought("budget", status=ThoughtStatus.PENDING, importance=5,
                                transition="Pour revenir au budget,", utterance="c'est 12 000 €, pas 15 000.")])
    board.publish("transcript", (Segment(5, "Inès", "Le surf, carrément !", 1.0, 3.0, final=True),))
    asyncio.run(JudgeAgent(board, LlmJudge(Recording())).run_once())
    fit = next(s for s in seen if "budget" in s and "Coming back" in s)
    assert "Pour revenir au budget, c'est 12 000 €" in fit


def test_thinkers_cannot_replace_findings_or_corrections():
    from kairos.agents.thinkers import Thinkers

    class Replacer(FakeLLM):
        async def json(self, system, user, *, purpose, temperature=0.4):
            if purpose == "thinker":
                return {"current_topic": "x", "obsolete": ["rf1", "fix", "minor"], "new_thoughts": []}
            return await super().json(system, user, purpose=purpose, temperature=temperature)

    board = board_with([thought("rf1", importance=4), thought("fix", importance=5), thought("minor", importance=2)])
    asyncio.run(Thinkers(board, Replacer(), "French").run_once())
    status = {t.id: t.status for t in board.snapshot().thoughts}
    assert status == {"rf1": ThoughtStatus.READY, "fix": ThoughtStatus.READY, "minor": ThoughtStatus.STALE}


def test_a_live_conversation_budget_lets_valid_thoughts_through_in_a_silence():
    board = board_with([thought("a")])
    harness = Harness()  # the live defaults
    first = decide(board.snapshot())
    assert harness.check(first, board.snapshot()) is not None
    assert harness.check(replace(first, t=first.t + 5, score=0.5), board.snapshot()) is None  # a weak one waits
    assert harness.check(replace(first, t=first.t + 5, score=0.71), board.snapshot()) is not None  # Jev > 0.55
    assert harness.check(replace(first, t=first.t + 16, score=0.5), board.snapshot()) is not None  # 10 s later


def test_an_intervention_cut_off_in_its_first_words_does_not_use_the_budget():
    board = board_with([thought("a")])
    harness = Harness()
    first = decide(board.snapshot())
    harness.check(first, board.snapshot())
    harness.forgive(first.t)  # "Le vol…" and someone spoke
    assert harness.check(replace(first, t=first.t + 2, score=0.6), board.snapshot()) is not None
