"""Wires the components together and runs a meeting.

Modes:
    closed   Kairos speaks; the replayed meeting waits while it speaks (closed loop).
    open     Kairos speaks; the replay carries on regardless, so people talk over
             it. Tests interruption handling. Needs a real clock.
    offline  Kairos never speaks; every decision it would have taken is recorded
             and compared with the real turn changes of the meeting.

With `speed=0` the meeting runs in lockstep on a simulated clock: after each
committed line, the background agents are awaited before time moves on. That is
an idealised latency (thinking takes no time) but it is fast and repeatable.
With `speed>0` the agents run in the background while the meeting plays.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field, replace
from typing import Callable

from .agents.common import AI_NAME, Worker, is_ai
from .addressing import NAME, address_strength, is_question, names_someone_else
from .agents.judge import JudgeAgent
from .agents.notes import NotesAgent
from .agents.relevance import Relevance
from .agents.background import ContextResearch
from .agents.dispatcher import ADDRESSED, TOPIC, Dispatcher
from .agents.researcher import Researcher
from .agents.workers import WorkerSupervisor
from .agents.scribe import Scribe
from .agents.thinkers import NUMERIC, Thinkers
from .agents.topic import TopicTracker
from .agents.cleaner import TranscriptCleaner
from .backchannel import is_backchannel
from .board import Board
from .clock import RealClock, SimClock
from .contracts import Decision, SpeechEvent, SpeechFinal, SpeechPartial, ThoughtStatus, UtterancePlan, VadStep
from .decide.judges import Judge
from .decide.policy import PolicyParams, decide, floor_open
from .harness import Harness, HarnessParams, decisive_only
from .llm import LLM
from .search import SearchProvider
from .sources.live import LiveSource
from .sources.replay import ReplaySource, Timeline
from .speaker import Speaker, SpeakerParams, SpeechOutcome

MODES = ("closed", "open", "offline")
#: Similarity above which a thought counts as already said: by Kairos itself (looser: its own paraphrases)
#: or by a participant (stricter: talking about the same topic is not saying the same thing).
#: Jev's P(addressed) under which a question the fast lane took is left to the people in the room
NOT_ADDRESSED = 0.3
#: silence after a question to Kairos before it takes the turn with an opener, when its answer is not ready
CUE_AFTER_S = 0.35
log = logging.getLogger("kairos.runtime")
KAIROS_SAID = 0.80  # only near-copies: same topic with other content (0.60-0.70) is left to the judge's "said"
HUMAN_SAID = 0.85


@dataclass(slots=True)
class RunConfig:
    mode: str = "closed"
    speed: float = 0.0              # 0 = lockstep on a simulated clock
    language: str = "English"
    notes_every_s: float = 20.0     # background mode
    notes_every_lines: int = 4      # lockstep mode
    thinker_delay_s: float = 1.5    # background mode: let a burst of lines land as one run
    role: str = "discreet"          # "discreet": facts and answers only; "active": also proposals and questions
    policy: PolicyParams = field(default_factory=PolicyParams)
    harness: HarnessParams = field(default_factory=HarnessParams)

    def __post_init__(self) -> None:
        if self.role == "active":
            # An active participant speaks up more readily and more often.
            self.policy = replace(self.policy, open_threshold=0.45, share_limit=0.25)
            self.harness = replace(self.harness, unsolicited_gap_s=6.0)


@dataclass(slots=True)
class Intervention:
    t: float                  # meeting time when Kairos started
    original_t: float         # the same moment in the original recording (before the replay waited)
    reason: str
    text: str
    planned: str
    latency_s: float          # silence before Kairos started
    decision: Decision
    outcome: SpeechOutcome | None = None
    after_speech_s: float | None = None    # end of the last human words -> Kairos's first word
    thought_ready_s: float | None = None   # how long the thought had been ready when Kairos spoke
    question_to_answer_s: float | None = None  # end of the question -> answer ready (negative: ready before)


class Runtime:
    def __init__(self, timeline: Timeline | None, memory: list[str], llm: LLM, judge: Judge,
                 config: RunConfig = RunConfig(), on_change: Callable[[], None] | None = None,
                 initial_notes: str = "", search: SearchProvider | None = None,
                 live_source=None, voice=None, flights=None, hotels=None) -> None:
        if config.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if timeline is None and config.speed == 0:
            raise ValueError("a live meeting needs a real clock (speed > 0)")
        if config.mode == "open" and config.speed == 0:
            raise ValueError("open loop needs a real clock (speed > 0): people must be able to talk over Kairos")
        self.timeline = timeline
        #: a live meeting's source other than typing (the microphone through Gradium), and Kairos's voice
        self.live_source = live_source
        self.voice = voice
        self.config = config
        self.llm = llm
        self.judge = judge
        self.board = Board()
        self.board.publish("long_term", tuple(memory))
        if initial_notes:
            self.board.publish("notes", initial_notes)  # short-term memory prepared before joining
        self.scribe = Scribe(self.board)
        # Live speech recognition mishears: each committed line is corrected from its context (not replays).
        self.cleaner = TranscriptCleaner(self.board, llm, self.scribe.revise) if live_source is not None else None
        self.relevance = Relevance(self.board, llm)
        self.judge_agent = JudgeAgent(self.board, judge, config.role)
        self.thinkers = Thinkers(self.board, llm, config.language, relevance=self.relevance, role=config.role)
        self.notes = NotesAgent(self.board, llm, config.language)
        self.search = search
        self.researcher = (Researcher(self.board, llm, search, config.language, relevance=self.relevance)
                           if search else None)
        if self.researcher:
            self.thinkers.request_research = self._request_research
        # Version 2: Jev dispatches work on each sentence; workers fill briefs, ask the room, and bring results.
        self.supervisor = WorkerSupervisor(self.board, llm, config.language, flights=flights,
                                           researcher=self.researcher, search=search, hotels=hotels)
        self.thinkers.work_status = self.supervisor.status_text
        self.dispatcher = (Dispatcher(self.board, judge, self._open_question, self._open_work, self._on_topic)
                           if hasattr(judge, "ask") else None)
        if self.dispatcher is not None:
            self.supervisor.room_questions = self.dispatcher.facts  # Jev's factual questions to the room
        # What the room talks about now: Jev notices a change of subject, the writing model names it.
        self.topic = TopicTracker(self.board, llm, config.language) if self.dispatcher else None
        self._topic_task: asyncio.Task | None = None
        if self.topic is not None:
            self.thinkers.owns_topic = False
        self._dispatching: dict[int, asyncio.Task] = {}
        # Context research: reads up on the meeting's subject in the background (facts for the thinkers).
        self.context = ContextResearch(self.board, llm, search, config.language) if self.dispatcher else None
        if self.dispatcher is not None:
            self.thinkers.research_from_thoughts = False  # searches start from the dispatcher, not from ideas
        self._cued: set[int] = set()  # questions that already got a "Mmh…"
        self.harness = Harness(config.harness)
        self.decisions: dict[float | None, Decision] = {}
        self.decision_log: list[Decision] = []
        self.cues: list[tuple[float, int, str]] = []  # (meeting time, line, opener) said while an answer was prepared
        self._last_why = ""
        self.interventions: list[Intervention] = []
        self.finished = False
        self._on_change = on_change or (lambda: None)
        self._lines = 0
        self._last_think = float("-inf")
        self._kicked_words: dict[int, int] = {}
        self._checked_words: dict[int, int] = {}  # segment -> words the checker already read (partials)
        self._answering: set[int] = set()  # questions to Kairos already being answered
        self._answer_started: set[int] = set()
        #: the meeting's participants, known before they speak (from the script; from the setup in v2)
        self._roster: set[str | None] = {name for _, _, name in timeline.speech} if timeline else set()
        if live_source is not None:
            # The person at the microphone is in the room from the start: before they speak, a typed "toi tu
            # pars aussi ?" from someone else is for them, not for Kairos.
            self._roster.add(getattr(live_source, "speaker", None))
            # One microphone does not separate voices: other people may talk into it. A "tu" question is then
            # not surely for Kairos; Jev decides ("c'est malheureux ou pas ?" was for the table).
            self._roster.add(None)
        self.judge_agent.on_addressed = self._open_question
        self._speak_task: asyncio.Task | None = None
        self.clock = None
        self.source: ReplaySource | None = None
        self.speaker: Speaker | None = None
        self._workers: list[Worker] = []

    # -- running ----------------------------------------------------------------------

    def meeting_time(self) -> float:
        return self.clock.now() - self._t0

    async def run(self) -> None:
        lockstep = self.config.speed == 0
        self.clock = SimClock() if lockstep else RealClock(self.config.speed)
        self._t0 = self.clock.now()
        if self.timeline is not None:
            self.source = ReplaySource(self.timeline.events, self.clock)
        elif self.live_source is not None:
            self.live_source.attach(self.clock, self.meeting_time)
            self.source = self.live_source
        else:
            self.source = LiveSource(self.clock)
        closed = self.config.mode == "closed"
        self.speaker = Speaker(self.board, self.scribe, self.clock, self.meeting_time,
                               SpeakerParams(language=self.config.language,
                                             words_per_s=2.85 if self.voice else 2.75),  # Gradium's French pace
                               on_word=self.source.yield_floor if closed else None, voice=self.voice)
        if self.voice is not None:
            self.voice.warm_up()
            asyncio.create_task(self.voice.prepare_cues())  # the openers are ready before the first question
        tasks = []
        if not lockstep:
            relevance = Worker("relevance", self.relevance.run_once, self.clock)
            judge = Worker("judge", self.judge_agent.run_once, self.clock)

            async def think() -> None:
                await self.thinkers.run_once()
                relevance.kick()
                judge.kick()

            thinkers = Worker("thinkers", think, self.clock, delay_s=self.config.thinker_delay_s)
            self._workers = [relevance, judge, thinkers]
            if self.researcher and self.dispatcher is None:  # without Jev, the planner decides searches
                async def research() -> None:
                    await self.researcher.run_once()
                    relevance.kick()
                    judge.kick()

                self._workers.append(Worker("researcher", research, self.clock, delay_s=0.2))
            tasks = [asyncio.create_task(w.loop()) for w in self._workers]
            tasks.append(asyncio.create_task(self._notes_loop()))
        try:
            async for event in self.source.events():
                await self._on_event(event)
            if self._speak_task:
                await self._speak_task
            if lockstep or self.config.mode == "offline":
                await self.notes.run_once()
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            self.finished = True
            self._on_change()

    def stop(self) -> None:
        """End a live meeting: the source stops, the run finishes normally."""
        if self.timeline is None and self.source is not None:
            self.source.stop()
        if self.voice is not None:
            self.voice.close()

    async def _notes_loop(self) -> None:
        while True:
            await self.clock.sleep(self.config.notes_every_s)
            try:
                await self.notes.run_once()
            except Exception:
                pass
            if self.context is not None:
                # After the notes: the subject is freshest. Runs aside, never blocking the notes.
                asyncio.create_task(self._context_research())

    def _on_topic(self, p_new: float) -> None:
        """Jev read the last lines: name the subject if there is none yet, confirm it if Jev sees a change."""
        if self.topic is None or (self._topic_task is not None and not self._topic_task.done()):
            return
        if p_new >= TOPIC or self.topic.wants_name():
            self._topic_task = asyncio.create_task(self._topic_check())

    async def _topic_check(self) -> None:
        try:
            changed = await self.topic.check()
        except Exception:
            log.exception("topic check failed")
            return
        if not changed:
            return
        # A new subject: the thinkers start from it, the notes follow it, the research reads up on it now.
        for worker in self._workers:
            if worker.name == "thinkers":
                worker.kick()
        asyncio.create_task(self.notes.run_once())
        if self.context is not None:
            self.context.new_subject()
            asyncio.create_task(self._context_research())

    async def _context_research(self) -> None:
        try:
            await self.context.run_once()
        except Exception:
            log.exception("context research failed")
            return
        for worker in self._workers:
            if worker.name == "thinkers":
                worker.kick()  # new facts: the thinkers may now have something concrete to offer

    # -- events -------------------------------------------------------------------------

    async def _on_event(self, event: SpeechEvent) -> None:
        self.scribe.on_event(event)
        if isinstance(event, (SpeechPartial, SpeechFinal)) and self.speaker.speaking:
            self.speaker.hear(event)
        if isinstance(event, SpeechPartial) and event.speaker != AI_NAME and self.config.speed > 0:
            self._on_partial(event)
        if isinstance(event, SpeechFinal):
            await self._after_line(event)
        elif isinstance(event, VadStep):
            await self._tick()
        self._on_change()

    def _on_partial(self, event: SpeechPartial) -> None:
        """Push speech to the background agents as it arrives, so their work is ready at the gap."""
        words = len(event.text.split())
        last = self._kicked_words.get(event.segment, 0)
        ends_clause = event.text.rstrip().endswith((".", "?", "!", ",", ";", ":"))
        if (self.config.speed > 0 and event.speaker != AI_NAME and ends_clause and words >= 5
                and words - self._checked_words.get(event.segment, 0) >= 5 and NUMERIC.search(event.text)):
            # A figure in a sentence still being spoken: check it now, so a correction is ready at the pause
            # ("donc à quatre aller-retour ça ferait 240 euros, c'est jouable ?": the check starts at the comma).
            self._checked_words[event.segment] = words
            asyncio.create_task(self._check(event.segment))
        if ends_clause or words - last >= 4:
            self._kicked_words[event.segment] = words
            for worker in self._workers:
                if worker.name in ("relevance", "judge"):
                    worker.kick()
        if event.text.rstrip().endswith((".", "?", "!")):
            for worker in self._workers:
                if worker.name == "thinkers":
                    worker.kick()
        if self.dispatcher is not None and self.dispatcher.wants(event.segment, event.text, final=False):
            self._dispatch(event.segment, event.text)
        if event.text.rstrip().endswith("?"):
            # A question is being asked: start answering and researching before the line is committed.
            if self._address(event.segment, event.text, event.speaker) == "strong":
                self._open_question(event.segment)
            for worker in self._workers:
                if worker.name == "researcher":
                    worker.kick()

    def _address(self, segment: int, text: str, speaker: str | None) -> str | None:
        """Is the line for Kairos? The words first; then the flow: a question right after Kairos spoke is a
        follow-up to what it said ("Aujourd'hui, c'est quand ?" after "ouverte jusqu'à 20 h aujourd'hui")."""
        others = self._others(speaker)
        strength = address_strength(text, self._human_count(), others)
        if (strength != "strong" and is_question(text) and not names_someone_else(text, others)
                and self._follows_kairos(segment)):
            return "strong"
        return strength

    def _follows_kairos(self, segment: int, within_s: float = 20.0) -> bool:
        """Kairos spoke the line just before this one, a moment ago."""
        lines = [s for s in self.board.snapshot().transcript if s.final or s.id == segment]
        line = next((s for s in lines if s.id == segment), None)
        if line is None:
            return False
        before = [s for s in lines if s.id != segment and s.t_start < line.t_start]
        return bool(before) and is_ai(before[-1]) and line.t_start - before[-1].t_end <= within_s

    def _participants(self) -> set[str | None]:
        """Everyone known to take part: the meeting's roster (the script's speakers) and whoever spoke since."""
        seen = {seg.speaker for seg in self.board.snapshot().transcript if not is_ai(seg)}
        return self._roster | seen

    def _others(self, speaker: str | None) -> set[str]:
        """Names of the other people in the meeting: a line naming one of them is for them."""
        return {name for name in self._participants() if name and name != speaker}

    def _human_count(self) -> int:
        """How many people take part (unknown speakers count as a group)."""
        people = self._participants()
        return 2 if None in people else len(people)

    def _open_question(self, segment: int) -> None:
        """Fast lane: a line naming Kairos. Treat it as a question at once; the judge confirms or cancels."""
        if segment in self._answering:
            return
        self._answering.add(segment)
        # A new question replaces the answers to earlier ones still waiting (cut off, or not said yet):
        # the person has moved on, the old answer must not come back later.
        older = {t.id: {"status": ThoughtStatus.STALE, "note": "remplacée par la question suivante"}
                 for t in self.board.snapshot().thoughts
                 if t.answers is not None and t.answers != segment
                 and t.status in (ThoughtStatus.READY, ThoughtStatus.PENDING)}
        if older:
            self.board.update_thoughts(older, by="nouvelle question")
        self.board.publish("signals", replace(self.board.snapshot().signals, addressed=0.9,
                                              addressed_segment=segment, answered=False))
        if self.config.speed > 0:
            asyncio.create_task(self._answer(segment))

    async def _after_line(self, line: SpeechFinal) -> None:
        self._lines += 1
        strength = (self._address(line.segment, line.text, line.speaker)
                    if line.speaker != AI_NAME else None)
        if strength == "strong":
            self._open_question(line.segment)
        elif strength == "weak":
            self.judge_agent.address_candidates.add(line.segment)  # the judge decides
        said_by = f"dit par {line.speaker or 'quelqu’un'} à {line.t_end:.0f} s"
        if line.speaker != AI_NAME and not is_backchannel(line.text):
            self.supervisor.on_line(line.segment, line.speaker)
            if self.dispatcher is not None and self.dispatcher.wants(line.segment, line.text, final=True):
                if self.config.speed > 0:
                    self._dispatch(line.segment, line.text)
                else:
                    await self.dispatcher.run(line.segment, line.text)
        if self.config.speed > 0:
            if not is_backchannel(line.text):
                asyncio.create_task(self._retire(line.text, HUMAN_SAID, said_by))
                # Only what the microphone heard is cleaned: typed lines are exactly what was meant.
                if self.cleaner is not None and line.speaker == getattr(self.live_source, "speaker", None):
                    asyncio.create_task(self._clean_then_check(line.segment))
                else:
                    asyncio.create_task(self._check(line.segment))
            for worker in self._workers:
                worker.kick()
            return
        # Lockstep: the same work as the background workers, awaited, and spaced the same way.
        now = self.meeting_time()
        if not is_backchannel(line.text):
            await self._retire(line.text, HUMAN_SAID, said_by)
        await self.relevance.run_once()
        await self._answer_if_open(line.segment)
        backchannel = is_backchannel(line.text)
        if not backchannel:
            await self.thinkers.check(line.segment)
        if not backchannel and now - self._last_think >= self.config.thinker_delay_s:
            self._last_think = now
            await self.thinkers.run_once()
            if self.researcher:
                await self.researcher.run_once()
            await self.relevance.run_once()
        await self.judge_agent.run_once()
        await self._answer_if_open(line.segment)  # a "tu"/"vous" question the judge just confirmed
        if self._lines % self.config.notes_every_lines == 0:
            await self.notes.run_once()

    async def _answer_if_open(self, segment: int) -> None:
        """Lockstep: write the answer once the question is open (the background lane does it itself)."""
        if segment in self._answering and segment not in self._answer_started:
            self._answer_started.add(segment)
            await self.thinkers.answer(segment)

    async def _clean_then_check(self, segment: int) -> None:
        """Correct the line first: the checker reads figures, and a misheard figure is a false correction.
        The dispatcher and the answer lane already started on the raw line: they must not wait."""
        try:
            cleaned = await self.cleaner.clean(segment)
        except Exception:
            log.exception("transcript cleaning failed")
            cleaned = None
        await self._check(segment)
        if cleaned:
            for worker in self._workers:
                if worker.name in ("relevance", "judge", "thinkers"):
                    worker.kick()  # they now read the corrected words

    async def _check(self, segment: int) -> None:
        try:
            await self.thinkers.check(segment)
        except Exception:
            return  # a failed check costs nothing but this check
        for worker in self._workers:
            if worker.name in ("relevance", "judge"):
                worker.kick()

    def _dispatch(self, segment: int, text: str) -> None:
        self.dispatcher.claim(segment, text)
        self._dispatching[segment] = asyncio.create_task(self.dispatcher.run(segment, text))

    async def _open_work(self, kind: str, segment: int, text: str, addressed: bool) -> None:
        await self.supervisor.open(kind, segment, text, addressed)
        for worker in self._workers:
            if worker.name in ("relevance", "judge"):
                worker.kick()

    async def _answer(self, segment: int) -> None:
        self._answer_started.add(segment)
        pending = self._dispatching.get(segment)
        if pending is not None and not pending.done():
            with contextlib.suppress(Exception):
                await asyncio.wait_for(asyncio.shield(pending), timeout=1.2)  # Jev: is a worker taking it?
            if not pending.done() and "vols" in (self.dispatcher.routing.get(segment) or []):
                with contextlib.suppress(Exception):  # the flight brief is deciding whether it answers this line
                    await asyncio.wait_for(asyncio.shield(pending), timeout=6.0)
        if self._not_for_kairos(segment):
            return
        if self.supervisor.waiting() and await self.supervisor.absorb(segment):
            return  # the line answers a brief's question: the brief replies (one voice at a time)
        if self.supervisor.handles(segment):
            return  # a worker brief answers this line (a question for the room, then its result)
        try:
            await self.thinkers.answer(segment)
        except Exception:
            log.exception("answer lane failed on line %s", segment)  # never silent: the trace needs it
            return  # no answer: the policy lets another fitting thought reply after a moment
        for worker in self._workers:
            if worker.name in ("relevance", "judge"):
                worker.kick()
        self._on_change()

    def _surely_addressed(self, segment: int) -> bool:
        """Its name was said, or Jev read the line as addressed to it: an "Alors…" will be followed by an answer."""
        line = next((s for s in self.board.snapshot().transcript if s.id == segment), None)
        if line is not None and NAME.search(line.text):
            return True
        if self.dispatcher is None:
            return False
        read = next((d for d in reversed(self.dispatcher.log) if d.segment == segment), None)
        return read is not None and read.p.get("addressed", 0.0) >= ADDRESSED

    def _not_for_kairos(self, segment: int) -> bool:
        """The fast lane opened on a question; Jev read the line and says it was for the people in the room.
        Without Kairos's name, Jev wins: the question is closed and nothing is said."""
        if self.dispatcher is None:
            return False
        read = next((d for d in reversed(self.dispatcher.log) if d.segment == segment), None)
        if read is None or read.p.get("addressed", 1.0) >= NOT_ADDRESSED or NAME.search(read.text):
            return False
        signals = self.board.snapshot().signals
        if signals.addressed_segment == segment and not signals.answered:
            self.board.publish("signals", replace(signals, answered=True, addressed=read.p["addressed"]))
        return True

    async def _tick(self) -> None:
        if self._speak_task and not self._speak_task.done():
            return
        snap = self.board.snapshot()
        decision = decide(snap, self.config.policy)
        signals = snap.signals
        if (self.voice is not None and not decision.speak and not signals.answered
                and signals.addressed_segment is not None and signals.addressed_segment not in self._cued
                and not snap.room.speaking and not snap.ai.speaking and snap.room.silence_s >= CUE_AFTER_S
                and self._surely_addressed(signals.addressed_segment)):
            # Asked something, answer not ready: take the turn at once ("Alors…"), the answer follows.
            self._cued.add(signals.addressed_segment)
            asyncio.create_task(self._cue(snap.room.t, signals.addressed_segment))
        if floor_open(snap, self.config.policy):
            gap = snap.room.silence_since
            previous = self.decisions.get(gap)
            if previous is None or not previous.speak:
                self.decisions[gap] = decision
            if decision.why != self._last_why:
                # Every change of mind, for the console and the Monitor: one decision per silence hid why a result
                # waited after Kairos had spoken.
                self._last_why = decision.why
                self.decision_log.append(decision)
                del self.decision_log[:-300]
        if not decision.speak:
            return
        plan = self.harness.check(decision, snap)
        if plan is None and self.harness.refusals and self.harness.refusals[-1][1].startswith("budget"):
            # The budget holds the best idea back, not a correction or an answer: those may still pass.
            decisive = decisive_only(snap)
            fallback = decide(decisive, self.config.policy)
            if fallback.speak:
                decision, plan = fallback, self.harness.check(fallback, decisive)
                if plan is not None:
                    self.decisions[snap.room.silence_since] = decision
        if plan is None:
            return
        now = self.meeting_time()
        humans = [s for s in snap.transcript if not is_ai(s)]
        last_words = max((s.t_end for s in humans), default=None)
        primary = next((t for t in snap.thoughts if t.id == decision.primary), None)
        question = next((s for s in humans if primary and s.id == primary.answers), None)
        intervention = Intervention(
            t=now, original_t=now - self.source.offset, reason=plan.reason,
            text="", planned=" ".join(text for _, text in plan.parts),
            latency_s=snap.room.silence_s, decision=decision,
            after_speech_s=round(now - last_words, 2) if last_words is not None else None,
            thought_ready_s=round(snap.room.t - primary.created_at, 2) if primary else None,
            question_to_answer_s=round(primary.created_at - question.t_end, 2) if question else None)
        self.interventions.append(intervention)
        if self.config.mode == "offline":
            await self._pretend_spoken(plan)
        elif self.config.speed == 0:
            await self._speak(plan, intervention)
        else:
            self._speak_task = asyncio.create_task(self._speak(plan, intervention))

    async def _speak(self, plan: UtterancePlan, intervention: Intervention) -> None:
        outcome = await self.speaker.say(plan)
        intervention.outcome = outcome
        intervention.text = outcome.text
        if outcome.yielded and outcome.cut:
            self._after_cut(outcome.cut)
        if outcome.yielded and len(outcome.text.split()) < 6:
            self.harness.forgive(plan.t)  # cut off in its first words: not counted against the budget
        # Only what was said in full retires its duplicates: "Ils sont attendus le…" (cut off) did not say the date.
        said_in_full = " ".join(text for tid, text in plan.parts if tid in outcome.said)
        self.supervisor.on_spoken(outcome.said)  # a question for the room was asked: its brief waits for answers
        await self._retire(said_in_full, KAIROS_SAID, f"déjà dit par Kairos à {intervention.t:.0f} s")
        self._on_change()

    async def _cue(self, t: float, segment: int) -> None:
        """The opener said while the answer is prepared, kept so that the console and tests see it."""
        text = await self.voice.cue()
        if text:
            self.cues.append((t, segment, text))
            del self.cues[:-100]
            self._on_change()

    def _after_cut(self, cut_ids: list[str]) -> None:
        """Someone talked over Kairos and went on: what they said changes what Kairos should say. Its own cut
        idea or answer is not said again word for word later ("Qu'est-ce que tu cherches, et où ?" after
        "je suis à Paris"): an answer still owed is written again from the conversation as it is now.
        Search results and the workers' questions keep their turn: their content does not depend on it."""
        snap = self.board.snapshot()
        thoughts = {t.id: t for t in snap.thoughts}
        cut = [thoughts[tid] for tid in cut_ids if tid in thoughts and thoughts[tid].kind in ("idea", "answer")
               and thoughts[tid].brief is None and not thoughts[tid].ack]
        if not cut:
            return
        self.board.update_thoughts({t.id: {"status": ThoughtStatus.STALE,
                                           "note": "coupée : à reformuler après ce qui a été dit"} for t in cut},
                                   by="orateur")
        cut_at = snap.room.t
        for segment in {t.answers for t in cut if t.answers is not None}:
            asyncio.create_task(self._answer_again(segment, cut_at))

    async def _answer_again(self, segment: int, cut_at: float = 0.0) -> None:
        """The answer to this question was cut off: once the person has finished, write it again with what
        they just added. Nothing if another question replaced it or it was answered meanwhile, nor when the
        person only acknowledged it ("ah ok, je vois"): writing the same answer again, to be cut again, was
        heard as Kairos repeating itself."""
        for _ in range(100):  # up to 8 s for the interrupting sentence to end
            room = self.board.snapshot().room
            if not room.speaking and room.silence_s >= 0.4:
                break
            await self.clock.sleep(0.08)
        snap = self.board.snapshot()
        signals = snap.signals
        if signals.addressed_segment != segment or signals.answered:
            return
        said = [s.text for s in snap.transcript if s.final and not is_ai(s) and s.t_end >= cut_at]
        if not said or all(_acknowledges(text) for text in said):
            return
        try:
            await self.thinkers.answer(segment)
        except Exception:
            log.exception("answer lane failed again on line %s", segment)
        self._on_change()

    async def _pretend_spoken(self, plan: UtterancePlan) -> None:
        """Offline mode: record the decision as if it had been carried out, without speaking."""
        self.board.update_thoughts(by="orateur", changes={tid: {"status": ThoughtStatus.SPOKEN, "note": "dite (hors ligne)"}
                                    for tid, _ in plan.parts})
        if plan.reason == "asked":
            self.board.publish("signals", replace(self.board.snapshot().signals, answered=True))
        self.interventions[-1].text = self.interventions[-1].planned
        await self._retire(self.interventions[-1].planned, KAIROS_SAID,
                           f"déjà dit par Kairos à {self.interventions[-1].t:.0f} s")

    async def _request_research(self, question: str, query: str, topic: str, segment: int | None,
                                urgent: bool) -> None:
        """A thinker needs a fact, or Kairos was asked something it does not know: look it up now."""
        if self.dispatcher is not None and urgent:
            # Version 2: a brief, tracked like any other search; its result (or its failure) is owed to that line.
            await self.supervisor.request_web(question, query, segment)
            return
        if self.config.speed == 0:
            await self.researcher.request(question, query, topic, segment, urgent)
            return

        async def search_then_judge() -> None:
            await self.researcher.request(question, query, topic, segment, urgent)
            for worker in self._workers:
                if worker.name in ("relevance", "judge"):
                    worker.kick()
            self._on_change()

        asyncio.create_task(search_then_judge())

    async def _retire(self, text: str, threshold: float, note: str) -> None:
        """Keep the reservoir in step with the conversation: what was just said is no longer to say."""
        try:
            await self.relevance.retire_similar(text, threshold, note)
        except Exception:
            pass  # the judge's "already said" check remains as a second line

    # -- live participant (console) -------------------------------------------------------

    async def say_live(self, speaker: str, text: str, segment: int) -> None:
        """A participant types in the console: their words arrive at speech rate, like the others."""
        start = self.meeting_time()
        words = text.split()
        self.scribe.set_live_speaking(True, start)
        if self.timeline is None:
            self.source.set_speaking(True, start)
        spoken = []
        for word in words:
            await self.clock.sleep(1 / 2.75)
            spoken.append(word)
            event = SpeechPartial(t=self.meeting_time(), segment=segment, speaker=speaker,
                                  text=" ".join(spoken), t_start=start)
            await self._on_event(event)
        end = self.meeting_time()
        self.scribe.set_live_speaking(False, end)
        if self.timeline is None:
            self.source.set_speaking(False, end)
        await self._on_event(SpeechFinal(t=end, segment=segment, speaker=speaker, text=text,
                                         t_start=start, t_end=end))

    # -- reporting -------------------------------------------------------------------------

    def human_lines(self) -> int:
        return sum(1 for s in self.board.snapshot().transcript if s.final and not is_ai(s))


def _acknowledges(text: str) -> bool:
    """A short reply that takes in what Kairos said without asking or adding anything ("Ah, ok. Je vois.")."""
    return "?" not in text and len(text.split()) <= 5
