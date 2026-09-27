"""The researcher: looks things up on the web in the background, so Kairos is ready to speak.

1. After a line, a planning call decides whether anything is worth looking up:
   a factual question asked out loud that nobody answered, or a factual point
   that would change a decision, and that Kairos's memory does not cover.
2. The query goes out without names or internal details.
3. The finding is stored on the board (short-term memory, with its sources),
   and turned at once into a thought ready to be said, with a lead-in for
   when the discussion has moved on. The judge and the decider treat it like
   any other thought: nothing waits for the search.

A budget limits searches per meeting and spaces them out.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import replace

from ..board import Board
from ..contracts import Finding, Thought, ThoughtStatus
from ..llm import LLM, cosine
from ..search import SearchProvider
from .common import enforce_cap, last_human_segment, render_transcript

LEAD_IN = {"French": "J'ai vérifié pour {topic} :", "English": "I looked up {topic}:"}

PLAN_SYSTEM = """You decide whether Kairos, an AI assistant attending a live meeting, should look something up on the web now, so it can contribute with facts.
Return one JSON object: {"search": null} or
{"search": {"question": "the question as it came up", "query": "a short web search query", "topic": "2-4 words", "segment": 12, "unanswered": true}}.

Look something up only if ALL of these hold:
- the question was raised by a PARTICIPANT (not by Kairos) in the LAST 3 LINES of the transcript: a question asked out loud (to anyone, or to Kairos) that nobody answered yet, or a factual point the room is deciding on right now. Questions from the notes, from Kairos's own lines, or from earlier in the meeting do not count;
- it is about the outside world, and the public web can answer it: typical figures, standards, what users generally prefer, how existing products do it;
- Kairos's memory and past findings do not already answer it.
Never look up anything about this organisation, this project or product (its dates, plans, scope, prices it sets), its brand, its budget or the people present: the web cannot know it, and a generic answer ("a beta usually lasts 2 to 12 weeks") would mislead the room.
Never look up opinions. The query must contain no names and no confidential details. Never repeat or rephrase a past search.
Most of the time the answer is {"search": null}. "segment" is the id (number after L) of the line that raised it.
"unanswered" is true only if a participant asked this factual question out loud and nobody in the room has answered it since; false for a rhetorical question ("isn't it a bit expensive?"), an answered one, or a point Kairos raises itself."""

NOT_FOUND = re.compile(r"\b(couldn'?t find|could not find|no clear answer|not find|pas trouvé|aucune réponse claire)\b",
                       re.IGNORECASE)
MD_LINK = re.compile(r"\(?\[([^\]]+)\]\((https?://[^)]+)\)\)?")
BARE_URL = re.compile(r"https?://\S+")


SOURCE_CUES = ("selon", "d'après", "d’après", "according to", "from", "source :", "source:")


def spoken(answer: str, max_words: int = 35) -> str:
    """What Kairos says: no links or URLs (sources stay in memory), short."""
    def source(m: re.Match) -> str:
        # Keep the source's name only where the sentence needs it ("selon X"); drop trailing citations.
        before = answer[:m.start()].rstrip().lower()
        return m.group(1) if before.endswith(SOURCE_CUES) else ""

    text = MD_LINK.sub(source, answer)
    text = BARE_URL.sub("", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\s+([.,;:])", r"\1", re.sub(r"\s{2,}", " ", text)).strip()
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words]).rstrip(",;:") + "."


class Researcher:
    def __init__(self, board: Board, llm: LLM, provider: SearchProvider, language: str = "English",
                 max_searches: int = 6, min_interval_s: float = 20.0, relevance=None) -> None:
        self.board = board
        self.llm = llm
        self.provider = provider
        self.relevance = relevance  # optional: refuses a query too close to a past one
        self._past_queries: list[list[float]] = []
        self._asked_for: set = set()  # questions (line ids or texts) already looked up
        self._owed: int | None = None  # a question whose answer is the next finding
        self.skipped_repeats = 0
        self.language = language
        self.max_searches = max_searches
        self.min_interval_s = min_interval_s
        self._ids = itertools.count(1)
        self._last_search_at: float | None = None
        self._busy = False
        self.requests = 0  # searches asked for directly (thinkers, answers), without the planning call

    @property
    def searches(self) -> int:
        return len(self.board.snapshot().findings)

    async def run_once(self) -> None:
        snap = self.board.snapshot()
        now = snap.room.t
        if self._busy or self.searches >= self.max_searches:
            return
        if self._last_search_at is not None and now - self._last_search_at < self.min_interval_s:
            return
        line = last_human_segment(snap)
        if line is not None and snap.signals.addressed_segment == line.id and not snap.signals.answered:
            return  # a question to Kairos: the answer lane decides whether a search is needed
        memory = "\n".join(f"- {m}" for m in snap.long_term) or "(none)"
        past = "\n".join(f"- {f.question} -> {f.answer or f.status}" for f in snap.findings) or "(none)"
        user = (f"Kairos's memory:\n{memory}\n\nPast findings:\n{past}\n\n"
                f"Meeting notes:\n{snap.notes or '(none yet)'}\n\n"
                f"Transcript (latest last):\n{render_transcript(snap, last=10)}")
        plan = (await self.llm.json(PLAN_SYSTEM, user, purpose="research plan", temperature=0.2)).get("search")
        if not isinstance(plan, dict) or not str(plan.get("query") or "").strip():
            return
        await self._search(str(plan.get("question") or plan["query"]), str(plan["query"]),
                           str(plan.get("topic") or "this"), _int(plan.get("segment")), answers=False,
                           unanswered=plan.get("unanswered") is True)

    async def request(self, question: str, query: str, topic: str, segment: int | None = None,
                      urgent: bool = False, unanswered: bool = False) -> None:
        """A search asked for directly: by the thinkers (a fact they need) or by the answer lane (a question
        Kairos cannot answer). No planning call. An urgent request (someone is waiting) skips the spacing."""
        now = self.board.snapshot().room.t
        if self.searches >= self.max_searches or (self._busy and not urgent):
            return
        if not urgent and self._last_search_at is not None and now - self._last_search_at < self.min_interval_s / 2:
            return
        self.requests += 1
        await self._search(question, query, topic, segment, answers=urgent, unanswered=unanswered)

    async def _search(self, question: str, query: str, topic: str, segment: int | None, answers: bool,
                      unanswered: bool = False) -> None:
        now = self.board.snapshot().room.t
        # One search per question: the planner and a thinker may ask for the same thing at the same moment.
        key = segment if segment is not None else question.strip().lower()
        if key in self._asked_for or (self.relevance is not None and not await self._new_query(query)):
            # Already looked up (or being looked up). If someone is waiting for this answer, Kairos owes it:
            # the finding, ready or still coming, becomes the answer to their question.
            self.skipped_repeats += 1
            if answers and segment is not None:
                self._owe(segment, topic)
            return
        self._asked_for.add(key)
        self._busy = True
        self._last_search_at = now
        finding = Finding(id=f"f{next(self._ids)}", question=question, query=query, segment=segment,
                          status="searching", started_at=now)
        self._publish(finding)
        try:
            result = await self.provider.search(finding.query, finding.question, self.language)
        except Exception:
            self._publish(replace(finding, status="failed"))
            return
        finally:
            self._busy = False
        if not result.answer or NOT_FOUND.search(result.answer):
            # Nothing solid: keep the attempt in memory, but give Kairos nothing to say.
            self._publish(replace(finding, status="failed", answer=result.answer, seconds=round(result.seconds, 2)))
            return
        done = replace(finding, status="done", answer=result.answer, sources=result.sources,
                       seconds=round(result.seconds, 2))
        self._publish(done)
        self._ready_to_say(done, topic, answers, unanswered=unanswered)

    def _owe(self, segment: int, topic: str) -> None:
        done = [f for f in self.board.snapshot().findings if f.status == "done"]
        if done:  # the answer is already here: offer it as the reply
            self._ready_to_say(done[-1], topic, answers_segment=segment)
        else:     # still searching: the next finding answers this question
            self._owed = segment

    def _ready_to_say(self, finding: Finding, topic: str, answers_question: bool = False,
                      answers_segment: int | None = None, unanswered: bool = False) -> None:
        """Turn the finding into a thought Kairos can say at the next gap."""
        snap = self.board.snapshot()
        signals = snap.signals
        if answers_segment is None and self._owed is not None and not signals.answered:
            answers_segment, self._owed = self._owed, None
        asked = answers_question or (not signals.answered and signals.addressed_segment == finding.segment)
        answers = answers_segment if answers_segment is not None else (
            finding.segment if asked and finding.segment is not None else None)
        thought_id = f"r{finding.id}" if answers_segment is None else f"r{finding.id}a{answers_segment}"
        if any(t.id == thought_id for t in snap.thoughts):
            return
        lead_in = LEAD_IN.get(self.language, LEAD_IN["English"]).format(topic=topic)
        self.board.update_thoughts({}, (Thought(
            id=thought_id, topic=topic, content=spoken(finding.answer, 60), utterance=spoken(finding.answer),
            transition=lead_in,
            # a factual question the room asked and left open is as decisive as a question to Kairos
            importance=5.0 if answers is not None or unanswered else 4.0,
            relevance=0.0, fit_now=0.0, already_said=0.0, status=ThoughtStatus.READY,
            stimuli=(f"web:{finding.id}",) + ((f"L{finding.segment}",) if finding.segment is not None else ()),
            version=snap.version, created_at=snap.room.t, answers=answers, kind="answer" if answers is not None else "finding"),), by="chercheur")
        enforce_cap(self.board)

    async def _new_query(self, query: str, threshold: float = 0.80) -> bool:

        vector = (await self.llm.embed([query], purpose="embedding"))[0]
        if any(cosine(vector, past) >= threshold for past in self._past_queries):
            return False
        self._past_queries.append(vector)
        return True

    def _publish(self, finding: Finding) -> None:
        others = tuple(f for f in self.board.snapshot().findings if f.id != finding.id)
        self.board.publish("findings", others + (finding,))


def _int(value) -> int | None:
    try:
        return int(str(value).lstrip("L"))
    except (TypeError, ValueError):
        return None
