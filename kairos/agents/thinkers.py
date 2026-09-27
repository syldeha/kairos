"""The thinkers: form what Kairos could say, before it is time to say it.

Adapted from thoughtful-agents (Inner Thoughts, CHI 2025): thoughts are
inspired by the conversation, Kairos's long-term memory and its previous
thoughts, and carry an importance score. Differences:
- they read the live meeting notes (short-term memory), not only the last lines;
- each thought comes with the sentence to say and a lead-in for later;
- they do not decide when to speak: the "now?" question moved to the judge;
- thoughts whose topic the discussion left become PENDING instead of being lost.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import replace

from ..board import Board
from ..contracts import Thought, ThoughtStatus
from ..llm import LLM
from .common import active_thoughts, enforce_cap, is_ai, language_of, recent_ack, render_findings, render_transcript


SYSTEM = """You are the inner voice of Kairos, an AI assistant attending a live meeting with humans. You never speak here: you form thoughts Kairos might say later. A separate system decides whether and when Kairos speaks.

Return one JSON object:
{
  "current_topic": "what the room is discussing right now, in a few words",
  "moved_on": ["ids of listed READY thoughts whose topic the discussion has left"],
  "obsolete": ["ids of listed thoughts that someone already said, or that are no longer true or useful"],
  "research": null,
  "new_thoughts": [
    {
      "topic": "the topic it belongs to",
      "content": "the idea in one line",
      "utterance": "what Kairos would say: one or two short spoken sentences, under 30 words, natural and direct",
      "transition": "a short lead-in if it is said later, after the discussion moved on (for example: 'Going back to the price,')",
      "importance": 1,
      "stimuli": ["L12", "M3"]
    }
  ]
}

Rules:
- At most 2 new thoughts. Zero is fine and common. The reservoir holds at most 5 thoughts (listed below): add one only if it covers an angle none of them covers; list a weaker version you replace in "obsolete".
- "research": if a public fact (web) would let Kairos contribute and neither its memory nor its findings have it, ask for it: {"question": "...", "query": "short web query, no names", "topic": "2-4 words"}. Otherwise null. Never about this organisation, its project, product, dates, plans, budget or people: the web cannot know them (a generic "usually 2 to 12 weeks" misleads the room).
- A thought must add something: a fact from memory, an answer, a useful question, a risk nobody raised, a link between two points. Never restate what was said, never praise, never summarise the obvious.
- Kairos is the group's assistant, not one of them: it is not part of their plans, trips or lives, and never talks like a member ("il faudrait confirmer la date", "on devrait..."). It brings what an assistant can: a fact, a figure, a search result, a calculation, or an offer to look something up when someone has a concrete need ("je peux chercher les billets pour Groningen du 11 au 14 ?").
- importance, used sparingly: 5 = the room is about to decide on something wrong or is missing a decisive fact Kairos knows; 4 = specific information that directly helps the current point; 3 = a relevant but optional remark; 1-2 = minor. Most thoughts are 2 or 3. Generic advice ("we should keep X in mind", "it is important to...") is 1. A concrete fact from Kairos's memory (a date, a figure, a name) that the room is missing or implicitly asking about is 4 or 5.
- FIRST, compare the last lines with Kairos's memory: if someone states a figure, a date, a count or a plan that contradicts what Kairos knows (a budget, a number of people, a constraint, a person's needs), the thought is a short correction naming both values ("Attention, the budget is 12,000 €, not 15,000"), importance 5. Same if a factual question asked to the room can be answered from memory or findings.
- Once Kairos has raised a point (see its own lines), do not raise it again in other words unless the room decides against it; move on to what Kairos has not said yet.
- Ground every proposal in the transcript, Kairos's memory or its findings: never invent options (places, prices, names) nobody mentioned.
- The transcript comes from live speech recognition: a word that makes no sense there was probably misheard ("ça aurait déjà raison" for "t'as raison"). Understand the line by its sound and the conversation, and never react to the misheard word itself.
- Never do arithmetic yourself (totals, divisions, margins, units to sell): a separate checker computes numbers exactly.
- Write content, utterance and transition in the language of the meeting: {language}. The utterance is spoken aloud: plain words only, no slashes, asterisks, arrows, bullet points or abbreviations like "vs" (write "ou", "vers", "contre").
- Address the participants the way they address each other or Kairos: with "tu" if they say "tu", with "vous" otherwise (in French).
- stimuli: ids of transcript lines (L..) and memories (M..) that inspired the thought.
- Do not duplicate an existing thought, and do not repeat what Kairos already said."""

#: Rules that replace the default ones when Kairos is an active participant.
ACTIVE_RULES = """Rules (Kairos is an ACTIVE participant, like a sharp colleague, not a silent note-taker):
- At most 2 new thoughts per pass. The reservoir holds at most 5 thoughts (listed below): add one only if it covers an angle none of them covers; list a weaker version you replace in "obsolete". Zero new thoughts is fine.
- "research": if a public fact (web) would let Kairos contribute and neither its memory nor its findings have it, ask for it: {"question": "...", "query": "short web query, no names", "topic": "2-4 words"}. Otherwise null. Never about this organisation, its project, product, dates, plans, budget or people: the web cannot know them (a generic "usually 2 to 12 weeks" misleads the room).
- Kairos contributes in any of these ways: a fact from memory or from the web findings; a correction when someone says something that contradicts what Kairos knows (numbers, constraints, people's needs); a concrete proposal or option; a question that moves the discussion forward (clarifies a choice, surfaces a risk, asks for a decision); connecting the current point to a constraint the room forgot.
- Take a position when useful ("I'd suggest...", "careful, ..."). Be specific: name the figure, the constraint, the option.
- Never restate what was just said, never praise, never generic advice ("let's keep X in mind", "it's important to...").
- Kairos is an AI assistant, not a participant: it never says it will come, eat or travel, and has no tastes of its own. It never announces a search: searches are started by the system.
- importance: 5 = the room is about to decide something that conflicts with a known constraint or fact; 4 = a correction, a decisive fact, or a concrete proposal that changes the current point; 3 = a useful question or option; 1-2 = minor.
- FIRST, compare the last lines with Kairos's memory: if someone states a figure, a date, a count or a plan that contradicts what Kairos knows (a budget, a number of people, a constraint, a person's needs), the thought is a short correction naming both values ("Attention, the budget is 12,000 €, not 15,000"), importance 5. Same if a factual question asked to the room can be answered from memory or findings.
- Once Kairos has raised a point (see its own lines), do not raise it again in other words unless the room decides against it; move on to what Kairos has not said yet.
- Ground every proposal in the transcript, Kairos's memory or its findings: never invent options (places, prices, names) nobody mentioned.
- The transcript comes from live speech recognition: a word that makes no sense there was probably misheard ("ça aurait déjà raison" for "t'as raison"). Understand the line by its sound and the conversation, and never react to the misheard word itself.
- Never do arithmetic yourself (totals, divisions, margins, units to sell): a separate checker computes numbers exactly.
- Write content, utterance and transition in the language of the meeting: {language}. The utterance is spoken aloud: plain words only, no slashes, asterisks, arrows, bullet points or abbreviations like "vs" (write "ou", "vers", "contre").
- Address the participants the way they address each other or Kairos: with "tu" if they say "tu", with "vous" otherwise (in French).
- stimuli: ids of transcript lines (L..) and memories (M..) that inspired the thought.
- Do not duplicate an existing thought, and do not repeat what Kairos already said."""

ROLES = ("discreet", "active")


LEAD_IN_EXAMPLE = {"French": "Pour revenir au prix,", "English": "Going back to the price,"}


def system_prompt(role: str, language: str) -> str:
    text = SYSTEM if role != "active" else SYSTEM.split("Rules:")[0] + ACTIVE_RULES
    # The example lead-in is in the meeting's language, or the model copies the English one.
    text = text.replace("Going back to the price,", LEAD_IN_EXAMPLE.get(language, LEAD_IN_EXAMPLE["English"]))
    return text.replace("{language}", language)

ANSWER_SYSTEM = """Kairos, an AI assistant attending a live meeting, has just been asked something. Write its answer.
Return one JSON object: {"known": true, "topic": "a few words", "content": "the answer in one line", "utterance": "what Kairos says", "query": null}.
The utterance is one or two short spoken sentences, under 30 words, in {language}, direct, no preamble. Use "tu" if the person asking says "tu". The utterance is spoken aloud: plain words only, no slashes, asterisks, arrows, bullet points or abbreviations like "vs" (write "ou", "vers", "contre").
Use Kairos's memory, its web findings and the transcript. Answer even if Kairos mentioned it before.
The line comes from live speech recognition and may hold misheard words ("44 euros à l'heure auto ?" for "à l'aller-retour ?"): understand it by what makes sense in the conversation, and never answer about a word that makes no sense there.
When the people are testing Kairos itself (they say so, or ask it how it behaves), "the agent" or "the project" they talk about is Kairos: answer as Kairos, never judge it from outside ("le projet est très pertinent").
If someone asks how a search is going, answer truthfully from "Background work": what is running, what it waits for, what it found. Never claim a search runs when none is listed.
Kairos is an AI assistant, not a participant: it never says it will come, eat, travel, or has tastes ("je suis partant", "je n'ai pas de préférence"); it helps ("je peux vous proposer des adresses").
Never say you are searching or will search: when a search is needed, set "known": false with a "query"; the system launches it and says so.
If Kairos's latest line was cut off by the person ("…et dans quel"), do not start it again: answer taking into account what they said after it ("je suis à Paris" answers "dans quelle ville ?").
If Kairos already gave the answer in its latest lines, do not repeat it: confirm in a few words and offer a useful next step (the address, a booking, another option).
If the line is a question between participants about one person's own plans, trip or life ("toi tu pars de Paris aussi ?"), it is not for Kairos even with "tu": Kairos is not part of the plans. Return an empty utterance.
If the line asks the group for its own opinion or judgment ("do you think it will matter?") and does not name Kairos, return an empty utterance: that question is for the people in the room.
If the line asks nothing (a greeting, or someone telling Kairos about their plans), reply in a few words that acknowledge it and invite them to go on ("Bonjour ! Dis-moi ce que tu cherches."), with "known": true.
Only for a factual question whose answer is a public fact that Kairos does not have, set "known": false and give a short web search "query" (no names): Kairos will look it up.
If Kairos does not know and the web cannot know it either (internal matters), say so briefly with "known": true."""

CHECK_SYSTEM = """You check the last line of a live meeting against the facts: what Kairos, an AI assistant attending it, knows for sure (memory, web findings) and the figures clearly stated earlier (transcript, meeting notes).
Return one JSON object: {"correction": null} or {"correction": {"topic": "2-4 words", "kind": "correction|calculation", "expression": "...", "utterance": "..."}}.
Give one only if:
- the LAST LINE states a figure, a date, a count, a plan or a decision that contradicts Kairos's memory, its findings, or a figure clearly stated earlier in the transcript or the notes ("production cost was twenty two?" when 12.50 was said): kind "correction". A person's constraint counts too (someone cannot fly, needs step-free access). A plan phrased as a question still counts ("everyone flies on the 14th?" when two people do not fly);
- or the LAST LINE gives a wrong result for a calculation (a total, a margin, a share: "so that's half the budget" when 6 parts at 1.20 against a 10 euro budget is 72 percent), or the room is trying to work out a number (how many, how much, what margin, what total) that the figures Kairos has (its memory, the notes, the transcript) let it compute, and nobody has given the right result yet: kind "calculation". A question like "so how many units do we need to sell?" is exactly this.
"expression": when a number must be computed, the arithmetic expression with the figures as stated (e.g. "50000000 / (25 - 20.5)"); Kairos's code computes it. Put {result} in the utterance where the computed value goes, and never write that value yourself. Several values: a list of expressions, with {result}, {result2}, {result3} in the utterance. Otherwise "".
Compute the quantity the line is about (what the parts cost, not the margin). When the line is unclear about what a figure refers to ("eighty percent of the... amount"), never guess what the speaker meant: give the figure itself and its share of each plausible base, e.g. "Six parts at 1.20 is {result} euros, that is {result2} percent of the 10 euro budget and {result3} percent of the price."
The utterance is one or two short spoken sentences in {language}, plainly stating the fact or the calculation, e.g. "With a 4.50 euro margin, 50 million euros of profit means about {result} remotes, not 2 million." or "Attention, le budget validé est de 12 000 €, pas 15 000." Use "tu" if the participants say "tu". Refer to people by their names, never by a gendered pronoun. The utterance is spoken aloud: plain words only, no slashes, asterisks, arrows, bullet points or abbreviations like "vs".
A contradiction needs the same quantity: units to sell and chips to buy, a target and an estimate, a cost and a price are different things even with the same unit.
A figure that makes no sense in the conversation may be misheard ("dans 3 euros" after prices of 153 euros): no correction for it.
No correction for an approximation close to the fact ("une vingtaine ?" when Kairos knows 18 people), an opinion, a proposal that contradicts nothing, an information question Kairos cannot compute from its figures, or a result someone in the room already gave correctly. Most lines need none: {"correction": null}."""

#: A figure or a question about one: the checker looks at the line even when Kairos has no memory.
NUMERIC = re.compile(r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|twelve|twenty|thirty|forty|fifty|"
                     r"hundred|thousand|million|how much|how many|cost|price|margin|budget|percent|un|deux|trois|"
                     r"quatre|cinq|six|sept|huit|neuf|dix|douze|vingt|trente|cinquante|cent|mille|combien|coût|prix|"
                     r"marge|pourcent)\b", re.I)

#: What Kairos says at once when it has to look something up before answering.
CHECKING = {"French": ["Je regarde ça tout de suite.", "Attends, je vérifie.", "Je cherche ça.", "Laisse-moi regarder."],
            "English": ["Let me look that up.", "One moment, I'll check.", "Let me find that.", "Give me a second to look."]}


class Thinkers:
    def __init__(self, board: Board, llm: LLM, language: str = "English", relevance=None,
                 role: str = "discreet", request_research=None) -> None:
        self.board = board
        self.llm = llm
        self.language = language
        self.role = role
        #: optional async callback(question, query, topic, segment, urgent): research on demand
        self.request_research = request_research
        self.relevance = relevance  # optional: filters out ideas Kairos already had
        self.ruminations = 0
        #: whether the thinkers may order searches themselves (off when Jev dispatches the work)
        self.research_from_thoughts = True
        #: what the background workers are doing, for "where are you with it?" (set by the runtime)
        self.work_status = lambda: "(none)"
        #: without a topic tracker (no Jev), the thinkers' own reading of the topic is shown as the subject
        self.owns_topic = True
        self._checks = 0
        #: (time, what happened, thought): the reservoir's history, for the report
        self.log: list[tuple[float, str, str]] = []
        self._ids = itertools.count(1)

    def _context(self, snap) -> str:
        memory = "\n".join(f"M{i}: {m}" for i, m in enumerate(snap.long_term)) or "(none)"
        existing = "\n".join(f"{t.id} [{t.status}] ({t.topic}) {t.content}" for t in active_thoughts(snap)) or "(none)"
        return (f"{render_topic(snap)}Kairos's memory:\n{memory}\n\n"
                f"What Kairos looked up on the web during this meeting:\n{render_findings(snap)}\n\n"
                f"Background work of Kairos (searches running or waiting for details):\n{self.work_status()}\n\n"
                f"Meeting notes:\n{snap.notes or '(none yet)'}\n\n"
                f"Transcript (latest last; lines by Kairos are Kairos's own):\n{render_transcript(snap)}\n\n"
                f"Existing thoughts:\n{existing}")

    async def answer(self, segment_id: int) -> None:
        """Fast lane: a line named Kairos. One focused call, published as soon as it returns."""
        snap = self.board.snapshot()
        question = next((s for s in snap.transcript if s.id == segment_id), None)
        if question is None:
            return
        user = self._context(snap) + f"\n\nThe question, line L{segment_id}: {question.speaker}: {question.text}"
        language = language_of(question.text, self.language)  # reply in the language the person just used
        raw = await self.llm.json(_speak_in(language) + ANSWER_SYSTEM.replace("{language}", language), user,
                                  purpose="answer")
        if language_of(str(raw.get("utterance") or ""), None) not in (None, language):
            # The rest of the conversation pulled the model into another language: once more, insisting.
            raw = await self.llm.json(_speak_in(language) + ANSWER_SYSTEM.replace("{language}", language), user
                                      + f"\n\nWrite the utterance in {language}.", purpose="answer")
        utterance = str(raw.get("utterance") or "").strip()
        known = raw.get("known", True) is not False
        ack = False
        if not known and self.request_research is not None and str(raw.get("query") or "").strip():
            # Kairos does not know: say so in a word, and look it up now. The finding answers the question.
            options = CHECKING.get(language, CHECKING["English"])
            self._checks += 1
            # One "je regarde" per search: when a worker already announced it, look it up silently.
            utterance = "" if recent_ack(self.board.snapshot()) else options[(self._checks - 1) % len(options)]
            ack = True
            await self.request_research(question.text, str(raw["query"]), str(raw.get("topic") or "the question"),
                                        segment_id, True)
        if not utterance:
            return
        self.board.update_thoughts({}, (Thought(
            id=f"t{next(self._ids)}", topic=str(raw.get("topic") or "answer"),
            content=str(raw.get("content") or utterance).strip(), utterance=utterance, transition=None,
            importance=5.0, relevance=0.0,
            fit_now=1.0,  # written as a reply to that very line; the judge may still revise it
            already_said=0.0, status=ThoughtStatus.READY, stimuli=(f"L{segment_id}",),
            version=snap.version, created_at=self.board.snapshot().room.t, answers=segment_id, ack=ack, kind="answer"),), by="réponse")
        enforce_cap(self.board)

    async def check(self, segment_id: int) -> None:
        """The checker: one focused call per committed line. Does it contradict what Kairos knows?
        A correction is Kairos's most useful contribution, and the thinkers, busy with everything else,
        do not produce it reliably or fast enough."""
        snap = self.board.snapshot()
        line = next((s for s in snap.transcript if s.id == segment_id), None)
        if line is None or is_ai(line):
            return
        if not (snap.long_term or any(f.status == "done" for f in snap.findings) or NUMERIC.search(line.text)):
            return  # no figure in the line and nothing known to check it against
        # A figure in the line: tell the decider a correction may be on its way (it holds other thoughts ~1 s).
        numeric = bool(NUMERIC.search(line.text))
        if numeric:
            self._set_checking(segment_id)
        try:
            await self._check_line(snap, line, segment_id)
        finally:
            if numeric:
                self._set_checking(None, done=segment_id)

    def _set_checking(self, segment: int | None, done: int | None = None) -> None:
        signals = self.board.snapshot().signals
        if done is not None and signals.checking != done:
            return  # another line is being checked now
        self.board.publish("signals", replace(signals, checking=segment))

    async def _check_line(self, snap, line, segment_id: int) -> None:
        memory = "\n".join(f"- {m}" for m in snap.long_term) or "(none)"
        # The notes carry the figures the room settled earlier (the brief, a budget): context, less sure than memory.
        user = (f"What Kairos knows for sure:\n{memory}\n\nWeb findings:\n{render_findings(snap)}\n\n"
                f"Meeting notes (a summary of what was said earlier: figures the room stated plainly count as said "
                f"earlier; the summary's own interpretations do not):\n{snap.notes or '(none yet)'}\n\n"
                f"Transcript (latest last):\n{render_transcript(snap, last=6)}\n\n"
                f"The last line, L{line.id}: {line.speaker}: {line.text}")
        raw = await self.llm.json(CHECK_SYSTEM.replace("{language}", self.language), user, purpose="checker",
                                  temperature=0.2)
        correction = raw.get("correction")
        utterance = str(correction.get("utterance") or "").strip() if isinstance(correction, dict) else ""
        raw_expr = correction.get("expression") if isinstance(correction, dict) else None
        expressions = [str(e).strip() for e in (raw_expr if isinstance(raw_expr, list) else [raw_expr]) if e and str(e).strip()]
        if utterance and expressions:
            # The arithmetic is done here, never by the model: a wrong number said aloud is worse than silence.
            for i, expression in enumerate(expressions[:3]):
                slot = "{result}" if i == 0 else f"{{result{i + 1}}}"
                value = safe_eval(expression)
                if value is None or slot not in utterance:
                    utterance = ""
                    break
                utterance = utterance.replace(slot, spoken_number(value, self.language))
            if "{result" in utterance:
                utterance = ""  # a placeholder with no expression behind it
        now = self.board.snapshot().room.t
        if not utterance:
            return
        if self.relevance is not None:
            vector = (await self.relevance.novel([utterance], decisive=[True]))[0]
            if vector is None:
                self.log.append((now, "refused: too close to an existing or said thought", utterance))
                return
        thought_id = f"c{next(self._ids)}"
        if self.relevance is not None:
            self.relevance.remember(thought_id, vector)
        self.board.update_thoughts({}, (Thought(
            id=thought_id, topic=str(correction.get("topic") or "correction"), content=utterance,
            utterance=utterance, transition=None, importance=5.0, relevance=0.0, fit_now=0.0, already_said=0.0,
            status=ThoughtStatus.READY, stimuli=(f"L{segment_id}",), version=snap.version, created_at=now, kind="correction"),), by="vérificateur")
        self.log.append((now, "created by the checker", utterance))
        enforce_cap(self.board)

    async def run_once(self) -> None:
        snap = self.board.snapshot()
        try:
            last = [s for s in snap.transcript if not is_ai(s)]
            language = language_of(last[-1].text, self.language) if last else self.language
            result = await self.llm.json(system_prompt(self.role, language), self._context(snap),
                                         purpose="thinker")
        except Exception:
            result = {}  # a failed pass costs nothing but this pass

        now = self.board.snapshot().room.t
        changes: dict[str, dict] = {}
        current = {t.id: t for t in active_thoughts(self.board.snapshot())}
        # An answer to a question still open is not the thinkers' to retire: the room is waiting for it.
        signals = self.board.snapshot().signals
        waiting = signals.addressed_segment if not signals.answered else None
        current = {tid: t for tid, t in current.items() if t.answers is None or t.answers != waiting}
        # Web findings and decisive thoughts (corrections) are not the thinkers' to replace: only "already said"
        # or an explicit rejection retires them. The thinkers may still set them aside when the topic moves on.
        protected = {tid for tid, t in current.items() if tid.startswith("r") or t.importance >= 5}
        for thought_id in _ids(result.get("moved_on")):
            if thought_id in current and current[thought_id].status == ThoughtStatus.READY:
                changes[thought_id] = {"status": ThoughtStatus.PENDING}
        for thought_id in _ids(result.get("obsolete")):
            if thought_id in current and thought_id not in protected:
                changes[thought_id] = {"status": ThoughtStatus.STALE, "note": "remplacée par une meilleure idée"}
        research = result.get("research")
        if (isinstance(research, dict) and str(research.get("query") or "").strip() and self.request_research
                and self.research_from_thoughts):
            await self.request_research(str(research.get("question") or research["query"]), str(research["query"]),
                                        str(research.get("topic") or "this"), None, False)

        raw_thoughts = [r for r in (result.get("new_thoughts") or [])[:2]
                        if isinstance(r, dict) and str(r.get("utterance", "")).strip()]
        vectors: list = [None] * len(raw_thoughts)
        if self.relevance is not None and raw_thoughts:
            vectors = await self.relevance.novel([str(r["utterance"]) for r in raw_thoughts])
            self.ruminations += sum(v is None for v in vectors)
            self.log.extend((self.board.snapshot().room.t, "refused: too close to an existing or said thought", str(r["utterance"]))
                            for r, v in zip(raw_thoughts, vectors) if v is None)
            raw_thoughts = [r for r, v in zip(raw_thoughts, vectors) if v is not None]
            vectors = [v for v in vectors if v is not None]
        added = []
        for raw, vector in zip(raw_thoughts, vectors):
            thought_id = f"t{next(self._ids)}"
            if vector is not None:
                self.relevance.remember(thought_id, vector)
            added.append(Thought(
                id=thought_id,
                topic=str(raw.get("topic") or result.get("current_topic") or ""),
                content=str(raw.get("content") or raw["utterance"]).strip(),
                utterance=str(raw["utterance"]).strip(),
                transition=(str(raw.get("transition")).strip() or None) if raw.get("transition") else None,
                # 5 is for what cannot wait: answers and the checker's corrections. The thinkers propose;
                # a "5" from them would skip the talk budget and come back every time they rate it so.
                importance=_clamp(raw.get("importance"), 1.0, 4.0, 3.0),
                relevance=0.0, fit_now=0.0, already_said=0.0,  # unknown until measured: fails towards silence
                status=ThoughtStatus.READY,
                stimuli=tuple(str(s) for s in raw.get("stimuli") or ()),
                version=snap.version,
                created_at=now,
            ))

        if changes or added:
            self.board.update_thoughts(changes, tuple(added), by="penseurs")
        enforce_cap(self.board)  # what matters least right now leaves (pending points first)
        self.log.extend([(now, "created", t.utterance) for t in added] +
                        [(now, getattr(c.get("status"), "value", "updated") + (f" ({c['note']})" if c.get("note") else ""),
                          current[tid].utterance if tid in current else tid)
                         for tid, c in changes.items()])
        topic = str(result.get("current_topic") or "")
        if topic and self.owns_topic:
            self.board.publish("signals", replace(self.board.snapshot().signals, topic=topic))


def render_topic(snap) -> str:
    """The subject the room is on now, as tracked from the last lines, so that nobody reads new talk
    through an old subject ("before using this case to test the agent" about real train tickets)."""
    signals = snap.signals
    if not signals.topic:
        return ""
    previous = (f' The room was on another subject before ("{signals.previous_topic}"): read the latest lines '
                f"as they are, never as part of that earlier subject.") if signals.previous_topic else ""
    return f"CURRENT TOPIC (tracked from the latest lines): {signals.topic}.{previous}\n\n"


def _ids(value) -> list[str]:
    return [str(v) for v in value] if isinstance(value, list) else []


def _clamp(value, low: float, high: float, default: float) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return default


def _speak_in(language: str) -> str:
    """Put the language first: with a conversation in French, a model otherwise keeps answering in French."""
    return (f"LANGUAGE: the person's latest line is in {language}. Everything Kairos says (utterance, question) "
            f"must be in {language}, whatever the language of the rest of the conversation.\n\n")


def safe_eval(expression: str) -> float | None:
    """Evaluate plain arithmetic (numbers, + - * / ** and parentheses) without running any code."""
    import ast
    import operator

    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}

    def walk(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](walk(node.operand))
        raise ValueError("not plain arithmetic")

    try:
        value = walk(ast.parse(expression.replace(",", ".").replace("_", ""), mode="eval").body)
    except Exception:
        return None
    return value if abs(value) < 1e15 else None


def spoken_number(value: float, language: str = "English") -> str:
    """A computed value as a person would say it: "11.1 million", "4 millions", "12.50"."""
    french = language == "French"
    def fmt(x: float, digits: int) -> str:
        text = f"{x:.{digits}f}".rstrip("0").rstrip(".")
        return text.replace(".", ",") if french else text
    magnitude = abs(value)
    if magnitude >= 1e9:
        return fmt(value / 1e9, 2 if magnitude < 1e10 else 1) + (" milliards" if french else " billion")
    if magnitude >= 1e6:
        # 1.25 million stays 1.25: rounding it to 1.2 would be a wrong figure said aloud.
        return fmt(value / 1e6, 2 if magnitude < 1e7 else 1) + (" millions" if french else " million")
    if magnitude >= 1e4:
        return f"{round(value):,}".replace(",", " " if french else ",")
    return fmt(value, 2)

