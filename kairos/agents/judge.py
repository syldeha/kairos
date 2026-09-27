"""The judge agent: precomputes, in the background, what the decider needs to know.

It follows speech as it arrives. Each run looks at the latest human line, even
if it is still being spoken, and asks the judge:
- for each of the best thoughts: would saying it now be coherent and useful (fit_now)?
- has its specific content already been given (already_said)?
- once a line naming Kairos is committed: does it really address Kairos?

The answers are stored on the board, with how much of the line was seen. The
decider reads them at the next gap without calling any model, so the judgment
must be ready before the gap: that is why it runs on partial lines.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import replace

from ..addressing import NAME  # noqa: F401  (re-exported for the runtime)
from ..board import Board
from ..contracts import ThoughtStatus
from ..decide.judges import Judge
from .common import active_thoughts, last_human_segment, render_findings, render_transcript, spoken_text

MAX_JUDGED = 6
RATER_ROLE = {
    "discreet": ("Kairos is a discreet assistant: it speaks to answer what it was asked, to offer its help once when "
                 "someone expresses a concrete need (a trip, a restaurant, a ticket), to ask for the details a search "
                 "needs, to give a search result, to correct something wrong, or to bring a fact the room needs right "
                 "now. Otherwise it stays silent."),
    "active": ("Kairos is an active participant: it answers, corrects, brings facts, and also proposes options or asks "
               "questions that move the decision forward. It stays silent when it would only repeat or distract."),
}
OUTDATED = 0.85  # above this, the idea was rejected, contradicted or answered: it leaves the reservoir
SAME_TOPIC = 0.35  # a line can only outdate an idea it is about (the idea's relevance to the discussion)


class JudgeAgent:
    def __init__(self, board: Board, judge: Judge, role: str = "discreet") -> None:
        self.board = board
        self.judge = judge
        self.role = role
        #: lines the runtime flagged as possibly addressed to Kairos without naming it ("tu", "vous")
        self.address_candidates: set[int] = set()
        #: called with the line id when the judge confirms such a line is addressed to Kairos
        self.on_addressed = None
        self._addressed_checked: int | None = None
        self._last_key: tuple | None = None
        #: (time, line id, words heard, {thought id: [fit, said, chosen]}): why a thought was or was not usable
        self.history: list[tuple[float, int | None, int, dict[str, list]]] = []

    async def run_once(self) -> None:
        snap = self.board.snapshot()
        line = last_human_segment(snap)
        words = len(line.text.split()) if line else 0
        # Thoughts not yet judged against this line first, then the most important: every thought is
        # re-judged as the discussion moves, so no old "coherent" score lingers.
        current = line.id if line else None
        thoughts = sorted(active_thoughts(snap),
                          key=lambda t: (t.judged_line == current, -t.importance, -t.created_at))[:MAX_JUDGED]
        key = (line.id if line else None, words, line.final if line else None, tuple(t.id for t in thoughts))
        if key == self._last_key:
            return  # nothing new to judge
        statements: dict[str, str] = {}
        check_address = (line is not None and line.final and line.id != self._addressed_checked
                         and (bool(NAME.search(line.text)) or line.id in self.address_candidates))
        if check_address:
            statements["addressed"] = (
                f'The line "{line.text}" (said by {line.speaker or "someone"}) speaks directly to Kairos, the AI '
                f"assistant taking part in the meeting, and asks it a question or makes a request to it "
                f'("tu" / "vous" / "you" may be aimed at Kairos or at another participant: judge from context).')
        lines_by_id = {s.id: s for s in snap.transcript}
        for t in thoughts:
            if t.answers is not None:
                statements[f"fit:{t.id}"] = (
                    f'"{t.utterance}" is a direct reply to what was asked of Kairos in line L{t.answers} '
                    f"(judge only whether it replies to it, not whether it is correct).")
            elif t.kind == "correction":
                # A correction targets an earlier line ("two million remotes"), not whatever was said last.
                source = next((lines_by_id[int(s[1:])] for s in t.stimuli
                               if s.startswith("L") and s[1:].isdigit() and int(s[1:]) in lines_by_id), None)
                about = f' about line L{source.id} ("{source.text[:160]}")' if source is not None else ""
                statements[f"fit:{t.id}"] = (
                    f'Kairos would now say "{t.utterance}"{about}. Given the transcript, this correction or '
                    f"calculation is right, the room has not settled the point correctly since, and it is still "
                    f"useful to say it now, even if the last line is about something else.")
            elif t.status == ThoughtStatus.PENDING:
                # A point the discussion moved away from: it cannot "respond to the last line" by definition.
                statements[f"fit:{t.id}"] = (
                    f'Coming back now to this earlier point, as Kairos would say it: "{spoken_text(t)}", would be '
                    f"useful to the room: it still matters for what they are deciding (a correction of a wrong "
                    f"figure, a forgotten constraint, a fact they need). A minor or already settled point does not "
                    f"count.")
            elif self.role == "active":
                statements[f"fit:{t.id}"] = (
                    f'If Kairos said right after the last line "{spoken_text(t)}", it would fit the current '
                    f"discussion and move it forward: a relevant fact, a correction, a concrete proposal, or a "
                    f"question that helps the room decide. Vague or generic remarks, or points unrelated to what "
                    f"is being discussed, do not count.")
            else:
                statements[f"fit:{t.id}"] = (
                    f'If Kairos said right after the last line "{spoken_text(t)}", it would directly respond to '
                    f"that line and add specific new information or a useful question. Generic remarks, advice "
                    f"anyone could give, or points unrelated to the last line do not count.")
            statements[f"said:{t.id}"] = (
                f'In the transcript lines, someone (or Kairos) already said out loud the specific content of this '
                f'idea: "{t.content}". Only the spoken transcript counts: what Kairos merely knows from its memory '
                f"or found on the web has not been said. Merely mentioning the same topic does not count, and for a "
                f"calculation or a conclusion, saying the figures it is computed from does not count: only the "
                f"result or conclusion itself. A recap or reminder that only lists figures or points someone "
                f"already said (in any words, e.g. 'twelve fifty' for 12.50) counts as said.")
            words_seen = len(line.text.split()) if line is not None else 0
            # A correction or a decisive fact (importance 5) is never "outdated" by someone repeating the
            # mistake: it leaves only once said, or once someone fixes the figure ("already said").
            if t.answers is None and t.importance < 5 and line is not None and t.relevance >= SAME_TOPIC \
                    and (words_seen >= 6 or (line.final and words_seen >= 3)):
                # A real sentence can make an idea outdated, even before it is committed; a fragment
                # like "Côté" never does.
                statements[f"outdated:{t.id}"] = (
                    f'A participant has explicitly rejected, contradicted or already answered this idea, or '
                    f'decided the exact point otherwise: "{t.content}". The conversation simply moving on to '
                    f"another topic does NOT make it outdated; a correction of a wrong figure stays valid until "
                    f"someone fixes that figure; a person's constraint (someone cannot fly, needs step-free access) "
                    f"or a validated figure stays valid when the room decides against it: that is when it must "
                    f"be said.")
        if not statements:
            self._last_key = key
            return

        memory = "\n".join(f"- {m}" for m in snap.long_term) or "(none)"
        state = ("Kairos is an AI assistant attending this meeting. A line marked 'still speaking' may be "
                 "incomplete.\n\n"
                 + (f"CURRENT TOPIC: {snap.signals.topic}\n\n" if snap.signals.topic else "") +
                 f"TRANSCRIPT, the only thing said out loud in the meeting (latest last):\n"
                 f"{render_transcript(snap, last=10)}\n\n"
                 f"Meeting notes:\n{snap.notes or '(none yet)'}\n\n"
                 f"NOT SAID IN THE MEETING, only known to Kairos:\n"
                 f"Kairos's memory:\n{memory}\n"
                 f"Kairos's web findings:\n{render_findings(snap)}")
        p, rating = await asyncio.gather(self.judge.probabilities(state, statements), self._rate(state, snap))
        self._last_key = key

        changes: dict[str, dict] = {}
        if rating is not None:
            chosen, none_p, value, best = rating
            for tid, prob in chosen.items():
                changes[tid] = {"chosen": round(prob, 3)}
            if best is not None:
                changes[best]["value"] = round(value, 2)
        for t in thoughts:
            fit, said = p.get(f"fit:{t.id}", 0.0), p.get(f"said:{t.id}", 0.0)
            change = {"fit_now": round(fit, 3), "already_said": round(said, 3),
                      "judged_line": line.id if line else None}
            outdated = p.get(f"outdated:{t.id}", 0.0)
            if said >= 0.8 and t.answers is None:  # a question deserves an answer even if Kairos said it before
                change["status"] = ThoughtStatus.STALE
                change["note"] = "le juge : déjà dit dans la réunion"
            elif outdated >= OUTDATED and line is not None:
                quoted = f"« {line.text[:60]}{'…' if len(line.text) > 60 else ''} »"
                if t.status == ThoughtStatus.READY:
                    # First alert: set aside, not dropped. A second alert on a later line drops it.
                    change["status"] = ThoughtStatus.PENDING
                    change["note"] = f"mise en attente : semble dépassée après {quoted}"
                elif t.note.startswith("mise en attente") and not t.note.endswith(quoted):
                    change["status"] = ThoughtStatus.STALE
                    change["note"] = f"dépassée après {quoted}"
            changes[t.id] = {**changes.get(t.id, {}), **change}
        if changes:
            self.board.update_thoughts(changes, by="juge Jev")
        self.history.append((snap.room.t, current, words, {
            tid: [c.get("fit_now"), c.get("already_said"), c.get("chosen")] for tid, c in changes.items()}))
        del self.history[:-80]

        signals = self.board.snapshot().signals
        if line is not None:
            signals = replace(signals, judged_segment=line.id, judged_words=words)
        if check_address:
            self._addressed_checked = line.id
            addressed = p.get("addressed", 0.0)
            if addressed < 0.6 and signals.addressed_segment == line.id:
                # The name was mentioned but nobody asked Kairos anything: cancel the fast lane.
                signals = replace(signals, answered=True)
            signals = replace(signals, addressed=addressed)
            confirmed = addressed >= 0.6 and signals.addressed_segment != line.id
        else:
            confirmed = False
        if line is not None and line.final:
            signals = replace(signals, last_final=line.id)
        if rating is not None and line is not None:
            signals = replace(signals, rater_line=line.id, rater_none=round(rating[1], 3))
        self.board.publish("signals", signals)
        if confirmed and self.on_addressed is not None:
            self.on_addressed(line.id)  # a "tu"/"vous" question was for Kairos: open the fast lane

    async def _rate(self, state: str, snap) -> tuple[dict[str, float], float, float, str | None] | None:
        """Jev as rater: which thought to say at the next pause, or none, and how much it would help.
        Only with a judge that answers typed questions (Jev); otherwise the policy's own score decides."""
        ask = getattr(self.judge, "ask", None)
        rated = active_thoughts(snap)
        if ask is None or not rated:
            return None
        options = {t.id: f"{_role_of(t)}{spoken_text(t)[:220]}" for t in rated}
        options["none"] = "stay silent: nothing here is worth saying right now"
        try:
            answers = await ask(state, {
                "pick": {"type": "choice", "criteria": options,
                         "instructions": f"{RATER_ROLE.get(self.role, RATER_ROLE['discreet'])} Kairos adds, it never "
                                         "echoes: a sentence that restates figures the room just said, or re-asks a "
                                         "question a participant already asked, is worth nothing. At the next pause, "
                                         "right after the last line, which sentence should Kairos say, if any?"},
                "value": {"type": "score", "criteria": ["no value", "useful", "decision-changing"],
                          "instructions": "How much would saying the most fitting of these sentences right now "
                                          "help the room?"},
            }, purpose="rater")
        except Exception:
            return None
        probs = (answers.get("pick") or {}).get("probabilities") or {}
        chosen = {t.id: float(probs.get(t.id, 0.0)) for t in rated}
        best = max(chosen, key=chosen.get) if chosen else None
        value = float((answers.get("value") or {}).get("score", 0.0))
        return chosen, float(probs.get("none", 0.0)), value, best


def _role_of(t) -> str:
    """What the thought is, for the rater: an offer tied to a need is not a random remark."""
    if t.answers is not None:
        return "[reply owed to a line addressed to Kairos] "
    return {"question": "[offer of help for a need just expressed] ", "finding": "[result of a search] ",
            "correction": "[correction or calculation from the figures stated] "}.get(t.kind, "")

