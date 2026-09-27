"""The decision rule. Pure code, no model: it reads precomputed fields on the board.

Called on every voice-activity step (80 ms). It answers "does Kairos speak
now, and what does it say?" in well under a millisecond, because everything
slow (thinking, judging, relevance) was done in the background.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..agents.common import AI_NAME, language_of
from ..board import BoardSnapshot
from ..contracts import HORIZONS, Decision, Thought, ThoughtStatus


@dataclass(frozen=True, slots=True)
class PolicyParams:
    # When is the floor open?
    min_gap_s: float = 0.25          # never start before this much silence
    p_end: float = 0.60              # ...and the detector expects silence to last 1 s
    long_gap_s: float = 0.9          # or the silence is already this long
    # Which thought?
    fit_min: float = 0.70            # judge: coherent to say now
    said_max: float = 0.30           # judge: not already said
    open_threshold: float = 0.55     # score needed to speak unprompted
    asked_threshold: float = 0.30    # score needed to answer a direct question
    answer_fit_min: float = 0.30     # judge: the answer does answer the question
    asked_window_s: float = 20.0     # a question left unanswered this long has passed
    unjudged_words: int = 2          # the judge may lag this many words behind the last line
    share_limit: float = 0.20        # above this share of talk time...
    share_penalty: float = 0.15      # ...the threshold goes up by this much
    chain_threshold: float = 0.65    # score needed to chain a pending thought
    freshness_s: float = 120.0       # a thought loses 63 % of its weight in this time
    relevance_low: float = 0.15      # cosine similarity mapped to 0..1 between these bounds
    relevance_high: float = 0.55
    decisive_wait_s: float = 3.0  # how long a fresh correction may hold the floor while the judge sees it
    choice_min: float = 0.5       # Jev rater: the picked thought must be at least this likely (and beat "none")
    unfinished_gap_s: float = 1.0  # a line still being spoken: wait this long before taking the floor
    asked_unfinished_gap_s: float = 0.5  # ...but a question to Kairos ending with "?" is over sooner
    answer_patience_s: float = 5.0   # a question with no answer yet after this long...
    answer_fallback_min: float = 0.8  # ...may be replied to by a thought Jev picks this clearly
    check_wait_s: float = 1.5        # the checker is reading the last line: hold other thoughts this long
    correction_fit_min: float = 0.50  # judge: a correction's statement is compound (right, unsettled, useful now)


def floor_open(snap: BoardSnapshot, p: PolicyParams) -> bool:
    room = snap.room
    if room.speaking or snap.ai.speaking:
        return False
    silence = room.silence_s
    if silence < p.min_gap_s:
        return False
    return room.p_silence[HORIZONS.index(1.0)] >= p.p_end or silence >= p.long_gap_s


def score(t: Thought, now: float, p: PolicyParams) -> float:
    freshness = math.exp(-max(0.0, now - t.created_at) / p.freshness_s)
    if t.status == ThoughtStatus.PENDING:
        # Coming back to a past topic: relevance to the current line does not apply.
        return t.importance / 5 * freshness
    span = p.relevance_high - p.relevance_low
    relevance = min(1.0, max(0.0, (t.relevance - p.relevance_low) / span))
    return t.importance / 5 * (0.5 + 0.5 * relevance) * freshness


def fit_bar(t: Thought, p: PolicyParams) -> float:
    """How coherent a thought must be judged to be said now. A correction is judged on a compound statement
    (it is right, the room has not settled it, it is still useful), so its scores run lower; Jev's choice
    still has to pick it over "none"."""
    return p.correction_fit_min if t.kind == "correction" else p.fit_min


def ai_share(snap: BoardSnapshot) -> float:
    return snap.ai.total_speech_s / snap.room.t if snap.room.t > 0 else 0.0


def decide(snap: BoardSnapshot, p: PolicyParams = PolicyParams()) -> Decision:
    now = snap.room.t
    if not floor_open(snap, p):
        return Decision(now, False, "the floor is not open")

    room, ai, signals = snap.room, snap.ai, snap.signals
    question_line = next((seg for seg in snap.transcript if seg.id == signals.addressed_segment), None)
    open_question = (not signals.answered and question_line is not None
                     and now - question_line.t_end <= p.asked_window_s)
    follow_up = open_question and any(
        t.answers == signals.addressed_segment and not t.ack and t.fit_now >= p.answer_fit_min
        and t.status in (ThoughtStatus.READY, ThoughtStatus.PENDING) for t in snap.thoughts)
    if ai.last_spoke_at is not None and room.silence_since is not None and ai.last_spoke_at >= room.silence_since \
            and not follow_up:  # after "let me look that up", the answer may follow at once
        return Decision(now, False, "Kairos spoke last: waiting for someone else")
    humans = [s for s in snap.transcript if s.speaker != AI_NAME]
    if humans:
        last = max(humans, key=lambda s: s.t_start)
        question = not signals.answered and signals.addressed_segment == last.id
        # A question put to Kairos that ends with "?" is over: the reply need not wait as long.
        gap = p.asked_unfinished_gap_s if question and last.text.rstrip().endswith("?") else p.unfinished_gap_s
        if not last.final and room.silence_s < gap:
            # "…cheaper than that? I mean…": a pause inside a long turn is not the end of it.
            return Decision(now, False, "the speaker may go on: the line is not finished")
        seen = signals.judged_segment == last.id and signals.judged_words >= len(last.text.split()) - p.unjudged_words
        if not seen and not question:
            return Decision(now, False, "the judge has not seen the end of the last line yet")

    active = [t for t in snap.thoughts if t.status in (ThoughtStatus.READY, ThoughtStatus.PENDING)]
    # Only judgments made against the current last line count: an old "coherent" score is not evidence.
    current_line = max(humans, key=lambda s: s.t_start).id if humans else None
    usable = [t for t in active if t.fit_now >= fit_bar(t, p) and t.already_said <= p.said_max
              and (current_line is None or t.judged_line == current_line)]
    # A thought written before the conversation switched language is not said in the old language.
    spoken_now = language_of(max(humans, key=lambda s: s.t_start).text, None) if humans else None
    if spoken_now is not None:
        usable = [t for t in usable if language_of(t.utterance, None) in (None, spoken_now)]

    # 1. Someone asked Kairos something: answer, with a low bar, even if Kairos said it before.
    if open_question:
        answers = [t for t in active if t.answers == signals.addressed_segment and t.fit_now >= p.answer_fit_min]
        if answers:
            best = max(answers, key=lambda t: score(t, now, p))
            s = score(best, now, p)
            if s >= p.asked_threshold:
                return _with_chain(Decision(now, True, f"answering line L{signals.addressed_segment}",
                                            "asked", best.id, None, best.fit_now, s), best, usable, now, p)
        waited = now - question_line.t_end
        rated_best = max((t for t in usable if t.chosen is not None and signals.rater_line == current_line),
                         key=lambda t: t.chosen, default=None)
        if waited >= p.answer_patience_s and rated_best is not None and rated_best.chosen >= p.answer_fallback_min:
            # The answer lane has nothing yet, but Jev strongly picks a thought that replies: better than silence.
            return Decision(now, True, f"no answer after {waited:.0f} s: Jev picks '{rated_best.topic}' "
                                       f"({rated_best.chosen:.2f})", "asked", rated_best.id, None,
                            rated_best.fit_now, rated_best.chosen)
        return Decision(now, False, f"asked at L{signals.addressed_segment}, answer not ready yet")

    # The checker is still reading the last line (a figure in it): a correction may be one second away. Saying
    # something else now would bury it ("320 euros, not 240" arrived 1.2 s after Kairos spoke about lodging).
    if signals.checking is not None and signals.checking == current_line and room.silence_s < p.check_wait_s:
        return Decision(now, False, "the checker is reading the last line: waiting for it")

    # A correction just written and not yet judged on this line: wait a moment for the judge (under a second)
    # rather than say something less important in its place. The decider runs again every 80 ms.
    fresh = [t for t in active if t.importance >= 5 and t.judged_line != current_line
             and now - t.created_at <= p.decisive_wait_s]
    if fresh and not any(t.importance >= 5 for t in usable):
        return Decision(now, False, "a correction is being judged: waiting for it")

    # Jev rated the reservoir on this very line: its choice replaces the hand-made score. It picks one
    # thought or "none"; the code only checks the pick is clear enough and that the floor is free.
    rated = [t for t in usable if t.chosen is not None]
    if signals.rater_line is not None and signals.rater_line == current_line and rated:
        best = max(rated, key=lambda t: t.chosen)
        if best.chosen < p.choice_min or best.chosen <= signals.rater_none:
            return Decision(now, False, f"Jev: nothing to say now (best {best.chosen:.2f}, none {signals.rater_none:.2f})")
        reason = "pending" if best.status == ThoughtStatus.PENDING else "important"
        decision = Decision(now, True, f"Jev picks '{best.topic}' ({best.chosen:.2f} vs none {signals.rater_none:.2f})",
                            reason, best.id, None, best.fit_now, best.chosen)
        return _with_chain(decision, best, usable, now, p) if reason == "important" else decision

    threshold = p.open_threshold + (p.share_penalty if ai_share(snap) > p.share_limit else 0.0)
    # A correction or a decisive fact (importance 5) is not held back because Kairos already spoke a lot.
    decisive = [t for t in usable if t.importance >= 5]
    if decisive:
        threshold = min(threshold, p.open_threshold)

    # 2. A ready thought on the current topic.
    ready = [t for t in usable if t.status == ThoughtStatus.READY]
    if ready:
        best = max(ready, key=lambda t: score(t, now, p))
        s = score(best, now, p)
        if s >= threshold:
            return _with_chain(Decision(now, True, f"important thought (score {s:.2f} ≥ {threshold:.2f})",
                                        "important", best.id, None, best.fit_now, s), best, usable, now, p)

    # 3. A pending thought, said alone with its lead-in.
    pending = [t for t in usable if t.status == ThoughtStatus.PENDING]
    if pending:
        best = max(pending, key=lambda t: score(t, now, p))
        s = score(best, now, p)
        if s >= max(threshold, p.chain_threshold):
            return Decision(now, True, f"coming back to '{best.topic}' (score {s:.2f})",
                            "pending", best.id, None, best.fit_now, s)

    judged = [t for t in snap.thoughts if t.status in (ThoughtStatus.READY, ThoughtStatus.PENDING)]
    if not judged:
        return Decision(now, False, "floor open, no thought in reservoir")
    if not usable:
        return Decision(now, False, "floor open, no thought judged coherent now")
    return Decision(now, False, f"floor open, best score below {threshold:.2f}")


def _with_chain(decision: Decision, primary: Thought, usable: list[Thought], now: float, p: PolicyParams) -> Decision:
    chain = [t for t in usable if t.status == ThoughtStatus.PENDING and t.id != primary.id
             and t.topic != primary.topic and t.already_said <= p.said_max / 1.5
             and score(t, now, p) >= p.chain_threshold]
    if not chain:
        return decision
    best = max(chain, key=lambda t: score(t, now, p))
    return Decision(decision.t, True, decision.why + f", then back to '{best.topic}'", decision.reason,
                    decision.primary, best.id, decision.confidence, decision.score)
