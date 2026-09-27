import asyncio
from pathlib import Path

import pytest

from kairos.agents.scribe import Scribe
from kairos.board import Board
from kairos.clock import RealClock, SimClock
from kairos.contracts import SpeechFinal, SpeechPartial, VadStep
from kairos.sources.replay import ReplayParams, ReplaySource, build_timeline, find_gaps, parse_script

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "reunion_produit.txt"

SHORT = """
+0.5  A: Bonjour, ça va ?
+0.6  B: Oui très bien.
+-0.2 A: Tant mieux.
"""


def replay_transcript(script: str, clock_factory, **params) -> tuple[Board, list]:
    timeline = build_timeline(parse_script(script), ReplayParams(**params))
    board = Board()
    scribe = Scribe(board)
    delivered = []

    async def run():
        source = ReplaySource(timeline.events, clock_factory())
        async for event in source.events():
            scribe.on_event(event)
            delivered.append(event)

    asyncio.run(run())
    return board, delivered


# -- script and timeline -------------------------------------------------------------

def test_parse_script_reads_signed_gaps_and_skips_comments():
    lines = parse_script("# comment\n+0.5  A: Salut.\n\n+-0.3 B: Oui.\n+1,2 C: D'accord : oui.")
    assert [(l.gap, l.speaker, l.text) for l in lines] == [
        (0.5, "A", "Salut."), (-0.3, "B", "Oui."), (1.2, "C", "D'accord : oui.")]


def test_parse_script_rejects_malformed_lines():
    with pytest.raises(ValueError, match="line 1"):
        parse_script("A: no gap")


def test_partials_grow_word_by_word_then_final_matches_script():
    timeline = build_timeline(parse_script("+0.5 A: un deux trois"))
    partials = [e.text for e in timeline.events if isinstance(e, SpeechPartial)]
    finals = [e for e in timeline.events if isinstance(e, SpeechFinal)]
    assert partials == ["un", "un deux", "un deux trois"]
    assert len(finals) == 1 and finals[0].text == "un deux trois"
    last_partial = max(e.t for e in timeline.events if isinstance(e, SpeechPartial))
    assert finals[0].t > last_partial


def test_events_are_time_ordered_and_vad_reports_speech():
    timeline = build_timeline(parse_script(SHORT))
    times = [e.t for e in timeline.events]
    assert times == sorted(times)
    for step in (e for e in timeline.events if isinstance(e, VadStep)):
        truth = any(a <= step.t < b for a, b, _ in timeline.speech)
        assert step.speaking == truth
        assert all(0.0 < p < 1.0 for p in step.p_silence)


def test_negative_gap_makes_speakers_overlap():
    timeline = build_timeline(parse_script(SHORT))
    b = next(s for s in timeline.speech if s[2] == "B")
    a_again = [s for s in timeline.speech if s[2] == "A"][-1]
    assert a_again[0] < b[1]  # A starts again before B has finished


def test_comma_pause_is_a_hold_not_a_turn_change():
    timeline = build_timeline(parse_script("+0.5 A: bon, alors\n+1.0 B: oui"))
    gaps = [g for g in timeline.gaps(min_s=0.2) if g.after is not None]
    assert [(g.before, g.after, g.shift) for g in gaps] == [("A", "A", False), ("A", "B", True)]


def test_silence_probability_is_higher_before_a_turn_end_than_mid_speech():
    timeline = build_timeline(parse_script("+0.5 A: " + "mot " * 20 + "\n+2.0 B: oui"))
    a_end = max(e for _, e, s in timeline.speech if s == "A")
    steps = [e for e in timeline.events if isinstance(e, VadStep)]
    near_end = [s.p(1.0) for s in steps if a_end - 0.9 < s.t < a_end]
    mid = [s.p(1.0) for s in steps if 1.0 < s.t < a_end - 2.0]
    assert sum(near_end) / len(near_end) > sum(mid) / len(mid) + 0.4


def test_anonymous_replay_hides_speakers_but_keeps_ground_truth():
    timeline = build_timeline(parse_script(SHORT), ReplayParams(anonymous=True))
    assert all(e.speaker is None for e in timeline.events if isinstance(e, (SpeechPartial, SpeechFinal)))
    assert {s for _, _, s in timeline.speech} == {"A", "B"}


def test_find_gaps_ignores_overlaps():
    gaps = find_gaps([(0.0, 2.0, "A"), (1.5, 3.0, "B"), (3.5, 4.0, "A")], min_s=0.2)
    assert [(g.start, g.end, g.before, g.after) for g in gaps[:-1]] == [(3.0, 3.5, "B", "A")]


# -- board ----------------------------------------------------------------------------

def test_board_versions_each_zone_and_refuses_stale_writers():
    board = Board()
    base = board.snapshot().zone_versions["thoughts"]
    board.publish("transcript", ())  # another zone changing does not invalidate the thinker
    assert board.publish("thoughts", ("fast",), base_version=base) is not None
    assert board.publish("thoughts", ("slow",), base_version=base) is None
    assert board.snapshot().thoughts == ("fast",)


def test_board_snapshots_are_immutable_views():
    board = Board()
    before = board.snapshot()
    board.publish("notes", "hello")
    assert before.notes == "" and board.snapshot().notes == "hello"
    with pytest.raises(TypeError):
        board.snapshot().zone_versions["notes"] = 0


# -- scribe and replay -----------------------------------------------------------------

def test_scribe_transcript_matches_the_fixture():
    script = FIXTURE.read_text(encoding="utf-8")
    board, _ = replay_transcript(script, SimClock)
    transcript = board.snapshot().transcript
    assert all(s.final for s in transcript)
    assert [s.text for s in sorted(transcript, key=lambda s: s.id)] == [l.text for l in parse_script(script)]


def test_scribe_shows_provisional_text_before_the_final():
    timeline = build_timeline(parse_script("+0.5 A: un deux trois"))
    board, scribe = Board(), None
    scribe = Scribe(board)
    for event in timeline.events:
        scribe.on_event(event)
        if isinstance(event, SpeechPartial) and event.text == "un deux":
            segment = board.snapshot().transcript[0]
            assert segment.text == "un deux" and not segment.final
            break
    else:
        pytest.fail("partial not delivered")


def test_scribe_tracks_silence_duration():
    timeline = build_timeline(parse_script("+0.5 A: bonjour\n+2.0 B: oui"))
    board = Board()
    scribe = Scribe(board)
    a_end = max(e for _, e, s in timeline.speech if s == "A")
    for event in timeline.events:
        scribe.on_event(event)
        if isinstance(event, VadStep) and event.t >= a_end + 1.0:
            break
    room = board.snapshot().room
    assert not room.speaking
    assert room.last_speaker == "A"
    assert 0.9 <= room.silence_s <= 1.1


def test_real_time_and_instant_replays_give_the_same_transcript():
    instant, _ = replay_transcript(SHORT, SimClock)
    fast, _ = replay_transcript(SHORT, lambda: RealClock(speed=40))
    assert instant.snapshot().transcript == fast.snapshot().transcript


def test_yield_floor_pushes_the_rest_of_the_meeting_back():
    timeline = build_timeline(parse_script(SHORT))
    finals_before = [e.t for e in timeline.events if isinstance(e, SpeechFinal)]
    delivered = []

    async def run():
        source = ReplaySource(timeline.events, SimClock())
        async for event in source.events():
            delivered.append(event)
            if isinstance(event, SpeechFinal) and event.segment == 0:
                source.yield_floor(3.0)  # the AI speaks for three seconds

    asyncio.run(run())
    finals_after = [e.t for e in delivered if isinstance(e, SpeechFinal)]
    assert finals_after[0] == finals_before[0]
    assert finals_after[1:] == [round(t + 3.0, 3) for t in finals_before[1:]]


def test_yielding_never_moves_speech_that_was_already_heard():
    timeline = build_timeline(parse_script("+0.5 A: un deux trois\n+2.0 B: oui"))
    original = next(e for e in timeline.events if isinstance(e, SpeechFinal) and e.segment == 0)
    delivered = []

    async def run():
        source = ReplaySource(timeline.events, SimClock())
        yielded = False
        async for event in source.events():
            delivered.append(event)
            if isinstance(event, SpeechPartial) and event.text == "un deux trois" and not yielded:
                source.yield_floor(3.0)  # the AI speaks before A's line is committed
                yielded = True

    asyncio.run(run())
    final_a = next(e for e in delivered if isinstance(e, SpeechFinal) and e.segment == 0)
    assert final_a.t == round(original.t + 3.0, 3)                     # committed later
    assert (final_a.t_start, final_a.t_end) == (original.t_start, original.t_end)  # but said when it was said
    first_b = next(e for e in delivered if isinstance(e, SpeechPartial) and e.segment == 1)
    assert first_b.t_start > final_a.t_end + 3.0                       # B waited for the AI
