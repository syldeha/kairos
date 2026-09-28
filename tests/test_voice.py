"""The voice demo's pieces, without network: Gradium messages in, speech events out."""

from kairos.contracts import SpeechFinal, SpeechPartial, VadStep
from kairos.sources.gradium import GradiumSource
from kairos.voice import EchoGuard


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def source(is_echo=None):
    clock = Clock()
    src = GradiumSource("key", "wss://example/api/speech", is_echo=is_echo)
    src.attach(None, clock)
    src._origin = 10.0  # the microphone started 10 s into the meeting
    return src, clock


def step(p0, p3=None):
    p3 = p0 if p3 is None else p3
    return {"type": "step", "vad": [{"horizon_s": 0.5, "inactivity_prob": p0}, {"horizon_s": 1.0, "inactivity_prob": p0},
                                    {"horizon_s": 2.0, "inactivity_prob": p3}, {"horizon_s": 3.0, "inactivity_prob": p3}]}


def test_words_build_a_segment_that_ends_on_the_voice_activity_and_a_flush():
    src, clock = source()
    clock.t = 11.0
    [first] = src.handle({"type": "text", "text": "Tu", "start_s": 0.5})
    assert isinstance(first, SpeechPartial) and first.t_start == 10.5 and first.speaker == "Vous"
    clock.t = 11.4
    [second] = src.handle({"type": "text", "text": "sais s'il y a un train ?", "start_s": 0.8})
    assert second.text == "Tu sais s'il y a un train ?" and second.segment == first.segment
    src.handle({"type": "end_text", "stop_s": 1.9})
    for _ in range(2):
        [vad] = src.handle(step(0.9))
        assert isinstance(vad, VadStep) and not vad.speaking
    assert src.flush_due() is None
    src.handle(step(0.9))            # third step in a row: the sentence is over
    flush = src.flush_due()
    assert flush == 1 and src.flush_due() is None  # asked once
    clock.t = 12.0
    [final] = src.handle({"type": "flushed", "flush_id": 1})
    assert isinstance(final, SpeechFinal) and final.text == "Tu sais s'il y a un train ?"
    assert final.t_start == 10.5 and final.t_end == 11.9
    [partial] = src.handle({"type": "text", "text": "Et", "start_s": 3.0})
    assert partial.segment != final.segment  # the next words start a new line


def test_a_pause_mid_sentence_does_not_cut_it():
    src, _ = source()
    src.handle({"type": "text", "text": "Je pense que", "start_s": 0.0})
    for _ in range(5):
        src.handle(step(0.8, 0.2))  # short pause, but the speaker is not done (long horizon low)
    assert src.flush_due() is None


def test_a_late_flush_still_commits_the_segment():
    src, clock = source()
    src.handle({"type": "text", "text": "Bonjour.", "start_s": 0.0})
    for _ in range(3):
        src.handle(step(0.95))
    src.flush_due()
    clock.t += 2.0
    [final] = src.overdue()
    assert final.text == "Bonjour."


def test_steps_map_to_the_horizons():
    src, _ = source()
    [vad] = src.handle(step(0.2, 0.7))
    assert vad.p_silence == (0.2, 0.2, 0.7, 0.7) and vad.speaking


def test_kairos_heard_back_through_the_microphone_is_dropped():
    clock = Clock()
    echo = EchoGuard(clock)
    src, _ = source(is_echo=echo)
    src.attach(None, clock)
    echo.speaking("Le budget validé est de 12 000 €, pas 15 000.", audio_end=3.0)
    clock.t = 1.0
    assert src.handle({"type": "text", "text": "budget validé", "start_s": 0.5}) == []
    [vad] = src.handle(step(0.1))
    assert not vad.speaking  # the detector hears Kairos, not a person
    [human] = src.handle({"type": "text", "text": "Attends, non", "start_s": 0.9})
    assert human.text == "Attends, non"  # someone talking over Kairos is heard
    echo.speaking("Dis-moi quand tu pars.", audio_end=3.0)
    assert src.handle({"type": "text", "text": "Tu", "start_s": 1.0})  # too common to be taken for an echo
    clock.t = 10.0  # long after Kairos stopped: its words are not filtered any more
    assert src.handle({"type": "text", "text": "budget", "start_s": 9.0})


def test_stop_discards_a_full_audio_queue_and_remains_idempotent():
    src, _ = source()
    for _ in range(src._audio.maxsize):
        src.push_audio(b"audio")
    assert src._audio.full()

    src.stop()
    src.stop()

    assert src._stopped.is_set()
    assert src._audio.qsize() == 1
    assert src._audio.get_nowait() is None
    src.push_audio(b"late audio")
    assert src._audio.empty()


# -- Jev (Vercel AI Gateway), with a fake server --------------------------------------

def _jev(handler):
    import httpx

    from kairos.decide.judges import JevJudge
    jev = JevJudge("vck_test")
    jev._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return jev


def test_jev_reads_probabilities_and_cost():
    import asyncio
    import json

    import httpx

    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "model": "typesafe-ai/jev",
            "answers": {"q0": {"type": "boolean", "probability": 0.93}, "q1": {"type": "boolean", "probability": 0.02}},
            "usage": {"inputTokens": 300, "outputTokens": 20},
            "providerMetadata": {"gateway": {"cost": "0.0000123"}}})

    jev = _jev(handler)
    p = asyncio.run(jev.probabilities("state", {"fit:t1": "fits?", "said:t1": "said?"}))
    assert p == {"fit:t1": 0.93, "said:t1": 0.02}
    assert seen["body"]["providerOptions"]["gateway"]["zeroDataRetention"] is True
    assert abs(jev.tracer.cost_usd() - 0.0000123) < 1e-12


def test_a_failing_jev_hands_over_to_the_llm_judge():
    import asyncio

    import httpx

    from kairos.decide.judges import FallbackJudge

    class Llm:
        async def probabilities(self, state, statements):
            return {k: 0.7 for k in statements}

    jev = _jev(lambda request: httpx.Response(403, json={"error": {"message": "requires a valid credit card"}}))
    judge = FallbackJudge(jev, Llm())
    assert asyncio.run(judge.probabilities("state", {"a": "?"})) == {"a": 0.7}
    assert judge.answered == {"jev": 0, "llm": 1} and "credit card" in judge.last_error
    calls = len(jev.tracer.calls)
    asyncio.run(judge.probabilities("state", {"a": "?"}))
    assert len(jev.tracer.calls) == calls  # Jev rests after a failure instead of costing 0.4 s every time


def test_jev_direct_uses_typesafe_names():
    import asyncio
    import json

    import httpx

    from kairos.decide.judges import JevJudge
    seen = {}

    def handler(request):
        seen["url"], seen["body"] = str(request.url), json.loads(request.content)
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": {"q0": {"type": "noul", "noul": 0.91}},
                                         "usage": {"input_tokens": 291, "output_tokens": 38}})

    jev = JevJudge("apikey_test", direct=True)
    jev._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    assert asyncio.run(jev.probabilities("state", {"fit:t1": "fits?"})) == {"fit:t1": 0.91}
    assert seen["url"] == JevJudge.DIRECT_URL and seen["body"]["model"] == "jev-latest"
    assert seen["body"]["questions"]["q0"]["type"] == "noul" and "providerOptions" not in seen["body"]
    assert jev.tracer.calls[-1].prompt_tokens == 291


def test_spoken_text_has_no_symbols_read_out():
    from kairos.speakable import speakable
    assert speakable("Il faut valider l’accès (ascenseur/rampe) et le **budget**.") == \
        "Il faut valider l’accès (ascenseur ou rampe) et le budget."
    assert speakable("Pour arbitrer vol vs train : Paris → Lisbonne, ~25 h.") == \
        "Pour arbitrer vol contre train : Paris vers Lisbonne, environ 25 h."
    assert speakable("Concurrents : 8-9 € par mois – à vérifier.") == "Concurrents : 8 à 9 € par mois, à vérifier."
    assert speakable("- *Option A* : train\n- Option B : avion") == "Option A : train. Option B : avion"
    assert speakable("Selon [aeromed.fr](https://www.aeromed.fr/x), 2h30.") == "Selon aeromed.fr, 2h30."
    assert speakable("Le séminaire porte-à-porte du 14 au 16 mai.") == "Le séminaire porte-à-porte du 14 au 16 mai."
    assert speakable("Autour de 8 à 9,90 €/mois, trajets hôtel–activités.") ==         "Autour de 8 à 9,90 € par mois, trajets hôtel-activités."
