# Kairos

**A meeting assistant that knows when to speak, what to say, and when to stay quiet.**

Most voice assistants wait for a wake word or answer everything. Kairos sits in a live
conversation, listens, and joins in only when it has something useful to add: a correction of a
wrong figure, an answer to a question put to it, a search result someone needs, a calculation
nobody has done. The rest of the time it stays silent.

Kairos listens through Gradium (speech-to-text) and speaks through Gradium text-to-speech. Its
console and prompts are in French, and it replies in whatever language is being spoken.

Kairos builds on two ideas:

- **Inner thoughts** ([thoughtful-agents](https://github.com/xybruceliu/thoughtful-agents),
  Liu et al., CHI 2025). The agent keeps a reservoir of things it could say, prepared while people
  talk. Speaking is a separate decision, taken at each pause.
- **A judge that is not the writer.** A fast classifier, [Jev](https://typesafe.ai), answers
  yes/no questions about the conversation in about 0.25 s: is this line addressed to Kairos? did
  the subject change? which prepared thought fits right now, if any? Language models write; Jev
  and plain code decide.

## What it does

| Situation | Kairos |
|---|---|
| "À 35 euros chacun, à six ça fait 180 euros" | "À 35 euros chacun, à six, le budget est de 210 euros, pas 180 euros." About 1 s after the sentence ends, before anything else. |
| "Kairos, tu peux regarder les trains pour Lyon ?" | "Alors…" at once, then "Je regarde.", then the result from a live search. |
| "Faut qu'on trouve un resto vers Bastille" (not addressed to it) | Starts a web search, then offers what it found at a natural pause. |
| "Toi, tu pars de Paris aussi ?" (one person to another) | Stays silent: it is an assistant, not part of the group's plans. |
| The group moves from the budget to a birthday present | Notices the change of subject and drops the ideas it had prepared for the old one. |
| Someone talks over it | Stops. If the person only said "mm", it restarts its sentence; otherwise it rewrites its answer to take in what was just said. |

## How it works

Background agents write to a shared board. The decider only reads the board, and decides in under
a millisecond.

```
microphone ─▶ Gradium STT ─▶ Scribe ─▶ transcript ──┬─▶ Cleaner      fixes misheard words from context
                                                    ├─▶ Dispatcher   (Jev) addressed? search? flights? new subject?
                                                    ├─▶ Checker      a wrong figure or a calculation to do? (code computes)
                                                    ├─▶ Thinkers     prepared thoughts, with the sentence to say
                                                    ├─▶ Answer lane  a question to Kairos gets an answer at once
                                                    ├─▶ Workers      web (Exa), flights (Jinko): ask for missing details, search
                                                    ├─▶ Topic        what the room is talking about now
                                                    └─▶ Judge + rater (Jev) coherent now? already said? which one to say?
                                                                   │
                        speaker ◀── harness ◀── policy (code): is the floor open, is the pick clear?
                            │
                            └─▶ Gradium TTS ─▶ speakers
```

| Component | Job | Model |
|---|---|---|
| Scribe | transcript and room state from speech events | code |
| Cleaner | corrects each live line from its context ("des gars à laquelle" → "des gares auxquelles"); never adds words | luna |
| Dispatcher | per sentence: addressed to Kairos? search needed? flights? change of subject? | Jev |
| Topic tracker | names the current subject; retires thoughts about an old one | Jev + luna |
| Checker | contradictions with known figures, wrong calculations, calculations the room is attempting; the arithmetic runs in code | luna |
| Thinkers | prepared thoughts (idea, answer, correction, finding), each with the sentence to say | luna |
| Answer lane | a question put to Kairos gets a reply at once, or "je regarde" plus a search | luna |
| Workers | briefs with the details a search needs; one question to the room for what is missing; results owed to whoever asked | luna + Exa / Jinko |
| Context research | reads up on the meeting's subject in the background (facts with figures) | luna + Exa |
| Judge and rater | fit now, already said, outdated; picks one thought or "none" | Jev |
| Policy | is the floor open? is the pick clear enough? | code |
| Harness | talk budget, length limits; answers and corrections go first | code |
| Speaker | speaks word by word; handles barge-in, false starts and resuming | code + Gradium TTS |

The reservoir holds at most five thoughts. Every addition and status change is logged with who made
it and why (`reservoir_log` in `/api/state`, and in the console).

## Run it

Requires Python 3.11+.

```
pip install -e ".[dev]"
cp .env.example .env          # then fill in your keys
python -m kairos.server.app   # console: http://127.0.0.1:8765
python -m pytest              # 121 tests, no network needed
```

Keys (in `.env`):

| Key | Used for | Required |
|---|---|---|
| `OPENAI_API_KEY` | writing model (default `gpt-5.6-luna`, see `KAIROS_THINK_MODEL`) and embeddings | yes |
| `TYPESAFE_API_KEY` | Jev (dispatcher, judge, rater). Without it an LLM judge is used, slower | recommended |
| `GRADIUM_API_KEY` | voice: speech-to-text and text-to-speech | for voice |
| `EXA_API_KEY` | web search worker and context research | optional |
| `JINKO_API_KEY` | flight search worker | optional |

### In the console

- **Direct**: you and Kairos, live. Tick **voix (Gradium)**, pick a voice, press **Démarrer**, and
  allow the microphone. Wear headphones: an echo guard drops Kairos's own voice, but less reliably.
  You can also type as a participant.
- **Scripted meetings** (`fixtures/`) and **AMI excerpts** (see below) are replayed word by word, as a
  streaming recognizer would deliver them.
- **Modes**: `boucle fermée` pauses a replayed meeting while Kairos speaks; `boucle ouverte` lets the
  recording talk over it (to test interruptions); `hors ligne` records its decisions without
  speaking. In Direct, the first two behave the same.
- **Écoute avant transcription**: how far ahead Gradium listens before writing a word (16 frames =
  1.28 s). 12 is about 0.3 s faster with slightly lower accuracy; 8 degrades clearly.
- `POST /api/say {"speaker": "…", "text": "…"}` adds a typed participant to a live session: handy
  for testing with two "people" in the room.

### AMI meetings

Kairos can replay real meetings from the [AMI corpus](https://groups.inf.ed.ac.uk/ami/corpus/)
(CC BY 4.0). Download `ami_public_manual_1.6.2.zip` (manual annotations) into `data/ami_zip/`, then
run `python scripts/ami_mine.py`, which extracts the word files and rates passages where an assistant
could help. `python scripts/ami_bench.py` replays nine selected passages against the running console
and scores Kairos's interventions (recall of the expected moments, precision of what it said).

## Measured

- **AMI benchmark** (9 real passages, 3 of them controls where Kairos should stay silent): recall 3–4
  of 6 expected moments, precision 4/4 to 8/9, controls silent. Recall varies by ±1–2 between runs.
- **Live corrections**: said 0.7–1.1 s after the sentence with the wrong figure ends.
- **Questions to Kairos**: an opener ("Alors…") about 0.5–1 s after the question; the full answer
  2–3 s after it. Most of the delay is speech recognition: Gradium's look-ahead (1.28 s) and
  end-of-turn confirmation.
- **Jev**: about 0.25 s per request during live meetings.

## Scripts

| Script | What it does |
|---|---|
| `scripts/ami_bench.py` | AMI benchmark (needs the console running) |
| `scripts/ami_mine.py` | extracts the AMI corpus and finds passages where an assistant could help |
| `scripts/flow_eval.py` | short live conversations scored on fluidity (repetition, timing, relevance) |
| `scripts/live_matrix.py` | real-time scenarios (flights, a trip) against the console |
| `scripts/stt_replay.py` | replays a recorded microphone through Gradium at several look-ahead settings |
| `scripts/voice_check.py` | end-to-end voice timing with a synthetic participant |
| `scripts/judge_compare.py`, `scripts/model_compare.py` | compare judges and writing models on the same moments |

Voice sessions are recorded to `runs/voice/` (ignored by git).

## Known limits

- **One microphone, several people**: Gradium does not separate speakers, so everyone is "Vous".
  Kairos then relies on Jev to tell whether a question is for it. A recognizer with real-time speaker
  separation would help.
- **Speech recognition errors**: the cleaner fixes what the context makes certain, and the agents are
  told that a line may be misheard. A badly misheard sentence still gets through.
- **Latency**: answers are written in full before they are spoken. Speaking while the answer is being
  written would save about 1 s.

## Credits

- [thoughtful-agents](https://github.com/xybruceliu/thoughtful-agents): the inner-thoughts framework
  (Liu et al., CHI 2025).
- [meeting-sidecar](https://github.com/KpihX/meeting-sidecar): patterns for live notes and meeting
  memory.
- [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/): real meetings for evaluation (CC BY 4.0).
- [Gradium](https://gradium.ai) (voice), [Jev by TypeSafe](https://typesafe.ai) (judge),
  [Exa](https://exa.ai) (search), Jinko (flights).
