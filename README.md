# Kairos

**A meeting assistant that knows when to speak, what to say, and when to stay quiet.**

Most voice assistants wait for a wake word or answer everything. Kairos sits in a live conversation,
listens, and joins in only when it has something useful to add: an answer to a question put to it, a
correction of a wrong figure, a search result someone needs, a calculation nobody has done. The rest of
the time it stays silent. It speaks through the room's loudspeakers; no bot joins a call.

Kairos builds on two ideas:

- **Inner thoughts** ([thoughtful-agents](https://github.com/xybruceliu/thoughtful-agents), Liu et al.,
  CHI 2025). The agent keeps a reservoir of things it could say, prepared while people talk. Speaking is
  a separate decision, taken at each pause.
- **A judge that is not the writer.** A fast classifier, [Jev](https://typesafe.ai), answers questions
  about the conversation in about 0.25 s: is this line addressed to Kairos? which search would help? did
  the subject change? which prepared thought fits right now, if any? Language models write; Jev and plain
  code decide.

It comes with two interfaces: its own console (a view of every agent, for tuning), and the interface of
[Atlas](https://github.com/KpihX/atlas) (session library, chat, workflow canvas, Monitor, notes, board).

## What It Does

```text
Room audio
    |
    v
Gradium STT -> Jev dispatcher -> reservoir of prepared thoughts -> policy (code) -> Gradium TTS -> Room
                   |                  ^        ^        ^
                   |                  |        |        +-- Checker    wrong figures, calculations (code computes)
                   |                  |        +----------- Thinkers   what Kairos could say, with the sentence
                   |                  +-------------------- Workers    web (Exa), flights and hotels (Jinko)
                   +--> Topic tracker, Notes & Board, Monitor
```

| Situation | Kairos |
|---|---|
| "Kairos, trouve-nous un vol Paris Lisbonne le 16 octobre" | "Alors…" in about 0.5 s, "Je regarde.", then "Le vol direct le moins cher coûte 105 euros…" from Jinko. |
| "Kairos, regarde les hôtels là-bas, pour nous trois" | Takes the city, dates and people from the trip being discussed, searches Jinko's live rates, names the best-rated option and its price per night. |
| "À 35 euros chacun, à six ça fait 180 euros" | "À 35 euros chacun, à six, le budget est de 210 euros, pas 180." About 1 s after the sentence, without being asked. |
| "Il y a eu une catastrophe dernièrement, je ne me rappelle plus c'était quoi" | Searches at once and gives the answer at the next pause. |
| "On prend un hôtel ou un appartement ?" (friends deciding) | Stays silent: that question is for the people, not the assistant. |
| "Toi, tu pars de Paris aussi ?" (one person to another) | Stays silent: it is an assistant, not part of the group's plans. |
| The group moves from the budget to a birthday present | Notices the change of subject and drops the ideas prepared for the old one. |
| Someone talks over it | Stops at once. After a mere "mm" it resumes its sentence; otherwise it yields and does not repeat itself. |

Two behaviours, chosen when a session starts:

- **Speaks when called (meeting)**: answers when someone calls it by name, corrects a clearly wrong
  figure, answers a factual question put to the room; takes no other initiative.
- **Speaks when a thought passes the threshold**: also says a prepared thought as soon as Jev picks it
  above 0.55 in a silence. Suited to a one-to-one conversation or a test.

## Quick Start

Requires Python 3.11+ and [Bun](https://bun.sh) for the interface.

```bash
git clone https://github.com/syldeha/kairos.git
cd kairos
pip install -e ".[dev]"
cp .env.example .env              # then fill in your keys
cd frontend && bun install && bun run build && cd ..
python -m kairos.ui.server        # Atlas interface: http://127.0.0.1:8787
```

Open the page, pick the language, **Microphone** and the behaviour, then **New session**. The **Chat**
view shows the conversation and how long Kairos took to reply; **Monitor** shows every decision.

The tuning console is still there: `python -m kairos.server.app`, then http://127.0.0.1:8765.

## Architecture

Background agents write to a shared board. The decider only reads the board and decides in under a
millisecond; nothing waits for a model to decide whether to speak.

```text
microphone -> Gradium STT -> Scribe -> transcript --+-> Cleaner       fixes misheard words from context
                                                    +-> Dispatcher    (Jev) addressed? which search? factual question? new subject?
                                                    +-> Checker       a wrong figure or a calculation to do? (code computes)
                                                    +-> Thinkers      prepared thoughts, with the sentence to say
                                                    +-> Answer lane   a question to Kairos gets an answer at once
                                                    +-> Workers       web (Exa), flights and hotels (Jinko): details, search, result
                                                    +-> Topic         what the room is talking about now
                                                    +-> Judge + rater (Jev) coherent now? already said? which one?
                                                                   |
                        speaker <-- harness <-- policy (code): is the floor open, is the pick clear, may Kairos take the initiative?
                            |
                            +-> Gradium TTS -> loudspeakers
```

| Component | Job | Model |
|---|---|---|
| Scribe | transcript and room state from speech events | code |
| Cleaner | corrects each live line from its context; never adds words | writing model |
| Dispatcher | per sentence: addressed to Kairos? which search (flights, hotels, web, none)? a factual question to the room? change of subject? | Jev |
| Topic tracker | names the current subject; retires thoughts about an old one | Jev + writing model |
| Checker | contradictions with known figures, wrong calculations; the arithmetic runs in code | writing model |
| Thinkers | prepared thoughts (idea, answer, correction, finding), each with the sentence to say | writing model |
| Answer lane | a question put to Kairos gets a reply at once, or "je regarde" plus a search; knows today's date | writing model |
| Workers | briefs with the details a search needs, filled from the notes and the other searches; ask only what cannot be assumed; results owed to whoever asked | writing model + Exa / Jinko |
| Memory | embeddings: nothing prepared twice, nothing said twice | text-embedding-3-small |
| Judge and rater | fit now, already said; picks one thought or "none" | Jev |
| Policy | is the floor open? is the pick clear? is Kairos in the conversation? | code |
| Harness | talk budget, length limits; answers and corrections go first | code |
| Speaker | speaks word by word; handles barge-in, false starts and resuming | code + Gradium TTS |
| Notes & Board | structured notes and a desired-state board, reviewed and reconciled (Atlas's method) | writing model |

The reservoir holds at most five thoughts. Every addition and status change is logged with who made it
and why: the Monitor view shows the log, each line's trace from hearing to speaking, and every decision.

## Repository

```text
kairos/
|-- kairos/
|   |-- agents/          dispatcher, thinkers, checker, workers, judge, topic, notes, research
|   |-- decide/          policy and judges
|   |-- sources/         Gradium speech-to-text, scripted and AMI replays
|   |-- server/app.py    the tuning console
|   |-- ui/              the Atlas interface: server, projection, protocol, Notes & Board
|   |-- runtime.py       wires the agents and runs a meeting
|   |-- speaker.py       speech, barge-in and resuming
|   |-- harness.py       rules the decider cannot bend
|   |-- travel.py        Jinko flights and hotels
|   `-- search.py        Exa web search
|-- frontend/            the Atlas interface (React, Vite)
|-- web/index.html       the console page
|-- scripts/             live scenarios, benchmarks, speech replay
|-- tests/
`-- fixtures/            scripted meetings
```

## Persistence

```text
~/.local/share/kairos/sessions.db   sessions of the Atlas interface (KAIROS_DB to change it)
runs/voice/                         microphone of each voice session, 24 kHz WAV, to replay it
runs/ui_scenarios/                  results of the live scenarios
```

`runs/` and `.env` are ignored by git.

## Provider Configuration

Keys go in `.env` (this folder, or the one above it):

| Variable | Role | Required |
|---|---|---|
| `OPENAI_API_KEY` | writing model and embeddings | yes |
| `TYPESAFE_API_KEY` | Jev: dispatcher, judge, rater. Without it an LLM judge is used, slower | recommended |
| `GRADIUM_API_KEY` | voice: speech-to-text and text-to-speech | for voice |
| `EXA_API_KEY` | web search worker and context research | optional |
| `JINKO_API_KEY` | flights and hotels workers | optional |

Settings (environment variables):

| Variable | Default | Meaning |
|---|---|---|
| `KAIROS_THINK_MODEL` | `gpt-5.6-luna` | writing model; `gpt-6-luna` answered best in our comparison at the same speed |
| `KAIROS_JUDGE_MODEL` | `gpt-4o-mini` | fallback judge when Jev fails (needs token probabilities) |
| `KAIROS_STT_DELAY` | `16` | Gradium look-ahead in 80 ms frames: more is more accurate and later (24 = 1.9 s) |
| `KAIROS_PORT` | `8787` | port of the Atlas interface |
| `KAIROS_DB` | `~/.local/share/kairos/sessions.db` | session database |
| `KAIROS_FRONTEND_DIR` | `frontend/dist` | built interface |

## Commands

| Command | What it does |
|---|---|
| `python -m kairos.ui.server` | Kairos with the Atlas interface |
| `python -m kairos.server.app` | the tuning console (live, scripted meetings, AMI replays) |
| `python -m pytest` | 148 tests, no network needed |
| `cd frontend && bun run build` | build the interface (`node node_modules/vitest/vitest.mjs run` for its 17 tests) |
| `python scripts/ui_scenarios.py [name ...]` | live scenarios against the running interface, with monitoring and a relevance judge |
| `python scripts/stt_replay.py runs/voice/x.wav 16 24 32` | replay a recorded microphone at several look-ahead settings |
| `python scripts/ami_bench.py` | AMI benchmark against the console |

## Verification

- **Unit tests**: `python -m pytest` (148) and the interface's 17 component tests.
- **Live scenarios** (`scripts/ui_scenarios.py`): meetings where typed participants talk among themselves
  and call Kairos. Each run reports calls served, unprompted interventions, reaction and answer delays,
  repetitions, and a model's rating of every intervention in its context. Watch them in the Chat view.
- **Speech replay** (`scripts/stt_replay.py`): the same recorded audio through Gradium at several settings.
- **AMI benchmark** (`scripts/ami_bench.py`): real meetings from the
  [AMI corpus](https://groups.inf.ed.ac.uk/ami/corpus/) (CC BY 4.0); download
  `ami_public_manual_1.6.2.zip` into `data/ami_zip/` and run `python scripts/ami_mine.py` first.

## Measured

- **Reaction when called**: an opener ("Alors…") 0.4–0.8 s after the question; the answer 1.5–3 s after
  it when it needs no search, 3–6 s with a web or Jinko search.
- **Meeting behaviour** (27-line scripted meeting, latest full run): 21/27 lines as expected. The four
  calls were answered (flights, hotels, evening out, airport transfer) and the wrong total corrected, but it
  still spoke after 5 lines not meant for it; the fixes made since have not been measured on it yet.
- **Corrections**: said 0.7–1.8 s after the sentence with the wrong figure ends.
- **AMI benchmark** (9 real passages, 3 of them controls): recall 3–4 of 6 expected moments, precision
  4/4 to 8/9, controls silent.
- **Jev**: about 0.25 s per request.

## Known Limits

- **One microphone, several people**: Gradium does not separate speakers, so everyone is "Vous". Kairos
  relies on its name and on Jev to tell whether a line is for it; far voices and overlapping speech are
  transcribed poorly.
- **Speech recognition errors**: the cleaner fixes what the context makes certain, and every agent is told
  a line may be misheard. A badly misheard sentence still gets through.
- **Latency**: answers are written in full before they are spoken; streaming them would save about 1 s.
- **Jinko** covers flights and hotels; trains and activities go through web search.
- **Cost**: every session uses Gradium, OpenAI, Jev and search credits; the live scenarios and speech
  replays do too.

## Team and Credits

Kairos is by Sylvain Dehayem. The interface in `frontend/` and its data contract in
`kairos/ui/protocol.py` come from [Atlas](https://github.com/KpihX/atlas), by Ivann Harold Kamdem Pouokam,
Sylvain Dehayem and Pavel Wadoh (X-IA Rise of Agents hackathon), used with the team's agreement; so does
the method of the Notes & Board agent.

- [thoughtful-agents](https://github.com/xybruceliu/thoughtful-agents): the inner-thoughts framework
  (Liu et al., CHI 2025).
- [meeting-sidecar](https://github.com/KpihX/meeting-sidecar): patterns for live notes and meeting memory.
- [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/): real meetings for evaluation (CC BY 4.0).
- [Gradium](https://gradium.ai) (voice), [Jev by TypeSafe](https://typesafe.ai) (judge),
  [Exa](https://exa.ai) (search), [Jinko](https://gojinko.com) (flights and hotels).
