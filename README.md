# Kairos

Kairos is a voice assistant for meetings. It listens to a conversation through the microphone and speaks
through the loudspeakers, but only when it is useful: when someone asks it something, when a figure said in
the room is wrong, or when a question needs a quick search. The rest of the time it stays silent.

## What the agent does

Kairos follows the conversation line by line. For each line it decides whether it is being spoken to,
whether a search would help, and whether the subject has changed. In the background it prepares short
things it could say (answers, corrections, search results) and keeps them in a small reservoir. When there
is a pause in the conversation, it decides whether one of them is worth saying.

In practice:

- Call it by name and ask a question: "Kairos, trouve-nous un vol Paris Lisbonne le 16 octobre". It answers
  right away, or says "Je regarde" and gives the result a few seconds later.
- It can search flights and hotels (Jinko) and anything else on the web (Exa). For a hotel, it takes the
  city, dates and number of people from the trip being discussed.
- If someone says a wrong total ("40 euros each, five of us, that's 180"), it corrects it.
- If someone asks the room a factual question or can't remember something, it looks it up and answers.
- Questions people ask each other ("are you free on Saturday?") are left to them.
- If someone talks over it, it stops. After a short "mm" it continues its sentence.

When you start a session you choose how active it is:

- Speaks when called: for meetings with several people. It answers when called, corrects wrong figures and
  answers factual questions, and takes no other initiative.
- Speaks when a thought passes the threshold: for one-to-one use. It also shares a prepared idea when it
  judges it relevant.

During the session, the interface shows the conversation (Chat), notes and a board of key points that
update as people talk, and a Monitor view explaining why Kairos spoke or stayed silent.

## How it works

- Speech-to-text and text-to-speech: Gradium.
- Deciding who is spoken to, which search to run and what to say: Jev (TypeSafe), a fast classifier that
  answers in about 0.25 s.
- Writing answers, preparing ideas, checking figures, notes: an OpenAI model (`gpt-5.6-luna` by default).
  Numbers are always computed by code, not by the model.
- Searches: Exa for the web, Jinko for flights and hotels.
- Whether to speak is decided by code from these signals, never by a model.

The idea of a reservoir of prepared thoughts comes from
[thoughtful-agents](https://github.com/xybruceliu/thoughtful-agents) (Liu et al., CHI 2025).

## Installation

You need Python 3.11 or later, and [Bun](https://bun.sh) to build the interface.

```bash
git clone https://github.com/syldeha/kairos.git
cd kairos
pip install -e ".[dev]"
cd frontend && bun install && bun run build && cd ..
cp .env.example .env
```

Then open `.env` and add your keys:

| Key | Used for | Needed |
|---|---|---|
| `OPENAI_API_KEY` | writing and memory | yes |
| `TYPESAFE_API_KEY` | Jev, the decision classifier | recommended (without it, a slower fallback is used) |
| `GRADIUM_API_KEY` | listening and speaking | yes, for voice |
| `EXA_API_KEY` | web search | optional |
| `JINKO_API_KEY` | flights and hotels | optional |

## Running it

```bash
python -m kairos.ui.server
```

Open http://127.0.0.1:8787, choose the language, "Microphone" as audio source and how active Kairos should
be, then click "New session" and allow the microphone. Use headphones or a moderate volume so Kairos does
not hear itself.

Sessions are saved and listed on the home page. They are stored in `~/.local/share/kairos/sessions.db`,
and the microphone of each session is recorded in `runs/voice/` (both stay on your machine).

There is also a developer console showing every internal agent, useful for tuning:
`python -m kairos.server.app`, then http://127.0.0.1:8765. It can also replay scripted meetings from
`fixtures/`.

Optional settings, as environment variables:

- `KAIROS_THINK_MODEL`: the OpenAI model used for writing (default `gpt-5.6-luna`; `gpt-6-luna` also works).
- `KAIROS_STT_DELAY`: how far ahead speech recognition listens, in 80 ms steps (default 16). Higher is more
  accurate but slower.
- `KAIROS_PORT`: the port of the interface (default 8787).

## Tests

```bash
python -m pytest
```

The tests run without network access. The interface has its own tests
(`node node_modules/vitest/vitest.mjs run` in `frontend/`).

`scripts/ui_scenarios.py` plays scripted conversations against a running server (typed participants who
talk among themselves and call Kairos now and then) and reports how Kairos behaved: calls answered, times it
spoke when it should not have, response delays. You can watch it in the Chat view. These scenarios use real
API credits.

## Project structure

- `kairos/agents/`: the background agents (reading each line, preparing ideas, checking figures, searches,
  notes).
- `kairos/decide/`: the rules that decide when to speak.
- `kairos/speaker.py`, `kairos/voice.py`: speaking, interruptions, the Gradium voice.
- `kairos/ui/`: the server behind the web interface.
- `kairos/server/app.py`: the developer console.
- `frontend/`: the web interface.
- `scripts/`, `tests/`, `fixtures/`: scenarios, tests and scripted meetings.

## Limits

- With one microphone for several people, Kairos cannot tell who is speaking, and distant or overlapping
  voices are often transcribed badly. It relies on its name to know it is being called.
- Answers are written completely before being spoken, which adds about a second.
- Jinko covers flights and hotels only; trains and activities are searched on the web.
- Each session uses Gradium, OpenAI, TypeSafe and search credits.

## Credits

Kairos is by Sylvain Dehayem. The web interface in `frontend/` comes from
[Atlas](https://github.com/KpihX/atlas), built by Ivann Harold Kamdem Pouokam, Sylvain Dehayem and Pavel
Wadoh for the X-IA Rise of Agents hackathon, and is used with the team's agreement.

It also builds on [thoughtful-agents](https://github.com/xybruceliu/thoughtful-agents),
[meeting-sidecar](https://github.com/KpihX/meeting-sidecar) and the
[AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) (CC BY 4.0), and uses
[Gradium](https://gradium.ai), [Jev by TypeSafe](https://typesafe.ai), [Exa](https://exa.ai) and
[Jinko](https://gojinko.com).
