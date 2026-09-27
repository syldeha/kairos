"""Judges: turn a meeting state and yes/no statements into probabilities.

Judges run in the background, never on the critical path: their answers are
stored on the board (on each thought, and in the signals) and the decider only
reads them. Two implementations behind one interface:

    LlmJudge  one short completion per statement, probability from the yes/no
              token log-probabilities
    JevJudge  TypeSafe's Jev through Vercel AI Gateway: all statements in one
              request, as boolean questions with native probabilities

FallbackJudge puts Jev first and the LLM judge behind it: a failed Jev request
(network, quota, account) is answered by the LLM judge instead of "no" everywhere,
which would silence Kairos without any visible reason.
"""

from __future__ import annotations

import asyncio
import time
from typing import Mapping, Protocol

from ..llm import LLM, Call, Tracer

SYSTEM = ("You judge statements about a live meeting that an AI assistant named Kairos attends. "
          "Read the state, then answer the question with a single word: yes or no.")


class Judge(Protocol):
    name: str

    async def probabilities(self, state: str, statements: Mapping[str, str]) -> dict[str, float]: ...


class LlmJudge:
    name = "llm"

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    async def probabilities(self, state: str, statements: Mapping[str, str]) -> dict[str, float]:
        async def one(statement: str) -> float:
            user = f"{state}\n\nStatement: {statement}\nIs this statement true? Answer yes or no."
            return await self.llm.yes_probability(SYSTEM, user, purpose="judge")

        keys = list(statements)
        values = await asyncio.gather(*(one(statements[k]) for k in keys), return_exceptions=True)
        # A failed question counts as "no": every layer fails towards silence.
        return {k: (v if isinstance(v, float) else 0.0) for k, v in zip(keys, values)}


class JevJudge:
    """TypeSafe's Jev, called directly (`POST api.typesafe.ai/v1/systemone`, model `jev-latest`, yes/no
    questions of type "noul") or through Vercel AI Gateway (`POST /v1/evaluate`, model `typesafe-ai/jev`,
    type "boolean"). Same questions, same probabilities; only the names differ."""

    name = "jev"
    DIRECT_URL = "https://api.typesafe.ai/v1/systemone"
    GATEWAY_URL = "https://ai-gateway.vercel.sh/v1/evaluate"

    def __init__(self, api_key: str, tracer: Tracer | None = None, timeout_s: float = 5.0,
                 direct: bool = False, model: str | None = None) -> None:
        import httpx

        if not api_key:
            raise RuntimeError("TYPESAFE_API_KEY (or AI_GATEWAY_API_KEY) is not set in the demo's .env file")
        self.direct = direct
        self.url = self.DIRECT_URL if direct else self.GATEWAY_URL
        self.model = model or ("jev-latest" if direct else "typesafe-ai/jev")
        self._client = httpx.AsyncClient(timeout=timeout_s,
                                         headers={"Authorization": f"Bearer {api_key}"})
        self.tracer = tracer or Tracer()
        self.last_error: str | None = None  # shown in the console: a failing judge must not fail silently

    async def probabilities(self, state: str, statements: Mapping[str, str]) -> dict[str, float]:
        # Jev question names must be identifiers; map them back afterwards.
        names = {f"q{i}": key for i, key in enumerate(statements)}
        kind = "noul" if self.direct else "boolean"
        answers = await self.ask(state, {q: {"type": kind, "instructions": statements[k]} for q, k in names.items()})
        return {key: _probability(answers.get(q)) for q, key in names.items()}

    async def ask(self, state: str, questions: Mapping[str, dict], purpose: str = "judge") -> dict[str, dict]:
        """Typed questions in one request: yes/no ("noul" direct, "boolean" via the gateway), "choice" with
        `criteria` {option: description}, "score" with `criteria` [labels, lowest first]. Returns Jev's answers:
        {"probability" or "noul"}, {"choice", "probabilities"}, {"score", "probabilities"}."""
        questions = {q: ({**v, "type": "noul"} if self.direct and v.get("type") == "boolean" else
                         {**v, "type": "boolean"} if not self.direct and v.get("type") == "noul" else v)
                     for q, v in questions.items()}
        body = {"model": self.model, "state": state, "questions": dict(questions)}
        if not self.direct:
            # Meeting content: ask the gateway not to keep it, and to serve it from TypeSafe only.
            body["providerOptions"] = {"gateway": {"zeroDataRetention": True, "only": ["typesafe-ai"]}}
        start = time.perf_counter()
        call = Call(purpose, self.model, 0.0)
        try:
            response = await self._client.post(self.url, json=body)
            response.raise_for_status()
            data = response.json()
            usage = data.get("usage") or {}
            call.prompt_tokens = usage.get("inputTokens", usage.get("input_tokens", 0))
            call.completion_tokens = usage.get("outputTokens", usage.get("output_tokens", 0))
            cost = ((data.get("providerMetadata") or {}).get("gateway") or {}).get("cost")
            call.cost_usd = float(cost) if cost is not None else None
            answers = _answers(data)
            missing = [q for q in questions if not isinstance(answers.get(q), dict)]
            if missing:
                raise ValueError(f"no answer for {missing}")
            self.last_error = None
            return answers
        except Exception as exc:
            call.ok = False
            detail = getattr(getattr(exc, "response", None), "text", "")[:300]
            self.last_error = f"{type(exc).__name__}: {detail or exc}".strip()
            raise JudgeFailed(self.last_error) from exc
        finally:
            call.seconds = time.perf_counter() - start
            self.tracer.calls.append(call)


class JudgeFailed(RuntimeError):
    pass


class FallbackJudge:
    """Jev first; if a request fails, the LLM judge answers it. Records how often each one answered."""

    name = "jev"

    def __init__(self, primary: JevJudge, fallback: LlmJudge, retry_after_s: float = 60.0) -> None:
        self.primary, self.fallback = primary, fallback
        self.tracer = primary.tracer
        self.answered = {"jev": 0, "llm": 0}
        self.retry_after_s = retry_after_s  # after a failure, Jev is skipped this long (no 0.4 s lost each time)
        self._failed_at: float | None = None

    @property
    def last_error(self) -> str | None:
        return self.primary.last_error

    async def probabilities(self, state: str, statements: Mapping[str, str]) -> dict[str, float]:
        resting = self._failed_at is not None and time.monotonic() - self._failed_at < self.retry_after_s
        if not resting:
            try:
                result = await self.primary.probabilities(state, statements)
                self.answered["jev"] += 1
                self._failed_at = None
                return result
            except JudgeFailed:
                self._failed_at = time.monotonic()
        self.answered["llm"] += 1
        return await self.fallback.probabilities(state, statements)

    async def ask(self, state: str, questions: Mapping[str, dict], purpose: str = "judge") -> dict[str, dict]:
        """Typed questions only Jev can answer (choices, scores): no fallback, the caller does without."""
        if self._failed_at is not None and time.monotonic() - self._failed_at < self.retry_after_s:
            raise JudgeFailed("Jev is resting after a failure")
        try:
            return await self.primary.ask(state, questions, purpose)
        except JudgeFailed:
            self._failed_at = time.monotonic()
            raise


def _answers(data: dict) -> dict:
    for key in ("answers", "results", "output"):
        if isinstance(data.get(key), dict):
            return data[key]
    return data


def _probability(answer) -> float:
    if isinstance(answer, (int, float)):
        return float(answer)
    if isinstance(answer, dict):
        for key in ("probability", "noul", "p"):  # "noul": TypeSafe's own name for a yes/no probability
            if isinstance(answer.get(key), (int, float)):
                return float(answer[key])
    return 0.0
