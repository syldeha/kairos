"""Model access: chat completions returning JSON, yes/no probabilities, embeddings.

All calls are asynchronous (the thoughtful-agents library used a synchronous
client, so its "parallel" calls ran one after the other). Every call is traced
with its purpose, duration and token usage, so the console can show what the
system is doing and what it costs.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Protocol

# USD per million tokens (input, output), for the cost estimate shown in the console.
# Models missing here are counted as calls but not in the cost: add their price when known.
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.6-luna": (1.00, 6.00),
    "text-embedding-3-small": (0.02, 0.0),
}


@dataclass(slots=True)
class Call:
    purpose: str
    model: str
    seconds: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ok: bool = True
    cost_usd: float | None = None  # when the provider reports the cost itself (Vercel AI Gateway)


@dataclass(slots=True)
class Tracer:
    calls: list[Call] = field(default_factory=list)

    def cost_usd(self) -> float:
        total = 0.0
        for c in self.calls:
            if c.cost_usd is not None:
                total += c.cost_usd
                continue
            price_in, price_out = PRICES.get(c.model, (0.0, 0.0))
            total += (c.prompt_tokens * price_in + c.completion_tokens * price_out) / 1e6
        return total

    def unpriced(self) -> list[str]:
        return sorted({c.model for c in self.calls if c.model not in PRICES and c.cost_usd is None and c.ok})

    def summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.calls:
            out[c.purpose] = out.get(c.purpose, 0) + 1
        return out


class LLM(Protocol):
    tracer: Tracer

    async def json(self, system: str, user: str, *, purpose: str, temperature: float = 0.4) -> dict: ...
    async def yes_probability(self, system: str, user: str, *, purpose: str) -> float: ...
    async def embed(self, texts: list[str], *, purpose: str) -> list[list[float]]: ...


class OpenAILLM:
    def __init__(self, api_key: str, model: str, embedding_model: str, judge_model: str | None = None) -> None:
        from openai import AsyncOpenAI

        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (expected in the demo's .env file)")
        self._client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.judge_model = judge_model or model  # must return token log-probabilities
        self.embedding_model = embedding_model
        self.tracer = Tracer()

    async def json(self, system: str, user: str, *, purpose: str, temperature: float = 0.4) -> dict:
        start = time.perf_counter()
        call = Call(purpose, self.model, 0.0)
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_object"},
                **_sampling(self.model, temperature),
            )
            _usage(call, response)
            return json.loads(response.choices[0].message.content or "{}")
        except Exception:
            call.ok = False
            raise
        finally:
            call.seconds = time.perf_counter() - start
            self.tracer.calls.append(call)

    async def yes_probability(self, system: str, user: str, *, purpose: str) -> float:
        """P(yes) from the token log-probabilities of a one-token yes/no answer."""
        start = time.perf_counter()
        call = Call(purpose, self.judge_model, 0.0)
        try:
            response = await self._client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0,
                max_tokens=1,
                logprobs=True,
                top_logprobs=10,
            )
            _usage(call, response)
            yes = no = 0.0
            for candidate in response.choices[0].logprobs.content[0].top_logprobs:
                token = candidate.token.strip().lower()
                if token in ("yes", "oui"):
                    yes += math.exp(candidate.logprob)
                elif token in ("no", "non"):
                    no += math.exp(candidate.logprob)
            return yes / (yes + no) if yes + no > 0 else 0.5
        except Exception:
            call.ok = False
            raise
        finally:
            call.seconds = time.perf_counter() - start
            self.tracer.calls.append(call)

    async def embed(self, texts: list[str], *, purpose: str) -> list[list[float]]:
        start = time.perf_counter()
        call = Call(purpose, self.embedding_model, 0.0)
        try:
            response = await self._client.embeddings.create(model=self.embedding_model, input=texts)
            call.prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
            return [d.embedding for d in response.data]
        except Exception:
            call.ok = False
            raise
        finally:
            call.seconds = time.perf_counter() - start
            self.tracer.calls.append(call)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _sampling(model: str, temperature: float) -> dict:
    """GPT-5 models reason before answering: turn that off for speed; they ignore temperature."""
    if model.startswith("gpt-5"):
        newer = not model.startswith(("gpt-5-", "gpt-5.0")) and model != "gpt-5"
        return {"reasoning_effort": "none" if newer else "minimal"}
    return {"temperature": temperature}


def _usage(call: Call, response) -> None:
    usage = getattr(response, "usage", None)
    if usage is not None:
        call.prompt_tokens = usage.prompt_tokens or 0
        call.completion_tokens = usage.completion_tokens or 0
