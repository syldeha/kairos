"""Web search providers for the researcher, behind one interface.

    ExaSearch        Exa, results with highlights, summarised by the writing model (EXA_API_KEY; default)
    OpenAIWebSearch  OpenAI Responses API with the web_search tool (works with the OpenAI key)
    TavilySearch     Tavily, built for agents (needs TAVILY_API_KEY with credits)

Both return a short spoken answer in the meeting's language and the sources.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from .llm import Call, Tracer


@dataclass(frozen=True, slots=True)
class SearchResult:
    answer: str
    sources: tuple[str, ...]
    seconds: float


class SearchProvider(Protocol):
    name: str

    async def search(self, query: str, question: str, language: str) -> SearchResult: ...


def _instructions(question: str, query: str, language: str) -> str:
    return (f"Look this up on the web: {query}\n"
            f"The question raised in a meeting was: {question}\n"
            f"Answer in {language}, in one or two short spoken sentences (under 35 words), as a colleague "
            f"would say it in the meeting, and name the source briefly (for example 'according to ...'). "
            f"If the web gives no clear answer, say so in one short sentence.")


class OpenAIWebSearch:
    name = "openai"

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", tracer: Tracer | None = None) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.tracer = tracer or Tracer()

    async def search(self, query: str, question: str, language: str) -> SearchResult:
        start = time.perf_counter()
        call = Call("search", self.model, 0.0)
        try:
            response = await self._client.responses.create(
                model=self.model, tools=[{"type": "web_search"}], input=_instructions(question, query, language))
            usage = getattr(response, "usage", None)
            if usage is not None:
                call.prompt_tokens = getattr(usage, "input_tokens", 0) or 0
                call.completion_tokens = getattr(usage, "output_tokens", 0) or 0
            sources = []
            for item in response.output:
                for part in getattr(item, "content", None) or []:
                    for annotation in getattr(part, "annotations", None) or []:
                        url = getattr(annotation, "url", None)
                        if url and url not in sources:
                            sources.append(url)
            return SearchResult(response.output_text.strip(), tuple(sources[:3]), time.perf_counter() - start)
        except Exception:
            call.ok = False
            raise
        finally:
            call.seconds = time.perf_counter() - start
            self.tracer.calls.append(call)


class TavilySearch:
    name = "tavily"
    URL = "https://api.tavily.com/search"

    def __init__(self, api_key: str, tracer: Tracer | None = None) -> None:
        import httpx

        if not api_key:
            raise RuntimeError("TAVILY_API_KEY is not set")
        self._key = api_key
        self._client = httpx.AsyncClient(timeout=15.0)
        self.tracer = tracer or Tracer()

    async def search(self, query: str, question: str, language: str) -> SearchResult:
        start = time.perf_counter()
        call = Call("search", "tavily", 0.0)
        try:
            response = await self._client.post(self.URL, json={
                "api_key": self._key, "query": query, "max_results": 3, "include_answer": True,
                "search_depth": "basic"})
            response.raise_for_status()
            data = response.json()
            sources = tuple(r.get("url") for r in data.get("results", [])[:3] if r.get("url"))
            return SearchResult(str(data.get("answer") or "").strip(), sources, time.perf_counter() - start)
        except Exception:
            call.ok = False
            raise
        finally:
            call.seconds = time.perf_counter() - start
            self.tracer.calls.append(call)


class ExaSearch:
    """Exa web search: titles, URLs and highlights; a writing model turns them into a spoken answer.

    Exa's own answer is not used: the highlights are evidence, and the meeting needs one short
    sentence in its language, with the source named.
    """

    name = "exa"
    URL = "https://api.exa.ai/search"

    def __init__(self, api_key: str, llm, tracer: Tracer | None = None, num_results: int = 5) -> None:
        import httpx

        if not api_key:
            raise RuntimeError("EXA_API_KEY is not set")
        self._key = api_key
        self._client = httpx.AsyncClient(timeout=20.0)
        self.llm = llm
        self.num_results = num_results
        self.tracer = tracer or Tracer()

    async def search(self, query: str, question: str, language: str) -> SearchResult:
        start = time.perf_counter()
        call = Call("search", "exa", 0.0)
        try:
            response = await self._client.post(self.URL, headers={"x-api-key": self._key}, json={
                "query": query, "type": "auto", "numResults": self.num_results, "contents": {"highlights": True}})
            response.raise_for_status()
            data = response.json()
            cost = (data.get("costDollars") or {}).get("total")
            call.cost_usd = float(cost) if cost is not None else None
            results = [r for r in data.get("results", []) if isinstance(r, dict)]
        except Exception:
            call.ok = False
            raise
        finally:
            call.seconds = time.perf_counter() - start
            self.tracer.calls.append(call)
        if not results:
            return SearchResult("", (), time.perf_counter() - start)
        evidence = "\n".join(
            f"[{i + 1}] {r.get('title') or ''} ({_site(r.get('url'))}): " + " … ".join(r.get("highlights") or [])[:600]
            for i, r in enumerate(results))
        answer = await self.llm.json(
            "You turn web search results into what a colleague says in a meeting. Return JSON "
            '{"answer": "...", "used": [1, 2]}. The answer is one or two short spoken sentences (under 35 words) '
            f"in {language}, answering the question from the results only. Name places by their name (\"Le Grenier "
            "propose...\"); cite a source (\"selon Time Out\") only when it is a guide, a news or weather site distinct "
            "from what it describes. The question was said to Kairos, the assistant speaking: \"Kairos\" is who is "
            "asked, never a place nor the subject (never \"Kairos propose\"); \"là-bas\" or \"there\" is the place in "
            "the search query. Be concrete: name the specific places, neighbourhoods, venues or times the results "
            "give (\"du fado dans l'Alfama, les bars du Bairro Alto\"), never generic activities (\"dîner, prendre "
            "un verre\"). Answer exactly what was asked, the most common option first (\"how do we get from the "
            "airport to the centre\": the metro, the bus, a taxi, with time and price, before private transfers). "
            "No symbols, no URLs. If the results do not answer it, say so in one short sentence.",
            f"Question raised in the meeting: {question}\nSearch query: {query}\n\nResults:\n{evidence}",
            purpose="search summary", temperature=0.2)
        used = [int(i) - 1 for i in answer.get("used") or [] if str(i).isdigit() and 0 < int(i) <= len(results)]
        sources = tuple(results[i].get("url") for i in (used or [0, 1, 2]) if i < len(results) and results[i].get("url"))
        return SearchResult(str(answer.get("answer") or "").strip(), sources[:3], time.perf_counter() - start)


def _site(url: str | None) -> str:
    from urllib.parse import urlparse

    return (urlparse(url or "").netloc or "").removeprefix("www.")
