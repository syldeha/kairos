"""Builds a run from simple specs: a meeting source, a memory file, a judge."""

from __future__ import annotations

import json
from pathlib import Path

from .agents.archivist import archive, notes_so_far
from .config import PROJECT_ROOT, Settings
from .decide.judges import FallbackJudge, JevJudge, Judge, LlmJudge
from .llm import LLM, OpenAILLM
from .search import ExaSearch, OpenAIWebSearch, SearchProvider, TavilySearch
from .sources.ami import DATA_DIR, ami_timeline, read_words
from .sources.replay import ReplayParams, Timeline, build_timeline, parse_script

FIXTURES = PROJECT_ROOT / "fixtures"


def load_timeline(spec: str, anonymous: bool = False) -> Timeline | None:
    """`path/to/script.txt`, `ami:MEETING[:start_s[:duration_s]]` for an AMI excerpt, or `live` (None)."""
    if spec == "live":
        return None
    params = ReplayParams(anonymous=anonymous)
    if spec.startswith("ami:"):
        parts = spec.split(":")
        meeting = parts[1]
        start = float(parts[2]) if len(parts) > 2 else 540.0
        duration = float(parts[3]) if len(parts) > 3 else 180.0
        return ami_timeline(meeting, start, duration, params)
    path = Path(spec)
    if not path.exists():
        path = FIXTURES / spec
    return build_timeline(parse_script(path.read_text(encoding="utf-8")), params)


def load_memory(path: str | Path | None) -> list[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        p = FIXTURES / path
    return [line.strip() for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


MEMORY_FOR = {"reunion_produit.txt": "kairos_fr.txt", "seminaire_lisbonne.txt": "kairos_seminaire.txt"}


def default_memory(spec: str) -> str | None:
    """The memory file that goes with a meeting. A live meeting starts with none: it is yours to give."""
    if spec == "live":
        return None
    if spec.startswith("ami:"):
        return "kairos_ami.txt"
    return MEMORY_FOR.get(Path(spec).name)


def default_language(spec: str) -> str:
    return "English" if spec.startswith("ami:") else "French"


def ami_transcript(meeting: str, start_s: float = 0.0, end_s: float = float("inf"), gap_s: float = 0.6) -> str:
    """Plain transcript of an AMI meeting between two times: one line per utterance."""
    lines: list[list] = []
    last_by_speaker: dict[str, list] = {}
    for w in read_words(DATA_DIR / meeting):
        if not start_s <= w.start < end_s:
            continue
        current = last_by_speaker.get(w.speaker)
        if current is None or w.start - current[2] > gap_s:
            current = [w.speaker, [], 0.0, w.start]
            lines.append(current)
            last_by_speaker[w.speaker] = current
        current[1].append(w.text)
        current[2] = w.end
    lines.sort(key=lambda l: l[3])
    return "\n".join(f"{speaker}: {' '.join(words)}" for speaker, words, _, _ in lines)


async def prime_ami(spec: str, llm: LLM, language: str = "English") -> tuple[list[str], str]:
    """Memories from the team's previous meeting, and notes of what was said before the excerpt.

    Cached next to the meeting data, so the priming is paid for once.
    """
    parts = spec.split(":")
    meeting = parts[1]
    start = float(parts[2]) if len(parts) > 2 else 540.0
    cache = DATA_DIR / meeting / f"prime_{int(start)}.json"
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        return data["memories"], data["notes"]
    previous = meeting[:-1] + chr(ord(meeting[-1]) - 1)
    memories: list[str] = []
    if (DATA_DIR / previous).is_dir():
        memories = await archive(ami_transcript(previous), llm, language, label=f"From the previous meeting ({previous}): ")
    notes = await notes_so_far(ami_transcript(meeting, 0.0, start), llm, language) if start > 0 else ""
    cache.write_text(json.dumps({"memories": memories, "notes": notes}, ensure_ascii=False, indent=2), encoding="utf-8")
    return memories, notes


def make_llm(settings: Settings) -> LLM:
    return OpenAILLM(settings.openai_api_key, settings.model, settings.embedding_model, settings.judge_model)


def make_search(kind: str, settings: Settings, llm: LLM | None = None) -> SearchProvider | None:
    if kind == "exa" and settings.exa_api_key and llm is not None:
        return ExaSearch(settings.exa_api_key, llm)
    if kind in ("openai", "exa"):  # Exa without its key falls back to OpenAI
        return OpenAIWebSearch(settings.openai_api_key)
    if kind == "tavily":
        return TavilySearch(settings.tavily_api_key)
    return None


def make_judge(kind: str, settings: Settings, llm: LLM) -> Judge:
    if kind == "jev":
        jev = (JevJudge(settings.typesafe_api_key, direct=True, model=settings.jev_model)
               if settings.typesafe_api_key else JevJudge(settings.gateway_api_key))
        return FallbackJudge(jev, LlmJudge(llm))
    return LlmJudge(llm)
