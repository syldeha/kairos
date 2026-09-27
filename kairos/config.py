"""Settings, read from a .env file: in this project's folder, or one folder above it."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATHS = (PROJECT_ROOT / ".env", PROJECT_ROOT.parent / ".env")
ENV_PATH = next((p for p in ENV_PATHS if p.exists()), ENV_PATHS[0])


def load_env(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


load_env()


@dataclass(slots=True)
class Settings:
    openai_api_key: str = field(repr=False, default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    gateway_api_key: str = field(repr=False, default_factory=lambda: os.getenv("AI_GATEWAY_API_KEY", ""))
    #: TypeSafe's own key: Jev called directly (preferred over the gateway when present).
    typesafe_api_key: str = field(repr=False, default_factory=lambda: os.getenv("TYPESAFE_API_KEY", ""))
    jev_model: str = field(default_factory=lambda: os.getenv("KAIROS_JEV_MODEL", "jev-latest"))
    #: Thinking and writing (thinkers, answers, researcher, notes): fast, cheap, natural sentences.
    model: str = field(default_factory=lambda: os.getenv("KAIROS_THINK_MODEL", "gpt-5.6-luna"))
    #: Judging (yes/no with token probabilities): must support logprobs. Measured as the sharpest fast one.
    judge_model: str = field(default_factory=lambda: os.getenv("KAIROS_JUDGE_MODEL", "gpt-4o-mini"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    #: "llm" (default) or "jev" (TypeSafe Jev through Vercel AI Gateway).
    judge: str = field(default_factory=lambda: os.getenv("KAIROS_JUDGE", "llm"))
    #: Web search for the researcher: "openai" (default, uses the OpenAI key), "tavily", or "off".
    search: str = field(default_factory=lambda: os.getenv("KAIROS_SEARCH", "exa" if os.getenv("EXA_API_KEY") else "openai"))
    exa_api_key: str = field(repr=False, default_factory=lambda: os.getenv("EXA_API_KEY", ""))
    #: Jinko: flight search for the travel worker.
    jinko_api_key: str = field(repr=False, default_factory=lambda: os.getenv("JINKO_API_KEY", ""))
    jinko_url: str = field(default_factory=lambda: os.getenv("JINKO_URL", "https://api.gojinko.com"))
    tavily_api_key: str = field(repr=False, default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    #: Gradium: speech-to-text with semantic voice activity, and text-to-speech (the live voice demo).
    gradium_api_key: str = field(repr=False, default_factory=lambda: os.getenv("GRADIUM_API_KEY", ""))
    gradium_url: str = field(default_factory=lambda: os.getenv("GRADIUM_URL", "wss://eu.api.gradium.ai/api/speech"))
