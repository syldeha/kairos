"""Text as it should be heard: no markdown, no symbols a voice would read out.

Text-to-speech reads every character it is given: "vol/train" becomes "vol slash
train", "**budget**" becomes "étoile étoile budget". The models are asked to write
plain spoken sentences, and this is the safety net before the voice.
"""

from __future__ import annotations

import re

WORDS = {
    "French": {"/": " ou ", "→": " vers ", "->": " vers ", "=>": " donc ", "~": "environ ", "≈": "environ ",
               "&": " et ", "+": " plus ", "=": " égal ", " vs ": " contre ", " vs. ": " contre ",
               "≥": " au moins ", "≤": " au plus ", ">": " plus de ", "<": " moins de "},
    "English": {"/": " or ", "→": " to ", "->": " to ", "=>": " so ", "~": "about ", "≈": "about ",
                "&": " and ", "+": " plus ", "=": " equals ", " vs ": " versus ", " vs. ": " versus ",
                "≥": " at least ", "≤": " at most ", ">": " more than ", "<": " less than "},
}
URL = re.compile(r"\(?\[([^\]]+)\]\([^)]+\)\)?|https?://\S+|www\.\S+")
MARKUP = re.compile(r"[*_`#|]+")
BULLET = re.compile(r"(^|\n)\s*(?:[-•·]|\d+[.)])\s+")
QUOTES = re.compile(r"[“”«»\"]")
DASH = re.compile(r"\s+[–—-]\s+")
SPACES = re.compile(r"\s{2,}")


def speakable(text: str, language: str = "French") -> str:
    words = WORDS.get(language, WORDS["English"])
    text = URL.sub(lambda m: m.group(1) or "", text)  # a link is read as its label, a bare URL not at all
    text = BULLET.sub(r"\1", text)
    text = MARKUP.sub("", text)
    text = QUOTES.sub("", text)
    text = DASH.sub(", ", text)
    text = re.sub(r"(\d)\s*[-–]\s*(\d)", r"\1 à \2" if language == "French" else r"\1 to \2", text)  # 8-9 €
    text = re.sub(r"(\w)[–—](\w)", r"\1-\2", text)  # hôtel–activités
    per = " par " if language == "French" else " per "
    text = re.sub(r"\s*/\s*(?=(mois|an|année|jour|nuit|personne|pers|semaine|heure|h|month|year|day|night|person|week|hour)\b)",
                  per, text)  # 9 €/mois is "par mois", not "ou mois"
    for symbol in sorted(words, key=len, reverse=True):
        text = text.replace(symbol, words[symbol])
    text = text.replace("\n", ". ")
    text = re.sub(r"\s+([,.])", r"\1", text)  # French keeps its space before ; : ! ?
    text = re.sub(r"([,.;:])\1+", r"\1", text)
    return SPACES.sub(" ", text).strip()


def plain_text(text: str) -> str:
    """What Kairos will say, as written in the reservoir and the transcript: no markdown ("## Pharmacie de la
    Gare", "[Pharmacie](https://…)"), links read as their label. Symbols stay: the voice's safety net handles them."""
    text = URL.sub(lambda m: m.group(1) or "", text)
    text = BULLET.sub(r"\1", text)
    text = MARKUP.sub("", text).replace("\n", " ")
    return SPACES.sub(" ", text).strip()
