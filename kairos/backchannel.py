"""Listening signals: short words that do not ask for the floor.

A string lookup, not a model (Deepgram's approach): listeners backchannel far
more often than they interrupt, and the check has to be instant.
"""

from __future__ import annotations

import re

BACKCHANNELS = {
    # English
    "mm", "mhm", "mmhmm", "mm-hmm", "uh-huh", "uhhuh", "hm", "hmm", "yeah", "yes", "yep", "right",
    "okay", "ok", "sure", "uh", "um", "mm-mm", "uh-uh",
    # French
    "oui", "ouais", "d'accord", "dac", "ok", "ah", "oh", "euh", "hum", "voilà", "exact", "exactement",
}


def is_backchannel(text: str, max_words: int = 2) -> bool:
    words = [w for w in re.split(r"[\s,.!?;:]+", text.lower()) if w]
    return 0 < len(words) <= max_words and all(w in BACKCHANNELS for w in words)
