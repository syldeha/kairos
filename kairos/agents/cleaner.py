"""The cleaner: corrects each live speech-to-text line from its context, within a second or so.

Live recognition mishears ("la photo des groupes tickets pour tes dons"), stutters ("En gros, en gros,")
and drops punctuation. Every agent reads the transcript: a messy line gives messy thoughts and odd
answers ("what does 'passer deux fois ici' mean?"). The cleaner rewrites a committed line once, with the
lines before it, the current topic and the names it knows; the raw text is kept for display.

It only corrects what was said: no added content, no change of meaning, and a line it cannot make
sense of stays as it is. Replays (AMI, scripts) are exact transcripts and are not cleaned.
"""

from __future__ import annotations

from ..board import Board
from ..llm import LLM
from .common import is_ai

CLEAN = """You correct a live speech-to-text transcript of a conversation, in its own language. The recognizer mishears words, cuts one turn into pieces, repeats fragments and drops punctuation.
Return one JSON object: {"previous": "the corrected previous part, or null", "text": "the corrected line"}.
- Recognition errors SOUND like what was said: say the line aloud in your head and look for what the person most likely said, given what was just said before. A reply often agrees or disagrees with the line before it: after "le budget validé est de 12 000 euros, pas 15 000", "sait vrai, j'avais oublié les douze" is "c'est vrai, j'avais oublié, les 12 000".
- Fix a misheard word when the sound and the conversation together make the right word clear, and the heard words make no sense there ("des gars à laquelle ils vont arriver" -> "des gares auxquelles ils vont arriver"; "44 euros à l'heure auto ?" after prices -> "44 euros l'aller-retour ?"). Fix grammar slips ("je pense que je parler" -> "je pense que je parle"). "Kairos" is the AI assistant in the room.
- When a "Previous part of the same turn" is given, the recognizer cut one sentence in two: correct both parts together, as one sentence, and return the previous part corrected in "previous" (null if it needs no change).
- Remove stutters, repeated fragments and fillers ("en gros, en gros," -> "en gros,", "euh"), and punctuate.
- Never invent: no name, number, amount or unit that does not sound like what was heard ("50 choses" stays, it does not sound like "50 euros"). Never add a word the speaker did not say, even an obvious one ("on va aller le 17" stays, never "on va aller à Lyon le 17").
- Keep the speaker's meaning, words and register ("tu", slang); never answer the line.
- A correction sounds close to what was heard, syllable for syllable: never replace a phrase by a longer or different-sounding one ("c'est à quoi ?" stays, it does not sound like "c'est à quelle heure ?").
- When no sound-alike reading makes sense, keep the words as they are."""


class TranscriptCleaner:
    def __init__(self, board: Board, llm: LLM, revise, names: tuple[str, ...] = ()) -> None:
        self.board = board
        self.llm = llm
        self.revise = revise  # callback(segment id, corrected text): the Scribe replaces the line
        self.names = names
        self.log: list[tuple[float, int, str, str]] = []  # (time, segment, raw, corrected)

    async def clean(self, segment: int) -> str | None:
        """Correct one committed line. Returns the corrected text when it changed."""
        snap = self.board.snapshot()
        line = next((s for s in snap.transcript if s.id == segment and s.final), None)
        if line is None or is_ai(line) or len(line.text.split()) < 3:
            return None
        before = [s for s in snap.transcript if s.final and s.t_start < line.t_start][-6:]
        # The recognizer cuts a turn at a short pause ("Ah oui, oui, …" | "… raison, c'était prévu le 18"):
        # the part just before, same speaker, nobody in between, is corrected together with this one.
        previous = before[-1] if before and before[-1].speaker == line.speaker and \
            line.t_start - before[-1].t_end <= 2.5 else None
        context = before[:-1] if previous is not None else before
        user = (f"Topic: {snap.signals.topic or '(unknown)'}\n"
                f"Names: {', '.join(('Kairos',) + self.names)}\n\n"
                "Conversation before (already corrected):\n"
                + ("\n".join(f"{s.speaker or 'Someone'}: {s.text}" for s in context) or "(nothing yet)")
                + (f"\n\nPrevious part of the same turn ({line.speaker or 'Someone'}), cut by the recognizer: "
                   f"{previous.text}" if previous is not None else "")
                + f"\n\nLine to correct ({line.speaker or 'Someone'}): {line.text}")
        raw = await self.llm.json(CLEAN, user, purpose="cleaner", temperature=0.0)
        text = str(raw.get("text") or "").strip()
        if previous is not None:
            fixed = str(raw.get("previous") or "").strip()
            if _plausible(previous.text, fixed):
                self.log.append((snap.room.t, previous.id, previous.text, fixed))
                self.revise(previous.id, fixed)
        if not _plausible(line.text, text):
            return None
        self.log.append((snap.room.t, segment, line.text, text))
        del self.log[:-100]
        self.revise(segment, text)
        return text


def _plausible(heard: str, corrected: str) -> bool:
    """A rewrite much longer than what was said is an answer or an invention, not a correction."""
    return bool(corrected) and corrected != heard and len(corrected.split()) <= len(heard.split()) * 1.3 + 3
