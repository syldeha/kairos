"""AMI Meeting Corpus: real four-person meetings with a time stamp on every word.

Source: AMI manual annotations 1.6.2 (https://groups.inf.ed.ac.uk/ami/),
Creative Commons Attribution 4.0. We only read `words/<meeting>.<speaker>.words.xml`.

    python -m kairos.sources.ami extract path/to/ami_public_manual_1.6.2.zip ES2002b

copies one meeting's word files into `data/ami/<meeting>/`. `ami_timeline`
then turns an excerpt into the same events as a scripted replay: one partial
per word at its real end time, a final after each utterance, and the simulated
voice activity. Real pauses, overlaps and "mm-hmm"s are kept.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

from ..contracts import SpeechEvent, SpeechFinal, SpeechPartial
from .replay import ReplayParams, Timeline, vad_steps

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "ami"
NITE_ID = "{http://nite.sourceforge.net/}id"


@dataclass(frozen=True, slots=True)
class Word:
    speaker: str
    start: float
    end: float
    text: str


def read_words(meeting_dir: Path) -> list[Word]:
    words: list[Word] = []
    for path in sorted(meeting_dir.glob("*.words.xml")):
        speaker = path.name.split(".")[1]
        last: Word | None = None
        for element in ET.parse(path).getroot():
            if element.tag != "w" or element.get("starttime") is None or element.get("endtime") is None:
                continue
            text = (element.text or "").strip()
            if element.get("punc") == "true":
                if last is not None:  # attach punctuation to the previous word
                    last = Word(last.speaker, last.start, last.end, last.text + text)
                    words[-1] = last
                continue
            last = Word(speaker, float(element.get("starttime")), float(element.get("endtime")), text)
            words.append(last)
    return sorted(words, key=lambda w: (w.start, w.speaker))


def ami_timeline(meeting: str, start_s: float = 0.0, duration_s: float = 300.0,
                 params: ReplayParams = ReplayParams(), utterance_gap_s: float = 0.6,
                 data_dir: Path = DATA_DIR) -> Timeline:
    words = [w for w in read_words(data_dir / meeting) if start_s <= w.start < start_s + duration_s]
    if not words:
        raise ValueError(f"no words for {meeting} between {start_s} s and {start_s + duration_s} s")
    origin = words[0].start - 0.5

    # Group each speaker's words into utterances: a new one after a pause of utterance_gap_s.
    utterances: list[list[Word]] = []
    open_by_speaker: dict[str, list[Word]] = {}
    for raw in words:
        w = Word(raw.speaker, raw.start - origin, raw.end - origin, raw.text)
        current = open_by_speaker.get(w.speaker)
        if current is None or w.start - current[-1].end > utterance_gap_s:
            current = []
            utterances.append(current)
            open_by_speaker[w.speaker] = current
        current.append(w)
    utterances.sort(key=lambda u: u[0].start)

    events: list[SpeechEvent] = []
    speech: list[tuple[float, float, str]] = []
    pauses: list[tuple[float, float]] = []
    for seg_id, utterance in enumerate(utterances):
        who = None if params.anonymous else utterance[0].speaker
        t_start = round(utterance[0].start, 3)
        for i, w in enumerate(utterance):
            events.append(SpeechPartial(t=round(w.end, 3), segment=seg_id, speaker=who,
                                        text=" ".join(x.text for x in utterance[: i + 1]), t_start=t_start))
        run_start = utterance[0].start
        for prev, nxt in zip(utterance, utterance[1:]):
            if nxt.start - prev.end >= 0.15:
                speech.append((round(run_start, 3), round(prev.end, 3), utterance[0].speaker))
                pauses.append((round(prev.end, 3), round(nxt.start, 3)))
                run_start = nxt.start
        speech.append((round(run_start, 3), round(utterance[-1].end, 3), utterance[0].speaker))
        t_end = round(utterance[-1].end, 3)
        events.append(SpeechFinal(t=round(t_end + params.final_delay_s, 3), segment=seg_id, speaker=who,
                                  text=" ".join(w.text for w in utterance), t_start=t_start, t_end=t_end))

    events += vad_steps(speech, pauses, params)
    events.sort(key=lambda e: e.t)
    return Timeline(events=tuple(events), speech=tuple(sorted(speech)))


def extract(zip_path: Path, meeting: str, data_dir: Path = DATA_DIR) -> Path:
    target = data_dir / meeting
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        names = [n for n in archive.namelist() if n.startswith(f"words/{meeting}.") and n.endswith(".words.xml")]
        if not names:
            raise ValueError(f"meeting {meeting} not found in {zip_path}")
        for name in names:
            (target / Path(name).name).write_bytes(archive.read(name))
    return target


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "extract":
        print("usage: python -m kairos.sources.ami extract <ami_public_manual_1.6.2.zip> <meeting>")
        raise SystemExit(2)
    print(extract(Path(sys.argv[2]), sys.argv[3]))
