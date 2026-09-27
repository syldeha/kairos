"""The shared board: the only place where components meet.

Background components publish to it at their own pace; the decider and the
speaker only read the latest snapshot and never wait. Everything runs on one
asyncio loop, so there are no locks.

Each zone has its own version. A writer that started from an old version of a
zone is refused (optimistic concurrency): a slow thinker cannot overwrite the
thoughts published by a faster one while it was working. The transcript
changing every word does not invalidate a thinker's write, because the check is
per zone.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .speakable import plain_text
from .contracts import AiState, Finding, RoomState, Segment, Signals, Thought

ZONES = ("transcript", "room", "notes", "thoughts", "long_term", "signals", "ai", "findings")


@dataclass(frozen=True, slots=True)
class BoardSnapshot:
    version: int
    zone_versions: Mapping[str, int]
    transcript: tuple[Segment, ...]
    room: RoomState
    notes: str
    thoughts: tuple[Thought, ...]
    long_term: tuple[str, ...]
    signals: Signals
    ai: AiState
    findings: tuple[Finding, ...]


class Board:
    def __init__(self) -> None:
        self._values: dict[str, Any] = {
            "transcript": (),
            "room": RoomState(),
            "notes": "",
            "thoughts": (),
            "long_term": (),
            "signals": Signals(),
            "ai": AiState(),
            "findings": (),
        }
        self._zone_versions = dict.fromkeys(ZONES, 0)
        self._version = 0
        self._listeners: list[Callable[[BoardSnapshot], None]] = []
        #: the reservoir's history: (meeting time, thought id, what happened, by whom, why, the sentence)
        self.log: list[tuple[float, str, str, str, str, str]] = []
        self._snapshot = self._build()

    def snapshot(self) -> BoardSnapshot:
        return self._snapshot

    def publish(self, zone: str, value: Any, base_version: int | None = None) -> int | None:
        """Replace one zone. Returns the new board version, or None if `base_version` is stale.

        `base_version` is the zone version the writer read before computing
        `value` (snapshot.zone_versions[zone]). None skips the check, for
        single-writer zones like the transcript.
        """
        if zone not in self._zone_versions:
            raise KeyError(f"unknown zone {zone!r}; zones are {ZONES}")
        if base_version is not None and base_version != self._zone_versions[zone]:
            return None
        self._values[zone] = value
        self._version += 1
        self._zone_versions[zone] = self._version
        self._snapshot = self._build()
        for listener in self._listeners:
            listener(self._snapshot)
        return self._version

    def subscribe(self, listener: Callable[[BoardSnapshot], None]) -> None:
        self._listeners.append(listener)

    def _build(self) -> BoardSnapshot:
        return BoardSnapshot(
            version=self._version,
            zone_versions=MappingProxyType(dict(self._zone_versions)),
            transcript=self._values["transcript"],
            room=self._values["room"],
            notes=self._values["notes"],
            thoughts=self._values["thoughts"],
            long_term=self._values["long_term"],
            signals=self._values["signals"],
            ai=self._values["ai"],
            findings=self._values["findings"],
        )

    def update_thoughts(self, changes: Mapping[str, Mapping[str, Any]], added: tuple[Thought, ...] = (),
                        by: str = "") -> int:
        """Merge field changes into the current thoughts by id, and append new ones.

        Background agents compute on an old snapshot, then merge here without
        awaiting in between: whatever another agent published meanwhile is kept.
        This is the reservoir's gate: every addition and every status change is logged, with who
        made it and why, so the console can show where each thought came from and why it left.
        """
        t = self._values["room"].t
        current = self._values["thoughts"]
        # A search summary may come back as markdown: what enters the reservoir is plain spoken text.
        added = tuple(replace(th, utterance=plain_text(th.utterance)) if th.utterance != plain_text(th.utterance)
                      else th for th in added)
        for thought in added:
            self.log.append((t, thought.id, f"added ({thought.kind})", by, thought.note, thought.utterance))
        for thought in current:
            change = changes.get(thought.id)
            if change and "status" in change and change["status"] != thought.status:
                self.log.append((t, thought.id, str(change["status"]), by, str(change.get("note", "")),
                                 thought.utterance))
        del self.log[:-400]
        merged = tuple(replace(t, **changes[t.id]) if t.id in changes else t for t in current)
        return self.publish("thoughts", merged + tuple(added))
