"""Clocks. Only the pacing of a replay depends on them; reasoning uses event times."""

from __future__ import annotations

import asyncio
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float: ...
    async def sleep(self, seconds: float) -> None: ...


class RealClock:
    """Wall-clock time, optionally sped up (speed=2 plays a meeting twice as fast)."""

    def __init__(self, speed: float = 1.0) -> None:
        if speed <= 0:
            raise ValueError("speed must be positive; use SimClock for an instant replay")
        self.speed = speed
        self._origin = asyncio.get_running_loop().time()

    def now(self) -> float:
        return (asyncio.get_running_loop().time() - self._origin) * self.speed

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(max(0.0, seconds) / self.speed)


class SimClock:
    """Simulated time: sleeping advances the clock instantly.

    Deterministic when a single task drives time (the replay source does).
    """

    def __init__(self) -> None:
        self._t = 0.0

    def now(self) -> float:
        return self._t

    async def sleep(self, seconds: float) -> None:
        self._t += max(0.0, seconds)
        await asyncio.sleep(0)
