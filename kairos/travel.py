"""Flight search for the travel worker, through Jinko (ported from Meeting Sidecar's adapter).

`flight_calendar` returns indicative prices, computed by Jinko a few days earlier: good for
"is there a direct flight, roughly how much", not for booking. One request per destination
(Jinko returns only the cheapest destination when given several), with one or several
departure dates, one-way or round trip. The summary keeps what a meeting needs: how many
options, the cheapest per destination, the direct ones, the times.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from .llm import Call, Tracer


@dataclass(frozen=True, slots=True)
class FlightSummary:
    route: str
    date: str
    options: int
    cheapest_eur: float | None
    direct: int
    lines: tuple[str, ...]   # a few itineraries, for the notes and the console
    priced_at: str
    seconds: float

    def as_text(self) -> str:
        head = (f"{self.route} on {self.date}: {self.options} options, {self.direct} direct, "
                f"cheapest {self.cheapest_eur:.0f} EUR per person" if self.cheapest_eur is not None
                else f"{self.route} on {self.date}: no option found")
        return head + (f" (prices from {self.priced_at})" if self.priced_at else "") + "\n" + "\n".join(self.lines)


class JinkoFlights:
    name = "jinko"

    def __init__(self, api_key: str, base_url: str = "https://api.gojinko.com", tracer: Tracer | None = None) -> None:
        import httpx

        if not api_key:
            raise RuntimeError("JINKO_API_KEY is not set")
        self._key = api_key
        self._url = base_url.rstrip("/") + "/v1/flight_calendar"
        self._client = httpx.AsyncClient(timeout=30.0)
        self.tracer = tracer or Tracer()

    async def search(self, origin: str | list[str], destination: str | list[str], date: str | list[str],
                     adults: int = 1, return_date: str | None = None) -> FlightSummary:
        """IATA city or airport codes (PAR, LIS); dates as YYYY-MM-DD; a return date makes it a round trip."""
        origins = [origin] if isinstance(origin, str) else list(origin)
        destinations = [destination] if isinstance(destination, str) else list(destination)
        dates = [date] if isinstance(date, str) else list(date)
        start = time.perf_counter()
        per_destination = await asyncio.gather(
            *(self._one(origins, d, dates, adults, return_date) for d in destinations), return_exceptions=True)
        if all(isinstance(r, Exception) for r in per_destination) and len(dates) > 3:
            # A server error (502) on a long date list: once more with the three middle dates.
            middle = dates[len(dates) // 2 - 1: len(dates) // 2 + 2]
            await asyncio.sleep(0.5)
            per_destination = await asyncio.gather(
                *(self._one(origins, d, middle, adults, return_date) for d in destinations), return_exceptions=True)
            dates = middle
        itineraries = [i for r in per_destination if isinstance(r, list) for i in r]
        if not itineraries and any(isinstance(r, Exception) for r in per_destination):
            raise next(r for r in per_destination if isinstance(r, Exception))
        rows = []
        for it in itineraries:
            out = it.get("outbound") or {}
            back = it.get("inbound") or {}
            total = it.get("total") or {}
            price = total.get("value", 0) / 10 ** total.get("decimal_places", 2) if total.get("value") else 1e9
            rows.append({"price": price, "stops": out.get("stops", 0), "dep": out.get("departure_datetime", ""),
                         "arr": out.get("arrival_datetime", ""), "airline": out.get("airline_name", ""),
                         "from": out.get("origin", ""), "to": (it.get("destination") or {}).get("name", out.get("destination", "")),
                         "minutes": out.get("duration_minutes", 0), "back": back.get("departure_datetime", ""),
                         "priced_at": it.get("priced_at", "")})
        rows.sort(key=lambda r: (r["price"], r["stops"], r["minutes"]))
        direct = [r for r in rows if r["stops"] == 0]
        cheapest_by_place = {}
        for r in rows:
            cheapest_by_place.setdefault(r["to"], r)
        shown = list(dict.fromkeys([id(r) for r in direct[:1] + list(cheapest_by_place.values()) + rows[:3]]))
        by_id = {id(r): r for r in rows}
        lines = tuple(dict.fromkeys(
            f"- to {r['to']}: {r['airline']}, {'direct' if r['stops'] == 0 else f'{r['stops']} stop(s)'}, "
            f"departs {r['dep'][:10]} {r['dep'][11:16]} from {r['from']}, {r['minutes'] // 60}h{r['minutes'] % 60:02d}"
            + (f", return {r['back'][:10]}" if r["back"] else "") + f", {r['price']:.0f} EUR"
            for r in (by_id[i] for i in shown[:6])))[:5]
        route = f"{'/'.join(origins)}-{'/'.join(destinations)}"
        when = ", ".join(dates) + (f" (return {return_date})" if return_date else "")
        return FlightSummary(route=route, date=when, options=len(rows), cheapest_eur=rows[0]["price"] if rows else None,
                             direct=len(direct), lines=lines, priced_at=(rows[0]["priced_at"] or "")[:10] if rows else "",
                             seconds=time.perf_counter() - start)

    async def _one(self, origins: list[str], destination: str, dates: list[str], adults: int,
                   return_date: str | None) -> list[dict]:
        start = time.perf_counter()
        call = Call("flights", "jinko", 0.0)
        body = {"origins": origins, "destinations": [destination], "departure_dates": dates[:7],
                "adults": max(1, min(int(adults), 9)), "currency": "EUR",
                "trip_type": "roundtrip" if return_date else "oneway"}
        if return_date:
            body["return_dates"] = [return_date]
        try:
            response = await self._client.post(self._url, headers={"X-API-Key": self._key}, json=body)
            response.raise_for_status()
            return [i for i in response.json().get("itineraries", []) if isinstance(i, dict)]
        except Exception:
            call.ok = False
            raise
        finally:
            call.seconds = time.perf_counter() - start
            self.tracer.calls.append(call)
