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


@dataclass(frozen=True, slots=True)
class HotelSummary:
    city: str
    checkin: str
    checkout: str
    options: int
    cheapest_eur: float | None
    lines: tuple[str, ...]   # a few hotels, cheapest well-rated first
    seconds: float

    def as_text(self) -> str:
        nights = _nights(self.checkin, self.checkout)
        head = (f"Hotels in {self.city}, {self.checkin} to {self.checkout} ({nights} nights): {self.options} found, "
                f"cheapest {self.cheapest_eur:.0f} EUR in total" if self.cheapest_eur is not None
                else f"Hotels in {self.city}, {self.checkin} to {self.checkout}: none found")
        return head + "\n" + "\n".join(self.lines)


class JinkoHotels:
    """Hotel rooms with live rates through Jinko's hotel_search: a city (with its country code) and the dates."""

    name = "jinko hotels"

    def __init__(self, api_key: str, base_url: str = "https://api.gojinko.com", tracer: Tracer | None = None) -> None:
        import httpx

        if not api_key:
            raise RuntimeError("JINKO_API_KEY is not set")
        self._key = api_key
        self._url = base_url.rstrip("/") + "/v1/hotel_search"
        self._client = httpx.AsyncClient(timeout=40.0)
        self.tracer = tracer or Tracer()

    async def search(self, city: str, country_code: str, checkin: str, checkout: str, adults: int = 2,
                     min_stars: int = 0) -> HotelSummary:
        start = time.perf_counter()
        call = Call("hotels", "jinko", 0.0)
        body = {"city_name": city, "country_code": country_code.upper()[:2], "checkin": checkin, "checkout": checkout,
                "adults": max(1, min(int(adults), 8)), "currency": "EUR"}
        try:
            response = await self._client.post(self._url, headers={"X-API-Key": self._key}, json=body)
            retry = _suggested_city(response)
            if retry is not None:
                # "Lisbonne" is not in Jinko's catalog, "Lisbon" is: Jinko says so; once, with its suggestion.
                body |= retry
                city = retry["city_name"]
                response = await self._client.post(self._url, headers={"X-API-Key": self._key}, json=body)
            response.raise_for_status()
            hotels = [h for h in response.json().get("hotels", []) if isinstance(h, dict) and h.get("rooms")]
        except Exception:
            call.ok = False
            raise
        finally:
            call.seconds = time.perf_counter() - start
            self.tracer.calls.append(call)
        rows = []
        for hotel in hotels:
            rates = [rate for room in hotel.get("rooms", []) for rate in room.get("rates", []) if rate.get("total_amount")]
            if not rates:
                continue
            best = min(rates, key=lambda rate: rate["total_amount"])
            rows.append({"name": hotel.get("name", ""), "stars": hotel.get("star_rating"), "rating": hotel.get("rating"),
                         "reviews": hotel.get("review_count"), "price": float(best["total_amount"]),
                         "board": best.get("board_name", ""), "refundable": best.get("is_refundable", False)})
        if min_stars:
            rows = [r for r in rows if (r["stars"] or 0) >= min_stars]  # "un cinq étoiles"
        rows.sort(key=lambda r: r["price"])
        well_rated = [r for r in rows if (r["rating"] or 0) >= 8.0]
        shown = list({id(r): r for r in well_rated[:3] + rows[:2]}.values())[:4]
        nights = _nights(checkin, checkout)
        lines = tuple(
            f"- {r['name']}: {r['stars'] or '?'} stars, rated {r['rating']:.1f}/10" if r["rating"] else f"- {r['name']}"
            for r in shown)
        lines = tuple(line + f", {r['price']:.0f} EUR in total ({r['price'] / nights:.0f} EUR a night), {r['board']}"
                      + (", refundable" if r["refundable"] else "") for line, r in zip(lines, shown))
        return HotelSummary(city=city, checkin=checkin, checkout=checkout, options=len(rows),
                            cheapest_eur=rows[0]["price"] if rows else None, lines=lines,
                            seconds=time.perf_counter() - start)


def _nights(checkin: str, checkout: str) -> int:
    import datetime as dt

    try:
        return max(1, (dt.date.fromisoformat(checkout) - dt.date.fromisoformat(checkin)).days)
    except ValueError:
        return 1


def _suggested_city(response) -> dict | None:
    """Jinko's closest catalog city when the name given did not match confidently, else None."""
    if response.status_code != 422:
        return None
    try:
        error = response.json().get("error") or {}
    except ValueError:
        return None
    destination = (error.get("suggested_retry") or {}).get("destination") or {}
    if error.get("code") == "DESTINATION_LOW_CONFIDENCE" and destination.get("city_name"):
        return {"city_name": destination["city_name"], "country_code": destination.get("country_code", "")}
    return None
