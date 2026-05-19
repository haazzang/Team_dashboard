"""Scottish Mortgage Investment Trust (SMT LN) daily NAV fetcher.

Scrapes RNS "Net Asset Value(s)" announcements from Investegate, parses the
four standard NAV figures (cum/ex income x debt-at-par/fair-value), and
caches them to disk so subsequent calls hit the network only for new dates.

The NAV publication cadence is one RNS per business day, announced ~11:30-12:30
London time on the day AFTER the valuation date. So the NAV for swap-report
valuation date T is published on T+1 (business day).
"""

from __future__ import annotations

import datetime as dt
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup


LISTING_URL = "https://www.investegate.co.uk/company/SMT"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
)
NAV_LABEL_MAP = {
    "cum par nav": "cum_par",
    "cum fair nav": "cum_fair",
    "ex par nav": "ex_par",
    "ex fair nav": "ex_fair",
    # Long-form labels (occasionally appear)
    "net asset value per ordinary share (cum income, debt at par)": "cum_par",
    "net asset value per ordinary share (cum income, debt at fair value)": "cum_fair",
    "net asset value per ordinary share (ex income, debt at par)": "ex_par",
    "net asset value per ordinary share (ex income, debt at fair value)": "ex_fair",
}
MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


@dataclass
class NavRecord:
    valuation_date: str   # ISO yyyy-mm-dd
    announcement_date: str
    announcement_time: str | None
    cum_par: float | None
    cum_fair: float | None
    ex_par: float | None
    ex_fair: float | None
    source_url: str


def _parse_date(text: str) -> dt.date | None:
    text = text.strip()
    m = re.match(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", text)
    if not m:
        return None
    day, mon, year = m.group(1), m.group(2)[:3].lower(), m.group(3)
    if mon not in MONTHS:
        return None
    try:
        return dt.date(int(year), MONTHS[mon], int(day))
    except ValueError:
        return None


def _parse_pence(text: str) -> float | None:
    m = re.match(r"\(?\s*(-?\d{1,5}(?:,\d{3})*(?:\.\d+)?)\s*p?\s*\)?$", text.strip())
    if not m:
        return None
    raw = m.group(1).replace(",", "")
    try:
        value = float(raw)
    except ValueError:
        return None
    if text.strip().startswith("("):
        value = -value
    return value


def _http_get(url: str, *, timeout: int = 20) -> str:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def fetch_listing() -> list[dict]:
    """Return list of {date, time, title, url} for SMT NAV announcements."""
    html = _http_get(LISTING_URL)
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="table-investegate")
    if not table or not table.find("tbody"):
        return []
    out = []
    for tr in table.find("tbody").find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 4:
            continue
        link = tds[3].find("a", class_="announcement-link")
        if not link:
            continue
        title = link.get_text(strip=True)
        if not title.startswith("Net Asset Value"):
            continue
        out.append({
            "date": tds[0].get_text(strip=True),
            "time": tds[1].get_text(strip=True),
            "title": title,
            "url": link["href"],
        })
    return out


def parse_detail(url: str) -> NavRecord | None:
    html = _http_get(url)
    text = BeautifulSoup(html, "html.parser").get_text(separator="\n")
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

    dates: list[dt.date] = []
    for ln in lines:
        d = _parse_date(ln)
        if d and d.year >= 2020 and d not in dates:
            dates.append(d)
        if len(dates) >= 2:
            break
    if len(dates) < 2:
        return None
    announcement_date, valuation_date = dates[0], dates[1]
    if valuation_date > announcement_date:
        announcement_date, valuation_date = valuation_date, announcement_date

    navs: dict[str, float] = {}
    for idx, ln in enumerate(lines[:-1]):
        key = ln.lower().rstrip(":")
        col = NAV_LABEL_MAP.get(key)
        if not col:
            continue
        for look in range(idx + 1, min(idx + 4, len(lines))):
            value = _parse_pence(lines[look])
            if value is not None:
                navs[col] = value
                break
    if not navs:
        return None

    return NavRecord(
        valuation_date=valuation_date.isoformat(),
        announcement_date=announcement_date.isoformat(),
        announcement_time=None,
        cum_par=navs.get("cum_par"),
        cum_fair=navs.get("cum_fair"),
        ex_par=navs.get("ex_par"),
        ex_fair=navs.get("ex_fair"),
        source_url=url,
    )


def load_cache(path: Path) -> dict[str, NavRecord]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return {}
    out: dict[str, NavRecord] = {}
    for item in raw:
        try:
            rec = NavRecord(**item)
            out[rec.valuation_date] = rec
        except TypeError:
            continue
    return out


def save_cache(path: Path, records: Iterable[NavRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_records = sorted(records, key=lambda r: r.valuation_date)
    path.write_text(json.dumps([asdict(r) for r in sorted_records], indent=2))


def refresh_nav_history(
    cache_path: Path,
    *,
    max_new: int = 30,
    sleep_seconds: float = 0.4,
) -> tuple[dict[str, NavRecord], int]:
    """Pull latest RNS listing, fetch detail for any new valuation dates, persist.

    Returns the merged cache and the number of newly added records.
    """
    cache = load_cache(cache_path)
    known_announcement_urls = {r.source_url for r in cache.values()}

    listing = fetch_listing()
    added = 0
    for row in listing:
        if added >= max_new:
            break
        if row["url"] in known_announcement_urls:
            continue
        try:
            rec = parse_detail(row["url"])
        except Exception:
            continue
        if rec is None:
            continue
        # Preserve announcement time from listing
        rec.announcement_time = row.get("time")
        cache[rec.valuation_date] = rec
        added += 1
        time.sleep(sleep_seconds)
    if added:
        save_cache(cache_path, cache.values())
    return cache, added


def history_as_records(cache: dict[str, NavRecord]) -> list[dict]:
    return [asdict(rec) for rec in sorted(cache.values(), key=lambda r: r.valuation_date)]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="data/smt_nav_history.json")
    parser.add_argument("--max-new", type=int, default=30)
    args = parser.parse_args()

    cache_path = Path(args.cache)
    cache, added = refresh_nav_history(cache_path, max_new=args.max_new)
    print(f"Added {added} new records; total {len(cache)}.")
    for rec in sorted(cache.values(), key=lambda r: r.valuation_date)[-10:]:
        print(
            f"{rec.valuation_date} cum_fair={rec.cum_fair} cum_par={rec.cum_par}"
            f" ex_fair={rec.ex_fair} ex_par={rec.ex_par}"
        )
