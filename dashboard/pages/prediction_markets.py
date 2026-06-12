import json
import math
import re
from datetime import datetime
from html import escape
from typing import Any
from zoneinfo import ZoneInfo

import requests
import streamlit as st

SEOUL_TZ = ZoneInfo("Asia/Seoul")
POLY_GAMMA_BASE = "https://gamma-api.polymarket.com"
POLY_CLOB_BASE = "https://clob.polymarket.com"
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
HTTP_TIMEOUT = 20
HTTP_HEADERS = {"User-Agent": "TeamDashboard/1.0"}

POLYMARKET_GROUPS = [
    {
        "title": "US x Iran permanent peace deal by...?",
        "event_slug": "us-x-iran-permanent-peace-deal-by",
        "row_group_titles": ["June 30", "May 31", "April 30", "April 22"],
        "chart_label": None,
    },
    {
        "title": "Trump announces US blockade of Hormuz lifted by...?",
        "event_slug": "trump-announces-us-blockade-of-hormuz-lifted-by",
        "row_group_titles": ["May 31", "April 30"],
        "chart_label": None,
    },
    {
        "title": "US x Iran diplomatic meeting by...?",
        "event_slug": "us-x-iran-diplomatic-meeting-by-329",
        "row_group_titles": ["April 30", "April 22"],
        "chart_label": None,
    },
    {
        "title": "Trump announces end of military operations against Iran by ...?",
        "event_slug": "trump-announces-end-of-military-operations-against-iran-by",
        "row_group_titles": ["June 30", "May 31", "April 30"],
        "chart_label": None,
    },
    {
        "title": "Will the U.S. invade Iran before 2027?",
        "event_slug": "will-the-us-invade-iran-before-2027",
        "row_group_titles": [],
        "single_label": "Yes",
        "chart_label": None,
    },
    {
        "title": "Will the Iranian regime fall by ...?",
        "event_slug": None,
        "standalone_event_slugs": [
            "will-the-iranian-regime-fall-by-the-end-of-2026",
            "will-the-iranian-regime-fall-by-june-30",
        ],
        "chart_label": "Chart - Regime Fall?",
        "chart_url": "https://polymarket.com/event/will-the-iranian-regime-fall-by-the-end-of-2026",
    },
    {
        "title": "Iran leadership change by...?",
        "event_slug": "iran-leadership-change-by",
        "row_group_titles": ["April 30"],
        "chart_label": None,
    },
    {
        "title": "Strait of Hormuz traffic returns to normal by...?",
        "event_slug": None,
        "standalone_event_slugs": [
            "strait-of-hormuz-traffic-returns-to-normal-by-end-of-june",
            "strait-of-hormuz-traffic-returns-to-normal-by-end-of-may",
            "strait-of-hormuz-traffic-returns-to-normal-by-april-30",
        ],
        "chart_label": None,
    },
]

KALSHI_EVENT_TICKER = "KXHORMUZNORM-26MAR17"
KALSHI_ROW_ORDER = [
    "Before Jan 1, 2027",
    "Before Jul 1, 2026",
    "Before Jun 1, 2026",
    "Before May 15, 2026",
    "Before May 1, 2026",
]


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _format_date(value: Any) -> str:
    parsed = _parse_dt(value)
    if not parsed:
        return "-"
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(SEOUL_TZ)
    return f"{parsed.strftime('%B')} {parsed.day}, {parsed.year}"


def _group_item_label(market: dict[str, Any]) -> tuple[str, datetime]:
    group_title = str(market.get("groupItemTitle") or "").strip()
    end_year = (_parse_dt(market.get("endDate")) or datetime.now(tz=SEOUL_TZ)).year
    parsed = datetime.strptime(f"{group_title} {end_year}", "%B %d %Y")
    return f"{group_title}, {end_year}", parsed


def _format_time(value: Any) -> str:
    parsed = _parse_dt(value)
    if not parsed:
        return "--:--"
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(SEOUL_TZ)
    return parsed.strftime("%H:%M")


def _format_price_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def _format_change_pp(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:+.1f}"


def _format_cent(value: float | None) -> str:
    if value is None:
        return "-"
    cents = value * 100
    if math.isclose(cents, round(cents), abs_tol=0.05):
        return f"{int(round(cents))}\u00a2"
    return f"{cents:.1f}\u00a2"


def _format_volume(value: float | None) -> str:
    if value is None:
        return "-"
    abs_value = abs(value)
    for divisor, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs_value >= divisor:
            return f"{value / divisor:.0f}{suffix}"
    return f"{value:.0f}"


def _score_class(value: float | None) -> str:
    if value is None:
        return "muted"
    if value >= 0.5:
        return "bull"
    if value >= 0.15:
        return "mid"
    return "bear"


def _change_class(value: float | None) -> str:
    if value is None:
        return "muted"
    if value > 0:
        return "up"
    if value < 0:
        return "down"
    return "flat"


def _trend_symbol(value: float | None) -> str:
    if value is None or math.isclose(value, 0.0, abs_tol=1e-9):
        return "•"
    return "▲" if value > 0 else "▼"


def _range_position(value: float | None, low: float | None, high: float | None) -> float:
    if value is None or low is None or high is None:
        return 0.5
    if math.isclose(high, low, abs_tol=1e-9):
        return 0.5
    return max(0.0, min(1.0, (value - low) / (high - low)))


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "market"


def _polymarket_market_url(event_slug: str, market_slug: str, market_count: int) -> str:
    if market_count > 1:
        return f"https://polymarket.com/event/{event_slug}/{market_slug}"
    return f"https://polymarket.com/event/{event_slug}"


def _kalshi_event_url(event: dict[str, Any]) -> str:
    series_ticker = str(event.get("series_ticker", "")).lower()
    title_slug = _slugify(str(event.get("title", "")))
    event_ticker = str(event.get("event_ticker", "")).lower()
    return f"https://kalshi.com/markets/{series_ticker}/{title_slug}/{event_ticker}"


def _kalshi_market_url(event: dict[str, Any], market: dict[str, Any]) -> str:
    return f"{_kalshi_event_url(event)}/{str(market.get('ticker', '')).lower()}"


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_polymarket_event(event_slug: str) -> dict[str, Any]:
    response = requests.get(
        f"{POLY_GAMMA_BASE}/events",
        params={"slug": event_slug},
        headers=HTTP_HEADERS,
        timeout=HTTP_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload:
        raise ValueError(f"Polymarket event not found: {event_slug}")
    return payload[0]


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_polymarket_batch_history(token_ids: tuple[str, ...]) -> dict[str, list[dict[str, Any]]]:
    if not token_ids:
        return {}
    response = requests.post(
        f"{POLY_CLOB_BASE}/batch-prices-history",
        json={"markets": list(token_ids), "interval": "1w", "fidelity": 60},
        headers=HTTP_HEADERS,
        timeout=HTTP_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("history", {})


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_kalshi_event(event_ticker: str) -> dict[str, Any]:
    response = requests.get(
        f"{KALSHI_BASE}/events/{event_ticker}",
        headers=HTTP_HEADERS,
        timeout=HTTP_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_kalshi_candles(series_ticker: str, event_ticker: str) -> list[list[dict[str, Any]]]:
    end_ts = int(datetime.now(tz=SEOUL_TZ).timestamp())
    start_ts = end_ts - (8 * 24 * 60 * 60)
    response = requests.get(
        f"{KALSHI_BASE}/series/{series_ticker}/events/{event_ticker}/candlesticks",
        params={"start_ts": start_ts, "end_ts": end_ts, "period_interval": 1440},
        headers=HTTP_HEADERS,
        timeout=HTTP_TIMEOUT,
    )
    response.raise_for_status()
    return response.json().get("market_candlesticks", [])


def _history_range(history_points: list[dict[str, Any]], current_price: float | None) -> tuple[float | None, float | None]:
    prices = [_safe_float(point.get("p")) for point in history_points]
    prices = [price for price in prices if price is not None]
    if current_price is not None:
        prices.append(current_price)
    if not prices:
        return None, None
    return min(prices), max(prices)


def _kalshi_week_metrics(
    candle_sets: list[list[dict[str, Any]]],
    market_index: int,
    market: dict[str, Any],
) -> tuple[float | None, float | None]:
    candles = candle_sets[market_index] if market_index < len(candle_sets) else []
    lows: list[float] = []
    highs: list[float] = []
    closes: list[float] = []
    for candle in candles:
        price_block = candle.get("price", {})
        low = _safe_float(price_block.get("low_dollars"))
        high = _safe_float(price_block.get("high_dollars"))
        close = _safe_float(price_block.get("close_dollars"))
        if low is not None:
            lows.append(low)
        if high is not None:
            highs.append(high)
        if close is not None:
            closes.append(close)

    current_price = _safe_float(market.get("last_price_dollars"))
    if current_price is not None:
        lows.append(current_price)
        highs.append(current_price)

    low_value = min(lows) if lows else None
    high_value = max(highs) if highs else None
    week_change = None
    if current_price is not None and closes:
        baseline = closes[0]
        week_change = current_price - baseline
    return low_value, high_value, week_change


def _build_polymarket_groups() -> tuple[list[dict[str, Any]], list[datetime | None]]:
    groups: list[dict[str, Any]] = []
    token_rows: dict[str, dict[str, Any]] = {}
    updated_times: list[datetime | None] = []

    for config in POLYMARKET_GROUPS:
        rows: list[dict[str, Any]] = []
        chart_url = config.get("chart_url")

        if config.get("event_slug"):
            event = _fetch_polymarket_event(config["event_slug"])
            chart_url = chart_url or f"https://polymarket.com/event/{event['slug']}"
            markets = event.get("markets", [])
            market_count = len(markets)

            if config.get("row_group_titles"):
                selected = []
                wanted = set(config["row_group_titles"])
                for market in markets:
                    if market.get("groupItemTitle") in wanted:
                        selected.append(market)
                selected.sort(key=lambda item: _parse_dt(item.get("endDate")) or datetime.min)
                selected.reverse()
            else:
                selected = markets[:1]

            for market in selected:
                yes_token = json.loads(market.get("clobTokenIds") or "[]")[0]
                if config.get("row_group_titles"):
                    label, sort_key = _group_item_label(market)
                else:
                    label = config.get("single_label") or _format_date(market.get("endDate"))
                    sort_key = _parse_dt(market.get("endDate")) or datetime.min
                row = {
                    "label": label,
                    "sort_key": sort_key,
                    "price": _safe_float(market.get("lastTradePrice")),
                    "change_1d": _safe_float(market.get("oneDayPriceChange")),
                    "change_1w": _safe_float(market.get("oneWeekPriceChange")),
                    "range_low": None,
                    "range_high": None,
                    "volume": _safe_float(market.get("volume")),
                    "updated_time": _format_time(market.get("updatedAt")),
                    "updated_at_obj": _parse_dt(market.get("updatedAt")),
                    "source": "PolyM",
                    "url": _polymarket_market_url(event["slug"], market["slug"], market_count),
                    "token_id": yes_token,
                }
                rows.append(row)
                token_rows[yes_token] = row
                updated_times.append(row["updated_at_obj"])
        else:
            for event_slug in config.get("standalone_event_slugs", []):
                event = _fetch_polymarket_event(event_slug)
                market = event.get("markets", [])[0]
                yes_token = json.loads(market.get("clobTokenIds") or "[]")[0]
                row = {
                    "label": _format_date(market.get("endDate")),
                    "sort_key": _parse_dt(market.get("endDate")) or datetime.min,
                    "price": _safe_float(market.get("lastTradePrice")),
                    "change_1d": _safe_float(market.get("oneDayPriceChange")),
                    "change_1w": _safe_float(market.get("oneWeekPriceChange")),
                    "range_low": None,
                    "range_high": None,
                    "volume": _safe_float(market.get("volume")),
                    "updated_time": _format_time(market.get("updatedAt")),
                    "updated_at_obj": _parse_dt(market.get("updatedAt")),
                    "source": "PolyM",
                    "url": _polymarket_market_url(event["slug"], market["slug"], len(event.get("markets", []))),
                    "token_id": yes_token,
                }
                rows.append(row)
                token_rows[yes_token] = row
                updated_times.append(row["updated_at_obj"])

        if rows:
            rows.sort(key=lambda item: item.get("sort_key", datetime.min), reverse=True)
            groups.append(
                {
                    "title": config["title"],
                    "rows": rows,
                    "chart_label": config.get("chart_label"),
                    "chart_url": chart_url,
                }
            )

    history_payload = _fetch_polymarket_batch_history(tuple(token_rows.keys()))
    for token_id, row in token_rows.items():
        low_value, high_value = _history_range(history_payload.get(token_id, []), row["price"])
        row["range_low"] = low_value
        row["range_high"] = high_value

    return groups, updated_times


def _build_kalshi_group() -> tuple[dict[str, Any], list[datetime | None]]:
    payload = _fetch_kalshi_event(KALSHI_EVENT_TICKER)
    event = payload["event"]
    markets = payload.get("markets", [])
    candle_sets = _fetch_kalshi_candles(event["series_ticker"], event["event_ticker"])
    updated_times: list[datetime | None] = []

    market_by_label = {market.get("yes_sub_title"): (index, market) for index, market in enumerate(markets)}
    rows = []
    for yes_sub_title in KALSHI_ROW_ORDER:
        if yes_sub_title not in market_by_label:
            continue
        market_index, market = market_by_label[yes_sub_title]
        low_value, high_value, week_change = _kalshi_week_metrics(candle_sets, market_index, market)
        updated_at = _parse_dt(market.get("updated_time"))
        row = {
            "label": _kalshi_row_label(yes_sub_title),
            "price": _safe_float(market.get("last_price_dollars")),
            "change_1d": _change_from_prices(market.get("last_price_dollars"), market.get("previous_price_dollars")),
            "change_1w": week_change,
            "range_low": low_value,
            "range_high": high_value,
            "volume": _safe_float(market.get("volume_fp")),
            "updated_time": _format_time(market.get("updated_time")),
            "updated_at_obj": updated_at,
            "source": "Kalshi",
            "url": _kalshi_market_url(event, market),
        }
        rows.append(row)
        updated_times.append(updated_at)

    return (
        {
            "title": str(event.get("title", "")),
            "rows": rows,
            "chart_label": "Chart - Hormuz Normalize?",
            "chart_url": _kalshi_event_url(event),
        },
        updated_times,
    )


def _change_from_prices(current: Any, previous: Any) -> float | None:
    current_value = _safe_float(current)
    previous_value = _safe_float(previous)
    if current_value is None or previous_value is None:
        return None
    return current_value - previous_value


def _kalshi_row_label(yes_sub_title: str) -> str:
    raw = yes_sub_title.replace("Before ", "").strip()
    try:
        parsed = datetime.strptime(raw, "%b %d, %Y")
    except ValueError:
        try:
            parsed = datetime.strptime(raw, "%B %d, %Y")
        except ValueError:
            return f"Hormuz Avg >60 {yes_sub_title}?"

    if parsed.day == 1:
        return f"Hormuz Avg >60 by {parsed.strftime('%b')} {parsed.year}?"
    year_suffix = str(parsed.year)[-2:]
    return f"Hormuz Avg >60 by {parsed.strftime('%b')} {parsed.day} '{year_suffix}?"


def _latest_snapshot_time(*groups: dict[str, Any], extra_times: list[datetime | None]) -> str:
    timestamps = [stamp for stamp in extra_times if stamp is not None]
    for group in groups:
        for row in group.get("rows", []):
            stamp = row.get("updated_at_obj")
            if stamp is not None:
                timestamps.append(stamp)

    if not timestamps:
        return "-"

    latest = max(timestamps)
    if latest.tzinfo is not None:
        latest = latest.astimezone(SEOUL_TZ)
    return latest.strftime("%Y-%m-%d %H:%M:%S KST")


def _render_row(row: dict[str, Any]) -> str:
    price = row.get("price")
    low_value = row.get("range_low")
    high_value = row.get("range_high")
    pos = _range_position(price, low_value, high_value) * 100
    label_class = _score_class(price)
    change_1d = row.get("change_1d")
    change_1w = row.get("change_1w")

    return f"""
        <div class="pm-row">
            <div class="pm-cell pm-label {label_class}">{escape(str(row.get("label", "-")))}</div>
            <div class="pm-cell pm-link"><a href="{escape(str(row.get("url", "#")))}" target="_blank" rel="noopener noreferrer">&#128279;</a></div>
            <div class="pm-cell pm-price {label_class}">{_format_price_pct(price)}</div>
            <div class="pm-cell pm-change {_change_class(change_1d)}">{_format_change_pp(change_1d)}</div>
            <div class="pm-cell pm-change {_change_class(change_1w)}">{_format_change_pp(change_1w)}</div>
            <div class="pm-cell pm-bound low">{_format_cent(low_value)}</div>
            <div class="pm-cell pm-range">
                <div class="pm-range-track">
                    <div class="pm-range-marker" style="left: calc({pos:.2f}% - 4px);"></div>
                </div>
            </div>
            <div class="pm-cell pm-bound high">{_format_cent(high_value)}</div>
            <div class="pm-cell pm-volume">{_format_volume(row.get("volume"))}</div>
            <div class="pm-cell pm-time">{escape(str(row.get("updated_time", "--:--")))}</div>
            <div class="pm-cell pm-source">{escape(str(row.get("source", "")))}</div>
            <div class="pm-cell pm-arrow {_change_class(change_1d)}">{_trend_symbol(change_1d)}</div>
        </div>
    """


def _render_group(group: dict[str, Any]) -> str:
    rows_html = "".join(_render_row(row) for row in group.get("rows", []))
    chart_html = ""
    if group.get("chart_label") and group.get("chart_url"):
        chart_html = f"""
            <div class="pm-chart-row">
                <a href="{escape(str(group['chart_url']))}" target="_blank" rel="noopener noreferrer">
                    &raquo; {escape(str(group['chart_label']))}
                </a>
            </div>
        """
    return f"""
        <div class="pm-group-title">&#9662; {escape(str(group.get("title", "")))}</div>
        {rows_html}
        {chart_html}
    """


def _board_css() -> str:
    return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&family=IBM+Plex+Sans+Condensed:wght@400;600;700&display=swap');

        .prediction-board {
            background: #000000;
            color: #f2efe6;
            border: 1px solid #2f2f2f;
            border-radius: 14px;
            overflow: hidden;
            box-shadow: 0 18px 38px rgba(0, 0, 0, 0.28);
        }

        .prediction-scroll {
            overflow-x: auto;
        }

        .prediction-shell {
            min-width: 1080px;
            font-family: "IBM Plex Mono", monospace;
            font-size: 12px;
            line-height: 1.1;
            background: #000000;
        }

        .prediction-head {
            padding: 0;
        }

        .prediction-title {
            font-family: "IBM Plex Sans Condensed", sans-serif;
            background: #111111;
            color: #ffd94d;
            font-size: 30px;
            font-weight: 700;
            padding: 10px 14px 8px;
            letter-spacing: 0.02em;
        }

        .prediction-topic {
            font-family: "IBM Plex Sans Condensed", sans-serif;
            background: #2c1739;
            color: #bf7dff;
            font-size: 28px;
            font-weight: 700;
            padding: 8px 14px 7px;
        }

        .prediction-section {
            font-family: "IBM Plex Sans Condensed", sans-serif;
            color: #ff4ad7;
            font-size: 26px;
            font-weight: 700;
            padding: 10px 14px 8px;
        }

        .pm-group-title {
            background: #111111;
            color: #f8f5e8;
            font-family: "IBM Plex Sans Condensed", sans-serif;
            font-size: 17px;
            padding: 7px 12px 6px;
            border-top: 1px solid #1d1d1d;
        }

        .pm-row {
            display: grid;
            grid-template-columns: minmax(290px, 1.8fr) 30px 76px 56px 56px 62px 92px 62px 74px 54px 60px 26px;
            align-items: center;
            gap: 0;
            min-height: 31px;
            border-top: 1px solid #090909;
        }

        .pm-row:nth-of-type(odd) {
            background: #0b0b0b;
        }

        .pm-row:nth-of-type(even) {
            background: #161616;
        }

        .pm-cell {
            padding: 5px 8px;
            white-space: nowrap;
        }

        .pm-label {
            font-size: 13px;
            letter-spacing: 0.01em;
        }

        .pm-link a,
        .pm-chart-row a {
            color: #9cb6ff;
            text-decoration: none;
        }

        .pm-link {
            text-align: center;
            font-size: 13px;
        }

        .pm-price,
        .pm-change,
        .pm-bound,
        .pm-volume,
        .pm-time,
        .pm-source,
        .pm-arrow {
            text-align: right;
            font-variant-numeric: tabular-nums;
        }

        .pm-bound.low {
            color: #ffd44d;
        }

        .pm-bound.high {
            color: #90ff84;
        }

        .pm-volume {
            color: #ff60d0;
        }

        .pm-time {
            color: #b6b6b6;
        }

        .pm-source {
            color: #56d4ff;
        }

        .pm-range-track {
            position: relative;
            height: 12px;
            margin: 0 6px;
            border-left: 1px solid #6b5c16;
            border-right: 1px solid #355e19;
            background-image:
                repeating-linear-gradient(90deg, #d1bf48 0 2px, transparent 2px 9px);
            background-position: center;
            background-repeat: repeat-x;
            background-size: auto 2px;
        }

        .pm-range-track::before {
            content: "";
            position: absolute;
            inset: 50% 0 auto 0;
            border-top: 2px dotted #d1bf48;
            transform: translateY(-50%);
        }

        .pm-range-marker {
            position: absolute;
            top: 1px;
            width: 8px;
            height: 10px;
            background: #ffcc33;
            border: 1px solid #fff08d;
            box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.35);
        }

        .pm-chart-row {
            padding: 7px 12px 10px;
            color: #6ce6ff;
            font-style: italic;
        }

        .bull {
            color: #53f06e;
        }

        .mid {
            color: #ffe34b;
        }

        .bear {
            color: #ff5b66;
        }

        .up {
            color: #5df27e;
        }

        .down {
            color: #ff5564;
        }

        .flat,
        .muted {
            color: #b8b8b8;
        }

        .prediction-meta {
            color: #bec8d1;
            margin-bottom: 0.45rem;
        }

        @media (max-width: 960px) {
            .prediction-board {
                border-radius: 10px;
            }

            .prediction-title {
                font-size: 24px;
            }

            .prediction-topic,
            .prediction-section {
                font-size: 20px;
            }
        }
        </style>
    """


def _render_board_html(poly_groups: list[dict[str, Any]], kalshi_group: dict[str, Any]) -> str:
    group_html = "".join(_render_group(group) for group in [*poly_groups[:-1], kalshi_group, poly_groups[-1]])
    return f"""
        {_board_css()}
        <div class="prediction-board">
            <div class="prediction-scroll">
                <div class="prediction-shell">
                    <div class="prediction-head">
                        <div class="prediction-title">Geopolitics &amp; Conflict</div>
                        <div class="prediction-topic">Iran</div>
                        <div class="prediction-section">Most Active</div>
                    </div>
                    {group_html}
                </div>
            </div>
        </div>
    """


def render_prediction_markets_page():
    st.caption("Live snapshot from Polymarket and Kalshi official APIs. Times shown in KST.")

    try:
        poly_groups, poly_updates = _build_polymarket_groups()
        kalshi_group, kalshi_updates = _build_kalshi_group()
    except Exception as err:
        st.error(f"Prediction markets data load failed: {err}")
        return

    snapshot_time = _latest_snapshot_time(*poly_groups, kalshi_group, extra_times=[*poly_updates, *kalshi_updates])
    st.caption(f"Last verified snapshot: {snapshot_time}")
    st.markdown(_render_board_html(poly_groups, kalshi_group), unsafe_allow_html=True)
