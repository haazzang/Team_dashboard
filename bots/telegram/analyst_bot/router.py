from __future__ import annotations

import re


# Matches symbols like "TSLA", "AAPL", "BRK.B" (case-insensitive; we uppercase before matching).
# Use ASCII lookarounds so it still matches when adjacent to Korean text (e.g., "TSLA애널리스트").
_TICKER_RE = re.compile(r"(?<![A-Z0-9.])([A-Z]{1,6}(?:\.[A-Z]{1,3})?)(?![A-Z0-9.])")

_TICKER_STOPWORDS = {
    "OPENBB",
    "DEEPSEEK",
    "OPENAI",
    "API",
    "AI",
    "EPS",
    "PER",
    "PBR",
    "ROE",
    "YOY",
    "QOQ",
    "USD",
    "KRW",
    "NYSE",
    "NASDAQ",
}

_REVISION_KEYWORDS = {
    "revision",
    "revisions",
    "estimate revision",
    "revision trend",
    "upgrade",
    "upgrades",
    "downgrade",
    "downgrades",
    "rating",
    "리비전",
    "애널리스트",
    "애널리스트리비전",
    "추정치",
    "추정",
    "컨센서스",
    "투자의견",
    "상향",
    "하향",
    "상향조정",
    "하향조정",
    "consensus",
    "analyst",
    "analysts",
}

_PRICE_KEYWORDS = {
    "price",
    "chart",
    "return",
    "performance",
    "주가",
    "차트",
    "수익률",
    "퍼포먼스",
}


def extract_symbol(text: str) -> str | None:
    candidates = [m.group(1) for m in _TICKER_RE.finditer(text.upper())]
    if not candidates:
        return None
    for c in candidates:
        if c in _TICKER_STOPWORDS:
            continue
        return c
    return None


def route_intent(text: str) -> str:
    lower = text.lower()
    if any(k in lower for k in _REVISION_KEYWORDS):
        return "analyst_revisions"
    if any(k in lower for k in _PRICE_KEYWORDS):
        return "price_history"
    return "unknown"


def clean_group_query(text: str, bot_username: str | None) -> str:
    t = (text or "").strip()
    if t.lower().startswith("/ask"):
        parts = t.split(maxsplit=1)
        return parts[1].strip() if len(parts) > 1 else ""
    if bot_username:
        t = re.sub(rf"@{re.escape(bot_username)}\b", "", t, flags=re.IGNORECASE).strip()
    return t


def parse_lookback_days(text: str, default_days: int) -> int:
    """
    Parse Korean/English lookback hints like:
    - "최근 3개월", "6개월", "1년", "2주", "30일", "last 90 days"
    """
    lower = text.lower()
    m = re.search(r"(\d+)\s*(일|주|개월|달|년)\b", text)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit == "일":
            return max(1, n)
        if unit == "주":
            return max(1, n * 7)
        if unit in {"개월", "달"}:
            return max(1, n * 30)
        if unit == "년":
            return max(1, n * 365)

    m2 = re.search(r"(?:last|past)\s*(\d+)\s*(day|days|week|weeks|month|months|year|years)\b", lower)
    if m2:
        n = int(m2.group(1))
        unit = m2.group(2)
        if unit.startswith("day"):
            return max(1, n)
        if unit.startswith("week"):
            return max(1, n * 7)
        if unit.startswith("month"):
            return max(1, n * 30)
        if unit.startswith("year"):
            return max(1, n * 365)

    return default_days
