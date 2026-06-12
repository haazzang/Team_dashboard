#!/usr/bin/env python3
"""Fetch FMP data for upcoming earnings previews (stable API only).

KST today + next 3 days, filter by non-null estimates, mcap >= $5B, top 15.
Pulls quote / TTM ratios / key metrics / annual estimates / PT consensus /
grades / news / historical earnings / quarterly income in parallel.
Computes NTM blended multiples (0.75*FY+0.25*FY+1).
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ENV_PATH = Path("/Users/hyejinha/Desktop/Workspace/Team/.env")


def load_env():
    if not ENV_PATH.exists():
        return
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()
API_KEY = os.environ.get("FMP_API_KEY")
if not API_KEY:
    print("FMP_API_KEY missing", file=sys.stderr)
    sys.exit(1)

BASE = "https://financialmodelingprep.com/stable"


def get(path: str, params: dict | None = None, retries: int = 3, allow_402: bool = False):
    p = dict(params or {})
    p["apikey"] = API_KEY
    url = f"{BASE}/{path}?" + urllib.parse.urlencode(p)
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "earnings-preview/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as e:  # noqa: PERF203
            if e.code in (402, 403, 404):
                if allow_402:
                    return None
                last = e
                break
            last = e
            time.sleep(0.4 * (i + 1))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(0.4 * (i + 1))
    raise RuntimeError(f"GET failed: {path} :: {last}")


def kst_today():
    return datetime.now(timezone(timedelta(hours=9)))


def earnings_calendar(start: str, end: str):
    return get("earnings-calendar", {"from": start, "to": end}) or []


def market_cap(symbol: str):
    d = get("market-capitalization", {"symbol": symbol})
    if isinstance(d, list) and d:
        return float(d[0].get("marketCap") or 0) or None
    return None


def quote(symbol: str):
    d = get("quote", {"symbol": symbol})
    return d[0] if isinstance(d, list) and d else {}


def profile(symbol: str):
    d = get("profile", {"symbol": symbol})
    return d[0] if isinstance(d, list) and d else {}


def ratios_ttm(symbol: str):
    d = get("ratios-ttm", {"symbol": symbol})
    return d[0] if isinstance(d, list) and d else {}


def key_metrics_ttm(symbol: str):
    d = get("key-metrics-ttm", {"symbol": symbol})
    return d[0] if isinstance(d, list) and d else {}


def analyst_estimates(symbol: str, limit: int = 10):
    # API returns desc-sorted; we pull 10 and let compute_ntm find FY0/FY1
    return get("analyst-estimates", {"symbol": symbol, "period": "annual", "limit": limit}) or []


def price_target_consensus(symbol: str):
    d = get("price-target-consensus", {"symbol": symbol})
    return d[0] if isinstance(d, list) and d else {}


def grades_consensus(symbol: str):
    d = get("grades-consensus", {"symbol": symbol})
    return d[0] if isinstance(d, list) and d else {}


def stock_news(symbol: str, limit: int = 5):
    # News endpoints require higher plan; try graceful fallback
    for path in ("news/stock-latest", "news/press-releases-latest"):
        d = get(path, {"symbols": symbol, "limit": limit}, allow_402=True)
        if isinstance(d, list) and d:
            return d
    return []


def historical_earnings(symbol: str, limit: int = 8):
    return get("earnings", {"symbol": symbol, "limit": limit}) or []


def income_quarterly(symbol: str, limit: int = 5):
    return get("income-statement", {"symbol": symbol, "period": "quarter", "limit": limit}) or []


def fetch_all_for_symbol(symbol: str) -> dict:
    tasks = {
        "quote": (quote, [symbol]),
        "profile": (profile, [symbol]),
        "ratios_ttm": (ratios_ttm, [symbol]),
        "key_metrics_ttm": (key_metrics_ttm, [symbol]),
        "analyst_estimates": (analyst_estimates, [symbol]),
        "price_target": (price_target_consensus, [symbol]),
        "grades": (grades_consensus, [symbol]),
        "news": (stock_news, [symbol]),
        "historical_earnings": (historical_earnings, [symbol]),
        "income_quarterly": (income_quarterly, [symbol]),
    }
    out: dict = {"symbol": symbol, "errors": {}}
    with cf.ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(fn, *args): name for name, (fn, args) in tasks.items()}
        for fut in cf.as_completed(futs):
            name = futs[fut]
            try:
                out[name] = fut.result()
            except Exception as e:  # noqa: BLE001
                out["errors"][name] = str(e)
                out[name] = None
    return out


def compute_ntm(estimates: list[dict], current_year: int) -> dict:
    """Blend 0.75*FY0E + 0.25*FY1E. Stable endpoint fields: revenueAvg, epsAvg, ebitdaAvg, netIncomeAvg."""
    if not estimates:
        return {}

    def yr(e):
        d = e.get("date") or ""
        try:
            return int(d[:4])
        except Exception:
            return 0

    es = sorted(estimates, key=lambda e: e.get("date") or "")
    today_str = datetime.now(timezone(timedelta(hours=9))).date().isoformat()
    # FY0 = first fiscal year-end strictly AFTER today (upcoming fiscal year)
    fy0 = next((e for e in es if (e.get("date") or "") > today_str), None)
    if not fy0:
        return {}
    idx = es.index(fy0)
    fy1 = es[idx + 1] if idx + 1 < len(es) else None
    fy0_yr = yr(fy0)
    if not fy1:
        return {}

    def blend(key):
        v0 = fy0.get(key)
        v1 = fy1.get(key)
        if v0 is None or v1 is None:
            return None
        try:
            return 0.75 * float(v0) + 0.25 * float(v1)
        except Exception:
            return None

    return {
        "fy0_year": fy0_yr,
        "fy1_year": yr(fy1),
        "ntm_revenue": blend("revenueAvg"),
        "ntm_eps": blend("epsAvg"),
        "ntm_ebitda": blend("ebitdaAvg"),
        "ntm_net_income": blend("netIncomeAvg"),
        "fy0": fy0,
        "fy1": fy1,
    }


def main():
    today = kst_today().date()
    start = today.isoformat()
    end = (today + timedelta(days=3)).isoformat()
    print(f"[info] KST window: {start} .. {end}", file=sys.stderr)

    cal = earnings_calendar(start, end)
    print(f"[info] raw calendar rows: {len(cal)}", file=sys.stderr)

    filtered = []
    for row in cal:
        eps_e = row.get("epsEstimated")
        rev_e = row.get("revenueEstimated")
        if eps_e is None or rev_e is None:
            continue
        try:
            float(eps_e); float(rev_e)
        except Exception:
            continue
        filtered.append(row)
    print(f"[info] after estimate filter: {len(filtered)}", file=sys.stderr)

    syms = [r["symbol"] for r in filtered]
    caps: dict[str, float | None] = {}
    with cf.ThreadPoolExecutor(max_workers=15) as ex:
        futs = {ex.submit(market_cap, s): s for s in syms}
        for fut in cf.as_completed(futs):
            s = futs[fut]
            try:
                caps[s] = fut.result()
            except Exception:
                caps[s] = None
    for r in filtered:
        r["marketCap"] = caps.get(r["symbol"])

    big = [r for r in filtered if (r.get("marketCap") or 0) >= 5e9]
    big.sort(key=lambda r: r.get("marketCap") or 0, reverse=True)
    top = big[:15]
    print(f"[info] >=$5B: {len(big)}, taking top {len(top)}", file=sys.stderr)
    for r in top:
        print(f"  - {r['symbol']:<6} mcap=${r['marketCap']/1e9:>7.1f}B  date={r.get('date')}", file=sys.stderr)

    enriched = []
    for r in top:
        sym = r["symbol"]
        print(f"[info] fetching {sym}...", file=sys.stderr)
        bundle = fetch_all_for_symbol(sym)
        bundle["calendar"] = r
        bundle["ntm"] = compute_ntm(bundle.get("analyst_estimates") or [], today.year)
        enriched.append(bundle)

    out_path = Path("/Users/hyejinha/Desktop/Workspace/Team/analysis_outputs/earnings_preview_data.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "generated_at_kst": kst_today().isoformat(),
        "window": {"from": start, "to": end},
        "count": len(enriched),
        "items": enriched,
    }, default=str, indent=2))
    print(f"[ok] wrote {out_path}", file=sys.stderr)
    print(json.dumps({"path": str(out_path), "tickers": [b["symbol"] for b in enriched]}))


if __name__ == "__main__":
    main()
