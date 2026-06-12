#!/usr/bin/env python3
"""Pull FMP earnings calendar + quotes + historicals for a given date and
emit a JSON payload used to build the daily Earnings Recap Notion page.

Usage: earnings_recap_fetch.py YYYY-MM-DD
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.error
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


def get(path: str, params: dict | None = None, retries: int = 3, allow_fail: bool = False):
    p = dict(params or {})
    p["apikey"] = API_KEY
    url = f"{BASE}/{path}?" + urllib.parse.urlencode(p)
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "earnings-recap/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as e:
            if e.code in (402, 403, 404):
                if allow_fail:
                    return None
                last = e
                break
            last = e
            time.sleep(0.4 * (i + 1))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(0.4 * (i + 1))
    if allow_fail:
        return None
    raise RuntimeError(f"GET failed: {path} :: {last}")


def earnings_calendar(date_str: str):
    return get("earnings-calendar", {"from": date_str, "to": date_str}) or []


def quote(symbol: str):
    d = get("quote", {"symbol": symbol}, allow_fail=True)
    if isinstance(d, list) and d:
        return d[0]
    return None


def profile(symbol: str):
    d = get("profile", {"symbol": symbol}, allow_fail=True)
    if isinstance(d, list) and d:
        return d[0]
    return None


def historical_price(symbol: str, date_str: str):
    # +/- 7 days to capture prior trading day close
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    frm = (dt - timedelta(days=7)).strftime("%Y-%m-%d")
    to = (dt + timedelta(days=2)).strftime("%Y-%m-%d")
    d = get("historical-price-eod/light", {"symbol": symbol, "from": frm, "to": to}, allow_fail=True)
    if not isinstance(d, list):
        return None
    # Sort ascending
    rows = sorted(d, key=lambda r: r.get("date", ""))
    return rows


def aftermarket_quote(symbol: str):
    d = get("aftermarket-quote", {"symbol": symbol}, allow_fail=True)
    if isinstance(d, list) and d:
        return d[0]
    return None


def classify_timing(time_field: str | None, hour_field: str | None = None) -> str:
    t = (time_field or "").lower().strip()
    if t in ("bmo", "before-market-open", "before market open"):
        return "BMO"
    if t in ("amc", "after-market-close", "after market close"):
        return "AMC"
    if t == "dmh" or "during" in t:
        return "DMH"
    return "UNKNOWN"


def fetch_one(item: dict, date_str: str) -> dict:
    sym = item.get("symbol")
    out = {"symbol": sym, "raw_calendar": item}
    try:
        with cf.ThreadPoolExecutor(max_workers=4) as ex:
            f_q = ex.submit(quote, sym)
            f_p = ex.submit(profile, sym)
            f_h = ex.submit(historical_price, sym, date_str)
            f_a = ex.submit(aftermarket_quote, sym)
            out["quote"] = f_q.result()
            out["profile"] = f_p.result()
            out["historical"] = f_h.result()
            out["aftermarket"] = f_a.result()
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)
    return out


def main():
    if len(sys.argv) < 2:
        print("usage: earnings_recap_fetch.py YYYY-MM-DD", file=sys.stderr)
        sys.exit(2)
    date_str = sys.argv[1]
    cal = earnings_calendar(date_str)
    # Filter: epsActual non-null (actually reported)
    reported = [c for c in cal if c.get("epsActual") is not None]
    # Dedup by symbol
    seen = set()
    uniq = []
    for c in reported:
        s = c.get("symbol")
        if not s or s in seen:
            continue
        seen.add(s)
        uniq.append(c)

    results = []
    failures = []
    # Parallelize across symbols (each symbol fan-out inside fetch_one)
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_one, c, date_str): c.get("symbol") for c in uniq}
        for fut in cf.as_completed(futs):
            sym = futs[fut]
            try:
                r = fut.result()
                if r.get("error"):
                    failures.append({"symbol": sym, "error": r["error"]})
                results.append(r)
            except Exception as e:  # noqa: BLE001
                failures.append({"symbol": sym, "error": str(e)})

    payload = {
        "date": date_str,
        "n_calendar": len(cal),
        "n_reported": len(uniq),
        "results": results,
        "failures": failures,
    }
    out_path = Path(f"/Users/hyejinha/Desktop/Workspace/Team/analysis_outputs/earnings_recap_{date_str}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps({"date": date_str, "n_calendar": len(cal), "n_reported": len(uniq),
                      "n_failures": len(failures), "output": str(out_path)}))


if __name__ == "__main__":
    main()
