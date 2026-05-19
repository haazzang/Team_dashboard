#!/usr/bin/env python3
import argparse
import datetime as dt
import email
import email.header
import imaplib
import json
import math
import os
import re
import sys
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


EXCEL_EXTENSIONS = {".xlsx", ".xlsm"}
SUPPORTED_EXTENSIONS = EXCEL_EXTENSIONS | {".csv"}


ALIASES = {
    "ticker": [
        "ticker",
        "bbg ticker",
        "bloomberg ticker",
        "ric",
        "underlying",
        "component underlying",
        "component underlyings",
        "security",
        "instrument",
        "description",
    ],
    "quantity": [
        "quantity",
        "qty",
        "number of shares",
        "shares",
        "units",
        "position",
        "notional quantity",
    ],
    "trade_price": [
        "trade price",
        "book price",
        "initial price",
        "cost price",
        "open price",
        "price traded",
    ],
    "market_value": [
        "market value",
        "market val",
        "mv",
        "current market value",
        "notional market value",
        "market value usd",
        "usd market value",
    ],
    "current_price": [
        "market price",
        "current price",
        "close price",
        "closing price",
        "last price",
        "underlying price",
        "price",
    ],
    "country": [
        "country",
        "country code",
        "risk country",
        "exchange country",
        "domicile",
    ],
    "currency": [
        "currency",
        "ccy",
        "settlement ccy",
        "settlement_ccy",
        "market value currency",
        "market value ccy",
        "price currency",
    ],
}


@dataclass
class ReportPnL:
    report_date: dt.date
    report_file: str
    short_count: int
    short_market_value: float
    short_cost_market_value: float
    short_pnl: float
    smt_current_price: float | None
    smt_price_source: str | None
    smt_pnl: float | None
    pair_pnl: float
    short_basket_detail: list[dict[str, Any]] | None = None


class MarketDataUnavailable(Exception):
    pass


def die(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def compact(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", norm_text(value))


def norm_ticker(value: Any) -> str:
    text = str(value).strip().upper()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+(EQUITY|COMDTY|CURNCY|INDEX)$", "", text)
    return text


def parse_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "-"}:
        return None
    neg = text.startswith("(") and text.endswith(")")
    text = text.replace(",", "")
    text = re.sub(r"[^0-9.\-]", "", text)
    if text in {"", "-", ".", "-."}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return -abs(number) if neg else number


def first_number(values: Iterable[Any]) -> float | None:
    for value in values:
        number = parse_number(value)
        if number is not None:
            return number
    return None


def decode_header(value: str | None) -> str:
    if not value:
        return ""
    parts = email.header.decode_header(value)
    out = []
    for text, enc in parts:
        if isinstance(text, bytes):
            out.append(text.decode(enc or "utf-8", errors="replace"))
        else:
            out.append(text)
    return "".join(out)


def sanitize_filename(name: str) -> str:
    name = name.replace("/", "_").replace("\\", "_")
    return re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" .") or "attachment"


def parse_email_date(message: email.message.Message) -> dt.date:
    raw = message.get("Date")
    if raw:
        parsed = email.utils.parsedate_to_datetime(raw)
        if parsed:
            return parsed.date()
    return dt.date.today()


def fetch_gmail_attachments(config: dict[str, Any]) -> list[Path]:
    email_cfg = config["email"]
    user = os.environ.get(email_cfg.get("imap_user_env", "GMAIL_USER"), "")
    password = os.environ.get(email_cfg.get("imap_password_env", "GMAIL_APP_PASSWORD"), "")
    if not user or not password:
        die("Set Gmail IMAP credentials in environment variables before --fetch-gmail.")

    host = email_cfg.get("imap_host", "imap.gmail.com")
    port = int(email_cfg.get("imap_port", 993))
    mailbox = email_cfg.get("mailbox", "INBOX")
    subject = email_cfg["subject"]
    since_date = dt.date.fromisoformat(email_cfg.get("since", config["pair"]["short_start_date"]))
    since_imap = since_date.strftime("%d-%b-%Y")
    download_dir = Path(email_cfg.get("download_dir", "reports"))
    download_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    with imaplib.IMAP4_SSL(host, port) as imap:
        imap.login(user, password)
        imap.select(mailbox)
        status, data = imap.uid("search", None, f'(SINCE {since_imap} SUBJECT "{subject}")')
        if status != "OK":
            die("Gmail IMAP search failed.")
        uids = data[0].split()
        for uid in uids:
            status, msg_data = imap.uid("fetch", uid, "(RFC822)")
            if status != "OK":
                continue
            raw = next((part[1] for part in msg_data if isinstance(part, tuple)), None)
            if not raw:
                continue
            msg = email.message_from_bytes(raw)
            message_date = parse_email_date(msg)
            uid_text = uid.decode("ascii", errors="ignore")
            for part in msg.walk():
                if part.get_content_maintype() == "multipart":
                    continue
                filename = decode_header(part.get_filename())
                if not filename:
                    continue
                ext = Path(filename).suffix.lower()
                if ext not in SUPPORTED_EXTENSIONS and ext != ".xls":
                    continue
                clean_name = sanitize_filename(filename)
                out = download_dir / f"{message_date.isoformat()}__uid{uid_text}__{clean_name}"
                if out.exists():
                    saved.append(out)
                    continue
                payload = part.get_payload(decode=True)
                if payload:
                    out.write_bytes(payload)
                    saved.append(out)
        imap.logout()
    return saved


def sheet_name_match(available: list[str], target: str) -> str:
    target_c = compact(target)
    if not available:
        die("Workbook has no sheets.")
    scored: list[tuple[int, str]] = []
    for sheet in available:
        sheet_c = compact(sheet)
        score = 0
        if sheet_c == target_c:
            score = 100
        elif target_c in sheet_c or (len(sheet_c) > 3 and sheet_c in target_c):
            score = 80
        elif (
            "underlying" in sheet_c
            and ("component" in sheet_c or "componet" in sheet_c)
            and "underlying" in target_c
        ):
            score = 70
        elif sheet_c == "und" and target_c == "und":
            score = 100
        scored.append((score, sheet))
    scored.sort(reverse=True)
    if scored[0][0] <= 0:
        die(f"Could not find sheet matching '{target}'. Available sheets: {available}")
    return scored[0][1]


def read_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path, header=None)
    if ext == ".xls":
        die(".xls files need xlrd, which is not installed in the bundled runtime. Save as .xlsx/.xlsm.")
    return pd.read_excel(path, sheet_name=sheet_name, header=None, engine="openpyxl")


def workbook_sheets(path: Path) -> list[str]:
    ext = path.suffix.lower()
    if ext == ".csv":
        return ["csv"]
    if ext == ".xls":
        die(".xls files need xlrd, which is not installed in the bundled runtime. Save as .xlsx/.xlsm.")
    return list(pd.ExcelFile(path, engine="openpyxl").sheet_names)


def alias_score(cell: Any, aliases: list[str]) -> int:
    c = compact(cell)
    if not c:
        return 0
    score = 0
    for alias in aliases:
        a = compact(alias)
        if c == a:
            score = max(score, 10)
        elif a in c or c in a:
            score = max(score, 5)
    return score


def find_header_row(raw: pd.DataFrame, required: list[str]) -> int:
    best_row = 0
    best_score = -1
    max_rows = min(40, len(raw))
    for idx in range(max_rows):
        row = raw.iloc[idx].tolist()
        score = 0
        hits = 0
        for key in required:
            key_score = max(alias_score(cell, ALIASES[key]) for cell in row)
            if key_score:
                hits += 1
                score += key_score
        non_empty = sum(1 for cell in row if norm_text(cell))
        score += min(non_empty, 8)
        if hits >= min(2, len(required)) and score > best_score:
            best_score = score
            best_row = idx
    return best_row


def make_unique_headers(headers: list[Any]) -> list[str]:
    out: list[str] = []
    counts: dict[str, int] = {}
    for idx, header in enumerate(headers):
        base = norm_text(header)
        if not base or base == "nan":
            base = f"column_{idx + 1}"
        counts[base] = counts.get(base, 0) + 1
        out.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
    return out


def table_from_raw(raw: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    header_row = find_header_row(raw, required)
    headers = make_unique_headers(raw.iloc[header_row].tolist())
    table = raw.iloc[header_row + 1 :].copy()
    table.columns = headers
    table = table.dropna(how="all")
    return table.reset_index(drop=True)


def find_column(df: pd.DataFrame, key: str, required: bool = True) -> str | None:
    best_col = None
    best_score = 0
    for col in df.columns:
        score = alias_score(col, ALIASES[key])
        if score > best_score:
            best_col = col
            best_score = score
    if required and not best_col:
        die(f"Could not identify required column '{key}'. Columns: {list(df.columns)}")
    return best_col


def row_ticker(value: Any) -> str:
    ticker = norm_ticker(value)
    ticker = re.sub(r"\s+US$", " US", ticker)
    ticker = re.sub(r"\s+LN$", " LN", ticker)
    return ticker


def same_ticker(a: Any, b: Any) -> bool:
    aa = row_ticker(a)
    bb = row_ticker(b)
    return aa == bb or aa.replace(" ", "") == bb.replace(" ", "")


def report_date_from_path(path: Path) -> dt.date:
    text = path.name
    match = re.search(r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)", text)
    if match:
        y, m, d = map(int, match.groups())
        return dt.date(y, m, d)
    return dt.datetime.fromtimestamp(path.stat().st_mtime).date()


def parse_date(value: Any) -> dt.date | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def report_date_from_table(df: pd.DataFrame) -> dt.date | None:
    for col in df.columns:
        c = compact(col)
        if c in {"valdate", "valuationdate", "reportdate", "date"}:
            for value in df[col]:
                parsed = parse_date(value)
                if parsed:
                    return parsed
    return None


def signed_market_value(market_value: float, quantity: float) -> float:
    if quantity == 0:
        return market_value
    if market_value == 0:
        return market_value
    if math.copysign(1, market_value) == math.copysign(1, quantity):
        return market_value
    return math.copysign(abs(market_value), quantity)


def build_trade_price_map(und: pd.DataFrame) -> dict[str, float]:
    ticker_col = find_column(und, "ticker")
    trade_col = find_column(und, "trade_price")
    out: dict[str, float] = {}
    for _, row in und.iterrows():
        ticker = row_ticker(row[ticker_col])
        price = parse_number(row[trade_col])
        if ticker and price is not None:
            out[ticker] = price
            out[ticker.replace(" ", "")] = price
    return out


def lookup_trade_price(trade_prices: dict[str, float], ticker: str) -> float | None:
    return trade_prices.get(ticker) or trade_prices.get(ticker.replace(" ", ""))


def is_us_short_row(
    row: pd.Series,
    ticker_col: str,
    qty_col: str,
    country_col: str | None,
    currency_col: str | None,
    suffix: str,
) -> bool:
    qty = parse_number(row[qty_col])
    if qty is None or qty >= 0:
        return False
    if currency_col and norm_text(row[currency_col]) == "usd":
        return True
    ticker = row_ticker(row[ticker_col])
    if ticker.endswith(f" {suffix.upper()}"):
        return True
    if country_col:
        country = norm_text(row[country_col])
        return country in {"us", "usa", "united states", "united states of america"}
    return False


def find_smt_current_price(component: pd.DataFrame, cfg: dict[str, Any]) -> float | None:
    """Return SMT LN current price in the SAME unit as cfg['smt_initial_price'].

    The swap report quotes SMT in GBP (e.g. 14.39), while initial_price is stored
    in GBp (pence, e.g. 1433.5082). When the row currency is GBP we multiply by
    100 to convert to pence.
    """
    ticker_col = find_column(component, "ticker")
    market_value_col = find_column(component, "market_value", required=False)
    quantity_col = find_column(component, "quantity", required=False)
    currency_col = find_column(component, "currency", required=False)
    current_price_col = find_column(component, "current_price", required=False)
    if current_price_col and re.search(r"(trade|book|initial|cost|open)", compact(current_price_col)):
        current_price_col = None
    smt_ticker = cfg["smt_ticker"]
    smt_shares = float(cfg["smt_shares"])
    scale = float(cfg.get("smt_price_scale", 100.0))

    candidates = component[component[ticker_col].map(lambda x: same_ticker(x, smt_ticker))]
    if candidates.empty:
        return None
    row = candidates.iloc[0]
    currency = norm_text(row[currency_col]) if currency_col else ""
    pence_factor = 100.0 if currency == "gbp" else 1.0

    if current_price_col:
        price = parse_number(row[current_price_col])
        if price is not None:
            return price * pence_factor
    if market_value_col:
        mv = parse_number(row[market_value_col])
        if mv is not None:
            shares = smt_shares
            if quantity_col:
                qty = parse_number(row[quantity_col])
                if qty:
                    shares = abs(qty)
            return abs(mv) / shares * scale if shares else None
    return None


def extract_price_from_fmp_payload(data: Any, report_date: dt.date) -> float | None:
    if isinstance(data, dict):
        if isinstance(data.get("historical"), list):
            return extract_price_from_fmp_payload(data["historical"], report_date)
        for key in ("close", "price", "adjClose"):
            price = parse_number(data.get(key))
            if price is not None:
                return price
    if isinstance(data, list):
        for row in data:
            if not isinstance(row, dict):
                continue
            row_date = parse_date(row.get("date"))
            if row_date and row_date != report_date:
                continue
            for key in ("close", "price", "adjClose"):
                price = parse_number(row.get(key))
                if price is not None:
                    return price
    return None


def fetch_fmp_eod_price(symbol: str, report_date: dt.date, api_key: str) -> float:
    params = {
        "symbol": symbol,
        "from": report_date.isoformat(),
        "to": report_date.isoformat(),
        "apikey": api_key,
    }
    endpoints = [
        "https://financialmodelingprep.com/stable/historical-price-eod/light",
        "https://financialmodelingprep.com/stable/historical-price-eod/full",
    ]
    errors: list[str] = []
    for endpoint in endpoints:
        url = endpoint + "?" + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            errors.append(f"{endpoint.rsplit('/', 1)[-1]} HTTP {exc.code}")
            continue
        except Exception as exc:
            errors.append(f"{endpoint.rsplit('/', 1)[-1]} {type(exc).__name__}: {exc}")
            continue
        price = extract_price_from_fmp_payload(data, report_date)
        if price is not None:
            return price
        errors.append(f"{endpoint.rsplit('/', 1)[-1]} returned no close/price")
    raise MarketDataUnavailable(
        f"FMP EOD price unavailable for {symbol} on {report_date.isoformat()}: {'; '.join(errors)}"
    )


def fetch_yahoo_eod_price(symbol: str, report_date: dt.date) -> float:
    start = report_date - dt.timedelta(days=3)
    end = report_date + dt.timedelta(days=2)
    period1 = int(dt.datetime.combine(start, dt.time(), dt.timezone.utc).timestamp())
    period2 = int(dt.datetime.combine(end, dt.time(), dt.timezone.utc).timestamp())
    url = (
        "https://query2.finance.yahoo.com/v8/finance/chart/"
        + urllib.parse.quote(symbol)
        + "?"
        + urllib.parse.urlencode(
            {
                "period1": period1,
                "period2": period2,
                "interval": "1d",
                "events": "history",
                "includeAdjustedClose": "true",
            }
        )
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise MarketDataUnavailable(f"Yahoo EOD price unavailable for {symbol}: HTTP {exc.code}") from exc
    except Exception as exc:
        raise MarketDataUnavailable(
            f"Yahoo EOD price unavailable for {symbol}: {type(exc).__name__}: {exc}"
        ) from exc

    result = ((data.get("chart") or {}).get("result") or [None])[0]
    if not isinstance(result, dict):
        raise MarketDataUnavailable(f"Yahoo EOD price unavailable for {symbol}: empty chart result")
    timestamps = result.get("timestamp") or []
    quote = (((result.get("indicators") or {}).get("quote") or [{}])[0])
    closes = quote.get("close") or []
    for timestamp, close in zip(timestamps, closes):
        row_date = dt.datetime.fromtimestamp(timestamp, dt.timezone.utc).date()
        if row_date != report_date:
            continue
        price = parse_number(close)
        if price is not None:
            return price
    raise MarketDataUnavailable(f"Yahoo EOD price unavailable for {symbol} on {report_date.isoformat()}")


def fetch_market_price(config: dict[str, Any], report_date: dt.date) -> tuple[float | None, str | None]:
    market_cfg = config.get("market_data", {})
    provider = market_cfg.get("provider")
    fallback_provider = market_cfg.get("fallback_provider")
    errors: list[str] = []

    if provider == "fmp":
        env_name = market_cfg.get("fmp_api_key_env", "FMP_API_KEY")
        api_key = os.environ.get(env_name, "")
        symbol = market_cfg.get("smt_symbol", config["pair"].get("smt_ticker", "SMT.L"))
        if not api_key:
            errors.append(f"{env_name} is not set")
        else:
            try:
                return fetch_fmp_eod_price(symbol, report_date, api_key), f"FMP {symbol} close"
            except MarketDataUnavailable as exc:
                errors.append(str(exc))
    elif provider == "yahoo":
        symbol = market_cfg.get("yahoo_symbol", market_cfg.get("smt_symbol", "SMT.L"))
        try:
            return fetch_yahoo_eod_price(symbol, report_date), f"Yahoo {symbol} close"
        except MarketDataUnavailable as exc:
            errors.append(str(exc))
    elif provider:
        errors.append(f"Unsupported market data provider: {provider}")

    if fallback_provider == "yahoo":
        symbol = market_cfg.get("yahoo_symbol", market_cfg.get("smt_symbol", "SMT.L"))
        try:
            source = f"Yahoo {symbol} close"
            if errors:
                source += " (fallback)"
            return fetch_yahoo_eod_price(symbol, report_date), source
        except MarketDataUnavailable as exc:
            errors.append(str(exc))

    if market_cfg.get("required"):
        die("; ".join(errors) if errors else "Market data unavailable.")
    return None, None


def normalize_smt_current_price(price: float | None, pair_cfg: dict[str, Any]) -> float | None:
    if price is None:
        return None
    initial = float(pair_cfg["smt_initial_price"])
    if initial > 100 and 0 < price < initial / 10:
        return price * 100.0
    return price


def calculate_report(path: Path, config: dict[str, Any]) -> ReportPnL:
    pair_cfg = config["pair"]
    sheets = workbook_sheets(path)
    und_sheet = sheet_name_match(sheets, pair_cfg.get("und_sheet", "und"))
    comp_sheet = sheet_name_match(sheets, pair_cfg.get("component_sheet", "component underlyings"))

    und = table_from_raw(read_sheet(path, und_sheet), ["ticker", "trade_price"])
    component = table_from_raw(read_sheet(path, comp_sheet), ["ticker", "quantity", "market_value"])

    trade_prices = build_trade_price_map(und)
    ticker_col = find_column(component, "ticker")
    qty_col = find_column(component, "quantity")
    mv_col = find_column(component, "market_value")
    country_col = find_column(component, "country", required=False)
    currency_col = find_column(component, "currency", required=False)
    comp_trade_col = find_column(component, "trade_price", required=False)

    report_date = report_date_from_table(und) or report_date_from_path(path)
    short_start = dt.date.fromisoformat(pair_cfg.get("short_start_date", "2026-05-13"))
    short_price_scale = float(pair_cfg.get("short_price_scale", 1.0))
    short_suffix = pair_cfg.get("short_ticker_country_suffix", "US")

    short_count = 0
    short_mv = 0.0
    short_cost = 0.0
    short_detail: list[dict[str, Any]] = []
    name_col = None
    for col in component.columns:
        c = compact(col)
        if c == "name" or c == "securityname":
            name_col = col
            break
    current_price_col = find_column(component, "current_price", required=False)
    if current_price_col and re.search(r"(trade|book|initial|cost|open)", compact(current_price_col)):
        current_price_col = None
    if report_date >= short_start:
        for _, row in component.iterrows():
            if not is_us_short_row(row, ticker_col, qty_col, country_col, currency_col, short_suffix):
                continue
            ticker = row_ticker(row[ticker_col])
            qty = parse_number(row[qty_col])
            mv = parse_number(row[mv_col])
            if qty is None or mv is None:
                continue
            trade_price = lookup_trade_price(trade_prices, ticker)
            if trade_price is None and comp_trade_col:
                trade_price = parse_number(row[comp_trade_col])
            if trade_price is None:
                die(f"Missing trade price for short component {ticker} in {path.name}")
            short_count += 1
            signed_mv = signed_market_value(mv, qty)
            cost_mv = qty * trade_price / short_price_scale
            short_mv += signed_mv
            short_cost += cost_mv
            short_detail.append({
                "ticker": ticker,
                "name": str(row[name_col]) if name_col and pd.notna(row[name_col]) else "",
                "quantity": qty,
                "trade_price": trade_price,
                "current_price": parse_number(row[current_price_col]) if current_price_col else None,
                "market_value_usd": signed_mv,
                "cost_market_value_usd": cost_mv,
                "pnl_usd": signed_mv - cost_mv,
            })

    short_pnl = short_mv - short_cost

    smt_current_price, smt_price_source = fetch_market_price(config, report_date)
    if smt_current_price is None:
        smt_current_price = normalize_smt_current_price(find_smt_current_price(component, pair_cfg), pair_cfg)
        smt_price_source = "swap report" if smt_current_price is not None else None
    smt_pnl = None
    if smt_current_price is not None:
        smt_pnl = (
            (smt_current_price - float(pair_cfg["smt_initial_price"]))
            * float(pair_cfg["smt_shares"])
            / float(pair_cfg.get("smt_price_scale", 100.0))
        )
    pair_pnl = short_pnl + (smt_pnl or 0.0)
    return ReportPnL(
        report_date=report_date,
        report_file=str(path),
        short_count=short_count,
        short_market_value=short_mv,
        short_cost_market_value=short_cost,
        short_pnl=short_pnl,
        smt_current_price=smt_current_price,
        smt_price_source=smt_price_source,
        smt_pnl=smt_pnl,
        pair_pnl=pair_pnl,
        short_basket_detail=short_detail,
    )


def collect_reports(args: argparse.Namespace, config: dict[str, Any]) -> list[Path]:
    reports: list[Path] = []
    if args.fetch_gmail:
        reports.extend(fetch_gmail_attachments(config))
    if args.input:
        reports.extend(Path(p) for p in args.input)
    reports_dir = Path(args.reports_dir or config["email"].get("download_dir", "reports"))
    if reports_dir.exists():
        for path in reports_dir.iterdir():
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                reports.append(path)
    unique = sorted({p.resolve() for p in reports}, key=lambda p: (report_date_from_path(p), p.name))
    return unique


def results_to_frame(results: list[ReportPnL]) -> pd.DataFrame:
    rows = [
        {
            "report_date": r.report_date.isoformat(),
            "report_file": r.report_file,
            "short_count": r.short_count,
            "short_market_value": r.short_market_value,
            "short_cost_market_value": r.short_cost_market_value,
            "short_pnl": r.short_pnl,
            "smt_current_price": r.smt_current_price,
            "smt_price_source": r.smt_price_source,
            "smt_pnl": r.smt_pnl,
            "pair_pnl": r.pair_pnl,
        }
        for r in results
    ]
    df = pd.DataFrame(rows).sort_values(["report_date", "report_file"]).reset_index(drop=True)
    if not df.empty:
        df["daily_pair_pnl_change"] = df["pair_pnl"].diff()
        df.loc[0, "daily_pair_pnl_change"] = df.loc[0, "pair_pnl"]
    return df


def fmt_money(value: Any, currency: str = "USD") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{currency} {float(value):,.2f}"


def build_telegram_message(df: pd.DataFrame, config: dict[str, Any]) -> str:
    latest = df.iloc[-1]
    currency = config["pair"].get("currency", "USD")
    lines = [
        "Long/Short Pair PnL Update",
        f"Latest date: {latest['report_date']}",
        f"Latest pair PnL: {fmt_money(latest['pair_pnl'], currency)}",
        f"Latest daily change: {fmt_money(latest['daily_pair_pnl_change'], currency)}",
        f"Short names: {int(latest['short_count'])}",
        f"SMT price source: {latest['smt_price_source'] if pd.notna(latest['smt_price_source']) else 'n/a'}",
        "",
        "Date | Short | SMT | Pair | Daily chg",
    ]
    for _, row in df.tail(10).iterrows():
        lines.append(
            " | ".join(
                [
                    str(row["report_date"]),
                    fmt_money(row["short_pnl"], currency),
                    fmt_money(row["smt_pnl"], currency),
                    fmt_money(row["pair_pnl"], currency),
                    fmt_money(row["daily_pair_pnl_change"], currency),
                ]
            )
        )
    return "\n".join(lines)


def send_telegram(text: str, config: dict[str, Any]) -> None:
    telegram_cfg = config.get("telegram", {})
    token = os.environ.get(telegram_cfg.get("bot_token_env", "TELEGRAM_BOT_TOKEN"), "")
    chat_id = os.environ.get(telegram_cfg.get("chat_id_env", "TELEGRAM_CHAT_ID"), "")
    if not token or not chat_id:
        die("Set Telegram bot token/chat id environment variables before --send-telegram.")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST")
    with urllib.request.urlopen(req, timeout=20) as response:
        body = response.read().decode("utf-8", errors="replace")
        if response.status >= 300:
            die(f"Telegram send failed: {response.status} {body}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Calculate long/short pair PnL from swap EOD reports.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--fetch-gmail", action="store_true")
    parser.add_argument("--send-telegram", action="store_true")
    parser.add_argument("--reports-dir")
    parser.add_argument("--input", action="append", help="Path to a local report file. Can be used multiple times.")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        die(f"Config file not found: {config_path}. Copy config.example.json to config.json first.")
    config = load_config(config_path)
    reports = collect_reports(args, config)
    if not reports:
        die("No report files found. Use --fetch-gmail, --input, or place reports in the configured reports dir.")

    results = [calculate_report(path, config) for path in reports]
    df = results_to_frame(results)
    output_dir = Path(config.get("outputs", {}).get("dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / config.get("outputs", {}).get("csv", "pair_pnl.csv")
    df.to_csv(out_csv, index=False)

    message = build_telegram_message(df, config)
    print(message)
    print(f"CSV: {out_csv}")
    if args.send_telegram:
        send_telegram(message, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
