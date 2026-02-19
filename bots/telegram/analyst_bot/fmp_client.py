from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import pandas as pd


class FMPClientError(RuntimeError):
    pass


class FMPAPIError(FMPClientError):
    pass


def _normalize_date(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (dt.datetime, dt.date)):
        return value.strftime("%Y-%m-%d")
    return str(value).strip() or None


def _parse_date_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    for col in candidates:
        if col in df.columns:
            s = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_convert(None)
            if s.notna().any():
                return s
    return None


@dataclass(frozen=True)
class FMPQueryResult:
    df: pd.DataFrame
    endpoint: str


class FMPClient:
    def __init__(self, *, api_key: str, base_url: str = "https://financialmodelingprep.com") -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    @staticmethod
    def _extract_error_message(data: Any) -> str | None:
        if not isinstance(data, dict):
            return None
        for key in ("Error Message", "error", "message"):
            if key in data and data[key]:
                return str(data[key]).strip()
        return None

    def _get_json(self, path: str, params: dict[str, Any]) -> Any:
        try:
            import requests  # type: ignore
        except ModuleNotFoundError as e:  # pragma: no cover
            raise FMPClientError("Missing dependency: `requests`. Install with `pip install requests`.") from e

        url = f"{self._base_url}{path}"
        qp = dict(params)
        qp["apikey"] = self._api_key
        try:
            resp = requests.get(url, params=qp, timeout=25)
        except Exception as e:
            raise FMPAPIError(f"FMP request failed for endpoint '{path}': {e}") from e

        if resp.status_code != 200:
            body = (resp.text or "").strip()
            snippet = body[:300] + ("..." if len(body) > 300 else "")
            raise FMPAPIError(f"FMP API error {resp.status_code} for endpoint '{path}': {snippet}")

        try:
            data = resp.json()
        except Exception as e:
            raise FMPAPIError(f"Invalid JSON from FMP endpoint '{path}': {e}") from e

        err_msg = self._extract_error_message(data)
        if err_msg:
            raise FMPAPIError(f"FMP API error for endpoint '{path}': {err_msg}")
        return data

    def equity_analyst_revisions(
        self,
        symbol: str,
        *,
        start_date: str | dt.date | dt.datetime | None = None,
        end_date: str | dt.date | dt.datetime | None = None,
    ) -> FMPQueryResult:
        """
        Approximate "analyst revision trend" using FMP upgrades/downgrades feed.
        Docs: https://site.financialmodelingprep.com/developer/docs
        """
        start = _normalize_date(start_date)
        end = _normalize_date(end_date)
        # NOTE: FMP deprecated many /api/v3 legacy endpoints for new users (Aug 31, 2025).
        # Prefer the newer stable endpoints.
        candidates: list[tuple[str, dict[str, Any]]] = [
            ("/stable/upgrades-downgrades", {"symbol": symbol}),
            (f"/stable/upgrades-downgrades/{symbol}", {}),
            ("/api/v4/upgrades-downgrades", {"symbol": symbol}),
            ("/api/v3/upgrades-downgrades", {"symbol": symbol}),
        ]

        last_err: Exception | None = None
        for path, params in candidates:
            try:
                data = self._get_json(path, params)
                rows: Any = data
                if isinstance(data, dict) and "data" in data:
                    rows = data.get("data")
                if not isinstance(rows, list):
                    raise FMPAPIError(f"Unexpected response shape from '{path}': {type(data)}")
                df = pd.DataFrame(rows)
                if df.empty:
                    continue
                dt_s = _parse_date_series(df, ["publishedDate", "published_date", "date"])
                if dt_s is not None:
                    df = df.assign(_dt=dt_s).sort_values("_dt").drop(columns=["_dt"])
                    if start:
                        df = df[dt_s >= pd.to_datetime(start)]
                    if end:
                        df = df[dt_s <= pd.to_datetime(end)]
                return FMPQueryResult(df=df.reset_index(drop=True), endpoint=path)
            except Exception as e:
                last_err = e
                continue
        raise FMPAPIError(f"Failed to fetch upgrades/downgrades from FMP. Last error: {last_err}")

    def equity_price_history(
        self,
        symbol: str,
        *,
        start_date: str | dt.date | dt.datetime | None = None,
        end_date: str | dt.date | dt.datetime | None = None,
        interval: str | None = None,
    ) -> FMPQueryResult:
        start = _normalize_date(start_date)
        end = _normalize_date(end_date)
        candidates: list[tuple[str, dict[str, Any]]] = [
            ("/stable/historical-price-full", {"symbol": symbol}),
            (f"/stable/historical-price-full/{symbol}", {}),
            (f"/api/v3/historical-price-full/{symbol}", {}),
        ]

        base_params: dict[str, Any] = {}
        if start:
            base_params["from"] = start
        if end:
            base_params["to"] = end
        if interval:
            base_params["serietype"] = "line" if interval == "1d" else "line"

        last_err: Exception | None = None
        for path, params in candidates:
            try:
                merged = dict(base_params)
                merged.update(params)
                data = self._get_json(path, merged)
                if isinstance(data, dict) and "historical" in data and isinstance(data.get("historical"), list):
                    rows = data.get("historical")
                elif isinstance(data, dict) and "data" in data and isinstance(data.get("data"), list):
                    rows = data.get("data")
                elif isinstance(data, list):
                    rows = data
                else:
                    raise FMPAPIError(f"Unexpected response shape from '{path}': {type(data)}")

                df = pd.DataFrame(rows)
                if df.empty:
                    continue

                dt_s = _parse_date_series(df, ["date", "datetime", "timestamp"])
                if dt_s is not None:
                    df = df.assign(_dt=dt_s).sort_values("_dt").drop(columns=["_dt"])
                return FMPQueryResult(df=df.reset_index(drop=True), endpoint=path)
            except Exception as e:
                last_err = e
                continue

        raise FMPAPIError(f"Failed to fetch price history from FMP. Last error: {last_err}")
