from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _find_datetime_series(df: pd.DataFrame) -> pd.Series | None:
    if isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(df.index, name="index_datetime")
    for col in ("date", "datetime", "timestamp", "asof", "period"):
        if col in df.columns:
            s = pd.to_datetime(df[col], errors="coerce")
            if s.notna().any():
                return s
    return None


def _numeric_summary(df: pd.DataFrame, max_cols: int = 12) -> dict[str, Any]:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    out: dict[str, Any] = {}
    for col in numeric_cols[:max_cols]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        first = float(s.iloc[0])
        last = float(s.iloc[-1])
        delta = last - first
        out[col] = {
            "first": first,
            "last": last,
            "delta": delta,
        }
    return out


def build_dataframe_context(
    df: pd.DataFrame,
    *,
    endpoint: str,
    max_preview_rows: int = 12,
    max_preview_cols: int = 20,
) -> str:
    df2 = df.copy()
    if df2.shape[1] > max_preview_cols:
        df2 = df2.iloc[:, :max_preview_cols]

    dt_s = _find_datetime_series(df2)
    date_range = None
    if dt_s is not None:
        dt_s = pd.to_datetime(dt_s, errors="coerce")
        if isinstance(df2.index, pd.DatetimeIndex):
            df2 = df2.sort_index()
        else:
            df2 = df2.assign(_dt=dt_s).sort_values("_dt").drop(columns=["_dt"])
        dt_clean = dt_s[dt_s.notna()]
        if not dt_clean.empty:
            date_range = {"start": str(dt_clean.min().date()), "end": str(dt_clean.max().date())}

    preview = df2.tail(max_preview_rows).to_dict(orient="records")
    payload = {
        "endpoint": endpoint,
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "columns": list(map(str, df.columns.tolist())),
        "date_range": date_range,
        "numeric_summary": _numeric_summary(df2),
        "preview_tail": preview,
    }
    return _safe_json(payload)
