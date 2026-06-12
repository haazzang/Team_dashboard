from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


API_KEY = "JI9NAp3lfVUybz10CbT3cPr8uQjLiowG"
SYMBOL = "%5EGSPC"
OUTPUT_DIR = Path("/Users/hyejinha/Desktop/Workspace/Team/analysis_outputs/spx_gap_signal")
WINDOWS = [
    ("1950-01-01", "1965-12-31"),
    ("1966-01-01", "1980-12-31"),
    ("1981-01-01", "1995-12-31"),
    ("1996-01-01", "2010-12-31"),
    ("2011-01-01", "2026-04-08"),
]
HORIZONS = {
    "3d": 3,
    "1w": 5,
    "2w": 10,
    "3w": 15,
    "1m": 21,
    "3m": 63,
}


@dataclass(frozen=True)
class SignalSpec:
    key: str
    label: str
    description: str


STRICT_SIGNAL = SignalSpec(
    key="strict_open_above_current_ma_prev_close_below_prev_ma",
    label="Strict 4-event signal",
    description=(
        "Today's open > today's 50DMA and 200DMA, while the previous close "
        "was below the previous day's 50DMA and 200DMA."
    ),
)
LOOSE_SIGNAL = SignalSpec(
    key="close_above_current_ma_prev_close_below_prev_ma",
    label="Loose close-cross signal",
    description=(
        "Today's close > today's 50DMA and 200DMA, while the previous close "
        "was below the previous day's 50DMA and 200DMA."
    ),
)


def fetch_window(start: str, end: str) -> pd.DataFrame:
    url = (
        "https://financialmodelingprep.com/stable/historical-price-eod/full"
        f"?symbol={SYMBOL}&from={start}&to={end}&apikey={API_KEY}"
    )
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    rows = response.json()
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No rows returned for {start} to {end}")
    return df


def fetch_history() -> pd.DataFrame:
    parts = [fetch_window(start, end) for start, end in WINDOWS]
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = (
        df.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    for window in (50, 200):
        df[f"ma{window}"] = df["close"].rolling(window).mean()
        df[f"prev_ma{window}"] = df[f"ma{window}"].shift(1)
    for col in ("open", "high", "low", "close"):
        df[f"prev_{col}"] = df[col].shift(1)
    return df


def apply_signal(df: pd.DataFrame, spec: SignalSpec) -> pd.Series:
    if spec is STRICT_SIGNAL:
        return (
            (df["open"] > df["ma50"])
            & (df["open"] > df["ma200"])
            & (df["prev_close"] < df["prev_ma50"])
            & (df["prev_close"] < df["prev_ma200"])
        )
    if spec is LOOSE_SIGNAL:
        return (
            (df["close"] > df["ma50"])
            & (df["close"] > df["ma200"])
            & (df["prev_close"] < df["prev_ma50"])
            & (df["prev_close"] < df["prev_ma200"])
        )
    raise ValueError(f"Unsupported signal: {spec.key}")


def forward_return(df: pd.DataFrame, idx: int, sessions: int) -> float:
    target_idx = idx + sessions
    if target_idx >= len(df):
        return math.nan
    return (df.iloc[target_idx]["close"] / df.iloc[idx]["close"] - 1.0) * 100.0


def forward_drawdown(df: pd.DataFrame, idx: int, sessions: int) -> float:
    window = df.iloc[idx + 1 : idx + sessions + 1]
    if window.empty:
        return math.nan
    base = df.iloc[idx]["close"]
    return (window["low"].min() / base - 1.0) * 100.0


def build_event_table(df: pd.DataFrame, spec: SignalSpec) -> pd.DataFrame:
    mask = apply_signal(df, spec)
    events = df.loc[mask].copy()
    rows = []
    for idx, row in events.iterrows():
        record = {
            "signal": spec.label,
            "date": row["date"].date().isoformat(),
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "prev_close": row["prev_close"],
            "ma50": row["ma50"],
            "ma200": row["ma200"],
            "prev_ma50": row["prev_ma50"],
            "prev_ma200": row["prev_ma200"],
            "gap_vs_prev_close_pct": (row["open"] / row["prev_close"] - 1.0) * 100.0,
            "max_dd_3m_pct": forward_drawdown(df, idx, HORIZONS["3m"]),
        }
        for label, sessions in HORIZONS.items():
            record[f"ret_{label}_pct"] = forward_return(df, idx, sessions)
        rows.append(record)
    return pd.DataFrame(rows)


def format_numeric_table(df: pd.DataFrame, round_digits: int = 2) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].round(round_digits)
    return out


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    df = format_numeric_table(df).copy()
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            value = row[col]
            if pd.isna(value):
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def chart_events(
    df: pd.DataFrame,
    event_dates: Iterable[str],
    output_path: Path,
    title: str,
    months_before: int = 6,
    months_after: int = 6,
) -> None:
    dates = [pd.Timestamp(d) for d in event_dates]
    n = len(dates)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4.5 * rows), sharey=False)
    axes = np.atleast_1d(axes).flatten()

    for ax, event_date in zip(axes, dates):
        start = event_date - pd.DateOffset(months=months_before)
        end = event_date + pd.DateOffset(months=months_after)
        window = df[(df["date"] >= start) & (df["date"] <= end)].copy()
        ax.plot(window["date"], window["close"], label="Close", color="#1f3b73", lw=1.8)
        ax.plot(window["date"], window["ma50"], label="50DMA", color="#d97706", lw=1.3)
        ax.plot(window["date"], window["ma200"], label="200DMA", color="#9f1239", lw=1.3)
        event_row = df.loc[df["date"] == event_date].iloc[0]
        ax.scatter(
            [event_row["date"]],
            [event_row["close"]],
            color="#059669",
            s=45,
            zorder=5,
        )
        ax.axvline(event_row["date"], color="#059669", linestyle="--", lw=1)
        ax.set_title(event_date.date().isoformat())
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=45)

    for ax in axes[n:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(title, fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def chart_single_event(
    df: pd.DataFrame,
    event_date: str,
    output_path: Path,
    title: str,
    months_before: int = 6,
    months_after: int = 2,
) -> None:
    event_date_ts = pd.Timestamp(event_date)
    start = event_date_ts - pd.DateOffset(months=months_before)
    end = event_date_ts + pd.DateOffset(months=months_after)
    window = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    event_row = df.loc[df["date"] == event_date_ts].iloc[0]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(window["date"], window["close"], label="Close", color="#1f3b73", lw=2)
    ax.plot(window["date"], window["ma50"], label="50DMA", color="#d97706", lw=1.5)
    ax.plot(window["date"], window["ma200"], label="200DMA", color="#9f1239", lw=1.5)
    ax.scatter([event_row["date"]], [event_row["close"]], color="#059669", s=55, zorder=5)
    ax.axvline(event_row["date"], color="#059669", linestyle="--", lw=1)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_markdown_summary(
    strict_table: pd.DataFrame,
    loose_table: pd.DataFrame,
    comparison_df: pd.DataFrame,
    notes_path: Path,
) -> None:
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    with notes_path.open("w", encoding="utf-8") as f:
        f.write("# SPX gap-above-50DMA-and-200DMA verification\n\n")
        f.write("## Signal definitions\n\n")
        f.write(f"- **{STRICT_SIGNAL.label}**: {STRICT_SIGNAL.description}\n")
        f.write(f"- **{LOOSE_SIGNAL.label}**: {LOOSE_SIGNAL.description}\n\n")
        f.write("## Definition count comparison\n\n")
        f.write(dataframe_to_markdown(comparison_df))
        f.write("\n\n## Strict signal event table\n\n")
        f.write(dataframe_to_markdown(strict_table))
        f.write("\n\n## Loose signal event table\n\n")
        f.write(dataframe_to_markdown(loose_table))
        f.write("\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch_history()
    df.to_csv(OUTPUT_DIR / "spx_daily_fmp.csv", index=False)

    strict_table = build_event_table(df, STRICT_SIGNAL)
    loose_table = build_event_table(df, LOOSE_SIGNAL)
    strict_table.to_csv(OUTPUT_DIR / "strict_signal_events.csv", index=False)
    loose_table.to_csv(OUTPUT_DIR / "loose_signal_events.csv", index=False)

    comparison = pd.DataFrame(
        [
            {
                "definition": STRICT_SIGNAL.label,
                "count_since_1950": len(strict_table),
                "includes_2026_04_08": bool((strict_table["date"] == "2026-04-08").any()),
            },
            {
                "definition": LOOSE_SIGNAL.label,
                "count_since_1950": len(loose_table),
                "includes_2026_04_08": bool((loose_table["date"] == "2026-04-08").any()),
            },
        ]
    )
    comparison.to_csv(OUTPUT_DIR / "definition_comparison.csv", index=False)

    if not strict_table.empty:
        chart_events(
            df,
            strict_table["date"].tolist(),
            OUTPUT_DIR / "strict_signal_event_grid.png",
            "S&P 500 strict signal windows",
        )
    if (loose_table["date"] == "2026-04-08").any():
        chart_single_event(
            df,
            "2026-04-08",
            OUTPUT_DIR / "current_move_2026_04_08.png",
            "S&P 500 current move on 2026-04-08",
        )

    save_markdown_summary(
        strict_table,
        loose_table,
        comparison,
        OUTPUT_DIR / "README.md",
    )


if __name__ == "__main__":
    main()
