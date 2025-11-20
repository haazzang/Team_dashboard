import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import warnings
warnings.filterwarnings("ignore", message="YF.download() has changed argument auto_adjust")

def dl_series(sym, start, end, out_name):
    """yfinance에서 sym을 다운로드하여 단일 Series(Adj Close->Close 우선순위)로 반환."""
    import pandas as pd, yfinance as yf
    tmp = yf.download(sym, start=start, end=end, progress=False, auto_adjust=False)
    if not isinstance(tmp, pd.DataFrame) or tmp.empty:
        raise RuntimeError(f"Download failed or empty for {sym}")
    # 1) 일반 컬럼 시나리오
    if "Adj Close" in tmp.columns:
        ser = tmp["Adj Close"]
    elif "Close" in tmp.columns:
        ser = tmp["Close"]
    else:
        # 2) MultiIndex 컬럼 시나리오
        if hasattr(tmp.columns, "levels") and ("Adj Close" in tmp.columns.get_level_values(0)):
            ser = tmp.xs("Adj Close", axis=1, level=0).squeeze()
        elif hasattr(tmp.columns, "levels") and ("Close" in tmp.columns.get_level_values(0)):
            ser = tmp.xs("Close", axis=1, level=0).squeeze()
        else:
            raise KeyError(f"No Adj Close/Close in columns for {sym}: {list(tmp.columns)}")

    # 가끔 ser가 DataFrame 한 칼럼일 수 있으니 Series로 강제
    if hasattr(ser, "columns"):
        ser = ser.iloc[:, 0]
    ser = ser.dropna().astype(float)
    ser.name = out_name
    
    return ser

# --------------- Config ---------------
INPUT_XLSX = r"C:\Users\hyeji\Desktop\성과\2026_매매\Holdings2.xlsx"
OUT_DIR    = Path(r"C:\Users\hyeji\Desktop\성과\2026_매매")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Yahoo symbols
TICKERS_BENCH = {"US":"^GSPC", "HK":"^HSI", "JP":"^N225"}
FX_KRW = {"USD":"KRW=X", "HKD":"HKDKRW=X", "JPY":"JPYKRW=X"}
FX_USD = {"HKD":"HKD=X", "JPY":"JPY=X"}  # HKD/USD, JPY/USD rates

# --------------- Helpers ---------------
def winsorize(s, p=0.001):
    lo, hi = s.quantile(p), s.quantile(1-p)
    return s.clip(lower=lo, upper=hi)

def ann_stats(rets):
    rets = rets.dropna()
    if len(rets)==0:
        return pd.Series({"ann_return":np.nan,"ann_vol":np.nan,"sharpe":np.nan})
    mu = rets.mean()*252
    vol = rets.std(ddof=1)*np.sqrt(252)
    sr = mu/vol if vol and vol>0 else np.nan
    return pd.Series({"ann_return":mu,"ann_vol":vol,"sharpe":sr})

def ols_alpha_beta(y, x):
    # y: portfolio returns, x: benchmark returns
    df = pd.concat({"y":y, "x":x}, axis=1).dropna()
    if len(df)<20:
        return pd.Series({"alpha_ann":np.nan,"beta":np.nan,"r2":np.nan,"t_alpha":np.nan,"t_beta":np.nan})
    X = sm.add_constant(df["x"].values)
    model = sm.OLS(df["y"].values, X).fit()
    alpha_daily = model.params[0]
    beta = model.params[1]
    r2 = model.rsquared
    t_alpha = model.tvalues[0]
    t_beta = model.tvalues[1]
    # Annualize alpha (simple *252)
    return pd.Series({"alpha_ann":alpha_daily*252, "beta":beta, "r2":r2, "t_alpha":t_alpha, "t_beta":t_beta})

# --------------- Load holdings ---------------
xl = pd.ExcelFile(INPUT_XLSX)

# Load hedge futures separately
try:
    hedge_df = xl.parse("Hedge_futures")
    print(f"Loaded Hedge_futures sheet with {len(hedge_df)} rows")
    # Assume structure: Date in first column, Cumulative P&L in column E (index 4)
    hedge_df.columns = hedge_df.columns.str.strip()
    hedge_df = hedge_df.rename(columns={hedge_df.columns[0]: "Date", hedge_df.columns[4]: "Cumulative_PnL_KRW"})
    hedge_df["Date"] = pd.to_datetime(hedge_df["Date"])
    hedge_df = hedge_df[["Date", "Cumulative_PnL_KRW"]].dropna()
    hedge_df = hedge_df.sort_values("Date")
    print(f"Processed hedge data: {len(hedge_df)} dates")
except Exception as e:
    print(f"Warning: Could not load Hedge_futures sheet: {e}")
    hedge_df = pd.DataFrame(columns=["Date", "Cumulative_PnL_KRW"])

# Load equity sheets
sheet_names = [s for s in xl.sheet_names if s.lower()!="hedge_futures" and s.lower()!="hedge_futures".lower()]
eq = []
for s in sheet_names:
    df = xl.parse(s)
    eq.append(df.assign(Sheet=s))
eq = pd.concat(eq, ignore_index=True)

# Standardize columns
eq.rename(columns={"Ticker":"TICKER","Market Price":"Market_Price","Market Value":"Market_Value"}, inplace=True)
eq["Date"] = pd.to_datetime(eq["Date"])
eq["Currency"] = eq["Currency"].str.upper()
ccy_to_ctry = {"USD":"US","HKD":"HK","JPY":"JP"}
eq["Country"] = eq["Currency"].map(ccy_to_ctry).fillna("OTHER")

# --------------- Build daily P&L (local) ---------------
eq = eq.sort_values(["TICKER","Date"])
eq["Prev_Price"] = eq.groupby("TICKER")["Market_Price"].shift(1)
eq["Prev_Qty"]   = eq.groupby("TICKER")["Quantity"].shift(1)
eq["Prev_MV"] = eq.groupby("TICKER")["Market_Value"].shift(1)

# Daily P&L for each stock
eq["Daily_PnL_local"] = (eq["Prev_Qty"] * (eq["Market_Price"] - eq["Prev_Price"])).fillna(0.0)

# Stock-level returns (only for stocks that existed yesterday)
eq["Stock_Ret"] = np.where(
    (eq["Prev_MV"].notna()) & (eq["Prev_MV"] > 0),
    eq["Daily_PnL_local"] / eq["Prev_MV"],
    np.nan
)

# --------------- Date range for downloads ---------------
start = str(eq["Date"].min().date())
end   = str(eq["Date"].max().date())

# --------------- Download Benchmarks ---------------
bench = {}
for ctry, sym in TICKERS_BENCH.items():
    syms = [sym]
    if sym == "^N225":
        syms += ["^NI225", "1321.T"]  # fallback for Nikkei 225
    last_err = None
    for s in syms:
        try:
            bench[ctry] = dl_series(s, start, end, f"{ctry}_BENCH")
            break
        except Exception as e:
            last_err = e
            continue
    if ctry not in bench:
        raise RuntimeError(f"Failed to download benchmark for {ctry}: {last_err}")

bench_df = pd.concat(bench.values(), axis=1)
bench_ret = bench_df.pct_change(fill_method=None)

# --------------- Download FX to KRW ---------------
fx = {}
for ccy, sym in FX_KRW.items():
    fx[ccy] = dl_series(sym, start, end, f"{ccy}_KRW")
fx_df = pd.concat(fx.values(), axis=1).sort_index().ffill()

# --------------- Portfolio-level KRW ---------------
# Calculate portfolio return as value-weighted average of stock returns
# Group by date and calculate weighted returns
def calc_portfolio_ret(df):
    """Calculate portfolio return from individual stock returns"""
    # Only include stocks that have valid returns (existed yesterday)
    valid = df[df["Stock_Ret"].notna()].copy()
    if len(valid) == 0:
        return np.nan
    
    # Weight by previous day's market value
    total_prev_mv = valid["Prev_MV_KRW"].sum()
    if total_prev_mv == 0:
        return np.nan
    
    valid["weight"] = valid["Prev_MV_KRW"] / total_prev_mv
    port_ret = (valid["Stock_Ret"] * valid["weight"]).sum()
    return port_ret

# Attach KRW values to each stock
eq_krw = eq.copy()
for ccy in ["USD", "HKD", "JPY"]:
    ctry = ccy_to_ctry[ccy]
    mask = eq_krw["Currency"] == ccy
    eq_krw.loc[mask, "FX_KRW"] = eq_krw.loc[mask, "Date"].map(
        lambda d: fx_df.loc[fx_df.index == d, f"{ccy}_KRW"].iloc[0] if d in fx_df.index else np.nan
    )

eq_krw["FX_KRW"] = eq_krw.groupby("TICKER")["FX_KRW"].ffill()
eq_krw["MV_KRW"] = eq_krw["Market_Value"] * eq_krw["FX_KRW"]
eq_krw["Prev_MV_KRW"] = eq_krw["Prev_MV"] * eq_krw["FX_KRW"]
eq_krw["PnL_KRW"] = eq_krw["Daily_PnL_local"] * eq_krw["FX_KRW"]

# Calculate portfolio-level returns
port_daily = eq_krw.groupby("Date", group_keys=False).apply(calc_portfolio_ret, include_groups=False).rename("Port_Ret_KRW").to_frame().reset_index()

# Add total MV and PnL for reference
mv_pnl = eq_krw.groupby("Date", as_index=False).agg({"MV_KRW": "sum", "PnL_KRW": "sum"})
port_daily = port_daily.merge(mv_pnl, on="Date", how="left")

# --------------- Country-level analysis for alpha/beta ---------------
# Calculate country returns using value-weighted approach
def calc_country_ret(df, country):
    """Calculate country return from individual stock returns"""
    country_df = df[df["Country"] == country].copy()
    valid = country_df[country_df["Stock_Ret"].notna()].copy()
    if len(valid) == 0:
        return np.nan
    
    total_prev_mv = valid["Prev_MV_KRW"].sum()
    if total_prev_mv == 0:
        return np.nan
    
    valid["weight"] = valid["Prev_MV_KRW"] / total_prev_mv
    country_ret = (valid["Stock_Ret"] * valid["weight"]).sum()
    return country_ret

# Build country daily dataframe
ctry_daily = []
for date in eq_krw["Date"].unique():
    date_df = eq_krw[eq_krw["Date"] == date]
    for country in ["US", "HK", "JP"]:
        country_ret = calc_country_ret(date_df, country)
        mv_krw = date_df[date_df["Country"] == country]["MV_KRW"].sum()
        pnl_krw = date_df[date_df["Country"] == country]["PnL_KRW"].sum()
        ctry_daily.append({
            "Date": date,
            "Country": country,
            "MV_KRW": mv_krw,
            "PnL_KRW": pnl_krw,
            "Country_Ret_KRW": country_ret
        })

ctry_daily = pd.DataFrame(ctry_daily)

# --------------- Merge with Benchmarks (by country) ---------------
bench_ret = bench_ret.reset_index().rename(columns={"Date":"Date"})
bench_ret["Date"] = pd.to_datetime(bench_ret["Date"])

def attach_bench(df, country):
    cols = {"US":"US_BENCH","HK":"HK_BENCH","JP":"JP_BENCH"}
    col = cols[country]
    merged = pd.merge(df[df["Country"]==country], bench_ret[["Date", country+"_BENCH"]].rename(columns={country+"_BENCH":col}), on="Date", how="left")
    return merged

ctry_with_bench = pd.concat([attach_bench(ctry_daily, "US"), attach_bench(ctry_daily, "HK"), attach_bench(ctry_daily, "JP")], ignore_index=True)

# --------------- Alpha/Beta per country ---------------
def country_alpha_beta(country):
    df = ctry_with_bench[ctry_with_bench["Country"]==country].sort_values("Date")
    y = df["Country_Ret_KRW"]
    x = df[country+"_BENCH"].rename("Bench_Ret")
    ab = ols_alpha_beta(y, x)
    stats = ann_stats(y)
    out = pd.concat([ab, stats])
    out.name = country
    return out

ab_us = country_alpha_beta("US")
ab_hk = country_alpha_beta("HK")
ab_jp = country_alpha_beta("JP")
ab_table = pd.concat([ab_us, ab_hk, ab_jp], axis=1).T
ab_table.index.name = "Country"

# --------------- Portfolio vs Blended Benchmark ---------------
# Blend benchmark by last-available country MV weights (KRW)
last_mv = ctry_daily.sort_values("Date").groupby("Country").tail(1)[["Country","MV_KRW"]].set_index("Country")["MV_KRW"]
weights = last_mv/last_mv.sum()
blend = (bench_ret.set_index("Date")[["US_BENCH","HK_BENCH","JP_BENCH"]]
         .mul([weights.get("US",0), weights.get("HK",0), weights.get("JP",0)], axis=1)
         .sum(axis=1).rename("Blend_Bench"))
port = port_daily.set_index("Date")["Port_Ret_KRW"]
port_ab = ols_alpha_beta(port, blend)
port_stats = ann_stats(port)
port_summary = pd.concat([port_ab, port_stats]).to_frame("Portfolio_KRW")

# --------------- Optional: Sector contributions (best-effort) ---------------
# Try to fetch sector per equity ticker (yfinance .info), cache to CSV to avoid rate limits
eq_last = eq.sort_values("Date").groupby("TICKER").tail(1)[["TICKER","Country","Currency"]].drop_duplicates()
sector_cache_file = OUT_DIR/"ticker_sector_cache.csv"
try:
    cache = pd.read_csv(sector_cache_file)
except Exception:
    cache = pd.DataFrame(columns=["TICKER","sector","industry"])

need = set(eq_last["TICKER"]) - set(cache["TICKER"])
rows = []
for t in need:
    try:
        info = yf.Ticker(str(t)).fast_info  # fast_info has limited fields
        # fallback
        full = yf.Ticker(str(t)).info
        sector = full.get("sector")
        industry = full.get("industry")
        rows.append({"TICKER":t, "sector":sector, "industry":industry})
    except Exception as e:
        rows.append({"TICKER":t, "sector":None, "industry":None})
if rows:
    cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True).drop_duplicates(subset=["TICKER"], keep="last")
    cache.to_csv(sector_cache_file, index=False)

eq_sector = eq.merge(cache, on="TICKER", how="left")

# Sector daily PnL in KRW
eq_sector["Daily_PnL_KRW"] = eq_sector["Daily_PnL_local"] * eq_sector["Currency"].map({"USD":np.nan,"HKD":np.nan,"JPY":np.nan})
# Map FX dynamically by date - use fx_full to avoid overwriting fx_df
fx_raw = {}
for ccy, sym in {"USD":"KRW=X","HKD":"HKDKRW=X","JPY":"JPYKRW=X"}.items():
    fx_raw[ccy] = dl_series(sym, start, end, f"{ccy}_KRW")
fx_full = pd.concat(fx_raw.values(), axis=1).sort_index().ffill()
eq_sector = eq_sector.set_index("Date")
for ccy in ["USD","HKD","JPY"]:
    mask = (eq_sector["Currency"]==ccy)
    eq_sector.loc[mask, "FX_KRW"] = fx_full[ccy+"_KRW"]
eq_sector["FX_KRW"] = eq_sector.groupby("TICKER")["FX_KRW"].ffill()
eq_sector["Daily_PnL_KRW"] = eq_sector["Daily_PnL_local"] * eq_sector["FX_KRW"]
eq_sector = eq_sector.reset_index()

sector_contrib = eq_sector.groupby(["Date","sector"], as_index=False)["Daily_PnL_KRW"].sum()
sector_contrib["Prev_MV_KRW"] = np.nan  # MV by sector not in raw data; if needed can derive similarly
sector_pnl_total = sector_contrib.groupby("sector", as_index=False)["Daily_PnL_KRW"].sum().sort_values("Daily_PnL_KRW", ascending=False)

# --------------- USD-based Portfolio Analysis ---------------
print("\nDownloading FX rates to USD...")
fx_usd = {}
for ccy, sym in FX_USD.items():
    try:
        fx_usd[ccy] = dl_series(sym, start, end, f"{ccy}_USD")
    except Exception as e:
        print(f"Warning: Failed to download {sym}: {e}")
        # Create dummy series filled with NaN
        fx_usd[ccy] = pd.Series(index=pd.date_range(start, end), dtype=float, name=f"{ccy}_USD")

fx_usd_df = pd.concat(fx_usd.values(), axis=1).sort_index().ffill()

# CRITICAL FIX: JPY=X and HKD=X return USD/JPY and USD/HKD (e.g., 150 JPY per USD)
# We need JPY/USD and HKD/USD for conversion, so take reciprocal
fx_usd_df["HKD_USD"] = 1 / fx_usd_df["HKD_USD"]
fx_usd_df["JPY_USD"] = 1 / fx_usd_df["JPY_USD"]

# Attach USD values to each stock
eq_usd = eq.copy()
for ccy in ["HKD", "JPY"]:
    mask = eq_usd["Currency"] == ccy
    eq_usd.loc[mask, "FX_USD"] = eq_usd.loc[mask, "Date"].map(
        lambda d: fx_usd_df.loc[fx_usd_df.index == d, f"{ccy}_USD"].iloc[0] if d in fx_usd_df.index else np.nan
    )

# USD currency has FX = 1.0
eq_usd.loc[eq_usd["Currency"] == "USD", "FX_USD"] = 1.0
eq_usd["FX_USD"] = eq_usd.groupby("TICKER")["FX_USD"].ffill()

eq_usd["MV_USD"] = eq_usd["Market_Value"] * eq_usd["FX_USD"]
eq_usd["Prev_MV_USD"] = eq_usd["Prev_MV"] * eq_usd["FX_USD"]
eq_usd["PnL_USD"] = eq_usd["Daily_PnL_local"] * eq_usd["FX_USD"]

# Calculate portfolio-level returns in USD
def calc_portfolio_ret_usd(df):
    """Calculate portfolio return from individual stock returns"""
    valid = df[df["Stock_Ret"].notna()].copy()
    if len(valid) == 0:
        return np.nan
    
    total_prev_mv = valid["Prev_MV_USD"].sum()
    if total_prev_mv == 0:
        return np.nan
    
    valid["weight"] = valid["Prev_MV_USD"] / total_prev_mv
    port_ret = (valid["Stock_Ret"] * valid["weight"]).sum()
    return port_ret

port_daily_usd = eq_usd.groupby("Date", group_keys=False).apply(calc_portfolio_ret_usd, include_groups=False).rename("Port_Ret_USD").to_frame().reset_index()

# Add total MV and PnL for reference
mv_pnl_usd = eq_usd.groupby("Date", as_index=False).agg({"MV_USD": "sum", "PnL_USD": "sum"})
port_daily_usd = port_daily_usd.merge(mv_pnl_usd, on="Date", how="left")

# Calculate cumulative returns (in USD)
port_daily_usd["Cum_Ret_USD"] = (1 + port_daily_usd["Port_Ret_USD"].fillna(0)).cumprod() - 1

# Build country daily USD dataframe for exports
ctry_daily_usd = []
for date in eq_usd["Date"].unique():
    date_df = eq_usd[eq_usd["Date"] == date]
    for country in ["US", "HK", "JP"]:
        country_df = date_df[date_df["Country"] == country]
        valid = country_df[country_df["Stock_Ret"].notna()].copy()
        
        if len(valid) > 0:
            total_prev_mv = valid["Prev_MV_USD"].sum()
            if total_prev_mv > 0:
                valid["weight"] = valid["Prev_MV_USD"] / total_prev_mv
                country_ret = (valid["Stock_Ret"] * valid["weight"]).sum()
            else:
                country_ret = np.nan
        else:
            country_ret = np.nan
            
        mv_usd = country_df["MV_USD"].sum()
        pnl_usd = country_df["PnL_USD"].sum()
        
        ctry_daily_usd.append({
            "Date": date,
            "Country": country,
            "MV_USD": mv_usd,
            "PnL_USD": pnl_usd,
            "Country_Ret_USD": country_ret
        })

ctry_daily_usd = pd.DataFrame(ctry_daily_usd)


# --------------- Hedge-Adjusted Returns (USD ONLY) ---------------
print("\nCalculating USD returns with hedge...")

# --- 1) Equity Daily P&L & Previous MV in USD ---
equity_daily_usd = eq_usd.groupby("Date")["PnL_USD"].sum().rename("Equity_PnL_USD")
prev_mv_usd      = eq_usd.groupby("Date")["Prev_MV_USD"].sum().rename("Prev_MV_USD")

port_with_hedge_usd = pd.concat([equity_daily_usd, prev_mv_usd], axis=1).reset_index()
port_with_hedge_usd["Date"] = pd.to_datetime(port_with_hedge_usd["Date"])

# --- 2) Hedge: cumulative → daily PnL (KRW) ---
if not hedge_df.empty:
    hedge_df = hedge_df.sort_values("Date").copy()
    hedge_df["Daily_Hedge_PnL_KRW"] = hedge_df["Cumulative_PnL_KRW"].diff()
    hedge_df["Daily_Hedge_PnL_KRW"] = hedge_df["Daily_Hedge_PnL_KRW"].fillna(hedge_df["Cumulative_PnL_KRW"])
else:
    hedge_df = pd.DataFrame(columns=["Date","Cumulative_PnL_KRW","Daily_Hedge_PnL_KRW"])

# --- 3) Attach USD/KRW FX and convert hedge PnL to USD ---
fx_df_reset = fx_df.reset_index()
fx_df_reset.columns = ["Date", "USD_KRW", "HKD_KRW", "JPY_KRW"]

hedge_df = pd.merge(hedge_df, fx_df_reset[["Date","USD_KRW"]], on="Date", how="left")
hedge_df["USD_KRW"] = hedge_df["USD_KRW"].ffill().bfill()
hedge_df["Daily_Hedge_PnL_USD"] = hedge_df["Daily_Hedge_PnL_KRW"] / hedge_df["USD_KRW"]

# --- 4) Merge hedge into equity daily ---
port_with_hedge_usd = port_with_hedge_usd.merge(
    hedge_df[["Date","Daily_Hedge_PnL_USD"]],
    on="Date",
    how="left"
)
port_with_hedge_usd["Daily_Hedge_PnL_USD"] = port_with_hedge_usd["Daily_Hedge_PnL_USD"].fillna(0.0)

# --- 5) Daily returns: Equity only & Equity + Hedge ---
# 보호 로직: Prev_MV_USD가 0이거나 NaN이면 해당 일 수익률은 0으로 처리
valid_mv = port_with_hedge_usd["Prev_MV_USD"] > 0

port_with_hedge_usd["Ret_Equity"] = 0.0
port_with_hedge_usd.loc[valid_mv, "Ret_Equity"] = (
    port_with_hedge_usd.loc[valid_mv, "Equity_PnL_USD"] /
    port_with_hedge_usd.loc[valid_mv, "Prev_MV_USD"]
)

port_with_hedge_usd["Ret_Total"] = 0.0
port_with_hedge_usd.loc[valid_mv, "Ret_Total"] = (
    (port_with_hedge_usd.loc[valid_mv, "Equity_PnL_USD"] +
     port_with_hedge_usd.loc[valid_mv, "Daily_Hedge_PnL_USD"]) /
    port_with_hedge_usd.loc[valid_mv, "Prev_MV_USD"]
)

# --- 6) Cumulative returns (% 기준은 나중에 곱하기 100) ---
port_with_hedge_usd["Equity_Cum_Return"] = (1 + port_with_hedge_usd["Ret_Equity"]).cumprod() - 1
port_with_hedge_usd["Total_Cum_Return"]  = (1 + port_with_hedge_usd["Ret_Total"]).cumprod() - 1

# Hedge 누적 PnL (USD)
port_with_hedge_usd["Hedge_Cum_PnL_USD"] = port_with_hedge_usd["Daily_Hedge_PnL_USD"].cumsum()

# --- 7) 최종 성과 요약 출력 ---
final_equity_ret = port_with_hedge_usd["Equity_Cum_Return"].iloc[-1] * 100
final_total_ret  = port_with_hedge_usd["Total_Cum_Return"].iloc[-1] * 100
final_hedge_pnl  = port_with_hedge_usd["Hedge_Cum_PnL_USD"].iloc[-1]

print(f"\nFinal Results:")
print(f"  Equity Only Return: {final_equity_ret:.2f}%")
print(f"  With Hedge Return: {final_total_ret:.2f}%")
print(f"  Hedge Contribution: {final_total_ret - final_equity_ret:.2f}%")
print(f"  Hedge Cumulative P&L: ${final_hedge_pnl:,.0f}")

# --- 8) Chart: Cumulative Returns in USD (Equity vs Equity+Hedge) ---
print("Creating cumulative returns chart (USD)...")
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(
    port_with_hedge_usd["Date"],
    port_with_hedge_usd["Equity_Cum_Return"] * 100,
    linewidth=2.5,
    label="Equity Only",
    alpha=0.9
)
ax.plot(
    port_with_hedge_usd["Date"],
    port_with_hedge_usd["Total_Cum_Return"] * 100,
    linewidth=2.5,
    linestyle="--",
    label="Equity + Hedge",
    alpha=0.9
)
ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
ax.set_xlabel("Date", fontsize=12, fontweight="bold")
ax.set_ylabel("Cumulative Return (%)", fontsize=12, fontweight="bold")
ax.set_title("Portfolio Cumulative Returns (USD) - Equity vs Equity+Hedge", fontsize=14, fontweight="bold", pad=20)
ax.legend(loc="best", fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle="--")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)
plt.tight_layout()
chart_path = OUT_DIR / "cumulative_returns_USD_with_hedge.png"
plt.savefig(chart_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved chart: {chart_path.name}")

# --------------- Monthly Returns Table in USD ---------------
# --------------- Monthly Returns Table in USD ---------------
print("Creating monthly returns table...")
port_monthly_usd = port_daily_usd.set_index("Date").copy()
port_monthly_usd.index = pd.to_datetime(port_monthly_usd.index)

# Calculate monthly returns using the product of (1 + daily returns) - 1
monthly_ret = port_monthly_usd.groupby(pd.Grouper(freq='ME'))["Port_Ret_USD"].apply(
    lambda x: (1 + x.fillna(0)).prod() - 1
)

# Create a pivot table with years as rows and months as columns
monthly_ret_df = pd.DataFrame({
    'Year': monthly_ret.index.year,
    'Month': monthly_ret.index.month,
    'Return': monthly_ret.values
})

monthly_pivot = monthly_ret_df.pivot(index='Year', columns='Month', values='Return')

# Dynamically assign month names only for existing columns
month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
               7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
monthly_pivot.columns = [month_names.get(col, str(col)) for col in monthly_pivot.columns]

# Calculate YTD (Year-to-Date) returns
monthly_pivot['YTD'] = monthly_pivot.apply(
    lambda row: (1 + row.dropna()).prod() - 1, axis=1
)

# Format as percentages
monthly_pivot_pct = monthly_pivot * 100

# Save to CSV
monthly_table_path = OUT_DIR / "monthly_returns_USD.csv"
monthly_pivot_pct.to_csv(monthly_table_path)
print(f"Saved monthly returns table: {monthly_table_path.name}")

# Create a visual table
fig, ax = plt.subplots(figsize=(16, max(6, len(monthly_pivot_pct) * 0.6)))
ax.axis('tight')
ax.axis('off')

# Format cell values
cell_text = []
for idx, row in monthly_pivot_pct.iterrows():
    formatted_row = [f"{idx}"]  # Year
    for val in row:
        if pd.isna(val):
            formatted_row.append("-")
        else:
            formatted_row.append(f"{val:.2f}%")
    cell_text.append(formatted_row)

# Create table
columns = ['Year'] + list(monthly_pivot_pct.columns)
table = ax.table(cellText=cell_text, colLabels=columns, cellLoc='center', 
                 loc='center', bbox=[0, 0, 1, 1])

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Color code cells
for i in range(len(cell_text)):
    for j in range(1, len(columns)):  # Skip Year column
        cell = table[(i + 1, j)]
        val_str = cell_text[i][j]
        if val_str != "-":
            val = float(val_str.replace('%', ''))
            if val > 0:
                cell.set_facecolor('#d4edda')  # Light green
            elif val < 0:
                cell.set_facecolor('#f8d7da')  # Light red
            else:
                cell.set_facecolor('#ffffff')  # White

# Style header
for j in range(len(columns)):
    cell = table[(0, j)]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(weight='bold', color='white')

plt.title('Monthly Returns (USD, %)', fontsize=14, fontweight='bold', pad=20)
table_chart_path = OUT_DIR / "monthly_returns_table_USD.png"
plt.savefig(table_chart_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved monthly returns chart: {table_chart_path.name}")


# --------------- Alpha / Beta Contribution Charts ---------------
print("Creating alpha/beta contribution charts...")

try:
    # Country-level alpha (annual) bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(ab_table.index, ab_table["alpha_ann"] * 100.0)
    ax.set_xlabel("Country")
    ax.set_ylabel("Annual Alpha (%)")
    ax.set_title("Country-Level Annual Alpha vs Benchmark")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    alpha_chart_path = OUT_DIR / "alpha_by_country.png"
    plt.savefig(alpha_chart_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved chart: {alpha_chart_path.name}")

    # Country-level beta bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(ab_table.index, ab_table["beta"])
    ax.set_xlabel("Country")
    ax.set_ylabel("Beta")
    ax.set_title("Country-Level Beta vs Benchmark")
    ax.axhline(1.0, linestyle="--", linewidth=0.8)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    beta_chart_path = OUT_DIR / "beta_by_country.png"
    plt.savefig(beta_chart_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved chart: {beta_chart_path.name}")

except Exception as e:
    print(f"Warning: failed to create alpha/beta charts: {e}")

# --------------- Sector Contribution Charts ---------------
print("Creating sector contribution charts...")

try:
    # 1) Total PnL by sector (bar chart)
    sector_pnl_plot = sector_pnl_total.copy()
    sector_pnl_plot = sector_pnl_plot.sort_values("Daily_PnL_KRW", ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(6, len(sector_pnl_plot) * 0.3)))
    ax.barh(sector_pnl_plot["sector"].astype(str), sector_pnl_plot["Daily_PnL_KRW"])
    ax.set_xlabel("Total PnL (KRW)")
    ax.set_ylabel("Sector")
    ax.set_title("Total Sector Contribution (PnL, KRW)")
    ax.grid(True, linestyle="--", alpha=0.3, axis="x")
    plt.tight_layout()
    sector_bar_path = OUT_DIR / "sector_total_pnl_KRW.png"
    plt.savefig(sector_bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved chart: {sector_bar_path.name}")

    # 2) Cumulative PnL over time for top sectors
    sector_cum = sector_contrib.copy()
    sector_cum = sector_cum.sort_values("Date")
    sector_cum["Cum_PnL_KRW"] = sector_cum.groupby("sector")["Daily_PnL_KRW"].cumsum()

    # Pick top N sectors by absolute total PnL
    top_n = 6
    top_sectors = (
        sector_pnl_total.assign(abs_pnl=sector_pnl_total["Daily_PnL_KRW"].abs())
        .sort_values("abs_pnl", ascending=False)
        .head(top_n)["sector"]
        .tolist()
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    for sec in top_sectors:
        sub = sector_cum[sector_cum["sector"] == sec]
        if sub.empty:
            continue
        ax.plot(sub["Date"], sub["Cum_PnL_KRW"], label=str(sec))

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative PnL (KRW)")
    ax.set_title(f"Cumulative Sector PnL Over Time (Top {top_n} Sectors by |PnL|)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    sector_cum_path = OUT_DIR / "sector_cumulative_pnl_KRW.png"
    plt.savefig(sector_cum_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved chart: {sector_cum_path.name}")

except Exception as e:
    print(f"Warning: failed to create sector charts: {e}")

# --------------- Exports ---------------

ctry_daily.to_csv(OUT_DIR/"country_daily_KRW.csv", index=False)
ctry_daily_usd.to_csv(OUT_DIR/"country_daily_USD.csv", index=False)
bench_ret.to_csv(OUT_DIR/"benchmarks_daily_ret.csv", index=False)
ab_table.to_csv(OUT_DIR/"alpha_beta_by_country.csv")
port_summary.to_csv(OUT_DIR/"portfolio_vs_blend_alpha_beta.csv")
port_daily.to_csv(OUT_DIR/"portfolio_daily_KRW.csv", index=False)
port_daily_usd.to_csv(OUT_DIR/"portfolio_daily_USD.csv", index=False)
port_with_hedge_usd.to_csv(OUT_DIR/"portfolio_daily_USD_with_hedge.csv", index=False)
sector_pnl_total.to_csv(OUT_DIR/"sector_contribution_total_KRW.csv", index=False)
sector_contrib.to_csv(OUT_DIR/"sector_contribution_daily_KRW.csv", index=False)


# Quick printouts
print("\n" + "="*50)
print("Saved outputs to:", OUT_DIR)
print("="*50)
print("\nCSV Files:")
for p in sorted(OUT_DIR.glob("*.csv")):
    print(" -", p.name)
print("\nChart Files:")
for p in sorted(OUT_DIR.glob("*.png")):
    print(" -", p.name)
print("="*50)
