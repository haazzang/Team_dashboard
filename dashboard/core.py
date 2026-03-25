import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import requests
from scipy import stats
import json
import os
import sqlite3
import time
from io import BytesIO
import base64
from datetime import datetime, timezone
from pathlib import Path
import unicodedata

try:
    from google.oauth2.credentials import Credentials
    from google.oauth2.service_account import Credentials as ServiceAccountCredentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaIoBaseUpload
    from google_auth_oauthlib.flow import InstalledAppFlow
except ModuleNotFoundError:
    Credentials = None
    ServiceAccountCredentials = None
    Request = None
    build = None
    HttpError = Exception
    MediaIoBaseUpload = None
    InstalledAppFlow = None

try:
    import openai
except ModuleNotFoundError:
    openai = None  # Streamlit UI still loads; AI report generation will prompt to install

# --- Page Config ---
st.set_page_config(page_title="Portfolio Dashboard", layout="wide")
st.title("Team Portfolio Analysis Dashboard")
try:
    build_ts = datetime.fromtimestamp(Path(__file__).stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    st.sidebar.caption(f"Build: {build_ts}")
except Exception:
    pass

# ==============================================================================
# [Helper Functions] - Fixed Timezone Issues
# ==============================================================================
def _clean_symbol(symbol):
    if symbol is None:
        return None
    if isinstance(symbol, (int, np.integer)):
        return str(symbol)
    if isinstance(symbol, float) and symbol.is_integer():
        return str(int(symbol))
    s = str(symbol).strip()
    if s.lower() in ("nan", "none", ""):
        return None
    if s.endswith(".0") and s.replace(".", "", 1).isdigit():
        return s[:-2]
    return s

def _is_isin_like(value):
    s = _clean_symbol(value)
    if not s:
        return False
    s = s.upper()
    return len(s) == 12 and s[:2].isalpha() and s.isalnum()

def normalize_yf_ticker(symbol, currency=None):
    sym = _clean_symbol(symbol)
    if not sym:
        return None
    if _is_isin_like(sym):
        return None
    # Bloomberg-style values like "3033 HK Equity" -> "3033"
    sym = sym.split()[0].strip()
    if _is_isin_like(sym):
        return None
    if "." in sym:
        return sym
    curr = str(currency).strip().upper() if currency is not None else ""
    if sym.isdigit():
        if curr == "HKD":
            return f"{sym.zfill(4)}.HK"
        if curr == "JPY":
            return f"{sym.zfill(4)}.T"
    if curr == "HKD":
        return f"{sym}.HK"
    if curr == "JPY":
        return f"{sym}.T"
    return sym

def is_etf_value(value):
    if value is None:
        return False
    text = str(value).strip().upper()
    if text in ("", "NAN", "NONE"):
        return False
    keywords = ("ETF", "ETN", "EXCHANGE TRADED FUND", "INDEX FUND", "TRACKER FUND")
    return any(k in text for k in keywords)

def is_etf_product_type(value):
    if value is None:
        return False
    text = str(value).strip().upper()
    if text in ("ETF", "ETN", "TRUST", "FUND"):
        return True
    return is_etf_value(text)

def is_etf_from_info(info):
    if not isinstance(info, dict):
        return False

    quote_type = str(info.get("quoteType", "")).strip().upper()
    if quote_type in ("ETF", "ETN"):
        return True

    if is_etf_value(info.get("sector")):
        return True

    for key in ("longName", "shortName", "fundFamily", "category"):
        if is_etf_value(info.get(key)):
            return True
    return False

@st.cache_data(ttl=3600)
def fetch_sectors_cached(tickers):
    sector_map = {}
    for t in tickers:
        try:
            t_str = str(t).strip()
            if t_str:
                info = yf.Ticker(t_str).info
                if is_etf_from_info(info):
                    sector_map[t] = "ETF"
                else:
                    sector = info.get("sector")
                    sector_map[t] = str(sector).strip() if sector is not None and str(sector).strip() else "Unknown"
            else:
                sector_map[t] = 'Unknown'
        except:
            sector_map[t] = 'Unknown'
    return sector_map

@st.cache_data(ttl=3600)
def fetch_etf_flags_cached(tickers):
    etf_map = {}
    for t in tickers:
        try:
            t_str = str(t).strip()
            if not t_str:
                etf_map[t] = False
                continue
            info = yf.Ticker(t_str).info
            etf_map[t] = is_etf_from_info(info)
        except:
            etf_map[t] = False
    return etf_map

def _normalize_sp500_symbol(symbol):
    sym = _clean_symbol(symbol)
    if not sym:
        return None
    return sym.replace(".", "-")

@st.cache_data(ttl=3600)
def fetch_sp500_weights():
    url = "https://www.slickcharts.com/sp500"
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
    except Exception:
        return pd.DataFrame()

    try:
        from bs4 import BeautifulSoup
    except ModuleNotFoundError:
        return pd.DataFrame()

    soup = BeautifulSoup(resp.text, "html.parser")
    tables = soup.find_all("table")
    for table in tables:
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        if "Symbol" in headers and "Weight" in headers:
            symbol_idx = headers.index("Symbol")
            weight_idx = headers.index("Weight")
            rows = []
            for tr in table.find_all("tr")[1:]:
                cols = [td.get_text(strip=True) for td in tr.find_all("td")]
                if len(cols) <= max(symbol_idx, weight_idx):
                    continue
                symbol = cols[symbol_idx]
                weight_str = cols[weight_idx].replace("%", "").replace(",", "")
                try:
                    weight = float(weight_str) / 100.0
                except ValueError:
                    continue
                rows.append({"Symbol": symbol, "Weight": weight})
            return pd.DataFrame(rows)
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_sp500_sector_map():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
    except Exception:
        return pd.DataFrame()

    try:
        from bs4 import BeautifulSoup
    except ModuleNotFoundError:
        return pd.DataFrame()

    def _norm(text):
        return "".join(str(text).split()).lower()

    soup = BeautifulSoup(resp.text, "html.parser")
    for table in soup.find_all("table"):
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        if not headers:
            continue
        norm_headers = [_norm(h) for h in headers]
        if "symbol" in norm_headers and "gicssector" in norm_headers:
            sym_idx = norm_headers.index("symbol")
            sec_idx = norm_headers.index("gicssector")
            rows = []
            for tr in table.find_all("tr")[1:]:
                cols = [td.get_text(strip=True) for td in tr.find_all("td")]
                if len(cols) <= max(sym_idx, sec_idx):
                    continue
                rows.append({"Symbol": cols[sym_idx], "Sector": cols[sec_idx]})
            return pd.DataFrame(rows)
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_sp500_sector_weights():
    weights = fetch_sp500_weights()
    if weights.empty:
        return pd.Series(dtype=float)
    sector_map = fetch_sp500_sector_map()
    if sector_map.empty:
        return pd.Series(dtype=float)
    merged = weights.merge(sector_map, on="Symbol", how="left")
    merged["Sector"] = merged["Sector"].fillna("Unknown")
    return merged.groupby("Sector")["Weight"].sum().sort_values(ascending=False)

def _normalize_filename(name):
    try:
        return unicodedata.normalize("NFC", name)
    except Exception:
        return name

def _resolve_normalized_path(path):
    if not path:
        return None
    p = Path(path)
    if p.exists():
        return p
    parent = p.parent
    if parent.exists():
        target = _normalize_filename(p.name)
        for f in parent.glob("*.xlsx"):
            if _normalize_filename(f.name) == target:
                return f
    return None

def _find_file_by_name(target_name, search_dirs):
    target = _normalize_filename(target_name)
    for d in search_dirs:
        if d is None or not d.exists():
            continue
        for f in d.glob("*.xlsx"):
            if _normalize_filename(f.name) == target:
                return f
    return None

def get_file_signature(path):
    try:
        stat = Path(path).stat()
        return (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return None

def _read_excel_with_retries(source, *, sheet_name=0, header=None, engine="openpyxl", retries=4, delay=0.3):
    last_error = None
    for attempt in range(retries):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_excel(source, sheet_name=sheet_name, header=header, engine=engine)
        except Exception as exc:
            last_error = exc
            if attempt == retries - 1:
                break
            time.sleep(delay)
    raise last_error

def _parse_portfolio_snapshot_df(df_raw):
    try:
        h_idx = -1
        for i in range(min(20, len(df_raw))):
            row_vals = [str(x).strip() for x in df_raw.iloc[i].values]
            if "기준일자" in row_vals and ("종목명" in row_vals or "종목코드" in row_vals or "심볼" in row_vals):
                h_idx = i
                break
        if h_idx == -1:
            return None, "Header not found"
        raw_cols = [str(c).strip() for c in df_raw.iloc[h_idx].tolist()]
        valid_indices = [i for i, c in enumerate(raw_cols) if c not in ["nan", "None", ""]]
        new_cols = []
        seen = {}
        for i in valid_indices:
            c_str = raw_cols[i]
            if c_str in seen:
                seen[c_str] += 1
                new_cols.append(f"{c_str}_{seen[c_str]}")
            else:
                seen[c_str] = 0
                new_cols.append(c_str)

        df = df_raw.iloc[h_idx + 1 :, valid_indices].copy()
        df.columns = new_cols
        if "기준일자" in df.columns:
            df["기준일자"] = pd.to_datetime(df["기준일자"], errors="coerce")
            df = df.dropna(subset=["기준일자"])

        cols_num = [
            "원화평가금액",
            "원화총평가손익",
            "원화총매매손익",
            "환손익",
            "잔고수량",
            "외화평가금액",
            "평가환율",
        ]
        for c in cols_num:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        return df, None
    except Exception as e:
        return None, str(e)

def load_portfolio_snapshot_upload(uploaded_file):
    try:
        df_raw = _read_excel_with_retries(uploaded_file, sheet_name=0, header=None, engine="openpyxl")
        return _parse_portfolio_snapshot_df(df_raw)
    except Exception as e:
        return None, str(e)

@st.cache_data
def load_portfolio_snapshot(file_path, file_signature):
    try:
        df_raw = _read_excel_with_retries(file_path, sheet_name=0, header=None, engine="openpyxl")
        return _parse_portfolio_snapshot_df(df_raw)
    except Exception as e:
        return None, str(e)

@st.cache_data
def download_benchmarks_all(start_date, end_date):
    """Download benchmarks for US, KR, HK, JP"""
    tickers = {'US': '^GSPC', 'KR': '^KS11', 'HK': '^HSI', 'JP': '^N225'}
    try:
        data = yf.download(list(tickers.values()), start=start_date, end=end_date + pd.Timedelta(days=5), progress=False)
        if 'Adj Close' in data.columns: df = data['Adj Close']
        elif 'Close' in data.columns: df = data['Close']
        else: df = data
        
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # [FIX] Remove Timezone
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        inv_map = {v: k for k, v in tickers.items()}
        df.rename(columns=inv_map, inplace=True)
        return df.ffill().pct_change().fillna(0)
    except:
        return pd.DataFrame()

@st.cache_data
def download_replication_benchmarks(start_date, end_date):
    """Download benchmarks for replication analysis (SPX, NDX)"""
    tickers = {'SPX': '^GSPC', 'NDX': '^NDX'}
    try:
        data = yf.download(list(tickers.values()), start=start_date, end=end_date + pd.Timedelta(days=5), progress=False)
        if 'Adj Close' in data.columns:
            df = data['Adj Close']
        elif 'Close' in data.columns:
            df = data['Close']
        else:
            df = data

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # [FIX] Remove Timezone
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        inv_map = {v: k for k, v in tickers.items()}
        df.rename(columns=inv_map, inplace=True)
        return df.ffill().pct_change().fillna(0)
    except:
        return pd.DataFrame()

@st.cache_data
def download_usdkrw(start_date, end_date):
    """Download USD/KRW Exchange Rate"""
    try:
        fx = yf.download('KRW=X', start=start_date, end=end_date + pd.Timedelta(days=5), progress=False)
        if 'Adj Close' in fx.columns: fx = fx['Adj Close']
        elif 'Close' in fx.columns: fx = fx['Close']
        
        if isinstance(fx.columns, pd.MultiIndex): fx.columns = fx.columns.get_level_values(0)
        
        # [FIX] Remove Timezone
        if isinstance(fx.index, pd.DatetimeIndex) and fx.index.tz is not None:
            fx.index = fx.index.tz_localize(None)
        
        if isinstance(fx, pd.Series): 
            fx = fx.to_frame(name='USD_KRW')
        else: 
            fx.rename(columns={'KRW=X': 'USD_KRW'}, inplace=True)
            if 'USD_KRW' not in fx.columns and not fx.empty:
                 fx.columns = ['USD_KRW']
        
        return fx.ffill()
    except:
        return pd.DataFrame()

@st.cache_data
def download_cross_assets(start_date, end_date):
    assets = {
        'S&P 500': '^GSPC', 'Nasdaq': '^IXIC', 'KOSPI': '^KS11', 
        'USD/KRW': 'KRW=X', 'US 10Y Yield': '^TNX', 'Gold': 'GC=F', 'Crude Oil': 'CL=F'
    }
    try:
        data = yf.download(list(assets.values()), start=start_date, end=end_date + pd.Timedelta(days=5), progress=False)
        if 'Adj Close' in data.columns: df = data['Adj Close']
        elif 'Close' in data.columns: df = data['Close']
        else: df = data
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # [FIX] Remove Timezone
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df.rename(columns={v: k for k, v in assets.items()}, inplace=True)
        return df # Returns Prices
    except:
        return pd.DataFrame()

FACTOR_TARGET_COUNT = 15

@st.cache_data
def download_factors(start_date, end_date, return_prices=False):
    """Download diverse factor proxies (ETFs)"""
    factors = {
        'Global Mkt': 'ACWI', 'Value': 'VLUE', 'Growth': 'IWF', 'Momentum': 'MTUM',
        'Quality': 'QUAL', 'Low Vol': 'USMV', 'Small Cap': 'IWM', 'Emerging': 'EEM', 
        'Bond': 'TLT', 'USD': 'UUP', 'Gold': 'GLD', 'Oil': 'USO',
        'High Beta': 'SPHB', 'Meme': 'MEME', 'Spec Tech': 'ARKK'
    }
    try:
        if pd.isna(start_date) or pd.isna(end_date):
            return pd.DataFrame()
        data = yf.download(list(factors.values()), start=start_date, end=end_date + pd.Timedelta(days=5), progress=False)
        if 'Adj Close' in data.columns: df = data['Adj Close']
        elif 'Close' in data.columns: df = data['Close']
        else: df = data
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # [FIX] Remove Timezone
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        inv_map = {v: k for k, v in factors.items()}
        df.rename(columns=inv_map, inplace=True)
        df = df.ffill()
        if return_prices:
            return df
        return df.pct_change().fillna(0)
    except:
        return pd.DataFrame()

@st.cache_data
def download_global_indices(start_date, end_date):
    """Download global index prices for report context"""
    indices = {'SPX': '^GSPC', 'Hang Seng': '^HSI', 'Nikkei 225': '^N225', 'ACWI': 'ACWI'}
    try:
        data = yf.download(list(indices.values()), start=start_date, end=end_date + pd.Timedelta(days=5), progress=False)
        if 'Adj Close' in data.columns: df = data['Adj Close']
        elif 'Close' in data.columns: df = data['Close']
        else: df = data

        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # Align timezone handling with rest of app
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        df.rename(columns={v: k for k, v in indices.items()}, inplace=True)
        return df.ffill()
    except:
        return pd.DataFrame()

@st.cache_data
def download_price_history(tickers, start_date, end_date):
    tickers = [t for t in tickers if t]
    if not tickers or pd.isna(start_date) or pd.isna(end_date):
        return pd.DataFrame()
    try:
        data = yf.download(tickers, start=start_date, end=end_date + pd.Timedelta(days=5), progress=False)
        if 'Adj Close' in data.columns: df = data['Adj Close']
        elif 'Close' in data.columns: df = data['Close']
        else: df = data
        if isinstance(df, pd.Series):
            df = df.to_frame()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df.ffill()
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_latest_prices(tickers):
    """전일 종가 기준으로 가격 데이터 가져오기"""
    prices = {}
    for t in tickers:
        if not t:
            continue
        try:
            ticker = yf.Ticker(t)
            hist = ticker.history(period="5d")
            if not hist.empty:
                prices[t] = hist['Close'].iloc[-1]
        except:
            pass
    return prices

@st.cache_data(ttl=1800)
def fetch_prev_day_returns(tickers):
    """보유 종목의 최근 거래일 전일 대비 등락률 계산"""
    output_cols = ["YF_Symbol", "전일등락률", "최근거래일", "직전거래일", "최근종가", "직전종가"]
    clean_tickers = tuple(sorted({str(t).strip() for t in tickers if t and str(t).strip()}))
    if not clean_tickers:
        return pd.DataFrame(columns=output_cols)

    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.Timedelta(days=20)
    prices = download_price_history(list(clean_tickers), start_date, end_date)
    if prices.empty:
        return pd.DataFrame(columns=output_cols)

    rows = []
    for ticker in clean_tickers:
        if ticker not in prices.columns:
            continue

        s = prices[ticker].dropna()
        if len(s) < 2:
            rows.append({
                "YF_Symbol": ticker,
                "전일등락률": np.nan,
                "최근거래일": pd.Timestamp(s.index[-1]) if len(s) == 1 else pd.NaT,
                "직전거래일": pd.NaT,
                "최근종가": float(s.iloc[-1]) if len(s) == 1 else np.nan,
                "직전종가": np.nan,
            })
            continue

        latest_close = float(s.iloc[-1])
        prev_close = float(s.iloc[-2])
        if pd.isna(prev_close) or prev_close == 0:
            pct_chg = np.nan
        else:
            pct_chg = latest_close / prev_close - 1

        rows.append({
            "YF_Symbol": ticker,
            "전일등락률": pct_chg,
            "최근거래일": pd.Timestamp(s.index[-1]),
            "직전거래일": pd.Timestamp(s.index[-2]),
            "최근종가": latest_close,
            "직전종가": prev_close,
        })

    return pd.DataFrame(rows, columns=output_cols)

def simulate_portfolio_nav(holdings_df, weight_adjustments, new_positions, base_nav, simulation_days=30, additional_cash=0, original_nav=None):
    """
    포트폴리오 시뮬레이션 수행

    Args:
        holdings_df: 현재 보유 종목 DataFrame (YF_Symbol, Weight, 원화평가금액 포함)
        weight_adjustments: dict {YF_Symbol: new_weight} - 기존 종목 비중 조절
        new_positions: list of dict [{"ticker": str, "weight": float, "market": str}] - 신규 종목 추가
        base_nav: 시뮬레이션 기준 NAV (원화) - 추가 현금 포함된 금액
        simulation_days: 시뮬레이션 기간 (일)
        additional_cash: 추가 투입 현금 (원화)
        original_nav: 원래 NAV (추가 현금 미포함)

    Returns:
        dict with simulation results
    """
    # original_nav가 없으면 base_nav 사용
    if original_nav is None:
        original_nav = base_nav

    # 1. 모든 티커 수집 (기존 + 신규)
    all_tickers = list(holdings_df["YF_Symbol"].dropna().unique())
    for pos in new_positions:
        if pos["ticker"] and pos["ticker"] not in all_tickers:
            all_tickers.append(pos["ticker"])

    if not all_tickers:
        return None

    # 2. 과거 가격 데이터 다운로드
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.Timedelta(days=simulation_days + 10)

    prices = download_price_history(all_tickers, start_date, end_date)
    if prices.empty:
        return None

    # 최근 simulation_days 일만 사용
    prices = prices.tail(simulation_days + 1)

    # 3. 수익률 계산
    returns = prices.pct_change().fillna(0)

    # 4. 원래 비중으로 포트폴리오 수익률 계산
    original_weights = holdings_df.set_index("YF_Symbol")["Weight"].to_dict()

    # 5. 시뮬레이션 비중 계산 (추가 현금 반영)
    if additional_cash > 0:
        # 추가 현금 투입 시: 기존 포지션 유지 + 추가 매수
        # 기존 종목의 실제 금액 유지, 새로운 NAV 기준으로 비중 재계산
        sim_weights = {}

        # 기존 종목: 기존 금액 / 새 NAV = 새 비중
        for ticker, orig_weight in original_weights.items():
            original_value = orig_weight * original_nav  # 기존 금액
            sim_weights[ticker] = original_value / base_nav  # 새 NAV 기준 비중

        # 비중 조절이 있는 경우 (증가분만 추가 현금으로 매수)
        for ticker, new_weight in weight_adjustments.items():
            if ticker in sim_weights:
                old_weight_in_new_nav = sim_weights[ticker]
                # 목표 비중이 현재 비중보다 높으면 추가 매수
                if new_weight > old_weight_in_new_nav:
                    sim_weights[ticker] = new_weight
                else:
                    # 비중 축소는 매도
                    sim_weights[ticker] = new_weight

        # 신규 종목 추가 (추가 현금으로 매수)
        for pos in new_positions:
            if pos["ticker"] and pos["weight"] > 0:
                sim_weights[pos["ticker"]] = pos["weight"]
    else:
        # 추가 현금 없음: 기존 로직
        sim_weights = original_weights.copy()

        # 기존 종목 비중 조절 적용
        for ticker, new_weight in weight_adjustments.items():
            if ticker in sim_weights:
                sim_weights[ticker] = new_weight

        # 신규 종목 추가
        for pos in new_positions:
            if pos["ticker"] and pos["weight"] > 0:
                sim_weights[pos["ticker"]] = pos["weight"]

    # 비중 정규화 (합이 1이 되도록)
    total_weight = sum(sim_weights.values())
    if total_weight > 0:
        sim_weights = {k: v / total_weight for k, v in sim_weights.items()}

    # 6. 원래 포트폴리오 NAV 계산 (원래 NAV 기준)
    original_port_returns = pd.Series(0.0, index=returns.index)
    for ticker, weight in original_weights.items():
        if ticker in returns.columns:
            original_port_returns += returns[ticker] * weight

    orig_nav_series = original_nav * (1 + original_port_returns).cumprod()

    # 7. 시뮬레이션 포트폴리오 NAV 계산 (새 NAV 기준)
    sim_port_returns = pd.Series(0.0, index=returns.index)
    for ticker, weight in sim_weights.items():
        if ticker in returns.columns:
            sim_port_returns += returns[ticker] * weight

    sim_nav_series = base_nav * (1 + sim_port_returns).cumprod()

    # 8. 섹터 정보 수집
    original_sectors = holdings_df.set_index("YF_Symbol")["섹터"].to_dict() if "섹터" in holdings_df.columns else {}

    # 신규 종목 섹터 조회
    new_tickers_for_sector = [pos["ticker"] for pos in new_positions if pos["ticker"] and pos["ticker"] not in original_sectors]
    if new_tickers_for_sector:
        new_sector_map = fetch_sectors_cached(tuple(new_tickers_for_sector))
        original_sectors.update(new_sector_map)

    return {
        "original_nav": orig_nav_series,
        "sim_nav": sim_nav_series,
        "original_weights": original_weights,
        "sim_weights": sim_weights,
        "returns": returns,
        "prices": prices,
        "sector_map": original_sectors,
        "additional_cash": additional_cash,
        "base_nav": base_nav,
        "original_nav_value": original_nav,
    }

def calculate_factor_exposure(weights, returns, simulation_days=30):
    """
    팩터 익스포저 계산 (베타 기반)

    Args:
        weights: dict {ticker: weight}
        returns: DataFrame of returns
        simulation_days: 분석 기간

    Returns:
        dict of factor exposures
    """
    # 팩터 ETF 정의
    factor_etfs = {
        "Market (SPY)": "SPY",
        "Value (IWD)": "IWD",
        "Growth (IWF)": "IWF",
        "Momentum (MTUM)": "MTUM",
        "Quality (QUAL)": "QUAL",
        "Low Vol (USMV)": "USMV",
        "Small Cap (IWM)": "IWM",
        "EM (EEM)": "EEM",
    }

    # 팩터 가격 다운로드
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.Timedelta(days=simulation_days + 10)

    factor_prices = download_price_history(list(factor_etfs.values()), start_date, end_date)
    if factor_prices.empty:
        return {}

    factor_returns = factor_prices.pct_change().fillna(0).tail(simulation_days)

    # 포트폴리오 수익률 계산
    port_returns = pd.Series(0.0, index=returns.index)
    for ticker, weight in weights.items():
        if ticker in returns.columns:
            port_returns += returns[ticker] * weight

    port_returns = port_returns.tail(simulation_days)

    # 각 팩터에 대한 베타 계산
    exposures = {}
    for factor_name, factor_ticker in factor_etfs.items():
        if factor_ticker not in factor_returns.columns:
            continue
        factor_ret = factor_returns[factor_ticker]

        # 공통 인덱스
        common_idx = port_returns.index.intersection(factor_ret.index)
        if len(common_idx) < 10:
            continue

        p_ret = port_returns.loc[common_idx]
        f_ret = factor_ret.loc[common_idx]

        # 베타 계산
        cov = np.cov(p_ret, f_ret)[0, 1]
        var = np.var(f_ret)
        if var > 0:
            beta = cov / var
            exposures[factor_name] = beta

    return exposures

def calculate_portfolio_beta_multi_period(weights, lookback_periods=[30, 60, 90]):
    """
    여러 기간에 대한 포트폴리오 베타 계산 (국가별 벤치마크)

    Args:
        weights: dict {ticker: weight}
        lookback_periods: 베타 계산 기간 리스트 (일)

    Returns:
        dict with beta values for each benchmark and period
    """
    if not weights:
        return {}

    # 벤치마크 정의
    benchmarks = {
        "US (S&P 500)": "^GSPC",
        "KR (KOSPI)": "^KS11",
        "HK (HSI)": "^HSI",
        "JP (Nikkei)": "^N225",
    }

    tickers = [t for t in weights.keys() if t]
    if not tickers:
        return {}

    # 최대 기간 + 여유분 데이터 다운로드
    max_period = max(lookback_periods)
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.Timedelta(days=max_period + 20)

    # 포트폴리오 종목 + 벤치마크 가격 다운로드
    all_tickers = tickers + list(benchmarks.values())
    prices = download_price_history(all_tickers, start_date, end_date)
    if prices.empty:
        return {}

    returns = prices.pct_change().dropna()

    # 포트폴리오 수익률 계산
    port_returns_full = pd.Series(0.0, index=returns.index)
    for ticker, weight in weights.items():
        if ticker in returns.columns:
            port_returns_full += returns[ticker] * weight

    # 각 기간, 각 벤치마크에 대해 베타 계산
    results = {}
    for period in lookback_periods:
        period_key = f"{period}D"
        results[period_key] = {}

        port_returns = port_returns_full.tail(period)

        for bench_name, bench_ticker in benchmarks.items():
            if bench_ticker not in returns.columns:
                continue

            bench_returns = returns[bench_ticker].tail(period)

            # 공통 인덱스
            common_idx = port_returns.index.intersection(bench_returns.index)
            if len(common_idx) < 10:
                continue

            p_ret = port_returns.loc[common_idx]
            b_ret = bench_returns.loc[common_idx]

            # 베타 계산
            cov = np.cov(p_ret, b_ret)[0, 1]
            var = np.var(b_ret)
            if var > 0:
                beta = cov / var
                results[period_key][bench_name] = beta

    return results

def calculate_portfolio_factor_exposure(weights, lookback_days=60):
    """
    포트폴리오 팩터 익스포저 계산

    Args:
        weights: dict {ticker: weight}
        lookback_days: 분석 기간

    Returns:
        dict of factor exposures
    """
    if not weights:
        return {}

    # 팩터 ETF 정의
    factor_etfs = {
        "Market": "SPY",
        "Value": "IWD",
        "Growth": "IWF",
        "Momentum": "MTUM",
        "Quality": "QUAL",
        "Low Vol": "USMV",
        "Small Cap": "IWM",
        "EM": "EEM",
    }

    tickers = [t for t in weights.keys() if t]
    if not tickers:
        return {}

    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.Timedelta(days=lookback_days + 20)

    # 포트폴리오 종목 + 팩터 ETF 가격 다운로드
    all_tickers = tickers + list(factor_etfs.values())
    prices = download_price_history(all_tickers, start_date, end_date)
    if prices.empty:
        return {}

    returns = prices.pct_change().dropna().tail(lookback_days)

    # 포트폴리오 수익률 계산
    port_returns = pd.Series(0.0, index=returns.index)
    for ticker, weight in weights.items():
        if ticker in returns.columns:
            port_returns += returns[ticker] * weight

    # 각 팩터에 대한 베타 계산
    exposures = {}
    for factor_name, factor_ticker in factor_etfs.items():
        if factor_ticker not in returns.columns:
            continue

        factor_ret = returns[factor_ticker]
        common_idx = port_returns.index.intersection(factor_ret.index)
        if len(common_idx) < 10:
            continue

        p_ret = port_returns.loc[common_idx]
        f_ret = factor_ret.loc[common_idx]

        cov = np.cov(p_ret, f_ret)[0, 1]
        var = np.var(f_ret)
        if var > 0:
            exposures[factor_name] = cov / var

    return exposures

@st.cache_data(ttl=3600)
def get_exchange_rates():
    """환율 정보 가져오기 (USD 기준)"""
    fx_tickers = {
        "KRW": "KRW=X",   # USD/KRW
        "JPY": "JPY=X",   # USD/JPY
        "HKD": "HKD=X",   # USD/HKD
    }
    rates = {"USD": 1.0}  # USD는 기본값 1

    for currency, ticker in fx_tickers.items():
        try:
            data = yf.Ticker(ticker).history(period="5d")
            if not data.empty:
                rates[currency] = data['Close'].iloc[-1]
        except:
            # 기본값 설정
            if currency == "KRW":
                rates[currency] = 1400.0
            elif currency == "JPY":
                rates[currency] = 150.0
            elif currency == "HKD":
                rates[currency] = 7.8

    return rates

def calculate_trade_shares(original_weights, sim_weights, total_nav_krw, holdings_df, new_positions, original_nav=None, additional_cash=0):
    """
    매매해야 하는 주수 계산

    Args:
        original_weights: dict {ticker: weight} 원래 비중
        sim_weights: dict {ticker: weight} 시뮬레이션 비중
        total_nav_krw: 총 NAV (KRW) - 시뮬레이션 기준 (추가 현금 포함)
        holdings_df: 보유 종목 DataFrame
        new_positions: 신규 종목 리스트
        original_nav: 원래 NAV (추가 현금 미포함)
        additional_cash: 추가 투입 현금 (KRW)

    Returns:
        list of dict with trade details
    """
    trades = []

    # original_nav가 없으면 total_nav_krw 사용
    if original_nav is None:
        original_nav = total_nav_krw

    # 환율 가져오기
    fx_rates = get_exchange_rates()

    # 모든 티커 수집
    all_tickers = list(set(original_weights.keys()) | set(sim_weights.keys()))

    # 각 티커에 대해 최신 종가 가져오기
    ticker_prices = {}
    ticker_currencies = {}

    for ticker in all_tickers:
        try:
            data = yf.Ticker(ticker).history(period="5d")
            if not data.empty:
                ticker_prices[ticker] = data['Close'].iloc[-1]

            # 통화 결정
            if ticker.endswith(".T"):
                ticker_currencies[ticker] = "JPY"
            elif ticker.endswith(".HK"):
                ticker_currencies[ticker] = "HKD"
            elif ticker.endswith(".KS") or ticker.endswith(".KQ"):
                ticker_currencies[ticker] = "KRW"
            else:
                ticker_currencies[ticker] = "USD"
        except:
            pass

    # holdings_df에서 통화 정보 업데이트
    if "통화" in holdings_df.columns:
        for idx, row in holdings_df.iterrows():
            ticker = row.get("YF_Symbol")
            if ticker and ticker in ticker_currencies:
                currency = row.get("통화", "USD")
                if currency and str(currency).strip().upper() not in ["", "NAN", "N/A"]:
                    ticker_currencies[ticker] = str(currency).strip().upper()

    # 매매 계산
    for ticker in all_tickers:
        orig_w = original_weights.get(ticker, 0)
        sim_w = sim_weights.get(ticker, 0)
        weight_diff = sim_w - orig_w

        # 현지 통화 가격
        local_price = ticker_prices.get(ticker)
        if local_price is None:
            continue

        currency = ticker_currencies.get(ticker, "USD")

        # 환율 (1 USD = X 현지통화)
        fx_rate = fx_rates.get(currency, 1.0)

        # USD/KRW 환율
        usd_krw = fx_rates.get("KRW", 1400.0)

        # 목표 금액 변화 계산 (추가 현금 고려)
        if additional_cash > 0:
            # 추가 현금 모드: 원래 금액과 목표 금액의 차이 계산
            original_value_krw = orig_w * original_nav  # 기존 보유 금액
            target_value_krw = sim_w * total_nav_krw    # 목표 금액 (새 NAV 기준)
            target_value_change_krw = target_value_krw - original_value_krw
        else:
            # 일반 모드: 비중 차이로 계산
            target_value_change_krw = (sim_w - orig_w) * total_nav_krw

        if abs(target_value_change_krw) < 1000:  # 1000원 미만 변화는 무시
            continue

        # 현지 통화 금액 변화
        if currency == "KRW":
            target_value_change_local = target_value_change_krw
        else:
            # KRW -> USD -> 현지통화
            target_value_change_usd = target_value_change_krw / usd_krw
            if currency == "USD":
                target_value_change_local = target_value_change_usd
            else:
                target_value_change_local = target_value_change_usd * fx_rate

        # 주수 계산
        shares = target_value_change_local / local_price if local_price > 0 else 0
        shares_rounded = int(round(shares))

        # 종목명 찾기
        name_row = holdings_df[holdings_df["YF_Symbol"] == ticker]
        if len(name_row) > 0:
            name = name_row["종목명"].values[0]
        else:
            # 신규 종목인 경우
            name = ticker

        # 매매 방향
        if shares_rounded > 0:
            action = "매수"
        elif shares_rounded < 0:
            action = "매도"
        else:
            continue

        trades.append({
            "티커": ticker,
            "종목명": name,
            "매매": action,
            "주수": abs(shares_rounded),
            "현지통화가격": local_price,
            "통화": currency,
            "원래비중": orig_w,
            "목표비중": sim_w,
            "비중변화": weight_diff,
            "매매금액(현지)": abs(shares_rounded * local_price),
            "매매금액(KRW)": abs(target_value_change_krw),
        })

    return trades

def calculate_portfolio_volatility(weights, lookback_days=30):
    """
    포트폴리오 변동성 계산 (30일 기준 표준편차)

    Args:
        weights: dict {ticker: weight}
        lookback_days: 변동성 계산 기간 (기본 30일)

    Returns:
        dict with volatility metrics
    """
    if not weights:
        return None

    tickers = [t for t in weights.keys() if t]
    if not tickers:
        return None

    # 가격 데이터 다운로드
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.Timedelta(days=lookback_days + 10)

    prices = download_price_history(tickers, start_date, end_date)
    if prices.empty:
        return None

    # 최근 lookback_days 일만 사용
    prices = prices.tail(lookback_days + 1)
    returns = prices.pct_change().dropna()

    if returns.empty or len(returns) < 5:
        return None

    # 포트폴리오 일일 수익률 계산
    port_returns = pd.Series(0.0, index=returns.index)
    total_weight = 0
    for ticker, weight in weights.items():
        if ticker in returns.columns:
            port_returns += returns[ticker] * weight
            total_weight += weight

    if total_weight == 0:
        return None

    # 변동성 계산 (연율화)
    daily_vol = port_returns.std()
    annual_vol = daily_vol * np.sqrt(252)

    # 개별 종목 변동성
    individual_vols = {}
    for ticker in tickers:
        if ticker in returns.columns:
            ind_daily_vol = returns[ticker].std()
            individual_vols[ticker] = ind_daily_vol * np.sqrt(252)

    # 최대 손실 (기간 내)
    cumulative = (1 + port_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # VaR (95%, 99%)
    var_95 = np.percentile(port_returns, 5)
    var_99 = np.percentile(port_returns, 1)

    return {
        "daily_volatility": daily_vol,
        "annual_volatility": annual_vol,
        "max_drawdown": max_drawdown,
        "var_95": var_95,
        "var_99": var_99,
        "individual_vols": individual_vols,
        "returns": port_returns,
    }

def calculate_portfolio_returns(weights, lookback_days=60):
    """
    포트폴리오 일일 수익률 계산 (lookback_days 기준)
    """
    if not weights:
        return pd.Series(dtype=float)

    tickers = [t for t in weights.keys() if t]
    if not tickers:
        return pd.Series(dtype=float)

    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.Timedelta(days=lookback_days + 10)

    prices = download_price_history(tickers, start_date, end_date)
    if prices.empty:
        return pd.Series(dtype=float)

    prices = prices.tail(lookback_days + 1)
    returns = prices.pct_change().dropna()
    if returns.empty:
        return pd.Series(dtype=float)

    valid_weights = {t: w for t, w in weights.items() if t in returns.columns and w is not None}
    total_weight = sum(valid_weights.values())
    if total_weight == 0:
        return pd.Series(dtype=float)

    port_returns = pd.Series(0.0, index=returns.index)
    for ticker, weight in valid_weights.items():
        port_returns += returns[ticker] * (weight / total_weight)

    return port_returns.dropna()

def align_factor_returns(port_index, factor_prices):
    if factor_prices is None or factor_prices.empty or port_index is None or len(port_index) == 0:
        return pd.DataFrame()
    aligned = factor_prices.reindex(port_index, method='ffill').ffill().bfill()
    return aligned.pct_change().fillna(0)

def calculate_holdings_beta(curr_hold, benchmark_map, end_date=None, lookback_days=252, min_obs=30):
    if curr_hold is None or curr_hold.empty:
        return {}
    if '원화평가금액' not in curr_hold.columns:
        return {}

    hold = curr_hold.copy()
    if 'Symbol' in hold.columns and 'Ticker_ID' in hold.columns:
        base_symbol = hold['Symbol'].where(hold['Symbol'].notna(), hold['Ticker_ID'])
    elif 'Symbol' in hold.columns:
        base_symbol = hold['Symbol']
    else:
        base_symbol = hold['Ticker_ID']
    currencies = hold['통화'] if '통화' in hold.columns else [None] * len(hold)
    hold['YF_Ticker'] = [
        normalize_yf_ticker(sym, cur) for sym, cur in zip(base_symbol, currencies)
    ]
    hold = hold.dropna(subset=['YF_Ticker'])
    hold['원화평가금액'] = pd.to_numeric(hold['원화평가금액'], errors='coerce').fillna(0)
    weights = hold.groupby('YF_Ticker')['원화평가금액'].sum()
    weights = weights[weights != 0]
    total = weights.sum()
    if total == 0:
        return {}
    weights = weights / total

    end_dt = end_date
    if end_dt is None or pd.isna(end_dt):
        end_dt = pd.Timestamp.today().normalize()
    end_dt = min(end_dt, pd.Timestamp.today().normalize())
    start_dt = end_dt - pd.Timedelta(days=int(lookback_days * 1.6))

    tickers = sorted(set(list(weights.index) + list(benchmark_map.values())))
    prices = download_price_history(tickers, start_dt, end_dt)
    if prices.empty:
        return {}
    returns = prices.pct_change()

    betas = {}
    for label, bench_ticker in benchmark_map.items():
        if bench_ticker not in returns.columns:
            continue
        bench_ret = returns[bench_ticker]
        weighted_sum = 0.0
        weight_total = 0.0
        for ticker, weight in weights.items():
            if ticker not in returns.columns:
                continue
            df = pd.concat([returns[ticker], bench_ret], axis=1).dropna()
            if len(df) < min_obs:
                continue
            cov = df.iloc[:, 0].cov(df.iloc[:, 1])
            var = df.iloc[:, 1].var()
            if pd.isna(var) or var == 0:
                continue
            beta = cov / var
            weighted_sum += weight * beta
            weight_total += weight
        if weight_total > 0:
            betas[label] = weighted_sum / weight_total
    return betas

def perform_factor_regression(port_ret, factor_ret):
    try:
        df = pd.concat([port_ret, factor_ret], axis=1).dropna()
        if len(df) < 3:
            return None, None, None
        factor_cols = list(df.columns[1:])
        max_factors = min(FACTOR_TARGET_COUNT, len(df) - 2)
        if len(factor_cols) > max_factors:
            keep = df[factor_cols].var().sort_values(ascending=False).index[:max_factors]
            df = pd.concat([df.iloc[:, [0]], df[keep]], axis=1)
            factor_cols = list(keep)
        Y, X = df.iloc[:, 0].values, df.iloc[:, 1:].values
        X_w_const = np.column_stack([np.ones(len(X)), X])
        betas, residuals, rank, s = np.linalg.lstsq(X_w_const, Y, rcond=None)
        
        names = ['Alpha'] + list(df.columns[1:])
        exposures = pd.Series(betas, index=names)
        contrib = pd.DataFrame(index=df.index)
        for i, f in enumerate(names[1:]): contrib[f] = df.iloc[:, 1+i] * betas[i+1]
        contrib['Alpha'] = betas[0]
        contrib['Unexplained'] = Y - (X_w_const @ betas)
        
        ss_tot = np.sum((Y - np.mean(Y))**2)
        ss_res = np.sum((Y - (X_w_const @ betas)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
        return exposures, contrib, r2
    except: return None, None, None

def calculate_alpha_beta(port_ret, bench_ret):
    try:
        df = pd.concat([port_ret, bench_ret], axis=1).dropna()
        df.columns = ['Port', 'Bench']
        if len(df) < 10: return np.nan, np.nan, np.nan
        slope, intercept, r_value, p_value, std_err = stats.linregress(df['Bench'], df['Port'])
        return intercept * 252, slope, r_value**2
    except: return np.nan, np.nan, np.nan

def calculate_rolling_beta(port_ret, bench_ret, window=60):
    try:
        df = pd.concat([port_ret, bench_ret], axis=1).dropna()
        if len(df) < window:
            return pd.Series(dtype=float)
        df.columns = ['Port', 'Bench']
        rolling_cov = df['Port'].rolling(window).cov(df['Bench'])
        rolling_var = df['Bench'].rolling(window).var()
        beta = rolling_cov / rolling_var
        return beta.dropna()
    except Exception:
        return pd.Series(dtype=float)

def calculate_rolling_r2(port_ret, bench_ret, window=60):
    try:
        df = pd.concat([port_ret, bench_ret], axis=1).dropna()
        if len(df) < window:
            return pd.Series(dtype=float)
        df.columns = ['Port', 'Bench']
        rolling_corr = df['Port'].rolling(window).corr(df['Bench'])
        rolling_r2 = rolling_corr ** 2
        return rolling_r2.dropna()
    except Exception:
        return pd.Series(dtype=float)

def create_manual_html_table(df, title=None):
    html = ''
    if title: html += f'<h5 style="margin-top:20px; margin-bottom:10px;">{title}</h5>'
    html += '<div style="overflow-x:auto;"><table style="width:100%; border-collapse: collapse; font-size: 0.9rem;">'
    html += '<thead style="background-color: #f8f9fa; border-bottom: 2px solid #dee2e6; color: black;"><tr>' 
    for col in df.columns:
        html += f'<th style="padding: 12px; text-align: center; white-space: nowrap;">{col}</th>'
    html += '</tr></thead><tbody>'
    for _, row in df.iterrows():
        html += '<tr style="border-bottom: 1px solid #dee2e6;">'
        for i, val in enumerate(row):
            align = 'left' if i == 0 else 'right'
            color = 'inherit'
            val_str = str(val)
            if '%' in val_str:
                if '-' in val_str: color = '#dc3545'
                else: color = '#198754'
            html += f'<td style="padding: 10px; text-align: {align}; color: {color}; white-space: nowrap;">{val}</td>'
        html += '</tr>'
    html += '</tbody></table></div>'
    return html


# ==============================================================================
# [AI Helper Functions: OpenAI & DeepSeek]
# ==============================================================================
def get_openai_api_key():
    """Resolve OpenAI API key, preferring the sidebar input (user-provided)."""
    if openai is None:
        return None
    api_key = st.sidebar.text_input(
        '🔑 OpenAI API Key',
        type='password',
        help='Key is used only for report generation. Set OPENAI_API_KEY env/secret or enter here.',
        key='openai_api_key_input',
    )
    if api_key:
        openai.api_key = api_key
        return api_key

    if getattr(openai, 'api_key', None):
        return openai.api_key

    try:
        if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
            api_key = st.secrets['OPENAI_API_KEY']
    except Exception:
        api_key = None
    if not api_key:
        api_key = os.environ.get('OPENAI_API_KEY', '')
    if api_key:
        openai.api_key = api_key
    return api_key


def get_deepseek_api_key():
    """Resolve DeepSeek API key (sidebar input first)."""
    api_key = st.sidebar.text_input(
        '🧠 DeepSeek API Key',
        type='password',
        help='Key is used only for report generation. Set DEEPSEEK_API_KEY env/secret or enter here.',
        key='deepseek_api_key_input',
    )
    if api_key:
        return api_key
    try:
        if hasattr(st, 'secrets') and 'DEEPSEEK_API_KEY' in st.secrets:
            api_key = st.secrets['DEEPSEEK_API_KEY']
    except Exception:
        api_key = None
    if not api_key:
        api_key = os.environ.get('DEEPSEEK_API_KEY', '')
    return api_key


def _build_openai_compatible_client(api_key, base_url=None):
    """Return a client compatible with both openai>=1.0 and <=0.28 styles."""
    if openai is None:
        return None
    try:
        from openai import OpenAI  # type: ignore
        return OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    except Exception:
        openai.api_key = api_key
        if base_url:
            openai.api_base = base_url
        return openai


def _translate_openai_error(err, provider="openai"):
    """Map OpenAI/DeepSeek errors to clearer, user-facing messages."""
    msg = str(err)
    lower = msg.lower()
    if "insufficient_balance" in lower or "insufficient balance" in lower or "error code: 402" in lower:
        return RuntimeError(f"{provider.capitalize()} balance/credit is exhausted. Please top up or use another API key.")
    if "insufficient_quota" in lower or "exceeded your current quota" in lower:
        return RuntimeError(f"{provider.capitalize()} quota exceeded. Please check plan/billing or try another API key.")
    if "invalid_api_key" in lower or "incorrect api key" in lower:
        return RuntimeError("Invalid API key. Please re-enter a valid key in the sidebar.")
    return err if isinstance(err, RuntimeError) else RuntimeError(msg)


def _calc_risk_metrics(ret_series):
    try:
        if ret_series is None:
            return {}
        if isinstance(ret_series, pd.Series):
            clean = ret_series.dropna()
        else:
            clean = pd.Series(ret_series).dropna()
        if clean.size < 2:
            return {}
        std = clean.std()
        if std == 0 or np.isnan(std):
            sharpe = np.nan
        else:
            sharpe = (clean.mean() / std) * np.sqrt(252)
        cum = (1 + clean).cumprod()
        peak = cum.cummax()
        drawdown = cum / peak - 1
        max_dd = drawdown.min()
        ann_vol = std * np.sqrt(252)
        return {
            'sharpe': float(sharpe),
            'max_drawdown': float(max_dd),
            'annual_vol': float(ann_vol),
        }
    except Exception:
        return {}


def _calc_perf_metrics(ret_series):
    try:
        if ret_series is None:
            return {}
        clean = ret_series.dropna() if isinstance(ret_series, pd.Series) else pd.Series(ret_series).dropna()
        if clean.size < 2:
            return {}
        total_return = (1 + clean).prod() - 1
        ann_factor = 252 / clean.size
        cagr = (1 + total_return) ** ann_factor - 1 if ann_factor > 0 else np.nan
        vol = clean.std() * np.sqrt(252)
        sharpe = (clean.mean() / clean.std()) * np.sqrt(252) if clean.std() > 0 else np.nan
        downside = clean[clean < 0]
        downside_std = downside.std()
        sortino = (clean.mean() / downside_std) * np.sqrt(252) if downside_std and downside_std > 0 else np.nan
        cum = (1 + clean).cumprod()
        peak = cum.cummax()
        max_dd = (cum / peak - 1).min()
        calmar = cagr / abs(max_dd) if max_dd and max_dd != 0 else np.nan
        win_rate = (clean > 0).sum() / (clean != 0).sum() if (clean != 0).sum() > 0 else np.nan
        return {
            'Total Return': float(total_return),
            'CAGR': float(cagr),
            'Annualized Volatility': float(vol),
            'Sharpe Ratio': float(sharpe),
            'Sortino Ratio': float(sortino),
            'Max Drawdown': float(max_dd),
            'Calmar Ratio': float(calmar),
            'Win Rate': float(win_rate),
        }
    except Exception:
        return {}


def _clean_metric_map(metric_map):
    if not isinstance(metric_map, dict):
        return {}
    cleaned = {}
    for k, v in metric_map.items():
        try:
            if v is None:
                cleaned[k] = None
            elif isinstance(v, (float, np.floating)) and not np.isfinite(v):
                cleaned[k] = None
            else:
                cleaned[k] = float(v)
        except Exception:
            cleaned[k] = None
    return cleaned


def _clean_risk_metrics(risk):
    if not isinstance(risk, dict):
        return {}
    portfolio = _clean_metric_map(risk.get('portfolio', {}))
    benchmarks_raw = risk.get('benchmarks', {})
    benchmarks = {}
    if isinstance(benchmarks_raw, dict):
        for name, metric_map in benchmarks_raw.items():
            if isinstance(metric_map, dict):
                benchmarks[name] = _clean_metric_map(metric_map)
    return {'portfolio': portfolio, 'benchmarks': benchmarks}


def _safe_float_or_none(value):
    try:
        if value is None:
            return None
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return float(value)
    except Exception:
        return None


def _serialize_stats_for_gpt(stats_res):
    out = {}
    for period, d in stats_res.items():
        if not d:
            continue
        try:
            top5 = d.get('top5')
            bot5 = d.get('bot5')
            ctry = d.get('ctry')
            sect = d.get('sect')
            factor = d.get('factor')
            idx = d.get('indices')
            risk = d.get('risk')
            hedge_contrib = d.get('hedge_contrib')
            hedge_pnl_krw = d.get('hedge_pnl_krw')
            out[period] = {
                'label': d.get('label'),
                'total_return': float(d.get('ret', 0.0)),
                'total_pnl_krw': float(d.get('pnl', 0.0)),
                'top5_contributors': top5.to_dict(orient='records') if hasattr(top5, 'to_dict') and not top5.empty else [],
                'bottom5_detractors': bot5.to_dict(orient='records') if hasattr(bot5, 'to_dict') and not bot5.empty else [],
                'country_contrib': ctry.to_dict() if hasattr(ctry, 'to_dict') else {},
                'sector_contrib': sect.to_dict() if hasattr(sect, 'to_dict') else {},
                'factor_contrib': factor.to_dict() if hasattr(factor, 'to_dict') else {},
                'indices_return': idx.to_dict() if hasattr(idx, 'to_dict') else {},
                'hedge_contrib': _safe_float_or_none(hedge_contrib),
                'hedge_pnl_krw': _safe_float_or_none(hedge_pnl_krw),
                'risk_metrics': _clean_risk_metrics(risk),
            }
        except Exception:
            continue
    return out


def generate_ai_weekly_report(stats_res, report_date, user_comment='', provider='openai', language='English'):
    """Generate weekly report via selected LLM provider."""
    if provider == 'deepseek':
        api_key = get_deepseek_api_key()
        base_url = 'https://api.deepseek.com'
        model = 'deepseek-chat'
    else:
        api_key = get_openai_api_key()
        base_url = None
        model = 'gpt-4o-mini'

    if not api_key:
        raise RuntimeError("API key is missing. Please enter it in the sidebar.")

    client = _build_openai_compatible_client(api_key, base_url=base_url)
    if client is None:
        raise RuntimeError("OpenAI-compatible client not available. Install the 'openai' package.")

    payload = _serialize_stats_for_gpt(stats_res)
    system_prompt = (
        "You are a professional global equity long-short hedge fund PM and quant alpha researcher. "
        "You write weekly performance reports that are analytical, data-driven, and concise. "
        "You NEVER invent numbers; you only interpret the structured data you are given. "
        "Assume the base currency is KRW. The audience is an internal investment committee."
    )
    comment_block = ""
    if user_comment:
        comment_block = (
            "\n\nAdditional PM comments (can be Korean or English, keep the intent but clean up the wording):\n"
            + user_comment
        )
    norm_lang = language.strip().lower()
    is_korean = norm_lang.startswith("ko") or "한국" in norm_lang or "korean" in norm_lang or norm_lang == "kr"
    lang_hint = "Write the full report in Korean." if is_korean else "Write the full report in English."

    sections = [
        "1) Market & Macro Overview (very short, based only on factor/index performance you see in the data)",
        "2) Portfolio Performance Summary (WTD, MTD, QTD, YTD vs indices)",
        "3) Attribution (by country, sector, factor, hedge contribution, and key single names from top/bottom contributors)",
        "4) Risk & Volatility Review (beta/volatility impressions inferred from patterns in returns; call out realized volatility, drawdowns, Sharpe-style risk-adjusted performance, AND isolate/describe hedge PnL impact vs underlying equity)",
        "5) PM Action Items / Portfolio Changes (what to add, trim, hedge, or monitor, qualitatively)",
        "6) Quant / Signal Perspective (what seems to work: momentum, mean-reversion, factor tilts, etc.).",
    ]
    sections.append(
        "7) YTD Risk Metrics vs Benchmarks (report YTD Sharpe ratio, max drawdown, and annualized volatility for "
        "the portfolio and each benchmark, based on total hedged returns; compare and interpret; do not invent numbers "
        "if metrics are missing)."
    )
    sections.append(
        "8) Drawdown / Sharpe / Return Improvement Plan (build separate plans for YTD, QTD, and MTD performance; "
        "for each period, diagnose weaknesses and propose concrete actions; include country allocation adjustments "
        "and country-level volatility management; include an 'Expected Impact' summary with directional results; "
        "do not invent numbers if metrics are missing)."
    )
    deepseek_extra = (
        "\nUnder the YTD risk section, include a compact table with rows for Portfolio and each benchmark, "
        "and columns for Sharpe Ratio (YTD), Max Drawdown (YTD), and Annualized Volatility (YTD). "
        "Use the provided risk_metrics from the YTD block (computed on total hedged returns); if a metric is missing, "
        "write 'N/A' and do not guess. "
        "Then provide a short comparison vs benchmarks.\n"
        "For the improvement plan, create three subsections titled YTD, QTD, and MTD. "
        "In each subsection, include: (a) key weaknesses, (b) action items, "
        "(c) country allocation changes, and (d) country-level volatility control ideas. "
        "Also include a compact impact matrix per subsection with rows for Drawdown, Sharpe Ratio, and Return, "
        "and columns for Key Drivers, Proposed Actions, and Expected Direction. "
        "Base portfolio operation and improvement suggestions on the YTD risk comparison."
    )
    section_block = "\n".join(sections)

    user_prompt = (
        f"Today is the weekly review for the portfolio, with performance measured up to {report_date.date()}.\n"
        "Here is the structured performance & attribution data for WTD, MTD, QTD, and YTD in JSON format.\n"
        "Use ONLY this information to write a full weekly report. Do not fabricate any new figures.\n\n"
        f"Structured data:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
        f"{comment_block}\n\n"
        "Your report MUST be structured in the following sections:\n"
        f"{section_block}\n"
        f"{deepseek_extra}\n"
        "Write in a clear, bullet-point friendly style suitable for a weekly investment meeting note.\n"
        f"{lang_hint}"
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            temperature=0.3,
            max_tokens=1500,
        )
        return resp.choices[0].message.content.strip()
    except AttributeError:
        try:
            resp = client.ChatCompletion.create(
                model=model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ],
                temperature=0.3,
                max_tokens=1500,
            )
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as e_old:
            raise _translate_openai_error(e_old, provider)
    except Exception as e_new:
        raise _translate_openai_error(e_new, provider)


# ==============================================================================
# [PART 1] Team PNL Load
# ==============================================================================
@st.cache_data
def load_team_pnl_data(file):
    try:
        df_pnl_raw = pd.read_excel(file, sheet_name='PNL', header=None, engine='openpyxl')
        h_idx = -1
        for i in range(15):
            if '일자' in [str(x).strip() for x in df_pnl_raw.iloc[i].values]:
                h_idx = i; break
        if h_idx == -1: return None, None, "PNL Header Not Found"
        
        raw_cols = df_pnl_raw.iloc[h_idx].tolist()
        new_cols = []
        seen = {}
        for c in raw_cols:
            c_str = str(c).strip()
            if c_str in ['nan', 'None', '']: continue
            if c_str in seen: seen[c_str] += 1; new_cols.append(f"{c_str}_{seen[c_str]}")
            else: seen[c_str] = 0; new_cols.append(c_str)
            
        df_pnl = df_pnl_raw.iloc[h_idx+1:].copy()
        valid_indices = [i for i, c in enumerate(df_pnl_raw.iloc[h_idx]) if str(c).strip() not in ['nan', 'None', '']]
        df_pnl = df_pnl.iloc[:, valid_indices]
        df_pnl.columns = new_cols
        date_col = [c for c in df_pnl.columns if '일자' in c][0]
        df_pnl = df_pnl.set_index(date_col)
        df_pnl.index = pd.to_datetime(df_pnl.index, errors='coerce')
        df_pnl = df_pnl.dropna(how='all').apply(pd.to_numeric, errors='coerce').fillna(0)

        df_pos_raw = pd.read_excel(file, sheet_name='Position', header=None, engine='openpyxl')
        h_idx_pos = -1
        for i in range(15):
            if '일자' in [str(x).strip() for x in df_pos_raw.iloc[i].values]:
                h_idx_pos = i; break
        
        raw_cols_pos = df_pos_raw.iloc[h_idx_pos].tolist()
        new_cols_pos = []
        seen_pos = {}
        for c in raw_cols_pos:
            c_str = str(c).strip()
            if c_str in ['nan', 'None', '']: continue
            if c_str in seen_pos: seen_pos[c_str] += 1; new_cols_pos.append(f"{c_str}_{seen_pos[c_str]}")
            else: seen_pos[c_str] = 0; new_cols_pos.append(c_str)
            
        df_pos = df_pos_raw.iloc[h_idx_pos+1:].copy()
        valid_indices_pos = [i for i, c in enumerate(df_pos_raw.iloc[h_idx_pos]) if str(c).strip() not in ['nan', 'None', '']]
        df_pos = df_pos.iloc[:, valid_indices_pos]
        df_pos.columns = new_cols_pos
        date_col_pos = [c for c in df_pos.columns if '일자' in c][0]
        df_pos = df_pos.set_index(date_col_pos)
        df_pos.index = pd.to_datetime(df_pos.index, errors='coerce')
        df_pos = df_pos.dropna(how='all').apply(pd.to_numeric, errors='coerce').fillna(0)

        return df_pnl, df_pos, None
    except Exception as e: return None, None, f"Team PNL Error: {e}"

# ==============================================================================
# [PART 2] Cash Equity Load
# ==============================================================================
@st.cache_data
def load_cash_equity_data(file):
    debug_logs = []
    try:
        xls = pd.ExcelFile(file, engine='openpyxl')
        all_holdings = []
        df_hedge = pd.DataFrame()
        
        for sheet in xls.sheet_names:
            if 'hedge' in sheet.lower() or '헷지' in sheet:
                try:
                    df_h = pd.read_excel(file, sheet_name=sheet, header=None, engine='openpyxl')
                    h_idx = -1
                    for i in range(15):
                        if '기준일자' in [str(x).strip() for x in df_h.iloc[i].values]:
                            h_idx = i; break
                    if h_idx != -1:
                        raw_cols = [str(c).strip() for c in df_h.iloc[h_idx]]
                        new_cols = []
                        seen = {}
                        for c in raw_cols:
                            if c in seen: seen[c]+=1; new_cols.append(f"{c}_{seen[c]}")
                            else: seen[c]=0; new_cols.append(c)
                        df_h.columns = new_cols
                        df_h = df_h.iloc[h_idx+1:].copy()
                        df_h['기준일자'] = pd.to_datetime(df_h['기준일자'], errors='coerce')
                        df_h = df_h.dropna(subset=['기준일자']).set_index('기준일자').sort_index()
                        col_cum = next((c for c in df_h.columns if '누적' in c and '총손익' in c), None)
                        if col_cum:
                            df_h[col_cum] = pd.to_numeric(df_h[col_cum], errors='coerce').fillna(0)
                            daily_hedge = df_h[col_cum].diff().fillna(0)
                            if df_hedge.empty: df_hedge = daily_hedge.to_frame(name='Hedge_PnL_KRW')
                            else: df_hedge = df_hedge.add(daily_hedge.to_frame(name='Hedge_PnL_KRW'), fill_value=0)
                except: pass
            else:
                try:
                    df = pd.read_excel(file, sheet_name=sheet, header=None, engine='openpyxl')
                    h_idx = -1
                    for i in range(15):
                        row_vals = [str(x).strip() for x in df.iloc[i].values]
                        if '기준일자' in row_vals and ('종목명' in row_vals or '종목코드' in row_vals):
                            h_idx = i; break
                    if h_idx != -1:
                        raw_cols = [str(c).strip() for c in df.iloc[h_idx]]
                        new_cols = []
                        seen = {}
                        for c in raw_cols:
                            if c in seen: seen[c]+=1; new_cols.append(f"{c}_{seen[c]}")
                            else: seen[c]=0; new_cols.append(c)
                        df.columns = new_cols
                        df = df.iloc[h_idx+1:].copy()
                        if '기준일자' in df.columns: all_holdings.append(df)
                except: pass

        if not all_holdings: return None, None, None, None, None, None, "Holdings 데이터 없음"

        eq = pd.concat(all_holdings, ignore_index=True)
        eq['기준일자'] = pd.to_datetime(eq['기준일자'], errors='coerce')
        eq = eq.dropna(subset=['기준일자'])
        
        rename_map = {'평가단가': 'Market_Price', '외화평가금액': 'Market_Value', '종목코드': 'Ticker', '심볼': 'Symbol'}
        eq.rename(columns=rename_map, inplace=True)
        
        cols_num = ['원화평가금액', '원화총평가손익', '원화총매매손익', '환손익', '잔고수량', 'Market_Price', 'Market_Value', '외화평가손익', '외화총매매손익', '평가환율']
        for c in cols_num:
            if c in eq.columns: eq[c] = pd.to_numeric(eq[c], errors='coerce').fillna(0)

        id_col = 'Ticker' if 'Ticker' in eq.columns else 'Symbol'
        if id_col not in eq.columns: id_col = '종목명'
        eq['Ticker_ID'] = eq[id_col].fillna('Unknown')

        def _resolve_yf_symbol(row):
            candidates = [
                row.get("Symbol"),
                row.get("Ticker_ID"),
                row.get("Ticker"),
            ]
            for base in candidates:
                sym = normalize_yf_ticker(base, row.get("통화"))
                if sym:
                    return sym
            return None

        eq['YF_Symbol'] = eq.apply(_resolve_yf_symbol, axis=1)

        if '섹터' not in eq.columns:
            uniques = tuple(sorted(eq['YF_Symbol'].dropna().unique()))
            if uniques:
                sec_map = fetch_sectors_cached(uniques)
                eq['섹터'] = eq['YF_Symbol'].map(sec_map).fillna('Unknown')
            else:
                eq['섹터'] = 'Unknown'
        else:
            eq['섹터'] = eq['섹터'].fillna('Unknown')
            unknown_mask = (
                eq["섹터"].astype(str).str.strip().str.upper().isin(["", "UNKNOWN", "NAN", "NONE"])
            )
            unknown_tickers = tuple(sorted(eq.loc[unknown_mask, "YF_Symbol"].dropna().unique()))
            if unknown_tickers:
                sec_map = fetch_sectors_cached(unknown_tickers)
                refilled = eq.loc[unknown_mask, "YF_Symbol"].map(sec_map)
                eq.loc[unknown_mask, "섹터"] = refilled.fillna(eq.loc[unknown_mask, "섹터"])
            eq["섹터"] = eq["섹터"].replace("", "Unknown").fillna("Unknown")

        etf_mask = pd.Series(False, index=eq.index)
        if '상품구분' in eq.columns:
            etf_mask |= eq['상품구분'].apply(is_etf_product_type)
        if '종목명' in eq.columns:
            etf_mask |= eq['종목명'].apply(is_etf_value)
        etf_tickers = tuple(sorted(eq["YF_Symbol"].dropna().unique()))
        if etf_tickers:
            etf_symbol_map = fetch_etf_flags_cached(etf_tickers)
            etf_mask |= eq["YF_Symbol"].map(etf_symbol_map).fillna(False)
        eq.loc[etf_mask, '섹터'] = 'ETF'

        if '통화' in eq.columns:
            curr_map = {'USD': 'US', 'HKD': 'HK', 'JPY': 'JP', 'KRW': 'KR', 'CNY': 'CN'}
            eq['Country'] = eq['통화'].map(curr_map).fillna('Other')
        else: eq['Country'] = 'Other'

        all_dates = sorted(eq['기준일자'].unique())
        all_tickers = eq['Ticker_ID'].unique()
        idx = pd.MultiIndex.from_product([all_dates, all_tickers], names=['기준일자', 'Ticker_ID'])
        grid = pd.DataFrame(index=idx).reset_index()
        
        eq_dedup = eq.drop_duplicates(subset=['기준일자', 'Ticker_ID'])
        daily_totals = None
        if {'원화총평가손익', '원화총매매손익'}.issubset(eq_dedup.columns):
            pnl_cols = ['원화총평가손익', '원화총매매손익']
            agg_cols = pnl_cols + (['환손익'] if '환손익' in eq_dedup.columns else [])
            daily_totals = eq_dedup.groupby('기준일자')[agg_cols].sum().sort_index()
            daily_totals['Total_PnL_KRW_Cum'] = daily_totals['원화총평가손익'] + daily_totals['원화총매매손익']
            if '환손익' in daily_totals.columns:
                daily_totals['FX_PnL_KRW_Cum'] = daily_totals['환손익']
            else:
                daily_totals['FX_PnL_KRW_Cum'] = 0.0
            daily_totals['Local_PnL_KRW_Cum'] = daily_totals['Total_PnL_KRW_Cum'] - daily_totals['FX_PnL_KRW_Cum']
            daily_totals['Daily_PnL_KRW'] = daily_totals['Total_PnL_KRW_Cum'].diff().fillna(0)
            daily_totals['Daily_Local_PnL_KRW'] = daily_totals['Local_PnL_KRW_Cum'].diff().fillna(0)
        cols_keep = ['원화평가금액', '원화총평가손익', '원화총매매손익', '외화평가손익', '외화총매매손익', '통화', '섹터', 'Country', '종목명', 'Market_Value']
        cols_keep = [c for c in cols_keep if c in eq_dedup.columns]
        
        merged = pd.merge(grid, eq_dedup[['기준일자', 'Ticker_ID'] + cols_keep], on=['기준일자', 'Ticker_ID'], how='left')
        merged = merged.sort_values(['Ticker_ID', '기준일자'])

        for c in ['원화총매매손익', '외화총매매손익']:
            if c in merged.columns: merged[c] = merged.groupby('Ticker_ID')[c].ffill().fillna(0)
        for c in ['원화평가손익', '외화평가손익', '원화평가금액', 'Market_Value']:
            if c in merged.columns: merged[c] = merged[c].fillna(0)
        for c in ['통화', '섹터', 'Country', '종목명']:
            if c in merged.columns: merged[c] = merged.groupby('Ticker_ID')[c].ffill().fillna('Unknown')

        merged['Cum_PnL_KRW'] = merged['원화총평가손익'] + merged['원화총매매손익']
        merged['Daily_PnL_KRW'] = merged.groupby('Ticker_ID')['Cum_PnL_KRW'].diff().fillna(0)
        
        if 'Market_Value' in merged.columns:
            merged['Implied_FX'] = np.where((merged['원화평가금액']!=0) & (merged['Market_Value']!=0),
                                            merged['원화평가금액']/merged['Market_Value'], 0)
            if '통화' in merged.columns:
                merged.loc[merged['통화']=='KRW', 'Implied_FX'] = 1.0
        else: merged['Implied_FX'] = 0
            
        fx_daily = merged[merged['Implied_FX']>0].groupby(['기준일자', '통화'])['Implied_FX'].median().reset_index().rename(columns={'Implied_FX': 'Daily_FX'})
        merged = pd.merge(merged, fx_daily, on=['기준일자', '통화'], how='left')
        merged['Daily_FX'] = merged.groupby('Ticker_ID')['Daily_FX'].ffill().fillna(1.0)
        merged['FX_Ret'] = merged.groupby('Ticker_ID')['Daily_FX'].pct_change().fillna(0)
        
        merged['Prev_MV_KRW'] = merged.groupby('Ticker_ID')['원화평가금액'].shift(1).fillna(0)
        
        daily_agg = merged.groupby('기준일자').agg({
            '원화평가금액': 'sum',
            'Prev_MV_KRW': 'sum'
        }).rename(columns={'원화평가금액': 'Total_MV_KRW', 'Prev_MV_KRW': 'Total_Prev_MV_KRW'})
        if daily_totals is not None:
            daily_agg = daily_agg.join(
                daily_totals[['Daily_PnL_KRW', 'Daily_Local_PnL_KRW', 'Total_PnL_KRW_Cum',
                              'Local_PnL_KRW_Cum', 'FX_PnL_KRW_Cum']],
                how='left'
            )
        else:
            daily_pnl = merged.groupby('기준일자')['Daily_PnL_KRW'].sum().rename('Daily_PnL_KRW')
            daily_agg = daily_agg.join(daily_pnl, how='left')
        
        merged = merged.merge(daily_agg['Total_Prev_MV_KRW'].rename('Day_Total_Prev'), on='기준일자', how='left')
        merged['Contrib_KRW'] = np.where(merged['Day_Total_Prev'] > 0, 
                                         merged['Daily_PnL_KRW'] / merged['Day_Total_Prev'], 0)
        
        contrib_sector = merged.groupby(['기준일자', '섹터'])['Contrib_KRW'].sum().reset_index()
        contrib_country = merged.groupby(['기준일자', 'Country'])['Contrib_KRW'].sum().reset_index()
        
        merged['Weight'] = np.where(merged['Day_Total_Prev']>0, merged['Prev_MV_KRW']/merged['Day_Total_Prev'], 0)
        daily_fx_ret = merged.groupby('기준일자').apply(lambda x: (x['FX_Ret'] * x['Weight']).sum()).rename('Port_FX_Ret')
        
        country_daily = merged.groupby(['기준일자', 'Country']).agg({
            'Daily_PnL_KRW': 'sum', 'Prev_MV_KRW': 'sum'
        }).reset_index()
        country_daily['Country_Ret'] = np.where(country_daily['Prev_MV_KRW']>0, 
                                                country_daily['Daily_PnL_KRW']/country_daily['Prev_MV_KRW'], 0)

        df_perf = daily_agg.join(df_hedge, how='outer').fillna(0)
        if 'Hedge_PnL_KRW' not in df_perf.columns:
            df_perf['Hedge_PnL_KRW'] = 0.0
        df_perf = df_perf.join(daily_fx_ret, how='left').fillna(0)
        
        min_d, max_d = df_perf.index.min(), df_perf.index.max()
        usdkrw = download_usdkrw(min_d, max_d)
        df_perf = df_perf.join(usdkrw, how='left').fillna(method='ffill').fillna(1400.0)
        if 'USD_KRW' not in df_perf.columns: df_perf['USD_KRW'] = 1400.0
        
        df_perf['Hedge_PnL_USD'] = df_perf['Hedge_PnL_KRW'] / df_perf['USD_KRW']
        df_perf['Total_Prev_MV_USD'] = df_perf['Total_Prev_MV_KRW'] / df_perf['USD_KRW']
        df_perf['Total_PnL_KRW'] = df_perf['Daily_PnL_KRW'] + df_perf['Hedge_PnL_KRW']
        
        denom_krw = df_perf['Total_Prev_MV_KRW'].replace(0, np.nan)
        denom_usd = df_perf['Total_Prev_MV_USD'].replace(0, np.nan)
        
        df_perf['Ret_Equity_KRW'] = df_perf['Daily_PnL_KRW'] / denom_krw
        df_perf['Ret_Total_KRW'] = df_perf['Total_PnL_KRW'] / denom_krw

        if 'Daily_Local_PnL_KRW' in df_perf.columns:
            df_perf['Daily_Local_PnL_USD'] = df_perf['Daily_Local_PnL_KRW'] / df_perf['USD_KRW']
            df_perf['Ret_Equity_Local'] = df_perf['Daily_Local_PnL_USD'] / denom_usd
        else:
            df_perf['Ret_Equity_Local'] = (1 + df_perf['Ret_Equity_KRW'].fillna(0)) / (1 + df_perf['Port_FX_Ret'].fillna(0)) - 1
        df_perf['Ret_Hedge_Local'] = df_perf['Hedge_PnL_USD'] / denom_usd
        df_perf['Ret_Total_Local'] = df_perf['Ret_Equity_Local'] + df_perf['Ret_Hedge_Local'].fillna(0)
        
        df_perf.fillna(0, inplace=True)
        if len(df_perf) > 1:
            df_perf = df_perf.iloc[1:]
        
        for c in ['Ret_Equity_KRW', 'Ret_Total_KRW', 'Ret_Equity_Local', 'Ret_Total_Local']:
            df_perf[c.replace('Ret', 'Cum')] = (1 + df_perf[c]).cumprod() - 1
            
        df_last = eq.sort_values('기준일자').groupby('Ticker_ID').tail(1)
        df_last['Final_PnL'] = df_last['원화총평가손익'] + df_last['원화총매매손익']
        
        return df_perf, df_last, {'Sector':contrib_sector, 'Country':contrib_country}, country_daily, merged, debug_logs, None

    except Exception as e:
        return None, None, None, None, None, None, f"Process Error: {e}"

