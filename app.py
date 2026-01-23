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
from io import BytesIO
import base64
from datetime import datetime, timezone
from pathlib import Path

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

def normalize_yf_ticker(symbol, currency=None):
    sym = _clean_symbol(symbol)
    if not sym:
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
    return "ETF" in str(value).strip().upper()

@st.cache_data(ttl=3600)
def fetch_sectors_cached(tickers):
    sector_map = {}
    for t in tickers:
        try:
            t_str = str(t).strip()
            if t_str:
                info = yf.Ticker(t_str).info
                sector_map[t] = info.get('sector', 'Unknown')
            else:
                sector_map[t] = 'Unknown'
        except:
            sector_map[t] = 'Unknown'
    return sector_map

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

@st.cache_data
def load_portfolio_snapshot(file_path, mtime):
    try:
        df_raw = pd.read_excel(file_path, sheet_name=0, header=None, engine="openpyxl")
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

        if '섹터' not in eq.columns:
            if 'Symbol' in eq.columns:
                def _resolve_yf_symbol(row):
                    base = row.get('Symbol')
                    if base is None or (isinstance(base, str) and base.strip() == '') or pd.isna(base):
                        if 'Ticker_ID' in row:
                            base = row.get('Ticker_ID')
                    return normalize_yf_ticker(base, row.get('통화'))
                eq['YF_Symbol'] = eq.apply(_resolve_yf_symbol, axis=1)
                uniques = eq['YF_Symbol'].dropna().unique()
                sec_map = fetch_sectors_cached(tuple(uniques))
                eq['섹터'] = eq['YF_Symbol'].map(sec_map).fillna('Unknown')
            else:
                eq['섹터'] = 'Unknown'
        else:
            eq['섹터'] = eq['섹터'].fillna('Unknown')

        etf_mask = pd.Series(False, index=eq.index)
        if '상품구분' in eq.columns:
            etf_mask |= eq['상품구분'].apply(is_etf_value)
        if '종목명' in eq.columns:
            etf_mask |= eq['종목명'].apply(is_etf_value)
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


# ==============================================================================
# [MAIN UI]
# ==============================================================================
menu = st.sidebar.radio(
    "Dashboard Menu",
    ["📌 Portfolio Snapshot", "Total Portfolio (Team PNL)", "Cash Equity Analysis", "📑 Weekly Report Generator"],
)

if menu == "📌 Portfolio Snapshot":
    st.subheader("📌 Portfolio Snapshot (2026_멀티.xlsx)")
    script_dir = Path(__file__).resolve().parent
    candidates = []
    env_path = os.getenv("PORTFOLIO_XLSX_PATH")
    if env_path:
        candidates.append(Path(env_path))
    if hasattr(st, "secrets") and "PORTFOLIO_XLSX_PATH" in st.secrets:
        candidates.append(Path(st.secrets["PORTFOLIO_XLSX_PATH"]))
    candidates.extend([
        script_dir / "2026_멀티.xlsx",
        Path.cwd() / "2026_멀티.xlsx",
        Path.home() / "Desktop" / "Workspace" / "Team" / "2026_멀티.xlsx",
    ])
    data_path = next((p for p in candidates if p.exists()), None)
    if data_path is None:
        st.error("2026_멀티.xlsx 파일이 앱 폴더에 없습니다.")
        st.caption("컨테이너/배포 환경에서는 로컬 파일이 보이지 않을 수 있습니다.")
        st.caption(
            "해결: 1) 파일을 앱 폴더에 복사하거나 2) PORTFOLIO_XLSX_PATH 환경변수/시크릿으로 경로를 지정하세요."
        )
        st.caption("검색 경로: " + " , ".join(str(p) for p in candidates))
    else:
        with st.spinner("포트폴리오 현황 불러오는 중..."):
            df_snapshot, err = load_portfolio_snapshot(str(data_path), data_path.stat().st_mtime)
        if err or df_snapshot is None or df_snapshot.empty:
            st.error(f"데이터 로드 실패: {err}")
        else:
            latest_date = df_snapshot["기준일자"].max()
            latest_all = df_snapshot[df_snapshot["기준일자"] == latest_date].copy()

            if "원화평가금액" not in latest_all.columns and {"외화평가금액", "평가환율"}.issubset(latest_all.columns):
                latest_all["원화평가금액"] = latest_all["외화평가금액"] * latest_all["평가환율"]

            id_col = "심볼" if "심볼" in latest_all.columns else ("종목코드" if "종목코드" in latest_all.columns else "종목명")
            latest_all["Ticker_ID"] = latest_all[id_col].fillna(latest_all.get("종목명", latest_all[id_col]))
            if "종목명" not in latest_all.columns:
                latest_all["종목명"] = latest_all["Ticker_ID"]
            if "통화" not in latest_all.columns:
                latest_all["통화"] = "N/A"

            def _resolve_symbol(row):
                base = row.get(id_col)
                if base is None or (isinstance(base, str) and base.strip() == "") or pd.isna(base):
                    base = row.get("종목코드") if "종목코드" in latest_all.columns else row.get("Ticker_ID")
                return normalize_yf_ticker(base, row.get("통화"))
            latest_all["YF_Symbol"] = latest_all.apply(_resolve_symbol, axis=1)

            if "섹터" not in latest_all.columns:
                tickers = tuple(sorted(latest_all["YF_Symbol"].dropna().unique()))
                sector_map = fetch_sectors_cached(tickers)
                latest_all["섹터"] = latest_all["YF_Symbol"].map(sector_map).fillna("Unknown")
            else:
                latest_all["섹터"] = latest_all["섹터"].fillna("Unknown")

            etf_mask = pd.Series(False, index=latest_all.index)
            if "상품구분" in latest_all.columns:
                etf_mask |= latest_all["상품구분"].apply(is_etf_value)
            if "종목명" in latest_all.columns:
                etf_mask |= latest_all["종목명"].apply(is_etf_value)
            latest_all.loc[etf_mask, "섹터"] = "ETF"
            latest_all["Is_ETF"] = etf_mask

            if "원화평가금액" not in latest_all.columns:
                st.error("원화평가금액 컬럼이 없어 비중 계산이 불가능합니다.")
                latest_all = pd.DataFrame()

            if latest_all.empty:
                st.warning("최신일 데이터가 없습니다.")
                st.stop()

            latest_for_weights = latest_all[latest_all["원화평가금액"] != 0].copy()
            if latest_for_weights.empty:
                latest_for_weights = latest_all.copy()

            latest_for_weights["Group_ID"] = latest_for_weights["YF_Symbol"].fillna(latest_for_weights["Ticker_ID"])
            holdings = latest_for_weights.groupby("Group_ID", dropna=False).agg(
                원화평가금액=("원화평가금액", "sum"),
                종목명=("종목명", "first"),
                섹터=("섹터", "first"),
                통화=("통화", "first"),
                Ticker_ID=("Ticker_ID", "first"),
                Is_ETF=("Is_ETF", "first"),
            ).reset_index()
            total_mv = holdings["원화평가금액"].sum()
            holdings["Weight"] = np.where(total_mv > 0, holdings["원화평가금액"] / total_mv, 0)
            holdings["Label"] = holdings["종목명"].astype(str) + " (" + holdings["Group_ID"].astype(str) + ")"

            etf_weight = holdings.loc[holdings["섹터"] == "ETF", "Weight"].sum() if not holdings.empty else 0
            holdings_non_etf = holdings[holdings["섹터"] != "ETF"].copy()
            total_mv_non_etf = holdings_non_etf["원화평가금액"].sum()
            sector_weights = holdings_non_etf.groupby("섹터")["원화평가금액"].sum().sort_values(ascending=False)
            sector_weights_pct = sector_weights / total_mv_non_etf if total_mv_non_etf else sector_weights * 0

            currency_weights = holdings.groupby("통화")["원화평가금액"].sum().sort_values(ascending=False)
            currency_weights_pct = currency_weights / total_mv if total_mv else currency_weights * 0

            total_pnl = latest_all.get("원화총평가손익", pd.Series(0, index=latest_all.index)).sum() + \
                        latest_all.get("원화총매매손익", pd.Series(0, index=latest_all.index)).sum()
            fx_pnl = latest_all.get("환손익", pd.Series(0, index=latest_all.index)).sum()
            local_pnl = total_pnl - fx_pnl

            hhi = (holdings["Weight"] ** 2).sum() if not holdings.empty else 0
            eff_n = (1 / hhi) if hhi > 0 else 0
            top5_weight = holdings["Weight"].nlargest(5).sum() if not holdings.empty else 0

            # 시뮬레이션용 holdings 데이터 준비
            holdings["YF_Symbol"] = holdings["Group_ID"]

            # 탭 생성: 현황 / 시뮬레이션
            tab_snapshot, tab_simulation = st.tabs(["📊 포트폴리오 현황", "🔬 포트폴리오 시뮬레이션"])

            with tab_snapshot:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("기준일자", latest_date.strftime("%Y-%m-%d"))
                c2.metric("총 AUM (KRW)", f"{total_mv:,.0f}")
                c3.metric("Total PnL (KRW)", f"{total_pnl:,.0f}")
                c4.metric("Local PnL (KRW)", f"{local_pnl:,.0f}")

                c5, c6, c7, c8 = st.columns(4)
                c5.metric("보유 종목 수", f"{len(holdings):,}")
                c6.metric("Top 5 비중", f"{top5_weight:.2%}")
                c7.metric("HHI", f"{hhi:.4f}")
                c8.metric("유효 보유 종목 수", f"{eff_n:.1f}")

                # 포트폴리오 변동성 계산
                current_weights = holdings.set_index("YF_Symbol")["Weight"].to_dict()
                with st.spinner("변동성 계산 중..."):
                    vol_metrics = calculate_portfolio_volatility(current_weights, lookback_days=30)

                if vol_metrics:
                    c9, c10, c11, c12 = st.columns(4)
                    c9.metric("30일 변동성 (연율)", f"{vol_metrics['annual_volatility']:.2%}")
                    c10.metric("30일 MDD", f"{vol_metrics['max_drawdown']:.2%}")
                    c11.metric("VaR 95%", f"{vol_metrics['var_95']:.2%}")
                    c12.metric("VaR 99%", f"{vol_metrics['var_99']:.2%}")

                st.caption(f"ETF 비중: {etf_weight:.2%} (섹터 비중/비교는 ETF 제외 기준)")

                st.markdown("#### 🔎 보유 종목 비중")
                top_holdings = holdings.sort_values("Weight", ascending=False).head(15)
                fig_hold = go.Figure(
                    data=go.Bar(
                        x=top_holdings["Label"],
                        y=top_holdings["Weight"],
                        text=[f"{w:.2%}" for w in top_holdings["Weight"]],
                        textposition="auto",
                    )
                )
                fig_hold.update_layout(yaxis_tickformat=".1%", xaxis_title="", yaxis_title="Weight")
                st.plotly_chart(fig_hold, use_container_width=True)

                st.markdown("#### 🧭 섹터 비중")
                fig_sector = go.Figure(
                    data=go.Pie(labels=sector_weights_pct.index, values=sector_weights_pct.values, hole=0.45)
                )
                fig_sector.update_traces(textinfo="percent+label")
                st.plotly_chart(fig_sector, use_container_width=True)

                st.markdown("#### 💱 통화 비중")
                fig_fx = go.Figure(
                    data=go.Bar(
                        x=currency_weights_pct.index.astype(str),
                        y=currency_weights_pct.values,
                        text=[f"{w:.2%}" for w in currency_weights_pct.values],
                        textposition="auto",
                    )
                )
                fig_fx.update_layout(yaxis_tickformat=".1%", xaxis_title="", yaxis_title="Weight")
                st.plotly_chart(fig_fx, use_container_width=True)

                st.markdown("#### 🆚 S&P 500 섹터 Weight 차이 (Portfolio - SP500)")
                with st.spinner("S&P 500 섹터 가중치 계산 중..."):
                    sp_sector = fetch_sp500_sector_weights()
                if sp_sector.empty:
                    st.warning("S&P 500 섹터 데이터를 불러오지 못했습니다.")
                else:
                    port_sector = sector_weights_pct.copy()
                    if "Unknown" in port_sector.index:
                        port_sector = port_sector.drop("Unknown")
                    sp_sector = sp_sector.drop("Unknown", errors="ignore")
                    if port_sector.sum() > 0:
                        port_sector = port_sector / port_sector.sum()
                    if sp_sector.sum() > 0:
                        sp_sector = sp_sector / sp_sector.sum()

                    all_sectors = sorted(set(port_sector.index) | set(sp_sector.index))
                    diff = port_sector.reindex(all_sectors, fill_value=0) - sp_sector.reindex(all_sectors, fill_value=0)
                    colors = np.where(diff.values >= 0, "#16a34a", "#dc2626")
                    fig_diff = go.Figure(
                        data=go.Bar(x=diff.index, y=diff.values, marker_color=colors)
                    )
                    fig_diff.update_layout(yaxis_tickformat=".1%", xaxis_title="", yaxis_title="Weight Difference")
                    st.plotly_chart(fig_diff, use_container_width=True)

                    comp = pd.DataFrame({
                        "Portfolio": port_sector.reindex(all_sectors, fill_value=0),
                        "S&P 500": sp_sector.reindex(all_sectors, fill_value=0),
                    })
                    comp["Diff"] = comp["Portfolio"] - comp["S&P 500"]
                    st.dataframe(comp.style.format("{:.2%}"))

                st.markdown("#### 📋 보유 종목 상세")
                show_cols = ["Group_ID", "종목명", "섹터", "통화", "원화평가금액", "Weight"]
                show_cols = [c for c in show_cols if c in holdings.columns]
                st.dataframe(holdings.sort_values("Weight", ascending=False)[show_cols].style.format({
                    "원화평가금액": "{:,.0f}",
                    "Weight": "{:.2%}",
                }))

            with tab_simulation:
                st.markdown("### 🔬 포트폴리오 비중 시뮬레이션")
                st.caption("기존 종목의 비중을 조절하거나 신규 종목을 추가하여 NAV 변화를 시뮬레이션합니다. (전일 종가 기준)")

                # 시뮬레이션 설정
                col_sim_settings1, col_sim_settings2 = st.columns(2)

                with col_sim_settings1:
                    sim_days = st.slider("시뮬레이션 기간 (일)", min_value=5, max_value=90, value=30, step=5)

                with col_sim_settings2:
                    # 추가 현금 투입 옵션
                    use_additional_cash = st.checkbox("💰 추가 현금 투입", value=False,
                                                      help="비중 상향 시 기존 NAV를 유지하면서 추가 자금을 투입합니다.")

                additional_cash_krw = 0
                if use_additional_cash:
                    st.markdown("#### 💵 추가 현금 투입 설정")

                    cash_input_col1, cash_input_col2 = st.columns(2)
                    with cash_input_col1:
                        additional_cash_krw = st.number_input(
                            "추가 투입 금액 (KRW)",
                            min_value=0,
                            max_value=100_000_000_000,  # 1000억
                            value=0,
                            step=100_000_000,  # 1억 단위
                            format="%d",
                            help="추가로 투입할 현금 (원화)"
                        )
                    with cash_input_col2:
                        if additional_cash_krw > 0:
                            new_total_nav = total_mv + additional_cash_krw
                            st.metric("새로운 총 NAV", f"₩{new_total_nav:,.0f}")
                            st.caption(f"기존 NAV: ₩{total_mv:,.0f} + 추가: ₩{additional_cash_krw:,.0f}")

                st.markdown("---")

                # 두 개의 컬럼으로 나누기
                col_existing, col_new = st.columns(2)

                with col_existing:
                    st.markdown("#### 📈 기존 종목 비중 조절")
                    st.caption("비중을 조절할 종목을 선택하고 새로운 비중(%)을 입력하세요.")

                    # 기존 종목 리스트 (상위 20개)
                    top_20 = holdings.sort_values("Weight", ascending=False).head(20)

                    # 세션 상태 초기화
                    if "weight_adjustments" not in st.session_state:
                        st.session_state.weight_adjustments = {}

                    # 종목별 슬라이더
                    weight_adjustments = {}
                    for idx, row in top_20.iterrows():
                        ticker = row["YF_Symbol"]
                        current_weight = row["Weight"] * 100  # %로 변환
                        label = f"{row['종목명']} ({ticker})"

                        new_weight = st.number_input(
                            label,
                            min_value=0.0,
                            max_value=100.0,
                            value=float(current_weight),
                            step=0.5,
                            format="%.2f",
                            key=f"weight_{ticker}",
                            help=f"현재 비중: {current_weight:.2f}%"
                        )
                        if abs(new_weight - current_weight) > 0.01:
                            weight_adjustments[ticker] = new_weight / 100  # 비율로 변환

                with col_new:
                    st.markdown("#### ➕ 신규 종목 추가")
                    st.caption("추가할 종목 티커, 마켓, 비중(%)을 입력하세요.")

                    # 마켓 옵션
                    market_options = {
                        "US": "미국 (기본)",
                        "JP": "일본 (.T)",
                        "HK": "홍콩 (.HK)",
                        "KR": "한국 (.KS)"
                    }

                    # 신규 종목 입력 (최대 5개)
                    new_positions = []
                    for i in range(5):
                        c1, c2, c3 = st.columns([2, 1, 1])
                        with c1:
                            new_ticker_raw = st.text_input(
                                f"티커 {i+1}",
                                value="",
                                placeholder="예: AAPL, 7203, 0700",
                                key=f"new_ticker_{i}"
                            ).upper().strip()
                        with c2:
                            new_market = st.selectbox(
                                f"마켓 {i+1}",
                                options=list(market_options.keys()),
                                format_func=lambda x: market_options[x],
                                key=f"new_market_{i}"
                            )
                        with c3:
                            new_weight_pct = st.number_input(
                                f"비중 % {i+1}",
                                min_value=0.0,
                                max_value=50.0,
                                value=0.0,
                                step=0.5,
                                format="%.2f",
                                key=f"new_weight_{i}"
                            )

                        # 티커 변환 (마켓에 따라 suffix 추가)
                        if new_ticker_raw and new_weight_pct > 0:
                            if new_market == "JP":
                                final_ticker = f"{new_ticker_raw}.T" if not new_ticker_raw.endswith(".T") else new_ticker_raw
                            elif new_market == "HK":
                                # 홍콩은 4자리 숫자로 패딩
                                if new_ticker_raw.isdigit():
                                    final_ticker = f"{new_ticker_raw.zfill(4)}.HK"
                                elif not new_ticker_raw.endswith(".HK"):
                                    final_ticker = f"{new_ticker_raw}.HK"
                                else:
                                    final_ticker = new_ticker_raw
                            elif new_market == "KR":
                                final_ticker = f"{new_ticker_raw}.KS" if not new_ticker_raw.endswith(".KS") else new_ticker_raw
                            else:
                                final_ticker = new_ticker_raw

                            new_positions.append({
                                "ticker": final_ticker,
                                "weight": new_weight_pct / 100,
                                "market": new_market
                            })

                    if new_positions:
                        st.caption("**추가될 종목:**")
                        for pos in new_positions:
                            st.caption(f"  • {pos['ticker']} ({pos['weight']*100:.1f}%)")

                st.markdown("---")

                # 시뮬레이션 실행 버튼
                if st.button("🚀 시뮬레이션 실행", type="primary", use_container_width=True):
                    if not weight_adjustments and not new_positions and additional_cash_krw == 0:
                        st.warning("비중을 조절하거나 신규 종목을 추가하거나 추가 현금을 투입해주세요.")
                    else:
                        # 시뮬레이션 NAV 결정 (추가 현금 포함 여부)
                        sim_base_nav = total_mv + additional_cash_krw if use_additional_cash else total_mv

                        with st.spinner("시뮬레이션 실행 중..."):
                            result = simulate_portfolio_nav(
                                holdings_df=holdings,
                                weight_adjustments=weight_adjustments,
                                new_positions=new_positions,
                                base_nav=sim_base_nav,
                                simulation_days=sim_days,
                                additional_cash=additional_cash_krw if use_additional_cash else 0,
                                original_nav=total_mv
                            )

                        if result is None:
                            st.error("시뮬레이션 실행 실패. 가격 데이터를 가져올 수 없습니다.")
                        else:
                            st.success("시뮬레이션 완료!")

                            # 추가 현금 투입 시 안내 메시지
                            if use_additional_cash and additional_cash_krw > 0:
                                st.info(f"💰 **추가 현금 투입 모드**: 기존 NAV ₩{total_mv:,.0f} + 추가 현금 ₩{additional_cash_krw:,.0f} = 새 NAV ₩{sim_base_nav:,.0f}")

                            # 결과 표시
                            st.markdown("### 📊 시뮬레이션 결과")

                            # NAV 비교 차트
                            fig_nav = go.Figure()
                            fig_nav.add_trace(go.Scatter(
                                x=result["original_nav"].index,
                                y=result["original_nav"].values,
                                mode="lines",
                                name="원래 포트폴리오",
                                line=dict(color="#6366f1", width=2)
                            ))
                            fig_nav.add_trace(go.Scatter(
                                x=result["sim_nav"].index,
                                y=result["sim_nav"].values,
                                mode="lines",
                                name=f"시뮬레이션 포트폴리오{' (추가 현금)' if additional_cash_krw > 0 else ''}",
                                line=dict(color="#f97316", width=2, dash="dash")
                            ))
                            fig_nav.update_layout(
                                title="포트폴리오 NAV 비교",
                                xaxis_title="날짜",
                                yaxis_title="NAV (KRW)",
                                yaxis_tickformat=",",
                                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                                hovermode="x unified"
                            )
                            st.plotly_chart(fig_nav, use_container_width=True)

                            # 성과 비교 메트릭
                            orig_final = result["original_nav"].iloc[-1]
                            sim_final = result["sim_nav"].iloc[-1]
                            orig_return = (orig_final / total_mv - 1) * 100
                            sim_return = (sim_final / sim_base_nav - 1) * 100
                            nav_diff = sim_final - orig_final
                            return_diff = sim_return - orig_return

                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("원래 NAV", f"{orig_final:,.0f}")
                            m2.metric("시뮬레이션 NAV", f"{sim_final:,.0f}", delta=f"{nav_diff:,.0f}")
                            m3.metric("원래 수익률", f"{orig_return:.2f}%")
                            m4.metric("시뮬레이션 수익률", f"{sim_return:.2f}%", delta=f"{return_diff:+.2f}%")

                            # 변동성 비교
                            st.markdown("### 📉 변동성 비교 (30일 기준)")

                            with st.spinner("변동성 계산 중..."):
                                orig_vol = calculate_portfolio_volatility(result["original_weights"], lookback_days=30)
                                sim_vol = calculate_portfolio_volatility(result["sim_weights"], lookback_days=30)

                            if orig_vol and sim_vol:
                                vol_diff = sim_vol["annual_volatility"] - orig_vol["annual_volatility"]
                                mdd_diff = sim_vol["max_drawdown"] - orig_vol["max_drawdown"]
                                var95_diff = sim_vol["var_95"] - orig_vol["var_95"]

                                v1, v2, v3, v4 = st.columns(4)
                                v1.metric("원래 변동성 (연율)", f"{orig_vol['annual_volatility']:.2%}")
                                v2.metric("시뮬레이션 변동성", f"{sim_vol['annual_volatility']:.2%}",
                                         delta=f"{vol_diff:+.2%}",
                                         delta_color="inverse")  # 변동성 증가는 빨간색
                                v3.metric("원래 VaR 95%", f"{orig_vol['var_95']:.2%}")
                                v4.metric("시뮬레이션 VaR 95%", f"{sim_vol['var_95']:.2%}",
                                         delta=f"{var95_diff:+.2%}",
                                         delta_color="inverse")

                                v5, v6, v7, v8 = st.columns(4)
                                v5.metric("원래 MDD", f"{orig_vol['max_drawdown']:.2%}")
                                v6.metric("시뮬레이션 MDD", f"{sim_vol['max_drawdown']:.2%}",
                                         delta=f"{mdd_diff:+.2%}",
                                         delta_color="inverse")
                                v7.metric("원래 VaR 99%", f"{orig_vol['var_99']:.2%}")
                                v8.metric("시뮬레이션 VaR 99%", f"{sim_vol['var_99']:.2%}")

                                # 리스크/리턴 요약
                                st.markdown("#### 리스크-리턴 요약")
                                orig_sharpe = orig_return / (orig_vol['annual_volatility'] * 100) if orig_vol['annual_volatility'] > 0 else 0
                                sim_sharpe = sim_return / (sim_vol['annual_volatility'] * 100) if sim_vol['annual_volatility'] > 0 else 0
                                sharpe_diff = sim_sharpe - orig_sharpe

                                rs1, rs2, rs3 = st.columns(3)
                                rs1.metric("원래 샤프비율", f"{orig_sharpe:.3f}")
                                rs2.metric("시뮬레이션 샤프비율", f"{sim_sharpe:.3f}", delta=f"{sharpe_diff:+.3f}")
                                rs3.metric("리스크 조정 효과",
                                          "개선" if sharpe_diff > 0 else "악화" if sharpe_diff < 0 else "동일",
                                          delta=f"{sharpe_diff:+.3f}")
                            else:
                                st.warning("변동성을 계산할 수 없습니다.")

                            # 비중 변경 요약
                            st.markdown("### 📋 비중 변경 요약")

                            # 변경된 비중 테이블
                            changes = []
                            for ticker, new_w in result["sim_weights"].items():
                                orig_w = result["original_weights"].get(ticker, 0)
                                if abs(new_w - orig_w) > 0.0001:
                                    # 종목명 찾기
                                    name_row = holdings[holdings["YF_Symbol"] == ticker]
                                    name = name_row["종목명"].values[0] if len(name_row) > 0 else ticker
                                    changes.append({
                                        "티커": ticker,
                                        "종목명": name,
                                        "원래 비중": orig_w,
                                        "변경 비중": new_w,
                                        "변경폭": new_w - orig_w
                                    })

                            if changes:
                                df_changes = pd.DataFrame(changes)
                                df_changes = df_changes.sort_values("변경폭", ascending=False)
                                st.dataframe(
                                    df_changes.style.format({
                                        "원래 비중": "{:.2%}",
                                        "변경 비중": "{:.2%}",
                                        "변경폭": "{:+.2%}"
                                    }).background_gradient(subset=["변경폭"], cmap="RdYlGn", vmin=-0.1, vmax=0.1),
                                    use_container_width=True
                                )
                            else:
                                st.info("비중 변경 사항이 없습니다.")

                            # 매매 주수 계산
                            st.markdown("### 🛒 매매 주문 (Trade Orders)")
                            if use_additional_cash and additional_cash_krw > 0:
                                st.caption(f"목표 비중 달성을 위해 매매해야 하는 주수입니다. (새 NAV ₩{sim_base_nav:,.0f} 기준, 각 국가별 최종 영업일 종가)")
                            else:
                                st.caption("목표 비중 달성을 위해 매매해야 하는 주수입니다. (각 국가별 최종 영업일 종가 기준)")

                            with st.spinner("매매 주수 계산 중..."):
                                trades = calculate_trade_shares(
                                    result["original_weights"],
                                    result["sim_weights"],
                                    sim_base_nav,  # 추가 현금 포함된 NAV 사용
                                    holdings,
                                    new_positions,
                                    original_nav=total_mv,
                                    additional_cash=additional_cash_krw if use_additional_cash else 0
                                )

                            if trades:
                                df_trades = pd.DataFrame(trades)

                                # 매수/매도 분리
                                buy_trades = df_trades[df_trades["매매"] == "매수"].copy()
                                sell_trades = df_trades[df_trades["매매"] == "매도"].copy()

                                col_buy, col_sell = st.columns(2)

                                with col_buy:
                                    st.markdown("#### 🟢 매수 주문")
                                    if not buy_trades.empty:
                                        buy_display = buy_trades[["티커", "종목명", "주수", "현지통화가격", "통화", "매매금액(KRW)"]].copy()
                                        buy_display = buy_display.sort_values("매매금액(KRW)", ascending=False)
                                        st.dataframe(
                                            buy_display.style.format({
                                                "주수": "{:,.0f}",
                                                "현지통화가격": "{:,.2f}",
                                                "매매금액(KRW)": "{:,.0f}"
                                            }),
                                            use_container_width=True
                                        )
                                        total_buy_krw = buy_trades["매매금액(KRW)"].sum()
                                        st.metric("총 매수 금액 (KRW)", f"{total_buy_krw:,.0f}")
                                    else:
                                        st.info("매수할 종목이 없습니다.")

                                with col_sell:
                                    st.markdown("#### 🔴 매도 주문")
                                    if not sell_trades.empty:
                                        sell_display = sell_trades[["티커", "종목명", "주수", "현지통화가격", "통화", "매매금액(KRW)"]].copy()
                                        sell_display = sell_display.sort_values("매매금액(KRW)", ascending=False)
                                        st.dataframe(
                                            sell_display.style.format({
                                                "주수": "{:,.0f}",
                                                "현지통화가격": "{:,.2f}",
                                                "매매금액(KRW)": "{:,.0f}"
                                            }),
                                            use_container_width=True
                                        )
                                        total_sell_krw = sell_trades["매매금액(KRW)"].sum()
                                        st.metric("총 매도 금액 (KRW)", f"{total_sell_krw:,.0f}")
                                    else:
                                        st.info("매도할 종목이 없습니다.")

                                # 전체 매매 상세 테이블
                                with st.expander("📊 전체 매매 상세 보기"):
                                    df_trades_display = df_trades[[
                                        "티커", "종목명", "매매", "주수", "현지통화가격", "통화",
                                        "원래비중", "목표비중", "비중변화", "매매금액(현지)", "매매금액(KRW)"
                                    ]].copy()
                                    df_trades_display = df_trades_display.sort_values("매매금액(KRW)", ascending=False)

                                    st.dataframe(
                                        df_trades_display.style.format({
                                            "주수": "{:,.0f}",
                                            "현지통화가격": "{:,.2f}",
                                            "원래비중": "{:.2%}",
                                            "목표비중": "{:.2%}",
                                            "비중변화": "{:+.2%}",
                                            "매매금액(현지)": "{:,.2f}",
                                            "매매금액(KRW)": "{:,.0f}"
                                        }),
                                        use_container_width=True
                                    )

                                    # 순 현금 흐름
                                    total_buy = buy_trades["매매금액(KRW)"].sum() if not buy_trades.empty else 0
                                    total_sell = sell_trades["매매금액(KRW)"].sum() if not sell_trades.empty else 0
                                    net_cash = total_sell - total_buy

                                    st.markdown("---")
                                    nc1, nc2, nc3 = st.columns(3)
                                    nc1.metric("총 매수", f"₩{total_buy:,.0f}")
                                    nc2.metric("총 매도", f"₩{total_sell:,.0f}")
                                    nc3.metric("순 현금 흐름", f"₩{net_cash:,.0f}",
                                              delta="현금 유입" if net_cash > 0 else "현금 유출" if net_cash < 0 else "균형")
                            else:
                                st.info("매매할 종목이 없습니다.")

                            # 섹터 비중 비교
                            st.markdown("### 🧭 섹터 비중 비교")

                            sector_map = result.get("sector_map", {})

                            # 원래 포트폴리오 섹터 비중
                            orig_sector_weights = {}
                            for ticker, weight in result["original_weights"].items():
                                sector = sector_map.get(ticker, "Unknown")
                                orig_sector_weights[sector] = orig_sector_weights.get(sector, 0) + weight

                            # 시뮬레이션 포트폴리오 섹터 비중
                            sim_sector_weights = {}
                            for ticker, weight in result["sim_weights"].items():
                                sector = sector_map.get(ticker, "Unknown")
                                sim_sector_weights[sector] = sim_sector_weights.get(sector, 0) + weight

                            # 모든 섹터 합치기
                            all_sectors_sim = sorted(set(orig_sector_weights.keys()) | set(sim_sector_weights.keys()))

                            sector_comparison = []
                            for sector in all_sectors_sim:
                                orig_w = orig_sector_weights.get(sector, 0)
                                sim_w = sim_sector_weights.get(sector, 0)
                                sector_comparison.append({
                                    "섹터": sector,
                                    "원래 비중": orig_w,
                                    "시뮬레이션 비중": sim_w,
                                    "변경폭": sim_w - orig_w
                                })

                            df_sector_comp = pd.DataFrame(sector_comparison)
                            df_sector_comp = df_sector_comp.sort_values("시뮬레이션 비중", ascending=False)

                            # 섹터 비중 차트
                            col_sector1, col_sector2 = st.columns(2)

                            with col_sector1:
                                fig_sector_orig = go.Figure(data=go.Pie(
                                    labels=list(orig_sector_weights.keys()),
                                    values=list(orig_sector_weights.values()),
                                    hole=0.4,
                                    title="원래 포트폴리오"
                                ))
                                fig_sector_orig.update_traces(textinfo="percent+label")
                                st.plotly_chart(fig_sector_orig, use_container_width=True)

                            with col_sector2:
                                fig_sector_sim = go.Figure(data=go.Pie(
                                    labels=list(sim_sector_weights.keys()),
                                    values=list(sim_sector_weights.values()),
                                    hole=0.4,
                                    title="시뮬레이션 포트폴리오"
                                ))
                                fig_sector_sim.update_traces(textinfo="percent+label")
                                st.plotly_chart(fig_sector_sim, use_container_width=True)

                            # 섹터 비중 변화 바 차트
                            df_sector_diff = df_sector_comp[df_sector_comp["변경폭"].abs() > 0.0001].copy()
                            if not df_sector_diff.empty:
                                colors_sector = np.where(df_sector_diff["변경폭"].values >= 0, "#16a34a", "#dc2626")
                                fig_sector_diff = go.Figure(data=go.Bar(
                                    x=df_sector_diff["섹터"],
                                    y=df_sector_diff["변경폭"],
                                    marker_color=colors_sector,
                                    text=[f"{v:+.1%}" for v in df_sector_diff["변경폭"]],
                                    textposition="auto"
                                ))
                                fig_sector_diff.update_layout(
                                    title="섹터 비중 변화",
                                    yaxis_tickformat=".1%",
                                    xaxis_title="",
                                    yaxis_title="비중 변화"
                                )
                                st.plotly_chart(fig_sector_diff, use_container_width=True)

                            # 섹터 비중 테이블
                            st.dataframe(
                                df_sector_comp.style.format({
                                    "원래 비중": "{:.2%}",
                                    "시뮬레이션 비중": "{:.2%}",
                                    "변경폭": "{:+.2%}"
                                }).background_gradient(subset=["변경폭"], cmap="RdYlGn", vmin=-0.05, vmax=0.05),
                                use_container_width=True
                            )

                            # 팩터 익스포저
                            st.markdown("### 📈 팩터 익스포저 (Factor Exposure)")
                            st.caption("팩터 ETF 대비 베타로 측정한 익스포저입니다.")

                            with st.spinner("팩터 익스포저 계산 중..."):
                                orig_exposure = calculate_factor_exposure(
                                    result["original_weights"],
                                    result["returns"],
                                    sim_days
                                )
                                sim_exposure = calculate_factor_exposure(
                                    result["sim_weights"],
                                    result["returns"],
                                    sim_days
                                )

                            if orig_exposure or sim_exposure:
                                all_factors = sorted(set(orig_exposure.keys()) | set(sim_exposure.keys()))

                                factor_comparison = []
                                for factor in all_factors:
                                    orig_exp = orig_exposure.get(factor, 0)
                                    sim_exp = sim_exposure.get(factor, 0)
                                    factor_comparison.append({
                                        "팩터": factor,
                                        "원래 익스포저": orig_exp,
                                        "시뮬레이션 익스포저": sim_exp,
                                        "변경폭": sim_exp - orig_exp
                                    })

                                df_factor = pd.DataFrame(factor_comparison)

                                # 팩터 익스포저 비교 차트
                                fig_factor = go.Figure()
                                fig_factor.add_trace(go.Bar(
                                    name="원래 포트폴리오",
                                    x=df_factor["팩터"],
                                    y=df_factor["원래 익스포저"],
                                    marker_color="#6366f1"
                                ))
                                fig_factor.add_trace(go.Bar(
                                    name="시뮬레이션 포트폴리오",
                                    x=df_factor["팩터"],
                                    y=df_factor["시뮬레이션 익스포저"],
                                    marker_color="#f97316"
                                ))
                                fig_factor.update_layout(
                                    title="팩터 익스포저 비교 (베타)",
                                    barmode="group",
                                    xaxis_title="",
                                    yaxis_title="베타",
                                    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
                                )
                                st.plotly_chart(fig_factor, use_container_width=True)

                                # 팩터 익스포저 테이블
                                st.dataframe(
                                    df_factor.style.format({
                                        "원래 익스포저": "{:.3f}",
                                        "시뮬레이션 익스포저": "{:.3f}",
                                        "변경폭": "{:+.3f}"
                                    }).background_gradient(subset=["변경폭"], cmap="RdYlGn", vmin=-0.2, vmax=0.2),
                                    use_container_width=True
                                )
                            else:
                                st.warning("팩터 익스포저를 계산할 수 없습니다.")

elif menu == "Total Portfolio (Team PNL)":
    st.subheader("📊 Total Team Portfolio Dashboard")
    uploaded_file = st.sidebar.file_uploader("Upload 'Team_PNL.xlsx'", type=['xlsx'], key="pnl")
    
    if uploaded_file:
        df_pnl, df_pos, err = load_team_pnl_data(uploaded_file)
        if df_pnl is not None:
            common_idx = df_pnl.index.intersection(df_pos.index)
            common_cols = [c for c in df_pnl.columns if c in df_pos.columns]
            df_pnl = df_pnl.loc[common_idx, common_cols]
            df_pos = df_pos.loc[common_idx, common_cols]
            
            df_cum_pnl = df_pnl.cumsum()
            df_user_ret = df_cum_pnl.div(df_pos.replace(0, np.nan)).fillna(0)
            df_daily_ret = df_pnl.div(df_pos.replace(0, np.nan)).fillna(0)
            
            t1, t2, t3, t4, t5 = st.tabs(["📈 Chart", "📊 Analysis", "🔗 Correlation", "🌍 Cross Asset", "🧪 Simulation"])
            
            with t1:
                strat = st.selectbox("Select Strategy", df_user_ret.columns)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_user_ret.index, y=df_user_ret[strat], name=strat, line=dict(width=2)))
                
                # Add Benchmarks
                bm_returns = download_benchmarks_all(df_pnl.index.min(), df_pnl.index.max())
                if not bm_returns.empty:
                    bm_cum = (1 + bm_returns).cumprod() - 1
                    for col in ['US', 'KR']:
                         if col in bm_cum.columns:
                            fig.add_trace(go.Scatter(x=bm_cum.index, y=bm_cum[col], name=f"{col} BM", line=dict(width=1, dash='dash')))
                st.plotly_chart(fig, use_container_width=True)
            
            with t2:
                stats = pd.DataFrame(index=df_daily_ret.columns)
                stats['Volatility'] = df_daily_ret.std() * np.sqrt(252)
                stats['Sharpe'] = (df_daily_ret.mean() / df_daily_ret.std() * np.sqrt(252)).fillna(0)
                stats['Win Rate'] = (df_daily_ret > 0).sum() / (df_daily_ret != 0).sum()
                
                # Profit Factor
                gp = df_daily_ret[df_daily_ret > 0].sum()
                gl = df_daily_ret[df_daily_ret < 0].sum().abs()
                stats['Profit Factor'] = (gp / gl).fillna(0)
                
                stats['MDD'] = ((1+df_daily_ret).cumprod() / (1+df_daily_ret).cumprod().cummax() - 1).min()
                stats['Total Return'] = df_user_ret.iloc[-1]
                
                disp = stats.copy()
                for c in disp.columns:
                    if c in ['Sharpe', 'Profit Factor']: disp[c] = disp[c].apply(lambda x: f"{x:.2f}")
                    else: disp[c] = disp[c].apply(lambda x: f"{x:.2%}")
                
                disp.insert(0, 'Strategy', disp.index)
                disp['Strategy'] = disp['Strategy'].apply(lambda x: x.split('_')[0])
                st.markdown(create_manual_html_table(disp), unsafe_allow_html=True)

            with t3:
                corr = df_daily_ret.corr()
                fig_corr = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale='RdBu', zmin=-1, zmax=1))
                fig_corr.update_layout(height=700)
                st.plotly_chart(fig_corr, use_container_width=True)
                
            with t4:
                df_assets = download_cross_assets(df_pnl.index.min(), df_pnl.index.max())
                if not df_assets.empty:
                    df_assets = df_assets.reindex(df_user_ret.index, method='ffill').pct_change().fillna(0)
                    comb = pd.concat([df_daily_ret, df_assets], axis=1).corr()
                    sub_corr = comb.loc[df_daily_ret.columns, df_assets.columns]
                    fig_cross = go.Figure(data=go.Heatmap(z=sub_corr.values, x=sub_corr.columns, y=sub_corr.index, colorscale='RdBu', zmin=-1, zmax=1))
                    st.plotly_chart(fig_cross, use_container_width=True)
                else: st.write("Data not available.")
            
            with t5:
                st.subheader("Simulation")
                c_in, c_out = st.columns([1,3])
                with c_in:
                    weights = {}
                    for col in df_daily_ret.columns:
                        weights[col] = st.slider(col, 0.0, 1.0, 1.0/len(df_daily_ret.columns), 0.05, key=f"sim_{col}")
                with c_out:
                    w_series = pd.Series(weights)
                    sim_daily = df_daily_ret.mul(w_series, axis=1).sum(axis=1)
                    sim_cum = (1 + sim_daily).cumprod() - 1
                    
                    fig_sim = go.Figure()
                    fig_sim.add_trace(go.Scatter(x=sim_cum.index, y=sim_cum, name="Simulated", line=dict(color='red')))
                    act_daily = df_pnl.sum(axis=1).div(df_pos.sum(axis=1)).fillna(0)
                    act_cum = (1 + act_daily).cumprod() - 1
                    fig_sim.add_trace(go.Scatter(x=act_cum.index, y=act_cum, name="Actual", line=dict(color='grey', dash='dot')))
                    st.plotly_chart(fig_sim, use_container_width=True)
        else: st.error(err)

elif menu == "Cash Equity Analysis":
    st.subheader("📈 Cash Equity Portfolio Analysis")
    uploaded_file_ce = st.sidebar.file_uploader("Upload 'Holdings3.xlsx'", type=['xlsx'], key="ce")
    
    if uploaded_file_ce:
        with st.spinner("Processing Data & Fetching Factors..."):
            res = load_cash_equity_data(uploaded_file_ce)
            df_perf, df_last, df_contrib, country_daily, logs, err, _ = res
        
        if err: st.error(err)
        elif df_perf is not None:
            start_dt, end_dt = df_perf.index.min(), df_perf.index.max()
            bm_returns = download_benchmarks_all(start_dt, end_dt)
            factor_prices = download_factors(start_dt, end_dt, return_prices=True)
            
            view_opt = st.radio("Currency View", ["KRW", "Local Currency (USD Base)"], horizontal=True)

            max_perf_date = df_perf.index.max() if not df_perf.empty else pd.NaT
            max_hold_date = df_last['기준일자'].max() if df_last is not None else pd.NaT
            max_date = max_perf_date if pd.notna(max_perf_date) else max_hold_date
            if pd.notna(max_date):
                curr_hold = df_last[(df_last['기준일자'] == max_date) & (df_last['잔고수량'] > 0)]
            else:
                curr_hold = df_last[df_last['잔고수량'] > 0] if df_last is not None else pd.DataFrame()

            if 'Total_MV_KRW' in df_perf.columns and not df_perf.empty:
                curr_aum = df_perf.iloc[-1]['Total_MV_KRW']
            else:
                curr_aum = curr_hold['원화평가금액'].sum() if not curr_hold.empty else 0
            
            c1, c2, c3, c4 = st.columns(4)
            if df_perf.empty:
                c1.metric("Total Return (Hedged)", "N/A")
                c2.metric("Equity Return (Unhedged)", "N/A")
                c3.metric("Hedge Impact", "N/A")
                y_main = y_sub = None
                name_main = name_sub = None
                target_ret = pd.Series(dtype=float)
            elif view_opt == "KRW":
                last_day = df_perf.iloc[-1]
                c1.metric("Total Return (Hedged)", f"{last_day['Cum_Total_KRW']:.2%}")
                c2.metric("Equity Return (Unhedged)", f"{last_day['Cum_Equity_KRW']:.2%}")
                c3.metric("Hedge Impact", f"{(last_day['Cum_Total_KRW'] - last_day['Cum_Equity_KRW']):.2%}")
                y_main, y_sub = 'Cum_Total_KRW', 'Cum_Equity_KRW'
                name_main, name_sub = 'Total (Hedged)', 'Equity (KRW)'
                target_ret = df_perf['Ret_Total_KRW']
            else:
                last_day = df_perf.iloc[-1]
                c1.metric("Total Return (Hedged)", f"{last_day['Cum_Total_Local']:.2%}")
                c2.metric("Equity Return (Unhedged)", f"{last_day['Cum_Equity_Local']:.2%}")
                c3.metric("Hedge Impact", f"{(last_day['Cum_Total_Local'] - last_day['Cum_Equity_Local']):.2%}")
                y_main, y_sub = 'Cum_Total_Local', 'Cum_Equity_Local'
                name_main, name_sub = 'Total (Hedged)', 'Equity (Local/USD)'
                target_ret = df_perf['Ret_Total_Local']
            c4.metric("Current AUM", f"{curr_aum:,.0f} KRW")

            if not df_perf.empty and y_main:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_perf.index, y=df_perf[y_main], name=name_main, line=dict(color='#2563eb', width=3)))
                if y_sub: fig.add_trace(go.Scatter(x=df_perf.index, y=df_perf[y_sub], name=name_sub, line=dict(color='#60a5fa', dash='dot')))
                
                if not bm_returns.empty:
                    bm_cum = (1 + bm_returns).cumprod() - 1
                    for col in ['US', 'KR', 'HK', 'JP']:
                        if col in bm_cum.columns:
                            fig.add_trace(go.Scatter(x=bm_cum.index, y=bm_cum[col], name=col+' BM', line=dict(width=1, dash='dash')))
                st.plotly_chart(fig, use_container_width=True)

            factor_returns = align_factor_returns(target_ret.index, factor_prices)

            st.markdown("#### 📊 Risk Metrics (Hedged Total Returns)")
            rows = []
            if view_opt == "KRW":
                port_label = "Portfolio (Hedged, KRW)"
            else:
                port_label = "Portfolio (Hedged, Local/USD)"
                st.caption("Local currency view uses total hedged returns (includes hedge PnL).")
            port_metrics = _calc_perf_metrics(target_ret)
            if port_metrics:
                rows.append({"Asset": port_label, **port_metrics})
            if not bm_returns.empty:
                bm_aligned = bm_returns.reindex(df_perf.index).dropna(how='all')
                for col in bm_aligned.columns:
                    metrics = _calc_perf_metrics(bm_aligned[col].dropna())
                    if metrics:
                        rows.append({"Asset": f"Benchmark {col}", **metrics})
            if rows:
                metrics_df = pd.DataFrame(rows)
                metric_order = [
                    "Total Return",
                    "CAGR",
                    "Annualized Volatility",
                    "Sharpe Ratio",
                    "Sortino Ratio",
                    "Max Drawdown",
                    "Calmar Ratio",
                    "Win Rate",
                ]
                metrics_df = metrics_df[["Asset"] + metric_order]
                disp = metrics_df.copy()
                percent_cols = ["Total Return", "CAGR", "Annualized Volatility", "Max Drawdown", "Win Rate"]
                ratio_cols = ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"]
                for col in percent_cols:
                    disp[col] = disp[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")
                for col in ratio_cols:
                    disp[col] = disp[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
                st.markdown(create_manual_html_table(disp, title="Risk Metrics vs Benchmarks"), unsafe_allow_html=True)
            else:
                st.write("Risk metrics not available.")

            t1, t2, t3, t4 = st.tabs(["Factor Risk & Attribution", "Selection Effect", "Holdings", "Beta Trend"])
            
            with t1:
                st.markdown(f"#### 🧪 {FACTOR_TARGET_COUNT}-Factor Analysis (Risk & Attribution)")
                if not factor_returns.empty:
                    exposures, contrib, r2 = perform_factor_regression(target_ret, factor_returns)
                    
                    if exposures is not None:
                        st.write(f"**R-Squared:** {r2:.2f} (Explained by Factors)")
                        c_exp, c_attr = st.columns(2)
                        with c_exp:
                            st.markdown("**Factor Exposures (Beta)**")
                            fig_exp = px.bar(exposures, orientation='h', labels={'value':'Beta', 'index':'Factor'})
                            fig_exp.update_layout(showlegend=False)
                            st.plotly_chart(fig_exp, use_container_width=True)
                        with c_attr:
                            st.markdown("**Cumulative Factor Attribution**")
                            if not contrib.empty:
                                contrib_cum = (1 + contrib).cumprod() - 1
                                fig_attr = go.Figure()
                                for col in contrib_cum.columns:
                                    if col != 'Alpha' and col != 'Unexplained':
                                        fig_attr.add_trace(go.Scatter(x=contrib_cum.index, y=contrib_cum[col], name=col))
                                st.plotly_chart(fig_attr, use_container_width=True)
                        
                        st.markdown("#### 📅 Monthly Factor Attribution")
                        m_contrib = contrib.resample('ME').apply(lambda x: (1+x).prod()-1)
                        m_contrib.index = m_contrib.index.strftime('%Y-%m')
                        fig_heat = go.Figure(data=go.Heatmap(
                            z=m_contrib.T.values, x=m_contrib.index, y=m_contrib.columns,
                            colorscale='RdBu', zmin=-0.03, zmax=0.03
                        ))
                        fig_heat.update_layout(height=500)
                        st.plotly_chart(fig_heat, use_container_width=True)
                    else:
                        st.write("Insufficient data to compute factor regression.")
                else: st.warning("Factor data download failed.")

            with t2:
                st.markdown("#### 💹 Return Contribution")
                if df_contrib:
                    c_a, c_b = st.columns(2)
                    with c_a:
                        if not df_contrib['Country'].empty:
                            ctry_cont = df_contrib['Country'].groupby('Country')['Contrib_KRW'].sum().sort_values(ascending=False).reset_index()
                            st.plotly_chart(px.bar(ctry_cont, x='Contrib_KRW', y='Country', orientation='h', title="Contribution by Country", text_auto='.2%'))
                    with c_b:
                        if not df_contrib['Sector'].empty:
                            sec_cont = df_contrib['Sector'].groupby('섹터')['Contrib_KRW'].sum().sort_values(ascending=False).reset_index()
                            st.plotly_chart(px.bar(sec_cont, x='Contrib_KRW', y='섹터', orientation='h', title="Contribution by Sector", text_auto='.2%'))
                    
                    st.markdown("---")
                    st.markdown("#### 🥧 Current Allocation Breakdown")
                    if not curr_hold.empty:
                        st.plotly_chart(px.pie(curr_hold, values='원화평가금액', names='섹터', title="Sector Allocation", hole=0.4), use_container_width=True)
                        st.plotly_chart(px.pie(curr_hold, values='원화평가금액', names='Country', title="Country Allocation", hole=0.4), use_container_width=True)

            with t3:
                pnl_df = df_last.sort_values('Final_PnL', ascending=False)[['종목명','섹터','Country','Final_PnL']]
                cw, cl = st.columns(2)
                cw.success("Top Winners"); cw.dataframe(pnl_df.head(5).style.format({'Final_PnL':'{:,.0f}'}))
                cl.error("Top Losers"); cl.dataframe(pnl_df.tail(5).style.format({'Final_PnL':'{:,.0f}'}))
                with st.expander("Daily Data"): st.dataframe(df_perf)

            with t4:
                st.markdown("#### 📈 Rolling Beta Trend vs Benchmarks")
                if bm_returns.empty:
                    st.warning("Benchmark data download failed.")
                else:
                    beta_window = st.slider(
                        "Rolling window (trading days)",
                        min_value=20,
                        max_value=252,
                        value=60,
                        step=5,
                        key="beta_window",
                    )
                    beta_fig = go.Figure()
                    bench_map = {"SPX": "US", "Hang Seng": "HK", "Nikkei 225": "JP"}
                    for label, col in bench_map.items():
                        if col in bm_returns.columns:
                            beta_series = calculate_rolling_beta(target_ret, bm_returns[col], window=beta_window)
                            if not beta_series.empty:
                                beta_fig.add_trace(go.Scatter(x=beta_series.index, y=beta_series, name=f"{label} Beta"))
                    if beta_fig.data:
                        beta_fig.update_layout(yaxis_title="Beta", xaxis_title="Date")
                        st.plotly_chart(beta_fig, use_container_width=True)
                    else:
                        st.write("Insufficient data to compute rolling beta.")

                st.markdown("#### 🧮 Holdings-Weighted Beta (Latest)")
                bench_yf_map = {"S&P 500": "^GSPC", "Hang Seng": "^HSI", "Nikkei 225": "^N225", "KOSPI": "^KS11"}
                holdings_beta = calculate_holdings_beta(curr_hold, bench_yf_map, end_date=max_date)
                if holdings_beta:
                    beta_df = pd.Series(holdings_beta, name="Beta").sort_values().reset_index()
                    beta_df.columns = ["Benchmark", "Beta"]
                    fig_beta = px.bar(beta_df, x="Beta", y="Benchmark", orientation="h", title="Holdings-Weighted Beta")
                    fig_beta.update_layout(showlegend=False)
                    st.plotly_chart(fig_beta, use_container_width=True)
                else:
                    st.write("Insufficient data to compute holdings-weighted beta.")

elif menu == "📑 Weekly Report Generator":
    st.subheader("📑 Weekly Meeting Report Generator")
    uploaded_file_ce = st.sidebar.file_uploader("Upload 'Holdings3.xlsx' for Report", type=['xlsx'], key="rep")
    
    if uploaded_file_ce:
        with st.spinner("Generating Report Data..."):
            res = load_cash_equity_data(uploaded_file_ce)
            df_perf, df_last, df_contrib, country_daily, df_daily_stock, logs, err = res
            
        if err: st.error(err)
        elif df_perf is not None:
            view_opt = st.radio("Currency View", ["KRW (Hedged)", "Local (Hedged, USD base)"], horizontal=True, key="weekly_view_opt")
            if "Local" in view_opt:
                ret_col = "Ret_Total_Local"
                view_label = "Total Return (Local, hedged)"
            else:
                ret_col = "Ret_Total_KRW"
                view_label = "Total Return (KRW, hedged)"

            max_date = df_perf.index.max()
            report_date = st.date_input("Report Date", max_date)
            report_date = pd.to_datetime(report_date)
            
            factor_prices = download_factors(df_perf.index.min(), report_date, return_prices=True)
            factor_returns = align_factor_returns(df_perf[ret_col].index, factor_prices)
            _, factor_contrib, _ = perform_factor_regression(df_perf[ret_col], factor_returns)
            
            dates = {
                'WTD': df_perf.index[df_perf.index <= report_date][-1] - pd.to_timedelta(df_perf.index[df_perf.index <= report_date][-1].weekday(), unit='D'),
                'MTD': report_date.replace(day=1),
                'QTD': report_date.replace(month=((report_date.month-1)//3)*3+1, day=1),
                'YTD': report_date.replace(month=1, day=1)
            }
            
            global_px = download_global_indices(min(dates.values()), report_date)
            df_perf_cut = df_perf[df_perf.index <= report_date]
            df_stock_cut = df_daily_stock[df_daily_stock['기준일자'] <= report_date]
            if factor_contrib is not None:
                factor_contrib_cut = factor_contrib[factor_contrib.index <= report_date]
            else: factor_contrib_cut = None
            
            def calc_period_stats(start_dt, label, global_px):
                sub_perf = df_perf_cut[df_perf_cut.index >= start_dt]
                if sub_perf.empty: return None
                cum_ret = (1 + sub_perf[ret_col]).prod() - 1
                abs_pnl = sub_perf['Total_PnL_KRW'].sum()  # already includes hedge PnL in KRW
                sub_stock = df_stock_cut[df_stock_cut['기준일자'] >= start_dt]
                stock_contrib = sub_stock.groupby(['종목명', 'Ticker_ID'])['Contrib_KRW'].sum().reset_index()
                top5 = stock_contrib.sort_values('Contrib_KRW', ascending=False).head(5)
                bot5 = stock_contrib.sort_values('Contrib_KRW', ascending=True).head(5)
                ctry_contrib = sub_stock.groupby('Country')['Contrib_KRW'].sum().sort_values(ascending=False)
                sect_contrib = sub_stock.groupby('섹터')['Contrib_KRW'].sum().sort_values(ascending=False)
                f_cont = pd.Series(dtype=float)
                if factor_contrib_cut is not None:
                    sub_f = factor_contrib_cut[factor_contrib_cut.index >= start_dt]
                    if not sub_f.empty:
                        f_cont = sub_f.apply(lambda x: (1+x).prod()-1).sort_values(ascending=False)
                idx_ret = pd.Series(dtype=float)
                sub_px = None
                if global_px is not None and not global_px.empty:
                    sub_px = global_px[(global_px.index >= start_dt) & (global_px.index <= report_date)]
                    sub_px = sub_px.ffill().bfill()
                    if not sub_px.empty:
                        idx_ret = sub_px.iloc[-1] / sub_px.iloc[0] - 1
                hedge_contrib = None
                hedge_pnl_krw = None
                if 'Hedge_PnL_KRW' in sub_perf.columns:
                    hedge_pnl_krw = sub_perf['Hedge_PnL_KRW'].sum()
                if ret_col == "Ret_Total_Local":
                    if 'Ret_Hedge_Local' in sub_perf.columns:
                        hedge_contrib = sub_perf['Ret_Hedge_Local'].fillna(0).sum()
                else:
                    if 'Hedge_PnL_KRW' in sub_perf.columns and 'Total_Prev_MV_KRW' in sub_perf.columns:
                        denom = sub_perf['Total_Prev_MV_KRW'].replace(0, np.nan)
                        hedge_contrib = (sub_perf['Hedge_PnL_KRW'] / denom).fillna(0).sum()
                portfolio_risk = _calc_risk_metrics(sub_perf[ret_col])
                benchmark_risk = {}
                if sub_px is not None and not sub_px.empty:
                    bench_ret = sub_px.pct_change().dropna(how='all')
                    for col in bench_ret.columns:
                        metrics = _calc_risk_metrics(bench_ret[col])
                        if metrics:
                            benchmark_risk[col] = metrics
                return {'label': label, 'ret': cum_ret, 'pnl': abs_pnl, 'top5': top5, 'bot5': bot5, 
                        'ctry': ctry_contrib, 'sect': sect_contrib, 'factor': f_cont, 'indices': idx_ret,
                        'hedge_contrib': hedge_contrib, 'hedge_pnl_krw': hedge_pnl_krw,
                        'risk': {'portfolio': portfolio_risk, 'benchmarks': benchmark_risk}}

            tabs = st.tabs(["Summary Report", "WTD", "MTD", "QTD", "YTD"])
            stats_res = {}
            for p in ['WTD', 'MTD', 'QTD', 'YTD']:
                stats_res[p] = calc_period_stats(dates[p], p, global_px)
                with tabs[list(dates.keys()).index(p) + 1]:
                    if stats_res[p]:
                        st.markdown(f"### {p} Performance ({dates[p].date()} ~ {report_date.date()})")
                        if not stats_res[p]['indices'].empty:
                            idx_df = stats_res[p]['indices'].sort_values(ascending=False).reset_index()
                            idx_df.columns = ['Index', 'Return']
                            idx_df['Return'] = idx_df['Return'].apply(lambda x: f"{x:.2%}")
                            st.markdown(create_manual_html_table(idx_df, title="Global Index Returns"), unsafe_allow_html=True)
                        c1, c2 = st.columns(2)
                        c1.metric(view_label, f"{stats_res[p]['ret']:.2%}")
                        c2.metric("PnL (KRW)", f"{stats_res[p]['pnl']:,.0f}")
                        st.markdown("#### Top Contributors")
                        c3, c4 = st.columns(2)
                        with c3: st.table(stats_res[p]['top5'][['종목명', 'Contrib_KRW']].style.format({'Contrib_KRW': '{:.2%}'}))
                        with c4: st.table(stats_res[p]['bot5'][['종목명', 'Contrib_KRW']].style.format({'Contrib_KRW': '{:.2%}'}))
                        st.markdown("#### Attribution Analysis")
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.markdown("**Country**")
                            st.dataframe(stats_res[p]['ctry'].to_frame().style.format('{:.2%}'))
                        with col_b:
                            st.markdown("**Sector**")
                            st.dataframe(stats_res[p]['sect'].to_frame().style.format('{:.2%}'))
                        with col_c:
                            st.markdown("**Factor**")
                            if not stats_res[p]['factor'].empty:
                                st.dataframe(stats_res[p]['factor'].to_frame(name='Contrib').style.format('{:.2%}'))
                            else: st.write("No factor data")
                        hedge_val = stats_res[p].get('hedge_contrib')
                        hedge_display = f"{hedge_val:.2%}" if hedge_val is not None else "N/A"
                        st.markdown("**Hedge Contribution**")
                        st.metric("Hedge Contribution (Return)", hedge_display)
                    else: st.write("No data.")

            with tabs[0]:
                st.markdown("### 📝 Weekly Meeting Commentary")
                txt = f"**[Portfolio Weekly Update - {report_date.date()}]**\n\n"
                
                wtd = stats_res.get('WTD')
                if wtd:
                    txt += f"**1. WTD Performance:** {wtd['ret']:.2%} ({wtd['pnl']:,.0f} KRW)\n"
                    if not wtd['top5'].empty: txt += f"   - **Lead:** {wtd['top5'].iloc[0]['종목명']} (+{wtd['top5'].iloc[0]['Contrib_KRW']:.2%})\n"
                    if not wtd['factor'].empty: txt += f"   - **Factor:** Driven by {wtd['factor'].idxmax()} (+{wtd['factor'].max():.2%})\n"
                
                mtd = stats_res.get('MTD')
                if mtd: 
                    txt += f"**2. MTD:** {mtd['ret']:.2%}. Best Sector: {mtd['sect'].idxmax()}.\n"
                    if not mtd['factor'].empty: txt += f"   - Factor: {mtd['factor'].idxmax()} style worked well.\n"
                
                ytd = stats_res.get('YTD')
                if ytd: txt += f"**3. YTD:** {ytd['ret']:.2%}, Total PnL {ytd['pnl']:,.0f} KRW.\n"
                
                st.text_area("Copy this:", txt, height=300)

                st.markdown("#### 🤖 AI-Generated Weekly Report")
                llm_choice = st.radio(
                    "Select LLM",
                    ["OpenAI (gpt-4o-mini)", "DeepSeek (deepseek-chat)"],
                    horizontal=True,
                    key="llm_choice_radio",
                )
                lang_choice = st.radio(
                    "언어 / Language",
                    ["English", "한국어"],
                    horizontal=True,
                    key="ai_lang_choice_radio",
                )
                user_comment = st.text_area(
                    "Optional: Add your own market/positioning comments",
                    height=120,
                    key="ai_comment_input",
                )
                if st.button("Generate AI Report", key="ai_report_btn"):
                    provider = "deepseek" if "DeepSeek" in llm_choice else "openai"
                    try:
                        ai_text = generate_ai_weekly_report(
                            stats_res,
                            report_date,
                            user_comment,
                            provider=provider,
                            language=lang_choice,
                        )
                        st.text_area("AI Report", ai_text, height=400, key="ai_report_text")
                    except Exception as e:
                        st.error(f"Failed to generate AI report: {e}")
