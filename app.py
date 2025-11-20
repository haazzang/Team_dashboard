import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import glob
import os

# --- 페이지 설정 ---
st.set_page_config(page_title="Portfolio Dashboard", layout="wide")
st.title("🚀 Team Portfolio Analysis Dashboard (Holdings3 Based)")

# ==============================================================================
# [Helper Functions]
# ==============================================================================

@st.cache_data
def fetch_sectors_cached(tickers):
    """티커별 섹터 정보를 yfinance에서 가져옵니다."""
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

@st.cache_data
def download_benchmark(start_date, end_date):
    """벤치마크(S&P500, KOSPI) 데이터를 다운로드합니다."""
    try:
        # 날짜 버퍼 추가
        bm = yf.download(['^GSPC', '^KS11'], start=start_date, end=end_date + pd.Timedelta(days=5), progress=False)['Close']
        bm = bm.ffill()
        # 인덱스가 MultiIndex인 경우 처리
        if isinstance(bm.columns, pd.MultiIndex):
             bm.columns = bm.columns.droplevel(1) # Ticker 레벨 제거 가정
        return bm
    except Exception as e:
        st.error(f"Benchmark download failed: {e}")
        return pd.DataFrame()

def load_data():
    """
    Holdings3 패턴의 파일들과 Hedge 파일을 로드하고 전처리합니다.
    사용자가 제공한 컬럼 매핑을 적용합니다.
    """
    # 1. Holdings3 데이터 로드 (주식 포트폴리오)
    files = sorted(glob.glob("Holdings3*.csv"))
    # Hedge 파일은 별도로 처리하기 위해 제외
    stock_files = [f for f in files if "Hedge" not in f]
    
    df_list = []
    for f in stock_files:
        try:
            temp = pd.read_csv(f)
            df_list.append(temp)
        except Exception as e:
            st.warning(f"Skipping {f}: {e}")
            
    if not df_list:
        return pd.DataFrame(), pd.DataFrame()
        
    df = pd.concat(df_list, ignore_index=True)
    
    # 2. 컬럼 매핑 (Holdings3 -> Logic Standard)
    # 요청 매핑: 
    # date: 기준일자, isin: 종목코드, Ticker: 심볼, Quantity: 잔고수량
    # Book Price: 장부단가, Notional: 외화장부금액, Market Price: 평가단가
    # Market Value: 외화평가금액, 평가손익: 외화평가손익, 매매손익: 외화총매매손익, Currency: 통화
    
    rename_map = {
        '기준일자': 'Date',
        '종목코드': 'ISIN',
        '심볼': 'Ticker',
        '잔고수량': 'Quantity',
        '장부단가': 'Book Price',
        '외화장부금액': 'Notional_USD',       # USD Logic용
        '평가단가': 'Market Price',
        '외화평가금액': 'Market Value_USD',   # USD Logic용
        '외화평가손익': 'Unrealized_USD',     # USD Logic용
        '외화총매매손익': 'Realized_USD',     # USD Logic용 (Cumulative)
        '통화': 'Currency',
        # KRW Logic을 위해 원화 컬럼도 매핑
        '원화장부금액': 'Notional_KRW',
        '원화평가금액': 'Market Value_KRW',
        '원화평가손익': 'Unrealized_KRW',
        '원화총매매손익': 'Realized_KRW'      # (Cumulative)
    }
    
    # 존재하는 컬럼만 이름 변경
    existing_cols = set(df.columns)
    valid_rename = {k: v for k, v in rename_map.items() if k in existing_cols}
    df = df.rename(columns=valid_rename)
    
    # 날짜 변환
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # 숫자형 변환 (콤마 제거 등)
    numeric_cols = ['Notional_USD', 'Market Value_USD', 'Unrealized_USD', 'Realized_USD',
                    'Notional_KRW', 'Market Value_KRW', 'Unrealized_KRW', 'Realized_KRW']
    
    for col in numeric_cols:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '').astype(float)
            df[col] = df[col].fillna(0)

    # 3. Hedge 데이터 로드
    hedge_files = [f for f in files if "Hedge" in f]
    df_hedge = pd.DataFrame()
    if hedge_files:
        try:
            # 가장 최근 Hedge 파일 하나만 사용하거나 병합 (여기서는 병합 가정)
            h_list = []
            for hf in hedge_files:
                h_temp = pd.read_csv(hf)
                h_list.append(h_temp)
            df_hedge = pd.concat(h_list, ignore_index=True)
            
            # Hedge 파일 컬럼 처리
            if '기준일자' in df_hedge.columns:
                df_hedge['Date'] = pd.to_datetime(df_hedge['기준일자'])
            
            # 필요한 컬럼: 누적 총손익 (Cumulative Total PnL)
            # 컬럼명이 '누적 총손익'이라고 가정 (Holdings3 - Hedge.csv 기준)
            if '누적 총손익' in df_hedge.columns:
                df_hedge = df_hedge[['Date', '누적 총손익']].sort_values('Date')
                df_hedge = df_hedge.groupby('Date').last().reset_index() # 중복 날짜 제거
                df_hedge = df_hedge.rename(columns={'누적 총손익': 'Hedge_Cum_PnL'})
                
                # 숫자형 변환
                if df_hedge['Hedge_Cum_PnL'].dtype == 'object':
                     df_hedge['Hedge_Cum_PnL'] = df_hedge['Hedge_Cum_PnL'].astype(str).str.replace(',', '').astype(float)
            else:
                st.warning("Hedge file found but '누적 총손익' column missing.")
                df_hedge = pd.DataFrame()
                
        except Exception as e:
            st.warning(f"Error loading hedge file: {e}")
            df_hedge = pd.DataFrame()

    return df, df_hedge

# ==============================================================================
# [Main Logic]
# ==============================================================================

data_load_state = st.text('Loading data...')
df, df_hedge = load_data()
data_load_state.text('Loading data... done!')

if df.empty:
    st.error("No Holdings3 data found. Please upload files.")
    st.stop()

# --- Date Range Selection ---
min_date, max_date = df['Date'].min(), df['Date'].max()
st.sidebar.header("Settings")
date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

if len(date_range) == 2:
    start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    
    # 날짜 필터링
    df_filtered = df[(df['Date'] >= start_d) & (df['Date'] <= end_d)].copy()
    
    if df_hedge is not None and not df_hedge.empty:
        hedge_filtered = df_hedge[(df_hedge['Date'] >= start_d) & (df_hedge['Date'] <= end_d)].copy()
    else:
        hedge_filtered = pd.DataFrame()
        
    # ==============================================================================
    # [1] Daily Logic Calculation (NAV Based)
    # ==============================================================================
    
    # 일자별 합계 집계
    daily_agg = df_filtered.groupby('Date')[
        ['Notional_USD', 'Market Value_USD', 'Unrealized_USD', 'Realized_USD',
         'Notional_KRW', 'Market Value_KRW', 'Unrealized_KRW', 'Realized_KRW']
    ].sum().reset_index()
    
    # Hedge 데이터 병합
    if not hedge_filtered.empty:
        daily_agg = pd.merge(daily_agg, hedge_filtered[['Date', 'Hedge_Cum_PnL']], on='Date', how='left')
        daily_agg['Hedge_Cum_PnL'] = daily_agg['Hedge_Cum_PnL'].fillna(0)
    else:
        daily_agg['Hedge_Cum_PnL'] = 0

    # --- NAV & Return Calculation Logic ---
    # Logic: NAV = Book(Notional) + Unrealized + Realized_Cumulative
    # This correctly accounts for realized gains adding to the portfolio equity even if positions are closed.
    
    # 1. USD (Pure Assets)
    daily_agg['NAV_USD'] = daily_agg['Notional_USD'] + daily_agg['Unrealized_USD'] + daily_agg['Realized_USD']
    
    # 2. KRW Unhedged (Asset + FX)
    daily_agg['NAV_KRW_Unhedged'] = daily_agg['Notional_KRW'] + daily_agg['Unrealized_KRW'] + daily_agg['Realized_KRW']
    
    # 3. KRW Hedged (Asset + FX + Futures)
    # Hedged NAV = Unhedged NAV + Hedge PnL
    daily_agg['NAV_KRW_Hedged'] = daily_agg['NAV_KRW_Unhedged'] + daily_agg['Hedge_Cum_PnL']

    # --- Normalize for Cumulative Return Chart (Base 100 or 0%) ---
    # 여기서는 시작일 대비 누적 수익률(%)을 계산
    
    base_usd = daily_agg['NAV_USD'].iloc[0] if daily_agg['NAV_USD'].iloc[0] != 0 else 1
    base_krw_unhedged = daily_agg['NAV_KRW_Unhedged'].iloc[0] if daily_agg['NAV_KRW_Unhedged'].iloc[0] != 0 else 1
    # Hedged 시작점은 Unhedged와 동일하게 맞추거나, 혹은 해당 시점의 Hedged NAV 사용
    # 논리적으로 Hedge가 0일차부터 있었다면 해당 NAV 사용
    base_krw_hedged = daily_agg['NAV_KRW_Hedged'].iloc[0] if daily_agg['NAV_KRW_Hedged'].iloc[0] != 0 else 1
    
    daily_agg['Cum_Ret_USD'] = (daily_agg['NAV_USD'] / base_usd) - 1
    daily_agg['Cum_Ret_KRW_Unhedged'] = (daily_agg['NAV_KRW_Unhedged'] / base_krw_unhedged) - 1
    daily_agg['Cum_Ret_KRW_Hedged'] = (daily_agg['NAV_KRW_Hedged'] / base_krw_hedged) - 1

    # ==============================================================================
    # [2] Benchmark Comparison
    # ==============================================================================
    bm_data = download_benchmark(start_d, end_d)
    bm_cum = pd.DataFrame()
    
    if not bm_data.empty:
        # Reindex to match portfolio dates
        bm_data = bm_data.reindex(daily_agg['Date'], method='ffill')
        
        # Calculate cumulative returns
        for col in bm_data.columns:
            first_val = bm_data[col].dropna().iloc[0]
            bm_cum[col] = (bm_data[col] / first_val) - 1
        
        bm_cum['Date'] = bm_data.index
    
    # ==============================================================================
    # [3] Dashboard Layout
    # ==============================================================================
    
    # --- Summary Metrics (Latest Date) ---
    latest = daily_agg.iloc[-1]
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total NAV (USD)", f"${latest['NAV_USD']:,.0f}", delta=f"{latest['Cum_Ret_USD']*100:.2f}% (Cum)")
    m2.metric("Total NAV (KRW Unhedged)", f"₩{latest['NAV_KRW_Unhedged']:,.0f}", delta=f"{latest['Cum_Ret_KRW_Unhedged']*100:.2f}% (Cum)")
    m3.metric("Total NAV (KRW Hedged)", f"₩{latest['NAV_KRW_Hedged']:,.0f}", delta=f"{latest['Cum_Ret_KRW_Hedged']*100:.2f}% (Cum)")
    
    hedge_effect = latest['Cum_Ret_KRW_Hedged'] - latest['Cum_Ret_KRW_Unhedged']
    m4.metric("Hedge Effect", f"{hedge_effect*100:.2f}%p", delta_color="off")

    # --- Chart 1: Cumulative Returns ---
    st.subheader("📈 Cumulative Returns Comparison")
    
    fig = go.Figure()
    
    # Portfolio Lines
    fig.add_trace(go.Scatter(x=daily_agg['Date'], y=daily_agg['Cum_Ret_USD'], 
                             mode='lines', name='Portfolio (USD)', line=dict(color='green', width=2)))
    fig.add_trace(go.Scatter(x=daily_agg['Date'], y=daily_agg['Cum_Ret_KRW_Unhedged'], 
                             mode='lines', name='KRW (Unhedged)', line=dict(color='orange', width=2)))
    fig.add_trace(go.Scatter(x=daily_agg['Date'], y=daily_agg['Cum_Ret_KRW_Hedged'], 
                             mode='lines', name='KRW (With Hedge)', line=dict(color='blue', width=3)))
    
    # Benchmark Lines
    if not bm_cum.empty:
        if '^GSPC' in bm_cum.columns:
            fig.add_trace(go.Scatter(x=bm_cum['Date'], y=bm_cum['^GSPC'], 
                                     name='S&P 500', line=dict(color='gray', dash='dot')))
        if '^KS11' in bm_cum.columns:
            fig.add_trace(go.Scatter(x=bm_cum['Date'], y=bm_cum['^KS11'], 
                                     name='KOSPI', line=dict(color='silver', dash='dot')))

    fig.update_layout(
        hovermode="x unified",
        yaxis_tickformat='.1%',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Tabbed Analysis ---
    t1, t2, t3 = st.tabs(["Portfolio Detail", "Sector Analysis", "Raw Data"])
    
    with t1:
        # 날짜별 종목 비중 확인
        last_date = df_filtered['Date'].max()
        current_holdings = df_filtered[df_filtered['Date'] == last_date].copy()
        current_holdings = current_holdings[current_holdings['Quantity'] > 0] # 잔고 있는 것만
        
        st.markdown(f"**Holdings as of {last_date.date()}**")
        
        # 간단한 테이블
        disp_cols = ['Ticker', 'ISIN', 'Quantity', 'Book Price', 'Market Price', 'Market Value_USD', 'Unrealized_USD']
        # 매핑된 컬럼이 있는지 확인하고 출력
        valid_disp = [c for c in disp_cols if c in current_holdings.columns]
        
        st.dataframe(
            current_holdings[valid_disp].sort_values('Market Value_USD', ascending=False).style.format({
                'Market Value_USD': '${:,.0f}',
                'Unrealized_USD': '${:,.0f}',
                'Book Price': '${:.2f}',
                'Market Price': '${:.2f}'
            }),
            use_container_width=True
        )
        
    with t2:
        if not current_holdings.empty:
            tickers = current_holdings['Ticker'].unique()
            sec_map = fetch_sectors_cached(tickers)
            current_holdings['Sector']