import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import datetime

# --- 페이지 설정 ---
st.set_page_config(page_title="Portfolio Dashboard", layout="wide")
st.title("🚀 Team Portfolio Analysis Dashboard")

# ---------------------------------------------------------
# 1. 공통 및 기존 데이터 로드 함수 (Team_PNL.xlsx 용)
# ---------------------------------------------------------
@st.cache_data
def load_team_pnl_data(file):
    try:
        # PNL 시트 로드
        df_pnl_raw = pd.read_excel(file, sheet_name='PNL', header=None, engine='openpyxl')
        header_idx = -1
        for i in range(10):
            if '일자' in df_pnl_raw.iloc[i].astype(str).values:
                header_idx = i
                break
        if header_idx == -1: return None, None, "PNL 시트에서 '일자'를 찾을 수 없습니다."
        
        pnl_cols = df_pnl_raw.iloc[header_idx].tolist()
        df_pnl_raw.columns = pnl_cols
        df_pnl = df_pnl_raw.iloc[header_idx+1:].copy()
        
        # Position 시트 로드
        df_pos_raw = pd.read_excel(file, sheet_name='Position', header=None, engine='openpyxl')
        header_idx_pos = -1
        for i in range(10):
            if '일자' in df_pos_raw.iloc[i].astype(str).values:
                header_idx_pos = i
                break
        if header_idx_pos == -1: return None, None, "Position 시트에서 '일자'를 찾을 수 없습니다."
        
        pos_cols = df_pos_raw.iloc[header_idx_pos].tolist()
        df_pos_raw.columns = pos_cols
        df_pos = df_pos_raw.iloc[header_idx_pos+1:].copy()

        # 정제 함수
        def clean_df(df):
            date_col = next((c for c in df.columns if str(c).strip() == '일자'), None)
            if not date_col: return None
            df.set_index(date_col, inplace=True)
            df.index = pd.to_datetime(df.index, errors='coerce')
            df = df.dropna(how='all')
            df = df[df.index.notnull()]
            df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            new_cols = []
            seen = {}
            for col in df.columns:
                c = str(col).strip()
                if c in ['nan', 'None', '', 'NaT']: continue
                if c in seen:
                    seen[c] += 1
                    new_cols.append(f"{c}_{seen[c]}")
                else:
                    seen[c] = 0
                    new_cols.append(c)
            
            valid_idx = [i for i, c in enumerate(df.columns) if str(c).strip() not in ['nan', 'None', '', 'NaT']]
            df_final = df.iloc[:, valid_idx]
            df_final.columns = new_cols
            return df_final, new_cols

        df_pnl_clean, cols_pnl = clean_df(df_pnl)
        df_pos_clean, cols_pos = clean_df(df_pos)
        
        return df_pnl_clean, df_pos_clean, cols_pnl

    except Exception as e:
        return None, None, f"Team_PNL 파일 오류: {e}"

# ---------------------------------------------------------
# 2. Cash Equity 데이터 로드 함수 (Holdings3.xlsx 용)
# ---------------------------------------------------------
@st.cache_data
def load_cash_equity_data(file):
    try:
        # 모든 시트 읽기 (일별 Holdings 데이터가 여러 시트에 나뉘어 있을 수 있음)
        xls = pd.ExcelFile(file, engine='openpyxl')
        sheet_names = xls.sheet_names
        
        all_holdings = []
        df_hedge = pd.DataFrame()
        
        for sheet in sheet_names:
            # Hedge 시트 처리
            if 'hedge' in sheet.lower() or '헷지' in sheet:
                df_h = pd.read_excel(file, sheet_name=sheet, engine='openpyxl')
                # Hedge 시트 컬럼 정제
                # 가정: 기준일자, 매매손익(원화환산), 평가손익(원화환산) 컬럼이 존재
                # 헤더 찾기
                h_idx = -1
                for i in range(10):
                    if '기준일자' in df_h.iloc[i].astype(str).values:
                        h_idx = i
                        break
                if h_idx != -1:
                    df_h.columns = df_h.iloc[h_idx]
                    df_h = df_h.iloc[h_idx+1:].copy()
                    
                df_h['기준일자'] = pd.to_datetime(df_h['기준일자'], errors='coerce')
                df_h = df_h.dropna(subset=['기준일자'])
                
                # 숫자 변환
                cols_to_num = ['매매손익(원화환산)', '평가손익(원화환산)']
                for c in cols_to_num:
                    if c in df_h.columns:
                        df_h[c] = pd.to_numeric(df_h[c], errors='coerce').fillna(0)
                
                # Hedge PnL 합계 계산
                if '매매손익(원화환산)' in df_h.columns and '평가손익(원화환산)' in df_h.columns:
                    df_h['Hedge_PnL'] = df_h['매매손익(원화환산)'] + df_h['평가손익(원화환산)']
                else:
                    df_h['Hedge_PnL'] = 0
                
                df_hedge = df_h.groupby('기준일자')['Hedge_PnL'].sum()
                
            else:
                # Holdings 시트 처리 (1, 2, 3... etc)
                df = pd.read_excel(file, sheet_name=sheet, engine='openpyxl')
                
                # 헤더 찾기 ('기준일자', '종목코드' 등이 있는 행)
                h_idx = -1
                for i in range(10):
                    row_vals = df.iloc[i].astype(str).values
                    if '기준일자' in row_vals and '종목명' in row_vals:
                        h_idx = i
                        break
                
                if h_idx != -1:
                    df.columns = df.iloc[h_idx]
                    df = df.iloc[h_idx+1:].copy()
                    # 필요한 컬럼만 있으면 추가
                    if '기준일자' in df.columns and '종목명' in df.columns:
                        all_holdings.append(df)
        
        if not all_holdings:
            return None, None, "Holdings 데이터를 찾을 수 없습니다."
            
        # Holdings 데이터 병합
        df_holdings = pd.concat(all_holdings, ignore_index=True)
        
        # 데이터 전처리
        df_holdings['기준일자'] = pd.to_datetime(df_holdings['기준일자'], errors='coerce')
        df_holdings = df_holdings.dropna(subset=['기준일자'])
        
        # 숫자형 변환 대상 컬럼
        num_cols = ['외화평가손익', '외화총매매손익', '원화총평가손익', '원화총매매손익', '원화평가금액', '잔고수량', '평가환율']
        for c in num_cols:
            if c in df_holdings.columns:
                df_holdings[c] = pd.to_numeric(df_holdings[c], errors='coerce').fillna(0)
                
        # 1. 로컬통화 손익 = 외화평가손익 + 외화총매매손익
        df_holdings['Local_PnL'] = df_holdings['외화평가손익'] + df_holdings['외화총매매손익']
        
        # 2. KRW 손익 = 원화총평가손익 + 원화총매매손익
        df_holdings['KRW_PnL'] = df_holdings['원화총평가손익'] + df_holdings['원화총매매손익']
        
        return df_holdings, df_hedge, None

    except Exception as e:
        return None, None, f"Holdings 파일 오류: {e}"

# ---------------------------------------------------------
# 3. Yahoo Finance Sector Fetcher (캐싱)
# ---------------------------------------------------------
@st.cache_data
def fetch_sectors(tickers):
    sector_map = {}
    # 티커가 너무 많으면 오래 걸리므로, 진행상황 표시
    progress_text = "Fetching sector info from Yahoo Finance..."
    my_bar = st.progress(0, text=progress_text)
    
    unique_tickers = list(set(tickers))
    total = len(unique_tickers)
    
    for i, ticker in enumerate(unique_tickers):
        try:
            # 심볼 매핑 (예: 005930 -> 005930.KS, US 티커는 그대로)
            # 여기서는 엑셀의 '심볼' 컬럼을 사용한다고 가정
            search_ticker = ticker
            if str(ticker).isdigit(): # 홍콩/일본 등 숫자만 있는 경우 처리 로직 필요하지만 일단 패스
                 pass 
            
            info = yf.Ticker(search_ticker).info
            sector = info.get('sector', 'Unknown')
            sector_map[ticker] = sector
        except:
            sector_map[ticker] = 'Unknown'
        
        if i % 5 == 0:
            my_bar.progress((i + 1) / total, text=progress_text)
            
    my_bar.empty()
    return sector_map

# ---------------------------------------------------------
# 4. Cross Asset 다운로드 함수
# ---------------------------------------------------------
@st.cache_data
def download_cross_assets(start_date, end_date):
    assets = {
        'S&P 500': '^GSPC', 'Nasdaq': '^IXIC', 'KOSPI': '^KS11', 
        'USD/KRW': 'KRW=X', 'US 10Y Yield': '^TNX'
    }
    try:
        data = yf.download(list(assets.values()), start=start_date, end=end_date + pd.Timedelta(days=5), progress=False)
        if 'Adj Close' in data.columns: df_close = data['Adj Close']
        elif 'Close' in data.columns: df_close = data['Close']
        else: df_close = data
            
        if isinstance(df_close.columns, pd.MultiIndex):
            df_close.columns = df_close.columns.get_level_values(0)
            
        inv_assets = {v: k for k, v in assets.items()}
        df_close.rename(columns=inv_assets, inplace=True)
        return df_close
    except:
        return pd.DataFrame()

# =========================================================
# 메인 앱 로직
# =========================================================

# 사이드바 메뉴
menu = st.sidebar.radio("Select Dashboard", ["Total Portfolio (Team PNL)", "Cash Equity Analysis"])

if menu == "Total Portfolio (Team PNL)":
    st.subheader("📊 Total Team Portfolio Dashboard")
    uploaded_file = st.sidebar.file_uploader("Upload 'Team_PNL.xlsx'", type=['xlsx'], key="pnl")
    
    if uploaded_file:
        df_pnl, df_pos, pnl_cols = load_team_pnl_data(uploaded_file)
        if df_pnl is not None:
            # (기존 로직 유지)
            common_idx = df_pnl.index.intersection(df_pos.index)
            common_cols = [c for c in pnl_cols if c in df_pos.columns]
            df_pnl = df_pnl.loc[common_idx, common_cols]
            df_pos = df_pos.loc[common_idx, common_cols]
            
            df_cum_pnl = df_pnl.cumsum()
            df_user_ret = df_cum_pnl.div(df_pos.replace(0, np.nan)).fillna(0)
            
            # 간단 차트 표시
            st.line_chart(df_user_ret)
            st.write("자세한 분석은 Cash Equity 탭이나 코드를 확장하여 확인하세요.")
        else:
            st.error(pnl_cols)

elif menu == "Cash Equity Analysis":
    st.subheader("📈 Cash Equity Portfolio Analysis")
    st.markdown("""
    **Holdings3.xlsx** 파일을 업로드하세요. (포함 시트: 일별 Holdings, Hedge)
    * **로컬 손익:** 외화평가손익 + 외화총매매손익
    * **KRW 손익:** 원화총평가손익 + 원화총매매손익
    * **Hedge 반영:** 주식 포트폴리오 KRW 손익 + Hedge 시트 손익
    """)
    
    uploaded_file_ce = st.sidebar.file_uploader("Upload 'Holdings3.xlsx'", type=['xlsx'], key="ce")
    
    if uploaded_file_ce:
        df_holdings, df_hedge_daily, err_msg = load_cash_equity_data(uploaded_file_ce)
        
        if err_msg:
            st.error(err_msg)
        else:
            # -------------------------------------------------
            # 1. 일별 성과 집계 (Aggregation)
            # -------------------------------------------------
            # 일별 주식 PnL 및 평가금액 합계
            daily_stats = df_holdings.groupby('기준일자').agg({
                'KRW_PnL': 'sum',
                '원화평가금액': 'sum' # 이것을 Exposure(분모)로 사용
            })
            
            # Hedge 데이터 병합
            # 기준일자 인덱스 기준 병합 (Outer Join하여 데이터 누락 방지 후 0 처리)
            df_perf = daily_stats.join(df_hedge_daily, how='outer').fillna(0)
            df_perf.rename(columns={'Hedge_PnL': 'Hedge_Impact'}, inplace=True)
            
            # Total PnL (Hedged)
            df_perf['Total_PnL'] = df_perf['KRW_PnL'] + df_perf['Hedge_Impact']
            
            # 수익률 계산 (분모: 전일자 평가금액, 시초가 기준 가정)
            # T일 수익률 = T일 PnL / (T-1일 원화평가금액)
            # 만약 T-1일 데이터가 없으면(첫날), 당일 평가금액 등을 대용하거나 0 처리
            df_perf['Prev_MV'] = df_perf['원화평가금액'].shift(1)
            
            # 수익률 계산 (Equity Only)
            df_perf['Ret_Equity_Only'] = df_perf['KRW_PnL'] / df_perf['Prev_MV']
            
            # 수익률 계산 (Hedged)
            df_perf['Ret_Hedged'] = df_perf['Total_PnL'] / df_perf['Prev_MV']
            
            # 첫날 NaN 처리
            df_perf.fillna(0, inplace=True)
            
            # 누적 수익률
            df_perf['Cum_Equity_Only'] = (1 + df_perf['Ret_Equity_Only']).cumprod() - 1
            df_perf['Cum_Hedged'] = (1 + df_perf['Ret_Hedged']).cumprod() - 1

            # -------------------------------------------------
            # 2. 벤치마크 다운로드
            # -------------------------------------------------
            start_dt, end_dt = df_perf.index.min(), df_perf.index.max()
            df_assets = download_cross_assets(start_dt, end_dt)
            
            bm_cum = pd.DataFrame()
            if not df_assets.empty:
                df_assets.index = pd.to_datetime(df_assets.index).tz_localize(None)
                df_assets = df_assets.reindex(df_perf.index, method='ffill')
                asset_ret = df_assets.pct_change().fillna(0)
                
                if 'S&P 500' in asset_ret.columns:
                    bm_cum['SPX'] = (1 + asset_ret['S&P 500']).cumprod() - 1
                if 'KOSPI' in asset_ret.columns:
                    bm_cum['KOSPI'] = (1 + asset_ret['KOSPI']).cumprod() - 1

            # -------------------------------------------------
            # 3. 대시보드 시각화
            # -------------------------------------------------
            
            # (A) Summary Metrics
            st.markdown("### 📊 Performance Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            latest = df_perf.iloc[-1]
            total_ret = latest['Cum_Hedged']
            equity_ret = latest['Cum_Equity_Only']
            hedge_effect = total_ret - equity_ret
            current_aum = latest['원화평가금액']
            
            col1.metric("Total Return (Hedged)", f"{total_ret:.2%}", delta=f"{df_perf['Ret_Hedged'].iloc[-1]:.2%} (1D)")
            col2.metric("Equity Only Return", f"{equity_ret:.2%}")
            col3.metric("Hedge Impact", f"{hedge_effect:.2%}", delta_color="off")
            col4.metric("Current AUM", f"{current_aum:,.0f} KRW")

            # (B) Chart: Cumulative Return
            st.markdown("### 📈 Cumulative Return Comparison")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_perf.index, y=df_perf['Cum_Hedged'], name="Hedged Portfolio", line=dict(color='#2563eb', width=3)))
            fig.add_trace(go.Scatter(x=df_perf.index, y=df_perf['Cum_Equity_Only'], name="Equity Only", line=dict(color='#93c5fd', width=2, dash='dot')))
            
            if not bm_cum.empty:
                for col in bm_cum.columns:
                    fig.add_trace(go.Scatter(x=bm_cum.index, y=bm_cum[col], name=col, line=dict(color='grey', width=1, dash='dash')))
            
            fig.update_layout(template="plotly_white", height=500, yaxis_tickformat=".2%", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # (C) Sector & Country Analysis
            st.markdown("### 🌍 Allocation & Attribution")
            
            # 최신 일자 데이터만 추출하여 비중 분석
            latest_date = df_holdings['기준일자'].max()
            df_latest = df_holdings[df_holdings['기준일자'] == latest_date].copy()
            
            # 섹터 정보가 파일에 없다면, '심볼'을 이용하여 야후 파이낸스에서 가져오기 시도
            # 단, 파일에 '섹터' 컬럼이 있으면 그것을 우선 사용
            if '섹터' not in df_latest.columns:
                if '심볼' in df_latest.columns:
                    with st.spinner("Fetching Sector info from Yahoo Finance..."):
                        # 심볼 리스트 추출
                        symbols = df_latest['심볼'].dropna().unique().tolist()
                        sector_map = fetch_sectors(symbols)
                        df_latest['섹터'] = df_latest['심볼'].map(sector_map)
                else:
                    df_latest['섹터'] = 'Unknown'
            
            # 탭 구성
            tab1, tab2, tab3 = st.tabs(["Sector Breakdown", "Country Breakdown", "Top Movers"])
            
            with tab1:
                # 섹터별 비중 (파이차트)
                sector_alloc = df_latest.groupby('섹터')['원화평가금액'].sum().reset_index()
                fig_sec = px.pie(sector_alloc, values='원화평가금액', names='섹터', title=f"Sector Allocation ({latest_date.date()})", hole=0.4)
                st.plotly_chart(fig_sec, use_container_width=True)
                
                # 섹터별 PnL 기여도 (Bar Chart)
                # 기간 전체 PnL 합계 (종목별 -> 섹터별)
                # 주의: 종목이 기간 중 매매되었을 수 있으므로, 일별 데이터에서 집계해야 정확함
                # 여기서는 '전체 기간'에 대한 근사치로, df_holdings 전체에서 집계
                
                # 전체 기간 데이터에 섹터 매핑
                if '심볼' in df_holdings.columns:
                    # 위에서 만든 맵 활용 (캐싱됨)
                    all_symbols = df_holdings['심볼'].dropna().unique().tolist()
                    sec_map_all = fetch_sectors(all_symbols)
                    df_holdings['섹터'] = df_holdings['심볼'].map(sec_map_all)
                
                sector_pnl = df_holdings.groupby('섹터')['KRW_PnL'].sum().reset_index().sort_values('KRW_PnL', ascending=False)
                fig_sec_pnl = px.bar(sector_pnl, x='섹터', y='KRW_PnL', title="Total PnL Contribution by Sector", color='KRW_PnL', color_continuous_scale='RdBu')
                st.plotly_chart(fig_sec_pnl, use_container_width=True)

            with tab2:
                # 국가별 비중 (있다면)
                if '국가' in df_latest.columns or '통화' in df_latest.columns:
                    group_col = '국가' if '국가' in df_latest.columns else '통화'
                    country_alloc = df_latest.groupby(group_col)['원화평가금액'].sum().reset_index()
                    fig_ctry = px.treemap(country_alloc, path=[group_col], values='원화평가금액', title=f"Allocation by {group_col}")
                    st.plotly_chart(fig_ctry, use_container_width=True)
                else:
                    st.info("국가/통화 정보가 없습니다.")

            with tab3:
                st.markdown("#### Top 5 Winners & Losers (Total KRW PnL)")
                # 종목별 전체 기간 손익 합계
                stock_pnl = df_holdings.groupby(['종목명', '심볼'])['KRW_PnL'].sum().reset_index().sort_values('KRW_PnL', ascending=False)
                
                col_win, col_lose = st.columns(2)
                with col_win:
                    st.success("Top 5 Winners 🏆")
                    st.table(stock_pnl.head(5).style.format({"KRW_PnL": "{:,.0f}"}))
                
                with col_lose:
                    st.error("Top 5 Losers 📉")
                    st.table(stock_pnl.tail(5).sort_values('KRW_PnL').style.format({"KRW_PnL": "{:,.0f}"}))

    else:
        st.info("👈 왼쪽 사이드바에서 'Holdings3.xlsx' 파일을 업로드해주세요.")