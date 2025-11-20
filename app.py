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
        df_pnl_raw = pd.read_excel(file, sheet_name='PNL', header=None, engine='openpyxl')
        header_idx = -1
        for i in range(15): # 탐색 범위 확대
            row_str = df_pnl_raw.iloc[i].astype(str).values
            if any('일자' in s for s in row_str):
                header_idx = i
                break
        if header_idx == -1: return None, None, "PNL 시트에서 '일자'를 찾을 수 없습니다."
        
        pnl_cols = [str(x).strip() for x in df_pnl_raw.iloc[header_idx]]
        df_pnl_raw.columns = pnl_cols
        df_pnl = df_pnl_raw.iloc[header_idx+1:].copy()
        
        df_pos_raw = pd.read_excel(file, sheet_name='Position', header=None, engine='openpyxl')
        header_idx_pos = -1
        for i in range(15):
            row_str = df_pos_raw.iloc[i].astype(str).values
            if any('일자' in s for s in row_str):
                header_idx_pos = i
                break
        if header_idx_pos == -1: return None, None, "Position 시트에서 '일자'를 찾을 수 없습니다."
        
        pos_cols = [str(x).strip() for x in df_pos_raw.iloc[header_idx_pos]]
        df_pos_raw.columns = pos_cols
        df_pos = df_pos_raw.iloc[header_idx_pos+1:].copy()

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
# 2. Cash Equity 데이터 로드 함수 (Holdings3.xlsx 용) - 개선됨
# ---------------------------------------------------------
@st.cache_data
def load_cash_equity_data(file):
    debug_logs = []
    try:
        xls = pd.ExcelFile(file, engine='openpyxl')
        sheet_names = xls.sheet_names
        debug_logs.append(f"발견된 시트: {sheet_names}")
        
        all_holdings = []
        df_hedge = pd.DataFrame()
        
        for sheet in sheet_names:
            # Hedge 시트
            if 'hedge' in sheet.lower() or '헷지' in sheet:
                try:
                    df_h = pd.read_excel(file, sheet_name=sheet, header=None, engine='openpyxl')
                    h_idx = -1
                    for i in range(10):
                        # 공백 제거 후 '기준일자' 찾기
                        row_vals = [str(x).strip() for x in df_h.iloc[i].values]
                        if '기준일자' in row_vals:
                            h_idx = i
                            break
                    
                    if h_idx != -1:
                        df_h.columns = [str(x).strip() for x in df_h.iloc[h_idx]]
                        df_h = df_h.iloc[h_idx+1:].copy()
                        df_h['기준일자'] = pd.to_datetime(df_h['기준일자'], errors='coerce')
                        df_h = df_h.dropna(subset=['기준일자'])
                        
                        # 숫자 변환
                        for c in ['매매손익(원화환산)', '평가손익(원화환산)']:
                            if c in df_h.columns:
                                df_h[c] = pd.to_numeric(df_h[c], errors='coerce').fillna(0)
                        
                        # Hedge PnL 계산
                        if '매매손익(원화환산)' in df_h.columns and '평가손익(원화환산)' in df_h.columns:
                            df_h['Hedge_PnL'] = df_h['매매손익(원화환산)'] + df_h['평가손익(원화환산)']
                        else:
                            df_h['Hedge_PnL'] = 0
                            
                        # 일별 합계
                        df_hedge_part = df_h.groupby('기준일자')['Hedge_PnL'].sum()
                        df_hedge = df_hedge.add(df_hedge_part, fill_value=0) # 여러 Hedge 시트가 있을 경우 합산
                        debug_logs.append(f"✅ Hedge 시트 로드 성공: {sheet}")
                    else:
                        debug_logs.append(f"⚠️ {sheet} 시트에서 '기준일자' 헤더를 못 찾음")
                except Exception as e:
                    debug_logs.append(f"❌ {sheet} 처리 중 오류: {e}")

            # Holdings 시트 (Hedge가 아닌 나머지)
            else:
                try:
                    df = pd.read_excel(file, sheet_name=sheet, header=None, engine='openpyxl')
                    h_idx = -1
                    for i in range(15):
                        row_vals = [str(x).strip() for x in df.iloc[i].values]
                        if '기준일자' in row_vals and ('종목명' in row_vals or '종목코드' in row_vals):
                            h_idx = i
                            break
                    
                    if h_idx != -1:
                        df.columns = [str(x).strip() for x in df.iloc[h_idx]]
                        df = df.iloc[h_idx+1:].copy()
                        # 필수 컬럼 확인
                        if '기준일자' in df.columns:
                            all_holdings.append(df)
                            debug_logs.append(f"✅ Holdings 로드: {sheet} ({len(df)}행)")
                        else:
                            debug_logs.append(f"⚠️ {sheet}: '기준일자' 컬럼 유실")
                    else:
                        # Holdings 시트가 아닐 수 있음 (로그 생략 가능)
                        pass
                except Exception as e:
                    debug_logs.append(f"❌ {sheet} 로드 실패: {e}")
        
        if not all_holdings:
            return None, None, f"Holdings 데이터 없음. 로그: {debug_logs}"
            
        # 병합
        df_holdings = pd.concat(all_holdings, ignore_index=True)
        df_holdings['기준일자'] = pd.to_datetime(df_holdings['기준일자'], errors='coerce')
        df_holdings = df_holdings.dropna(subset=['기준일자'])
        
        # 숫자 변환
        target_cols = ['외화평가손익', '외화총매매손익', '원화총평가손익', '원화총매매손익', '원화평가금액']
        for c in target_cols:
            if c in df_holdings.columns:
                df_holdings[c] = pd.to_numeric(df_holdings[c], errors='coerce').fillna(0)
        
        # PnL 계산
        if '외화평가손익' in df_holdings.columns and '외화총매매손익' in df_holdings.columns:
            df_holdings['Local_PnL'] = df_holdings['외화평가손익'] + df_holdings['외화총매매손익']
        
        if '원화총평가손익' in df_holdings.columns and '원화총매매손익' in df_holdings.columns:
            df_holdings['KRW_PnL'] = df_holdings['원화총평가손익'] + df_holdings['원화총매매손익']
        
        return df_holdings, df_hedge, debug_logs

    except Exception as e:
        return None, None, f"치명적 오류: {e}"

# ---------------------------------------------------------
# 3. Yahoo Finance Sector Fetcher
# ---------------------------------------------------------
@st.cache_data
def fetch_sectors(tickers):
    sector_map = {}
    unique_tickers = list(set(tickers))
    # 10개 단위로 진행바 없이 조용히 처리 (속도 위해)
    for ticker in unique_tickers:
        try:
            # 단순 티커 매핑
            t = str(ticker).strip()
            info = yf.Ticker(t).info
            sector_map[t] = info.get('sector', 'Unknown')
        except:
            sector_map[ticker] = 'Unknown'
    return sector_map

# ---------------------------------------------------------
# 4. Cross Asset
# ---------------------------------------------------------
@st.cache_data
def download_cross_assets(start_date, end_date):
    assets = {'S&P 500': '^GSPC', 'KOSPI': '^KS11', 'USD/KRW': 'KRW=X'}
    try:
        data = yf.download(list(assets.values()), start=start_date, end=end_date + pd.Timedelta(days=5), progress=False)
        if 'Adj Close' in data.columns: df = data['Adj Close']
        elif 'Close' in data.columns: df = data['Close']
        else: df = data
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.rename(columns={v: k for k, v in assets.items()}, inplace=True)
        return df
    except:
        return pd.DataFrame()

# =========================================================
# 메인 앱 로직
# =========================================================

# 사이드바
st.sidebar.title("Dashboard Menu")
menu = st.sidebar.radio("Go to", ["Total Portfolio (Team PNL)", "Cash Equity Analysis"])

if menu == "Total Portfolio (Team PNL)":
    st.subheader("📊 Total Team Portfolio Dashboard")
    uploaded_file = st.sidebar.file_uploader("Upload 'Team_PNL.xlsx'", type=['xlsx'], key="pnl")
    
    if uploaded_file:
        df_pnl, df_pos, pnl_cols = load_team_pnl_data(uploaded_file)
        if df_pnl is not None:
            # 간단한 차트 표시 (기존 로직 생략)
            common_idx = df_pnl.index.intersection(df_pos.index)
            common_cols = [c for c in pnl_cols if c in df_pos.columns]
            df_pnl = df_pnl.loc[common_idx, common_cols]
            df_pos = df_pos.loc[common_idx, common_cols]
            df_cum = df_pnl.cumsum()
            df_ret = df_cum.div(df_pos.replace(0, np.nan)).fillna(0)
            
            st.line_chart(df_ret)
        else:
            st.error(pnl_cols)

elif menu == "Cash Equity Analysis":
    st.subheader("📈 Cash Equity Portfolio Analysis")
    st.info("왼쪽 사이드바에서 'Holdings3.xlsx' 파일을 업로드해주세요.")
    
    uploaded_file_ce = st.sidebar.file_uploader("Upload 'Holdings3.xlsx'", type=['xlsx'], key="ce")
    
    if uploaded_file_ce:
        df_holdings, df_hedge_daily, logs = load_cash_equity_data(uploaded_file_ce)
        
        # 디버그 로그 표시 (접기 가능)
        with st.expander("Data Loading Logs (Debug)", expanded=False):
            st.write(logs)
        
        if df_holdings is not None and not df_holdings.empty:
            # -------------------------------------------------
            # 1. 성과 집계
            # -------------------------------------------------
            # 일별 주식 PnL 및 Exposure
            daily_equity = df_holdings.groupby('기준일자')[['KRW_PnL', '원화평가금액']].sum()
            
            # Hedge 병합
            if isinstance(df_hedge_daily, pd.Series):
                df_hedge_daily = df_hedge_daily.to_frame(name='Hedge_PnL')
            elif isinstance(df_hedge_daily, pd.DataFrame) and 'Hedge_PnL' not in df_hedge_daily.columns:
                df_hedge_daily['Hedge_PnL'] = 0
                
            df_perf = daily_equity.join(df_hedge_daily, how='outer').fillna(0)
            df_perf['Total_PnL'] = df_perf['KRW_PnL'] + df_perf['Hedge_PnL']
            
            # 수익률 계산 (Time-Weighted Proxy)
            # Denominator: Previous Day's MV (Approx for capital base)
            df_perf['Prev_MV'] = df_perf['원화평가금액'].shift(1)
            # 첫날은 수익률 0 처리 (혹은 당일 MV 사용 가능하나 보수적으로)
            df_perf = df_perf.iloc[1:].copy() 
            
            # 0으로 나누기 방지
            df_perf['Ret_Equity'] = np.where(df_perf['Prev_MV'] > 0, df_perf['KRW_PnL'] / df_perf['Prev_MV'], 0)
            df_perf['Ret_Hedged'] = np.where(df_perf['Prev_MV'] > 0, df_perf['Total_PnL'] / df_perf['Prev_MV'], 0)
            
            # 누적
            df_perf['Cum_Equity'] = (1 + df_perf['Ret_Equity']).cumprod() - 1
            df_perf['Cum_Hedged'] = (1 + df_perf['Ret_Hedged']).cumprod() - 1

            # -------------------------------------------------
            # 2. 벤치마크
            # -------------------------------------------------
            if not df_perf.empty:
                s, e = df_perf.index.min(), df_perf.index.max()
                df_bm = download_cross_assets(s, e)
                bm_cum = pd.DataFrame()
                if not df_bm.empty:
                    df_bm = df_bm.reindex(df_perf.index, method='ffill')
                    bm_ret = df_bm.pct_change().fillna(0)
                    if 'KOSPI' in bm_ret.columns:
                        bm_cum['KOSPI'] = (1 + bm_ret['KOSPI']).cumprod() - 1

                # -------------------------------------------------
                # 3. 대시보드 UI
                # -------------------------------------------------
                
                # (A) Summary
                st.markdown("### Performance Summary")
                c1, c2, c3, c4 = st.columns(4)
                
                last_row = df_perf.iloc[-1]
                c1.metric("Total Return (Hedged)", f"{last_row['Cum_Hedged']:.2%}")
                c2.metric("Equity Only Return", f"{last_row['Cum_Equity']:.2%}")
                c3.metric("Hedge Impact", f"{(last_row['Cum_Hedged'] - last_row['Cum_Equity']):.2%}")
                c4.metric("Current AUM", f"{last_row['원화평가금액']:,.0f} ₩")
                
                # (B) Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_perf.index, y=df_perf['Cum_Hedged'], name='Hedged Portfolio', line=dict(color='blue', width=2)))
                fig.add_trace(go.Scatter(x=df_perf.index, y=df_perf['Cum_Equity'], name='Equity Only', line=dict(color='lightblue', dash='dot')))
                if 'KOSPI' in bm_cum.columns:
                    fig.add_trace(go.Scatter(x=bm_cum.index, y=bm_cum['KOSPI'], name='KOSPI', line=dict(color='grey', dash='dash')))
                
                fig.update_layout(title="Cumulative Return", template="plotly_white", height=500, yaxis_tickformat=".2%")
                st.plotly_chart(fig, use_container_width=True)
                
                # (C) Sector Analysis
                st.markdown("### Portfolio Breakdown")
                latest_dt = df_holdings['기준일자'].max()
                df_cur = df_holdings[df_holdings['기준일자'] == latest_dt].copy()
                
                # 섹터 정보 없으면 가져오기
                if '섹터' not in df_cur.columns:
                    if '심볼' in df_cur.columns:
                        with st.spinner("Fetching Sectors..."):
                            sec_map = fetch_sectors(df_cur['심볼'].dropna().unique())
                            df_cur['섹터'] = df_cur['심볼'].map(sec_map).fillna('Unknown')
                    else:
                        df_cur['섹터'] = 'Unknown'
                
                c_left, c_right = st.columns(2)
                
                with c_left:
                    sec_grp = df_cur.groupby('섹터')['원화평가금액'].sum().reset_index()
                    fig_pie = px.pie(sec_grp, values='원화평가금액', names='섹터', title="Sector Exposure")
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                with c_right:
                    # Top Gainers
                    st.write("Top 5 Holdings (Size)")
                    top_hold = df_cur.sort_values('원화평가금액', ascending=False).head(5)[['종목명', '섹터', '원화평가금액']]
                    st.dataframe(top_hold.style.format({'원화평가금액': '{:,.0f}'}), use_container_width=True)

        else:
            st.warning("데이터 로드 후 결과가 비어있습니다. 로그를 확인하세요.")