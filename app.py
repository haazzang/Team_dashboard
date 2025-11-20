import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf

# --- 페이지 설정 ---
st.set_page_config(page_title="Portfolio Dashboard", layout="wide")
st.title("🚀 Team Portfolio Analysis Dashboard")

# ---------------------------------------------------------
# Helper: Yahoo Finance Sector Fetcher (Cached)
# ---------------------------------------------------------
@st.cache_data
def fetch_sectors_cached(tickers):
    sector_map = {}
    for t in tickers:
        try:
            info = yf.Ticker(str(t)).info
            sector_map[t] = info.get('sector', 'Unknown')
        except:
            sector_map[t] = 'Unknown'
    return sector_map

# ---------------------------------------------------------
# Helper: Benchmark Downloader
# ---------------------------------------------------------
@st.cache_data
def download_benchmark(start_date, end_date):
    try:
        bm = yf.download(['^GSPC', '^KS11'], start=start_date, end=end_date + pd.Timedelta(days=5), progress=False)['Adj Close']
        bm = bm.ffill()
        return bm
    except:
        return pd.DataFrame()

# ---------------------------------------------------------
# 1. 데이터 로드 및 전처리 (Fix: Sector Mapping Order)
# ---------------------------------------------------------
@st.cache_data
def load_cash_equity_data(file):
    debug_logs = []
    try:
        xls = pd.ExcelFile(file, engine='openpyxl')
        all_holdings = []
        df_hedge = pd.DataFrame()
        
        # 1. 시트 로드 및 병합
        for sheet in xls.sheet_names:
            # (A) Hedge 시트
            if 'hedge' in sheet.lower() or '헷지' in sheet:
                try:
                    df_h = pd.read_excel(file, sheet_name=sheet, header=None, engine='openpyxl')
                    h_idx = -1
                    for i in range(15):
                        if '기준일자' in [str(x).strip() for x in df_h.iloc[i].values]:
                            h_idx = i; break
                    
                    if h_idx != -1:
                        df_h.columns = [str(c).strip() for c in df_h.iloc[h_idx]]
                        df_h = df_h.iloc[h_idx+1:].copy()
                        df_h['기준일자'] = pd.to_datetime(df_h['기준일자'], errors='coerce')
                        df_h = df_h.dropna(subset=['기준일자']).sort_values('기준일자').set_index('기준일자')
                        
                        col_cum_pnl = next((c for c in df_h.columns if '누적' in c and '총손익' in c), None)
                        if col_cum_pnl:
                            df_h[col_cum_pnl] = pd.to_numeric(df_h[col_cum_pnl], errors='coerce').fillna(0)
                            daily_hedge = df_h[col_cum_pnl].diff().fillna(0)
                            
                            if df_hedge.empty: df_hedge = daily_hedge.to_frame(name='Hedge_PnL_KRW')
                            else: df_hedge = df_hedge.add(daily_hedge.to_frame(name='Hedge_PnL_KRW'), fill_value=0)
                            debug_logs.append(f"✅ Hedge: {sheet}")
                except Exception as e: debug_logs.append(f"❌ Hedge Error: {e}")

            # (B) Equity 시트
            else:
                try:
                    df = pd.read_excel(file, sheet_name=sheet, header=None, engine='openpyxl')
                    h_idx = -1
                    for i in range(15):
                        row_vals = [str(x).strip() for x in df.iloc[i].values]
                        if '기준일자' in row_vals and ('종목명' in row_vals or '종목코드' in row_vals):
                            h_idx = i; break
                    
                    if h_idx != -1:
                        df.columns = [str(c).strip() for c in df.iloc[h_idx]]
                        df = df.iloc[h_idx+1:].copy()
                        if '기준일자' in df.columns: all_holdings.append(df)
                except Exception as e: debug_logs.append(f"❌ Sheet Error: {e}")

        if not all_holdings: return None, None, None, f"No Holdings Found. Logs: {debug_logs}"

        # 2. 전체 데이터 병합
        eq = pd.concat(all_holdings, ignore_index=True)
        eq['기준일자'] = pd.to_datetime(eq['기준일자'], errors='coerce')
        eq = eq.dropna(subset=['기준일자'])
        
        # 숫자 변환
        cols_num = ['원화평가금액', '원화총평가손익', '원화총매매손익', '잔고수량', 'Market Price']
        for c in cols_num:
            if c in eq.columns: eq[c] = pd.to_numeric(eq[c], errors='coerce').fillna(0)
        
        # -------------------------------------------------
        # [Fix] 섹터 매핑을 가장 먼저 수행 (KeyError 방지)
        # -------------------------------------------------
        id_col = '심볼' if '심볼' in eq.columns else '종목코드'
        
        if '섹터' not in eq.columns:
            if '심볼' in eq.columns:
                # 팁: 전체 기간의 유니크 심볼에 대해 한 번만 호출
                uniques = eq['심볼'].dropna().unique()
                sec_map = fetch_sectors_cached(tuple(uniques))
                eq['섹터'] = eq['심볼'].map(sec_map).fillna('Unknown')
            else:
                eq['섹터'] = 'Unknown'
                
        # 3. 데이터 정렬 및 전일 데이터 계산
        eq = eq.sort_values([id_col, '기준일자'])
        eq['Prev_Qty'] = eq.groupby(id_col)['잔고수량'].shift(1).fillna(0)
        eq['Prev_MV'] = eq.groupby(id_col)['원화평가금액'].shift(1).fillna(0)
        
        # (A) KRW PnL (Excel Logic)
        # 누적 개념으로 Daily 유도 (매매 반영 위함)
        eq['Total_Cum_PnL'] = eq['원화총평가손익'] + eq['원화총매매손익']
        
        # (B) Local Return (Pure Price Return)
        # Local Price가 'Market Price' 컬럼에 있다고 가정
        if 'Market Price' in eq.columns:
            eq['Prev_Price_Local'] = eq.groupby(id_col)['Market Price'].shift(1)
            # Local Return = (Price_t - Price_t-1) / Price_t-1
            eq['Ret_Local_Stock'] = np.where(eq['Prev_Price_Local'] > 0, 
                                            (eq['Market Price'] - eq['Prev_Price_Local']) / eq['Prev_Price_Local'], 
                                            0)
        else:
            eq['Ret_Local_Stock'] = 0

        # 4. 일별 집계 (Aggregation)
        # (1) KRW Aggregation
        daily_agg = eq.groupby('기준일자').agg({
            'Total_Cum_PnL': 'sum',
            '원화평가금액': 'sum',
            'Prev_MV': 'sum'
        })
        daily_agg['Daily_PnL_KRW'] = daily_agg['Total_Cum_PnL'].diff().fillna(0)
        
        # (2) Local Return Aggregation (Weighted Average)
        # Weight_i = Prev_MV_KRW_i / Total_Prev_MV_KRW
        # 먼저 일별 총 Prev_MV를 각 행에 붙여야 함
        daily_total_prev = eq.groupby('기준일자')['Prev_MV'].sum().rename('Day_Total_Prev_MV')
        eq = eq.join(daily_total_prev, on='기준일자')
        
        eq['Weight'] = np.where(eq['Day_Total_Prev_MV'] > 0, eq['Prev_MV'] / eq['Day_Total_Prev_MV'], 0)
        eq['W_Ret_Local'] = eq['Ret_Local_Stock'] * eq['Weight']
        
        daily_local_ret = eq.groupby('기준일자')['W_Ret_Local'].sum().rename('Ret_Equity_Local')
        
        # 5. 최종 병합
        df_perf = daily_agg.join(df_hedge, how='outer').fillna(0)
        df_perf = df_perf.join(daily_local_ret, how='left').fillna(0)
        
        # Total PnL (KRW)
        df_perf['Total_PnL_KRW'] = df_perf['Daily_PnL_KRW'] + df_perf['Hedge_PnL_KRW']
        
        # 수익률 계산 (KRW)
        df_perf['Ret_Equity_KRW'] = np.where(df_perf['Prev_MV'] > 0, df_perf['Daily_PnL_KRW'] / df_perf['Prev_MV'], 0)
        df_perf['Ret_Total_KRW'] = np.where(df_perf['Prev_MV'] > 0, df_perf['Total_PnL_KRW'] / df_perf['Prev_MV'], 0)
        
        # 첫날 제외 (수익률 왜곡 방지)
        df_perf = df_perf.iloc[1:]
        
        # 누적 수익률
        df_perf['Cum_Equity_KRW'] = (1 + df_perf['Ret_Equity_KRW']).cumprod() - 1
        df_perf['Cum_Total_KRW'] = (1 + df_perf['Ret_Total_KRW']).cumprod() - 1
        df_perf['Cum_Equity_Local'] = (1 + df_perf['Ret_Equity_Local']).cumprod() - 1

        # 6. 종목별 최종 상태 (Top/Bottom 분석용)
        # 섹터가 이미 매핑되어 있으므로 안전함
        df_last_seen = eq.sort_values('기준일자').groupby(id_col).tail(1)
        df_last_seen['Final_PnL'] = df_last_seen['원화총평가손익'] + df_last_seen['원화총매매손익']

        return df_perf, df_last_seen, debug_logs

    except Exception as e:
        return None, None, None, f"Error: {e}"

# =========================================================
# 메인 앱 UI
# =========================================================

menu = st.sidebar.radio("Menu", ["Cash Equity Analysis"])

if menu == "Cash Equity Analysis":
    st.subheader("📈 Cash Equity Portfolio Analysis")
    
    uploaded_file = st.sidebar.file_uploader("Upload 'Holdings3.xlsx'", type=['xlsx'])
    
    if uploaded_file:
        with st.spinner("Analyzing..."):
            df_perf, df_last, logs = load_cash_equity_data(uploaded_file)[0:3]
            
        with st.expander("Debug Logs"):
            st.write(logs)
            
        if df_perf is not None:
            # --- View Option (KRW vs Local) ---
            view_opt = st.radio("Currency View", ["KRW (Unhedged / Hedged)", "Local Currency (Price Return Only)"], horizontal=True)
            
            last_day = df_perf.iloc[-1]
            curr_aum = df_perf.iloc[-1]['원화평가금액']
            
            # (A) Summary
            st.markdown("### 📊 Performance Summary")
            c1, c2, c3, c4 = st.columns(4)
            
            if view_opt.startswith("KRW"):
                c1.metric("Total Return (Hedged)", f"{last_day['Cum_Total_KRW']:.2%}")
                c2.metric("Equity Return (KRW)", f"{last_day['Cum_Equity_KRW']:.2%}")
                c3.metric("Hedge Impact", f"{(last_day['Cum_Total_KRW'] - last_day['Cum_Equity_KRW']):.2%}")
                c4.metric("Current AUM", f"{curr_aum:,.0f} KRW")
                
                y_col_main = 'Cum_Total_KRW'
                y_col_sub = 'Cum_Equity_KRW'
                name_main = 'Total (Hedged)'
                name_sub = 'Equity (KRW)'
            else:
                c1.metric("Local Return (Price Only)", f"{last_day['Cum_Equity_Local']:.2%}")
                c2.metric("Equity Return (KRW)", f"{last_day['Cum_Equity_KRW']:.2%}")
                c3.metric("FX Effect (Approx)", f"{(last_day['Cum_Equity_KRW'] - last_day['Cum_Equity_Local']):.2%}")
                c4.metric("Current AUM", f"{curr_aum:,.0f} KRW")
                
                y_col_main = 'Cum_Equity_Local'
                y_col_sub = None
                name_main = 'Equity (Local)'
                name_sub = None

            # (B) Chart
            st.markdown("### 📈 Cumulative Return")
            fig = go.Figure()
            
            # Main Line
            fig.add_trace(go.Scatter(x=df_perf.index, y=df_perf[y_col_main], name=name_main, line=dict(color='#2563eb', width=3)))
            
            # Sub Line (Only for KRW view or comparison)
            if y_col_sub:
                fig.add_trace(go.Scatter(x=df_perf.index, y=df_perf[y_col_sub], name=name_sub, line=dict(color='#60a5fa', dash='dot')))
            
            # Benchmark
            bm_df = download_benchmark(df_perf.index.min(), df_perf.index.max())
            if not bm_df.empty:
                bm_df = bm_df.reindex(df_perf.index, method='ffill').pct_change().fillna(0)
                bm_cum = (1 + bm_df).cumprod() - 1
                if '^GSPC' in bm_cum.columns:
                    fig.add_trace(go.Scatter(x=bm_cum.index, y=bm_cum['^GSPC'], name='S&P 500', line=dict(color='grey', width=1, dash='dash')))
                if '^KS11' in bm_cum.columns:
                    fig.add_trace(go.Scatter(x=bm_cum.index, y=bm_cum['^KS11'], name='KOSPI', line=dict(color='silver', width=1, dash='dash')))

            fig.update_layout(template="plotly_white", height=500, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # (C) Details
            st.markdown("### 🌍 Portfolio Details")
            t1, t2 = st.tabs(["Sector Allocation", "Top Movers (KRW)"])
            
            # Current Holdings
            df_curr = df_last[df_last['잔고수량'] > 0]
            
            with t1:
                if not df_curr.empty:
                    # Sector Pie
                    fig_pie = px.pie(df_curr, values='원화평가금액', names='섹터', title="Current Sector Exposure")
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.write("현재 보유 종목이 없습니다.")
                    
            with t2:
                # PnL Table (All time)
                cols = ['종목명', '섹터', 'Final_PnL']
                df_pnl = df_last[cols].sort_values('Final_PnL', ascending=False)
                
                cw, cl = st.columns(2)
                cw.success("Top 5 Winners")
                cw.dataframe(df_pnl.head(5).style.format({'Final_PnL': '{:,.0f}'}))
                
                cl.error("Top 5 Losers")
                cl.dataframe(df_pnl.tail(5).sort_values('Final_PnL').style.format({'Final_PnL': '{:,.0f}'}))

            with st.expander("Daily Data View"):
                st.dataframe(df_perf)

    else:
        st.info("Upload File.")