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
# 1. 데이터 로드 및 전처리 (Code-Based Logic + Excel Data)
# ---------------------------------------------------------
@st.cache_data
def load_cash_equity_data(file):
    debug_logs = []
    try:
        xls = pd.ExcelFile(file, engine='openpyxl')
        
        all_holdings = []
        df_hedge = pd.DataFrame()
        
        # 1. 모든 시트 순회
        for sheet in xls.sheet_names:
            # (A) Hedge 시트 처리
            if 'hedge' in sheet.lower() or '헷지' in sheet:
                try:
                    df_h = pd.read_excel(file, sheet_name=sheet, header=None, engine='openpyxl')
                    # 헤더 찾기 ('기준일자', '누적 총손익' 등)
                    h_idx = -1
                    for i in range(15):
                        row_vals = [str(x).strip() for x in df_h.iloc[i].values]
                        if '기준일자' in row_vals:
                            h_idx = i
                            break
                    
                    if h_idx != -1:
                        df_h.columns = [str(c).strip() for c in df_h.iloc[h_idx]]
                        df_h = df_h.iloc[h_idx+1:].copy()
                        
                        # 날짜 변환
                        df_h['기준일자'] = pd.to_datetime(df_h['기준일자'], errors='coerce')
                        df_h = df_h.dropna(subset=['기준일자']).sort_values('기준일자')
                        
                        # 누적 총손익 컬럼 찾기 (보통 '누적 총손익' 또는 '누적총손익')
                        col_cum_pnl = next((c for c in df_h.columns if '누적' in c and '총손익' in c), None)
                        
                        if col_cum_pnl:
                            df_h[col_cum_pnl] = pd.to_numeric(df_h[col_cum_pnl], errors='coerce').fillna(0)
                            
                            # 코드 로직: Daily Hedge PnL = Cumulative PnL Diff
                            df_h = df_h.set_index('기준일자')
                            # 일별 변동분 계산 (첫날은 0 혹은 누적값 그대로? 보통 Diff 사용)
                            daily_hedge = df_h[col_cum_pnl].diff().fillna(0)
                            
                            # 기존 데이터가 있으면 합산 (시트가 여러 개일 경우 대비)
                            df_hedge = df_hedge.add(daily_hedge, fill_value=0)
                            debug_logs.append(f"✅ Hedge 시트 로드: {sheet}")
                        else:
                            debug_logs.append(f"⚠️ {sheet}: '누적 총손익' 컬럼 없음")
                except Exception as e:
                    debug_logs.append(f"❌ Hedge 시트 오류 ({sheet}): {e}")

            # (B) Equity Holdings 시트 처리 (나머지 시트)
            else:
                try:
                    df = pd.read_excel(file, sheet_name=sheet, header=None, engine='openpyxl')
                    # 헤더 찾기
                    h_idx = -1
                    for i in range(15):
                        row_vals = [str(x).strip() for x in df.iloc[i].values]
                        if '기준일자' in row_vals and ('종목명' in row_vals or '종목코드' in row_vals):
                            h_idx = i
                            break
                    
                    if h_idx != -1:
                        df.columns = [str(c).strip() for c in df.iloc[h_idx]]
                        df = df.iloc[h_idx+1:].copy()
                        
                        # 필수 데이터 확인
                        if '기준일자' in df.columns:
                            all_holdings.append(df)
                        else:
                            pass # 기준일자 없으면 데이터 아님
                except Exception as e:
                    debug_logs.append(f"❌ Sheet 오류 ({sheet}): {e}")

        if not all_holdings:
            return None, None, f"주식 보유 내역을 찾을 수 없습니다. 로그: {debug_logs}"

        # 2. 주식 데이터 병합 및 전처리
        eq = pd.concat(all_holdings, ignore_index=True)
        eq['기준일자'] = pd.to_datetime(eq['기준일자'], errors='coerce')
        eq = eq.dropna(subset=['기준일자'])
        
        # 숫자 변환 (원화 컬럼 활용)
        cols_to_num = ['잔고수량', '원화평가금액', '원화총평가손익', '원화총매매손익']
        for c in cols_to_num:
            if c in eq.columns:
                eq[c] = pd.to_numeric(eq[c], errors='coerce').fillna(0)

        # 3. 수익률 계산 (핵심: 코드 로직 구현)
        # Logic: Daily PnL = Prev_Qty * (Price_t - Price_t-1)
        # 여기서 Price는 "원화 환산 주가"를 의미함 (환율 효과 포함)
        # 원화 주가 = 원화평가금액 / 잔고수량
        
        eq['KRW_Unit_Price'] = np.where(eq['잔고수량'] != 0, eq['원화평가금액'] / eq['잔고수량'], 0)
        
        # 정렬 (종목별, 날짜별) -> 종목 식별자는 '종목코드' 또는 '심볼' 사용
        # 심볼이 없으면 종목코드 사용
        id_col = '심볼' if '심볼' in eq.columns else '종목코드'
        eq = eq.sort_values([id_col, '기준일자'])
        
        # 전일 데이터 가져오기 (Shift)
        eq['Prev_Qty'] = eq.groupby(id_col)['잔고수량'].shift(1).fillna(0)
        eq['Prev_Price'] = eq.groupby(id_col)['KRW_Unit_Price'].shift(1).fillna(0)
        eq['Prev_MV'] = eq.groupby(id_col)['원화평가금액'].shift(1).fillna(0)
        
        # Daily PnL 계산 (코드 로직: 보유분에 대한 평가 차손익)
        # 매매가 일어난 당일의 매매손익은 이 로직(Price Diff)으로는 정확히 잡히지 않으나,
        # 코드의 로직(Time-Weighted Return 근사)을 따름
        eq['Daily_PnL_KRW'] = eq['Prev_Qty'] * (eq['KRW_Unit_Price'] - eq['Prev_Price'])
        
        # 4. 섹터 정보 매핑 (파일에 없으면 Yfinance)
        if '섹터' not in eq.columns:
            if '심볼' in eq.columns:
                unique_tickers = eq['심볼'].dropna().unique()
                # 캐싱된 함수 호출
                sec_map = fetch_sectors_cached(tuple(unique_tickers)) # 튜플로 변환해 해시 가능하게
                eq['섹터'] = eq['심볼'].map(sec_map).fillna('Unknown')
            else:
                eq['섹터'] = 'Unknown'

        return eq, df_hedge, debug_logs

    except Exception as e:
        return None, None, f"처리 중 치명적 오류: {e}"

# ---------------------------------------------------------
# 2. Yahoo Finance Sector Fetcher (Helper)
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
# 3. 벤치마크 다운로드 (비교용)
# ---------------------------------------------------------
@st.cache_data
def download_benchmark(start_date, end_date):
    try:
        bm = yf.download(['^GSPC', '^KS11'], start=start_date, end=end_date + pd.Timedelta(days=5), progress=False)['Adj Close']
        bm = bm.ffill()
        return bm
    except:
        return pd.DataFrame()

# =========================================================
# 메인 앱
# =========================================================

# 메뉴 선택
menu = st.sidebar.radio("Dashboard Menu", ["Total Portfolio (Team PNL)", "Cash Equity Analysis"])

if menu == "Total Portfolio (Team PNL)":
    st.info("기존 기능(Team_PNL.xlsx)은 유지됩니다. (코드 생략)")

elif menu == "Cash Equity Analysis":
    st.subheader("📈 Cash Equity Portfolio Analysis")
    
    uploaded_file = st.sidebar.file_uploader("Upload 'Holdings3.xlsx'", type=['xlsx'])
    
    if uploaded_file:
        with st.spinner("데이터 처리 중... (엑셀 파일 읽기 & 수익률 계산)"):
            df_eq, df_hedge, logs = load_cash_equity_data(uploaded_file)
        
        # 디버그 로그 (필요시 확인)
        with st.expander("Debug Logs"):
            st.write(logs)
            
        if df_eq is not None and not df_eq.empty:
            # -------------------------------------------
            # 1. 일별 성과 집계 (Aggregation)
            # -------------------------------------------
            # 일자별 Equity 합계 (PnL, Prev_MV)
            daily_agg = df_eq.groupby('기준일자')[['Daily_PnL_KRW', 'Prev_MV']].sum()
            
            # Hedge 데이터 병합
            # df_hedge는 Series 형태 (Index: 날짜, Value: Daily PnL)
            if isinstance(df_hedge, pd.Series):
                df_hedge = df_hedge.to_frame(name='Hedge_PnL_KRW')
            elif df_hedge.empty:
                df_hedge = pd.DataFrame(columns=['Hedge_PnL_KRW'])

            # 날짜 인덱스 맞추기
            df_perf = daily_agg.join(df_hedge, how='outer').fillna(0)
            
            # Total PnL
            df_perf['Total_PnL_KRW'] = df_perf['Daily_PnL_KRW'] + df_perf['Hedge_PnL_KRW']
            
            # 수익률 계산 (분모: 전일 Equity 평가금액)
            # 시초가(Prev_MV)가 0인 경우(첫날 등) 수익률 0 처리
            df_perf['Ret_Equity'] = np.where(df_perf['Prev_MV'] > 0, df_perf['Daily_PnL_KRW'] / df_perf['Prev_MV'], 0)
            df_perf['Ret_Total'] = np.where(df_perf['Prev_MV'] > 0, df_perf['Total_PnL_KRW'] / df_perf['Prev_MV'], 0)
            
            # 누적 수익률
            df_perf['Cum_Equity'] = (1 + df_perf['Ret_Equity']).cumprod() - 1
            df_perf['Cum_Total'] = (1 + df_perf['Ret_Total']).cumprod() - 1

            # -------------------------------------------
            # 2. 대시보드 UI
            # -------------------------------------------
            min_date, max_date = df_perf.index.min(), df_perf.index.max()
            
            # (A) 요약 지표
            st.markdown("### 📊 Performance Summary")
            last_day = df_perf.iloc[-1]
            # 현재 평가금액 (가장 최근 일자의 원화평가금액 합계)
            curr_aum = df_eq[df_eq['기준일자'] == max_date]['원화평가금액'].sum()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Return (Hedged)", f"{last_day['Cum_Total']:.2%}")
            col2.metric("Equity Return (Unhedged)", f"{last_day['Cum_Equity']:.2%}")
            col3.metric("Hedge Effect", f"{(last_day['Cum_Total'] - last_day['Cum_Equity']):.2%}")
            col4.metric("Current Equity AUM", f"{curr_aum:,.0f} KRW")

            # (B) 수익률 차트
            st.markdown("### 📈 Cumulative Return Comparison")
            
            # 벤치마크 로드
            bm_df = download_benchmark(min_date, max_date)
            if not bm_df.empty:
                bm_df = bm_df.reindex(df_perf.index, method='ffill').pct_change().fillna(0)
                bm_cum = (1 + bm_df).cumprod() - 1
            else:
                bm_cum = pd.DataFrame()

            fig = go.Figure()
            # Total (Hedged)
            fig.add_trace(go.Scatter(x=df_perf.index, y=df_perf['Cum_Total'], name='Total (Hedged)', line=dict(color='#2563eb', width=3)))
            # Equity Only
            fig.add_trace(go.Scatter(x=df_perf.index, y=df_perf['Cum_Equity'], name='Equity Only', line=dict(color='#60a5fa', dash='dot')))
            
            # Benchmarks
            if not bm_cum.empty:
                if '^GSPC' in bm_cum.columns:
                    fig.add_trace(go.Scatter(x=bm_cum.index, y=bm_cum['^GSPC'], name='S&P 500', line=dict(color='grey', width=1, dash='dash')))
                if '^KS11' in bm_cum.columns:
                    fig.add_trace(go.Scatter(x=bm_cum.index, y=bm_cum['^KS11'], name='KOSPI', line=dict(color='silver', width=1, dash='dash')))

            fig.update_layout(template="plotly_white", height=500, yaxis_tickformat=".2%", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # (C) 섹터 및 종목 분석
            st.markdown("### 🌍 Portfolio Breakdown & Attribution")
            
            tab1, tab2 = st.tabs(["Sector Allocation", "Top Movers"])
            
            # 최신 데이터 기준
            df_latest = df_eq[df_eq['기준일자'] == max_date].copy()
            
            with tab1:
                if not df_latest.empty:
                    sec_grp = df_latest.groupby('섹터')['원화평가금액'].sum().reset_index()
                    fig_pie = px.pie(sec_grp, values='원화평가금액', names='섹터', title=f"Sector Exposure ({max_date.date()})", hole=0.4)
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.write("최신 일자 데이터가 없습니다.")
            
            with tab2:
                # 전체 기간 누적 PnL 상위 종목
                # 종목별 Daily PnL 합계
                stock_pnl = df_eq.groupby(['종목명', '섹터'])['Daily_PnL_KRW'].sum().reset_index()
                stock_pnl = stock_pnl.sort_values('Daily_PnL_KRW', ascending=False)
                
                c_win, c_lose = st.columns(2)
                with c_win:
                    st.success("🏆 Top 5 Contributors (KRW)")
                    st.dataframe(stock_pnl.head(5).style.format({'Daily_PnL_KRW': '{:,.0f}'}))
                with c_lose:
                    st.error("📉 Bottom 5 Contributors (KRW)")
                    st.dataframe(stock_pnl.tail(5).sort_values('Daily_PnL_KRW').style.format({'Daily_PnL_KRW': '{:,.0f}'}))
                    
            # (D) 데이터 테이블 보기
            with st.expander("View Daily Performance Data"):
                st.dataframe(df_perf.style.format("{:.4%}", subset=['Ret_Equity', 'Ret_Total', 'Cum_Equity', 'Cum_Total'])
                             .format("{:,.0f}", subset=['Daily_PnL_KRW', 'Hedge_PnL_KRW', 'Total_PnL_KRW', 'Prev_MV']))

        else:
            st.warning("데이터를 불러왔으나 내용이 비어있습니다.")
    else:
        st.info("파일을 업로드해주세요.")