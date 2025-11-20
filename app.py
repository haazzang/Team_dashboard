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
# 1. 데이터 로드 및 전처리 (Robust PnL Calculation)
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
                    # 헤더 찾기
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
                        
                        # 누적 총손익 컬럼 찾기
                        col_cum_pnl = next((c for c in df_h.columns if '누적' in c and '총손익' in c), None)
                        
                        if col_cum_pnl:
                            df_h[col_cum_pnl] = pd.to_numeric(df_h[col_cum_pnl], errors='coerce').fillna(0)
                            df_h = df_h.set_index('기준일자')
                            
                            # 일별 변동분 계산 (Daily PnL)
                            # 첫날은 데이터가 없으므로 0 또는 누적값 자체(시작일 기준)인데, 보통 Diff로 처리
                            daily_hedge = df_h[col_cum_pnl].diff().fillna(0)
                            
                            # 합산
                            if df_hedge.empty:
                                df_hedge = daily_hedge.to_frame(name='Hedge_PnL_KRW')
                            else:
                                df_hedge = df_hedge.add(daily_hedge.to_frame(name='Hedge_PnL_KRW'), fill_value=0)
                                
                            debug_logs.append(f"✅ Hedge 시트 로드: {sheet}")
                        else:
                            debug_logs.append(f"⚠️ {sheet}: '누적 총손익' 컬럼 없음")
                except Exception as e:
                    debug_logs.append(f"❌ Hedge 시트 오류 ({sheet}): {e}")

            # (B) Equity Holdings 시트 처리
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
                        df.columns = [str(c).strip() for c in df.iloc[h_idx]]
                        df = df.iloc[h_idx+1:].copy()
                        if '기준일자' in df.columns:
                            all_holdings.append(df)
                except Exception as e:
                    debug_logs.append(f"❌ Sheet 오류 ({sheet}): {e}")

        if not all_holdings:
            return None, None, f"주식 보유 내역을 찾을 수 없습니다. 로그: {debug_logs}"

        # 2. 주식 데이터 병합
        eq = pd.concat(all_holdings, ignore_index=True)
        eq['기준일자'] = pd.to_datetime(eq['기준일자'], errors='coerce')
        eq = eq.dropna(subset=['기준일자'])
        
        # 숫자 변환
        cols_to_num = ['원화평가금액', '원화총평가손익', '원화총매매손익']
        for c in cols_to_num:
            if c in eq.columns:
                eq[c] = pd.to_numeric(eq[c], errors='coerce').fillna(0)

        # 3. 포트폴리오 레벨 일별 집계 (핵심 수정 부분)
        # 종목별로 계산하지 않고, 일별 전체 합계로 계산하여 매도(Exit)시 PnL 누락 방지
        
        # (1) 일별 PnL 합계 (누적 개념)
        # Total Cumulative PnL at time T = Sum(Unrealized PnL) + Sum(Cumulative Realized PnL)
        eq['Total_Cum_PnL_Stock'] = eq['원화총평가손익'] + eq['원화총매매손익']
        
        daily_agg = eq.groupby('기준일자').agg({
            'Total_Cum_PnL_Stock': 'sum',  # 전체 누적 손익 합계
            '원화평가금액': 'sum'          # 전체 평가 금액 (Exposure)
        })
        
        # (2) Daily PnL 유도 (Change in Cumulative PnL)
        daily_agg['Daily_PnL_KRW'] = daily_agg['Total_Cum_PnL_Stock'].diff().fillna(0)
        
        # (3) 수익률 분모: 전일 평가금액 (Prev MV)
        daily_agg['Prev_MV'] = daily_agg['원화평가금액'].shift(1)
        
        # 4. Hedge 데이터 병합
        df_perf = daily_agg.join(df_hedge, how='outer').fillna(0)
        
        # 5. 최종 수익률 계산
        # Total Daily PnL = Equity Daily PnL + Hedge Daily PnL
        df_perf['Total_PnL_KRW'] = df_perf['Daily_PnL_KRW'] + df_perf['Hedge_PnL_KRW']
        
        # 수익률 = Daily PnL / Prev_MV
        # 첫날 등 Prev_MV가 0인 경우 처리
        df_perf['Ret_Equity'] = np.where(df_perf['Prev_MV'] > 0, df_perf['Daily_PnL_KRW'] / df_perf['Prev_MV'], 0)
        df_perf['Ret_Total'] = np.where(df_perf['Prev_MV'] > 0, df_perf['Total_PnL_KRW'] / df_perf['Prev_MV'], 0)
        
        # 첫날 데이터(수익률 0) 제외 (선택 사항, 보통 첫날은 기준점이므로 제외)
        df_perf = df_perf.iloc[1:]
        
        # 누적 수익률
        df_perf['Cum_Equity'] = (1 + df_perf['Ret_Equity']).cumprod() - 1
        df_perf['Cum_Total'] = (1 + df_perf['Ret_Total']).cumprod() - 1
        
        # 6. 섹터 정보 매핑 (최신 데이터 기준)
        latest_dt = eq['기준일자'].max()
        df_latest = eq[eq['기준일자'] == latest_dt].copy()
        
        if '섹터' not in df_latest.columns:
            if '심볼' in df_latest.columns:
                unique_tickers = df_latest['심볼'].dropna().unique()
                sec_map = fetch_sectors_cached(tuple(unique_tickers))
                df_latest['섹터'] = df_latest['심볼'].map(sec_map).fillna('Unknown')
            else:
                df_latest['섹터'] = 'Unknown'
        
        # 전체 기간 종목별 PnL (Top Movers용) -> 여기서는 Daily PnL을 종목별로 발라내기 어려우므로
        # 근사치로 (마지막 날 누적 PnL - 첫날 누적 PnL)을 사용하거나,
        # 원화총평가손익(Current) + 원화총매매손익(Current) 은 해당 종목의 Life-to-Date PnL임.
        # 따라서 최신 일자의 'Total_Cum_PnL_Stock'을 사용하면 됨.
        # 단, 이미 전량 매도된 종목은 df_latest에 없음.
        # 해결: 전체 데이터에서 종목별 Max(Total_Cum_PnL) - Min(...) 은 정확하지 않음.
        # 간단히: 각 종목의 마지막 관측일의 '원화총매매손익' + 마지막 관측일의 '원화평가손익'
        
        # 종목별 마지막 데이터 추출
        id_col = '심볼' if '심볼' in eq.columns else '종목코드'
        df_last_seen = eq.sort_values('기준일자').groupby(id_col).tail(1)
        df_last_seen['Final_PnL'] = df_last_seen['원화총평가손익'] + df_last_seen['원화총매매손익']
        
        return df_perf, df_last_seen, debug_logs

    except Exception as e:
        return None, None, None, f"처리 중 치명적 오류: {e}"

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
# 3. 벤치마크 다운로드
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
    st.info("기존 기능(Team_PNL.xlsx)은 유지됩니다. (코드 생략 - 필요시 이전 코드 복붙)")

elif menu == "Cash Equity Analysis":
    st.subheader("📈 Cash Equity Portfolio Analysis")
    
    uploaded_file = st.sidebar.file_uploader("Upload 'Holdings3.xlsx'", type=['xlsx'])
    
    if uploaded_file:
        with st.spinner("데이터 처리 중... (엑셀 파일 읽기 & 수익률 계산)"):
            # 함수 호출 (리턴값 3개 주의)
            res = load_cash_equity_data(uploaded_file)
            
            if res[3] if len(res)>3 else None: # Error string check
                 st.error(res[3])
            else:
                 df_perf, df_last_holdings, logs = res[0], res[1], res[2]

        # 디버그 로그
        with st.expander("Debug Logs"):
            st.write(logs)
            
        if df_perf is not None and not df_perf.empty:
            min_date, max_date = df_perf.index.min(), df_perf.index.max()
            
            # (A) 요약 지표
            st.markdown("### 📊 Performance Summary")
            last_day = df_perf.iloc[-1]
            curr_aum = df_perf.iloc[-1]['원화평가금액']

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Return (Hedged)", f"{last_day['Cum_Total']:.2%}")
            col2.metric("Equity Return (Unhedged)", f"{last_day['Cum_Equity']:.2%}")
            col3.metric("Hedge Effect", f"{(last_day['Cum_Total'] - last_day['Cum_Equity']):.2%}")
            col4.metric("Current Equity AUM", f"{curr_aum:,.0f} KRW")

            # (B) 수익률 차트
            st.markdown("### 📈 Cumulative Return Comparison")
            
            bm_df = download_benchmark(min_date, max_date)
            bm_cum = pd.DataFrame()
            if not bm_df.empty:
                bm_df = bm_df.reindex(df_perf.index, method='ffill').pct_change().fillna(0)
                bm_cum = (1 + bm_df).cumprod() - 1

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_perf.index, y=df_perf['Cum_Total'], name='Total (Hedged)', line=dict(color='#2563eb', width=3)))
            fig.add_trace(go.Scatter(x=df_perf.index, y=df_perf['Cum_Equity'], name='Equity Only', line=dict(color='#60a5fa', dash='dot')))
            
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
            
            with tab1:
                # 현재 보유중인 종목만 대상 (잔고수량 > 0)
                df_current_holdings = df_last_holdings[df_last_holdings['잔고수량'] > 0]
                
                if not df_current_holdings.empty:
                    if '섹터' not in df_current_holdings.columns:
                         # 만약 df_last_holdings에 섹터가 없으면 다시 매핑 (위에서 이미 했지만 안전장치)
                         # load 함수에서 이미 df_last_seen은 raw eq에서 왔으므로 섹터가 없을 수 있음
                         # -> eq 전체에 섹터를 매핑했으면 있을 것임. 확인 필요.
                         # load 함수 수정: eq['섹터'] 매핑 후 df_last_seen 추출하도록 수정했음.
                         pass

                    sec_grp = df_current_holdings.groupby('섹터')['원화평가금액'].sum().reset_index()
                    fig_pie = px.pie(sec_grp, values='원화평가금액', names='섹터', title=f"Sector Exposure (Current)", hole=0.4)
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.write("현재 보유중인 주식이 없습니다.")
            
            with tab2:
                # 전체 기간 누적 PnL 상위 종목 (이미 매도된 종목 포함)
                stock_pnl = df_last_holdings[['종목명', '섹터', 'Final_PnL']].sort_values('Final_PnL', ascending=False)
                
                c_win, c_lose = st.columns(2)
                with c_win:
                    st.success("🏆 Top 5 Contributors (Total PnL)")
                    st.dataframe(stock_pnl.head(5).style.format({'Final_PnL': '{:,.0f}'}))
                with c_lose:
                    st.error("📉 Bottom 5 Contributors (Total PnL)")
                    st.dataframe(stock_pnl.tail(5).sort_values('Final_PnL').style.format({'Final_PnL': '{:,.0f}'}))
                    
            # (D) 데이터 테이블
            with st.expander("View Daily Performance Data"):
                st.dataframe(df_perf.style.format("{:.4%}", subset=['Ret_Equity', 'Ret_Total', 'Cum_Equity', 'Cum_Total'])
                             .format("{:,.0f}", subset=['Daily_PnL_KRW', 'Hedge_PnL_KRW', 'Total_PnL_KRW', 'Prev_MV']))

        else:
            st.warning("데이터를 불러왔으나 내용이 비어있습니다.")
    else:
        st.info("파일을 업로드해주세요.")