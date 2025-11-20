import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf

# --- 페이지 설정 ---
st.set_page_config(page_title="Portfolio Dashboard", layout="wide")
st.title("🚀 Team Portfolio Analysis (Code-Based Logic)")

# ---------------------------------------------------------
# 1. 데이터 로드 및 전처리 (Python 코드 로직 반영)
# ---------------------------------------------------------
@st.cache_data
def process_portfolio_data(file):
    try:
        # 1. 엑셀 파일 로드
        xls = pd.ExcelFile(file, engine='openpyxl')
        
        # 시트 분류
        hedge_sheet_name = None
        equity_sheets = []
        
        for s in xls.sheet_names:
            if 'hedge' in s.lower() or 'futures' in s.lower():
                hedge_sheet_name = s
            else:
                equity_sheets.append(s)
                
        # -----------------------------------------------------
        # 2. Equity 데이터 처리 (코드 라인 96~126 참고)
        # -----------------------------------------------------
        df_list = []
        for s in equity_sheets:
            # 헤더 찾기 (Date, Ticker 등이 있는 행)
            d = pd.read_excel(file, sheet_name=s, header=None, engine='openpyxl')
            header_idx = -1
            for i in range(10):
                row_vals = [str(x).strip() for x in d.iloc[i].values]
                if 'Date' in row_vals or 'Ticker' in row_vals or 'Symbol' in row_vals:
                    header_idx = i
                    break
            
            if header_idx != -1:
                d.columns = d.iloc[header_idx]
                d = d.iloc[header_idx+1:].copy()
                # 필수 컬럼이 있는 경우만 추가
                # 코드에서는 'Ticker', 'Market Price', 'Market Value', 'Quantity', 'Date', 'Currency' 사용
                # 실제 파일 컬럼명 매핑 필요
                # 파일 예시: Date, ISIN, Ticker, Quantity, Book Price, Notional, Market Price, Market Value, ...
                
                # 컬럼 정규화 (공백 제거)
                d.columns = [str(c).strip() for c in d.columns]
                
                # 필요한 컬럼 존재 여부 확인
                required = ['Date', 'Ticker', 'Market Price', 'Market Value', 'Quantity', 'Currency']
                if all(r in d.columns for r in required):
                     df_list.append(d)

        if not df_list:
            return None, None, "유효한 Equity 시트를 찾지 못했습니다. (컬럼명: Date, Ticker, Quantity, Market Price, Market Value, Currency 확인 필요)"

        eq = pd.concat(df_list, ignore_index=True)
        
        # 숫자 변환 및 날짜 처리
        eq['Date'] = pd.to_datetime(eq['Date'], errors='coerce')
        eq = eq.dropna(subset=['Date'])
        
        num_cols = ['Market Price', 'Market Value', 'Quantity']
        for c in num_cols:
            eq[c] = pd.to_numeric(eq[c], errors='coerce').fillna(0)
            
        eq['Currency'] = eq['Currency'].astype(str).str.upper().str.strip()
        
        # 정렬 (Ticker, Date 순)
        eq = eq.sort_values(['Ticker', 'Date'])
        
        # --- 핵심 로직: 전일 데이터 계산 (코드 113~118행) ---
        # 그룹별 Shift를 통해 전일 가격, 전일 수량, 전일 평가금액 확보
        eq['Prev_Price'] = eq.groupby('Ticker')['Market Price'].shift(1)
        eq['Prev_Qty'] = eq.groupby('Ticker')['Quantity'].shift(1)
        eq['Prev_MV'] = eq.groupby('Ticker')['Market Value'].shift(1)
        
        # 일별 PnL (로컬 통화 기준) = 전일 수량 * (당일 가격 - 전일 가격)
        # *주의: 매매(Buy/Sell)로 인한 수량 변화가 없는 보유분(Holding)에 대한 손익만 계산됨 (Time-Weighted Return 방식)
        eq['Daily_PnL_Local'] = eq['Prev_Qty'] * (eq['Market Price'] - eq['Prev_Price'])
        eq['Daily_PnL_Local'] = eq['Daily_PnL_Local'].fillna(0)
        
        # 개별 주식 수익률 (로컬)
        # eq['Stock_Ret'] = np.where(eq['Prev_MV'] > 0, eq['Daily_PnL_Local'] / eq['Prev_MV'], 0)

        # -----------------------------------------------------
        # 3. 환율 데이터 다운로드 및 적용 (코드 151~188행)
        # -----------------------------------------------------
        # 날짜 범위
        start_dt = eq['Date'].min()
        end_dt = eq['Date'].max()
        
        # 필요한 통화 목록
        currencies = eq['Currency'].unique()
        fx_map = {}
        
        # 야후 파이낸스 티커 매핑
        fx_tickers = {
            'USD': 'KRW=X', 
            'HKD': 'HKDKRW=X', 
            'JPY': 'JPYKRW=X',
            'KRW': None
        }
        
        dl_list = [fx_tickers[c] for c in currencies if c in fx_tickers and fx_tickers[c]]
        
        if dl_list:
            fx_data = yf.download(dl_list, start=start_dt, end=end_dt + pd.Timedelta(days=5), progress=False)
            if 'Adj Close' in fx_data.columns: fx_df = fx_data['Adj Close']
            elif 'Close' in fx_data.columns: fx_df = fx_data['Close']
            else: fx_df = fx_data
            
            # MultiIndex 처리
            if isinstance(fx_df.columns, pd.MultiIndex):
                fx_df.columns = fx_df.columns.get_level_values(0)
                
            # ffill로 휴장일 데이터 채움
            fx_df = fx_df.ffill()
        else:
            fx_df = pd.DataFrame()

        # 환율 적용 함수
        def get_fx_rate(row):
            ccy = row['Currency']
            date = row['Date']
            if ccy == 'KRW': return 1.0
            
            ticker = fx_tickers.get(ccy)
            if not ticker: return 1.0 # 매핑 없으면 1.0 처리
            
            if ticker in fx_df.columns and date in fx_df.index:
                return fx_df.loc[date, ticker]
            else:
                # 날짜 매칭 안되면 가장 최근 데이터라도 사용 (ffill 이미 됨)
                # 혹은 1.0
                return 1.0 

        # 속도를 위해 map 사용 (apply보다 빠름)
        # 먼저 환율 테이블을 긴 형태(melt)로 변환하여 merge하는 것이 효율적임
        fx_long = fx_df.reset_index().melt(id_vars='Date', var_name='Ticker', value_name='Rate')
        
        # 통화 -> 티커 매핑
        curr_to_ticker = {k:v for k,v in fx_tickers.items() if v}
        eq['FX_Ticker'] = eq['Currency'].map(curr_to_ticker)
        
        # Merge FX
        eq_merged = pd.merge(eq, fx_long, left_on=['Date', 'FX_Ticker'], right_on=['Date', 'Ticker'], how='left')
        eq_merged['Rate'] = eq_merged['Rate'].fillna(1.0) # KRW or missing
        
        # KRW 환산 (코드 186~188행)
        # 코드 로직: 오늘 환율을 사용하여 PnL과 MV를 환산
        eq_merged['PnL_KRW'] = eq_merged['Daily_PnL_Local'] * eq_merged['Rate']
        eq_merged['Prev_MV_KRW'] = eq_merged['Prev_MV'] * eq_merged['Rate']
        
        # -----------------------------------------------------
        # 4. 포트폴리오 레벨 집계 (Equity) (코드 190~196행)
        # -----------------------------------------------------
        # 일자별 합계
        port_daily = eq_merged.groupby('Date')[['PnL_KRW', 'Prev_MV_KRW']].sum()
        
        # Equity 수익률 = Equity PnL Sum / Equity Prev MV Sum
        port_daily['Ret_Equity'] = np.where(port_daily['Prev_MV_KRW'] > 0, 
                                            port_daily['PnL_KRW'] / port_daily['Prev_MV_KRW'], 0)
        
        # -----------------------------------------------------
        # 5. Hedge 데이터 처리 (코드 420~440행)
        # -----------------------------------------------------
        hedge_daily = pd.DataFrame(index=port_daily.index)
        hedge_daily['Hedge_PnL_KRW'] = 0.0

        if hedge_sheet_name:
            df_h = pd.read_excel(file, sheet_name=hedge_sheet_name, header=None, engine='openpyxl')
            # 헤더 찾기 ('기준일자', '누적 총손익' 등)
            h_idx = -1
            for i in range(10):
                row_vals = [str(x).strip() for x in df_h.iloc[i].values]
                # 파일 예시: 기준일자, 매매손익(원화환산)..., 누적 총손익
                # 코드 예시: Date in col 0, Cumulative PnL in col 4
                if '기준일자' in row_vals or 'Date' in row_vals:
                    h_idx = i
                    break
            
            if h_idx != -1:
                df_h.columns = [str(c).strip() for c in df_h.iloc[h_idx]]
                df_h = df_h.iloc[h_idx+1:].copy()
                
                col_date = [c for c in df_h.columns if '일자' in c or 'Date' in c][0]
                # 누적 손익 컬럼 찾기 (예: '누적 총손익' or 5번째 컬럼)
                # 코드에서는 index 4 (5번째) 사용. 여기서는 이름으로 찾거나 index 사용
                if len(df_h.columns) > 4:
                    col_cum_pnl = df_h.columns[4] # 5번째 컬럼 가정 (코드 로직)
                else:
                    col_cum_pnl = [c for c in df_h.columns if '누적' in c and '손익' in c][-1]

                df_h[col_date] = pd.to_datetime(df_h[col_date], errors='coerce')
                df_h = df_h.dropna(subset=[col_date]).sort_values(col_date)
                df_h[col_cum_pnl] = pd.to_numeric(df_h[col_cum_pnl], errors='coerce').fillna(0)
                
                # 일별 변동분 계산 (Diff)
                df_h.set_index(col_date, inplace=True)
                daily_hedge = df_h[col_cum_pnl].diff().fillna(df_h[col_cum_pnl]) # 첫날은 누적값 그대로
                
                # 메인 날짜 인덱스에 맞춤 (reindex)
                hedge_daily['Hedge_PnL_KRW'] = daily_hedge.reindex(port_daily.index).fillna(0)

        # -----------------------------------------------------
        # 6. 최종 통합 (Total Return) (코드 453~458행)
        # -----------------------------------------------------
        final_df = port_daily.join(hedge_daily)
        final_df['Total_PnL_KRW'] = final_df['PnL_KRW'] + final_df['Hedge_PnL_KRW']
        
        # Total Return = (Equity PnL + Hedge PnL) / Equity Prev MV
        # (Hedge는 증거금만 사용하므로 분모(Exposure)는 주식 평가금액 기준)
        final_df['Ret_Total'] = np.where(final_df['Prev_MV_KRW'] > 0,
                                         final_df['Total_PnL_KRW'] / final_df['Prev_MV_KRW'], 0)
        
        # 누적 수익률
        final_df['Cum_Equity'] = (1 + final_df['Ret_Equity']).cumprod() - 1
        final_df['Cum_Total'] = (1 + final_df['Ret_Total']).cumprod() - 1
        
        return final_df, None

    except Exception as e:
        return None, None, f"데이터 처리 중 오류 발생: {e}"

# =========================================================
# 메인 앱 UI
# =========================================================

uploaded_file = st.sidebar.file_uploader("Upload 'Holdings2.xlsx' (or similar)", type=['xlsx'])

if uploaded_file:
    with st.spinner("데이터 분석 및 환율 다운로드 중..."):
        df_res, logs, err = process_portfolio_data(uploaded_file)
    
    if err:
        st.error(err)
    else:
        # 날짜 필터링
        min_date, max_date = df_res.index.min(), df_res.index.max()
        st.sidebar.write(f"📅 Data Range: {min_date.date()} ~ {max_date.date()}")
        
        # --- 1. 성과 요약 (Summary) ---
        st.markdown("### 📊 Portfolio Performance Summary")
        last_day = df_res.iloc[-1]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Cumulative Return", f"{last_day['Cum_Total']:.2%}")
        c2.metric("Equity Only Return", f"{last_day['Cum_Equity']:.2%}")
        c3.metric("Hedge Contribution", f"{(last_day['Cum_Total'] - last_day['Cum_Equity']):.2%}")
        c4.metric("Current AUM (KRW)", f"{last_day['Prev_MV_KRW']:,.0f}") # 전일 MV 기준이므로 근사치

        # --- 2. 차트 (Chart) ---
        st.markdown("### 📈 Cumulative Return (Equity vs Total)")
        
        # 벤치마크 다운로드 (비교용)
        bm_data = yf.download(['^GSPC', '^KS11'], start=min_date, end=max_date + pd.Timedelta(days=1), progress=False)['Adj Close']
        if not bm_data.empty:
             bm_data = bm_data.ffill().reindex(df_res.index).pct_change().fillna(0)
             bm_cum = (1 + bm_data).cumprod() - 1
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_res.index, y=df_res['Cum_Total'], name='Total Portfolio (Hedged)', line=dict(color='#2563eb', width=3)))
        fig.add_trace(go.Scatter(x=df_res.index, y=df_res['Cum_Equity'], name='Equity Only', line=dict(color='lightblue', width=2, dash='dot')))
        
        if not bm_data.empty:
            if '^GSPC' in bm_cum.columns:
                fig.add_trace(go.Scatter(x=bm_cum.index, y=bm_cum['^GSPC'], name='S&P 500', line=dict(color='grey', width=1, dash='dash')))
            if '^KS11' in bm_cum.columns:
                fig.add_trace(go.Scatter(x=bm_cum.index, y=bm_cum['^KS11'], name='KOSPI', line=dict(color='lightgrey', width=1, dash='dash')))

        fig.update_layout(template="plotly_white", height=500, yaxis_tickformat=".2%", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        # --- 3. 데이터 테이블 ---
        with st.expander("View Daily Data"):
            st.dataframe(df_res.style.format("{:,.0f}", subset=['PnL_KRW', 'Prev_MV_KRW', 'Hedge_PnL_KRW', 'Total_PnL_KRW'])
                         .format("{:.4%}", subset=['Ret_Equity', 'Ret_Total', 'Cum_Equity', 'Cum_Total']))

else:
    st.info("👈 왼쪽 사이드바에서 엑셀 파일(Holdings2.xlsx)을 업로드해주세요.")
    st.markdown("""
    #### 📌 수익률 계산 로직 (Python Script 기반)
    1. **Equity PnL (Local):** `전일수량(Quantity_t-1)` × (`당일주가(Price_t)` - `전일주가(Price_t-1)`)
    2. **Equity PnL (KRW):** `Equity PnL (Local)` × `당일환율(FX_t)`
    3. **Equity MV (KRW):** `전일평가금액(MV_Local_t-1)` × `당일환율(FX_t)`
    4. **Hedge PnL (KRW):** Hedge 시트의 `누적손익(Cumulative)` 차분(Diff)
    5. **Total Return:** (`Equity PnL KRW` + `Hedge PnL KRW`) ÷ `Equity MV KRW`
    """)