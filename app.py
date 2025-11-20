import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf

# --- 페이지 설정 ---
st.set_page_config(page_title="Portfolio Dashboard", layout="wide")
st.title("🚀 Team Portfolio Analysis Dashboard")

# ==============================================================================
# [Helper Functions] 공통 함수
# ==============================================================================

@st.cache_data
def fetch_sectors_cached(tickers):
    sector_map = {}
    # 진행바 없이 조용히 처리 (속도 최적화)
    for t in tickers:
        try:
            # 티커가 문자열인지 확인
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
    try:
        bm = yf.download(['^GSPC', '^KS11'], start=start_date, end=end_date + pd.Timedelta(days=5), progress=False)['Adj Close']
        bm = bm.ffill()
        return bm
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
        df.rename(columns={v: k for k, v in assets.items()}, inplace=True)
        return df
    except:
        return pd.DataFrame()

def create_manual_html_table(df, title=None):
    html = ''
    if title: html += f'<h5 style="margin-top:20px; margin-bottom:10px;">{title}</h5>'
    html += '<div style="overflow-x:auto;"><table style="width:100%; border-collapse: collapse; font-size: 0.9rem;">'
    html += '<thead style="background-color: #f8f9fa; border-bottom: 2px solid #dee2e6; color: black;"><tr>' # 헤더 검정색 강제
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
# [PART 1] Total Portfolio (Team PNL) 로드 함수
# ==============================================================================
@st.cache_data
def load_team_pnl_data(file):
    try:
        df_pnl_raw = pd.read_excel(file, sheet_name='PNL', header=None, engine='openpyxl')
        h_idx = -1
        for i in range(15):
            if '일자' in [str(x).strip() for x in df_pnl_raw.iloc[i].values]:
                h_idx = i; break
        if h_idx == -1: return None, None, "PNL 시트 헤더 없음"
        
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

        # Position Sheet
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

    except Exception as e:
        return None, None, f"Load Error: {e}"

# ==============================================================================
# [PART 2] Cash Equity 데이터 로드 (로직 수정됨: Group Diff -> Sum)
# ==============================================================================
@st.cache_data
def load_cash_equity_data(file):
    debug_logs = []
    try:
        xls = pd.ExcelFile(file, engine='openpyxl')
        all_holdings = []
        df_hedge = pd.DataFrame()
        
        for sheet in xls.sheet_names:
            # [A] Hedge Sheet
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
                        df_h = df_h.dropna(subset=['기준일자']).set_index('기준일자').sort_index()
                        
                        col_cum = next((c for c in df_h.columns if '누적' in c and '총손익' in c), None)
                        if col_cum:
                            df_h[col_cum] = pd.to_numeric(df_h[col_cum], errors='coerce').fillna(0)
                            # Hedge 일별 PnL
                            daily_hedge = df_h[col_cum].diff().fillna(0)
                            if df_hedge.empty: df_hedge = daily_hedge.to_frame(name='Hedge_PnL_KRW')
                            else: df_hedge = df_hedge.add(daily_hedge.to_frame(name='Hedge_PnL_KRW'), fill_value=0)
                except: pass
            
            # [B] Equity Sheet
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
                except: pass

        if not all_holdings: return None, None, None, "Holdings 데이터가 없습니다."

        # 2. 병합 및 전처리
        eq = pd.concat(all_holdings, ignore_index=True)
        eq['기준일자'] = pd.to_datetime(eq['기준일자'], errors='coerce')
        eq = eq.dropna(subset=['기준일자'])
        
        # 숫자 변환
        cols_num = ['원화평가금액', '원화총평가손익', '원화총매매손익', '잔고수량', 'Market Price', '외화평가금액']
        for c in cols_num:
            if c in eq.columns: eq[c] = pd.to_numeric(eq[c], errors='coerce').fillna(0)

        # [중요] 종목 식별자
        id_col = '심볼' if '심볼' in eq.columns else '종목코드'
        
        # 3. 섹터 매핑 (필수)
        # 여기서 미리 매핑해야 나중에 groupby할 때 에러 안 남
        if '섹터' not in eq.columns:
            if '심볼' in eq.columns:
                uniques = eq['심볼'].dropna().unique()
                sec_map = fetch_sectors_cached(tuple(uniques))
                eq['섹터'] = eq['심볼'].map(sec_map).fillna('Unknown')
            else:
                eq['섹터'] = 'Unknown'
        else:
            eq['섹터'] = eq['섹터'].fillna('Unknown')

        # 4. 수익률 계산 핵심 로직 (Group Diff -> Sum 방식)
        # (1) 종목별 정렬
        eq = eq.sort_values([id_col, '기준일자'])
        
        # (2) 종목별 누적 PnL 계산 (Unrealized + Realized)
        eq['Stock_Cum_PnL'] = eq['원화총평가손익'] + eq['원화총매매손익']
        
        # (3) 종목별 Daily PnL (차분)
        # 첫날의 Diff는 NaN이 되므로 0 처리 (보수적 접근)
        eq['Stock_Daily_PnL'] = eq.groupby(id_col)['Stock_Cum_PnL'].diff().fillna(0)
        
        # (4) Local Return 계산 (Market Price 변동분)
        if 'Market Price' in eq.columns:
            eq['Prev_Price'] = eq.groupby(id_col)['Market Price'].shift(1)
            eq['Stock_Ret_Local'] = np.where(eq['Prev_Price'] > 0, 
                                           (eq['Market Price'] - eq['Prev_Price']) / eq['Prev_Price'], 
                                           0)
            # Local Weighting용 전일 평가금액
            eq['Prev_MV'] = eq.groupby(id_col)['원화평가금액'].shift(1).fillna(0)
        else:
            eq['Stock_Ret_Local'] = 0
            eq['Prev_MV'] = 0

        # (5) 일별 Aggregation (단순 합산)
        # 여기서 종목이 리스트에서 사라지면, 그날의 합산에 포함되지 않을 뿐
        # 거대한 마이너스 값(Drop)이 발생하지 않음.
        daily_agg = eq.groupby('기준일자').agg({
            'Stock_Daily_PnL': 'sum',  # 이게 진짜 Daily PnL 합계
            '원화평가금액': 'sum',         # 당일 Exposure
            'Prev_MV': 'sum'           # 전일 Exposure 합계
        }).rename(columns={'Stock_Daily_PnL': 'Daily_PnL_KRW'})
        
        # (6) Local Return Aggregation (Weighted Average)
        # 가중치 = 개별종목 전일MV / 전체 전일MV
        daily_total_prev = daily_agg['Prev_MV'] # 이미 일별 합계임
        
        # 원래 데이터프레임에 일별 총 Prev_MV 붙이기
        eq = eq.merge(daily_total_prev.rename('Total_Prev_MV'), on='기준일자', how='left')
        eq['Weight'] = np.where(eq['Total_Prev_MV'] > 0, eq['Prev_MV'] / eq['Total_Prev_MV'], 0)
        eq['W_Ret_Local'] = eq['Stock_Ret_Local'] * eq['Weight']
        
        daily_local_ret = eq.groupby('기준일자')['W_Ret_Local'].sum().rename('Ret_Equity_Local')
        
        # 5. Hedge 병합 및 최종 수익률
        df_perf = daily_agg.join(df_hedge, how='outer').fillna(0)
        df_perf = df_perf.join(daily_local_ret, how='left').fillna(0)
        
        # Total PnL
        df_perf['Total_PnL_KRW'] = df_perf['Daily_PnL_KRW'] + df_perf['Hedge_PnL_KRW']
        
        # Exposure (분모): 전일 평가금액 (Prev_MV)
        # Prev_MV는 eq.groupby...shift(1) 합계로 구함. 
        # (주의: 오늘 신규 매수한 종목은 Prev_MV가 0이므로 수익률 기여도가 당일엔 0에 수렴하거나 PnL만 분자에 더해짐. 이는 타당함)
        
        # 수익률 계산
        # 분모가 0인 경우(첫날 등) 처리
        df_perf['Ret_Equity_KRW'] = np.where(df_perf['Prev_MV'] > 0, df_perf['Daily_PnL_KRW'] / df_perf['Prev_MV'], 0)
        df_perf['Ret_Total_KRW'] = np.where(df_perf['Prev_MV'] > 0, df_perf['Total_PnL_KRW'] / df_perf['Prev_MV'], 0)
        
        # 첫날 제외
        df_perf = df_perf.iloc[1:]
        
        # 누적
        df_perf['Cum_Equity_KRW'] = (1 + df_perf['Ret_Equity_KRW']).cumprod() - 1
        df_perf['Cum_Total_KRW'] = (1 + df_perf['Ret_Total_KRW']).cumprod() - 1
        df_perf['Cum_Equity_Local'] = (1 + df_perf['Ret_Equity_Local']).cumprod() - 1
        
        # 종목별 최종 상태 (Top Movers용)
        df_last = eq.sort_values('기준일자').groupby(id_col).tail(1)
        df_last['Final_PnL'] = df_last['원화총평가손익'] + df_last['원화총매매손익']
        
        return df_perf, df_last, debug_logs, None

    except Exception as e:
        return None, None, None, f"Process Error: {e}"


# ==============================================================================
# [MAIN UI]
# ==============================================================================

menu = st.sidebar.radio("Dashboard Menu", ["Total Portfolio (Team PNL)", "Cash Equity Analysis"])

# ------------------------------------------------------------------------------
# MENU 1: Total Portfolio (Team PNL)
# ------------------------------------------------------------------------------
if menu == "Total Portfolio (Team PNL)":
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
            
            # Benchmarks
            with st.spinner("Fetching Market Data..."):
                df_assets = download_cross_assets(df_pnl.index.min(), df_pnl.index.max())
                bm_cum = pd.DataFrame(index=df_user_ret.index)
                if not df_assets.empty:
                    df_assets = df_assets.reindex(df_user_ret.index, method='ffill')
                    df_asset_ret = df_assets.pct_change().fillna(0)
                    if 'S&P 500' in df_assets.columns: bm_cum['SPX'] = (1 + df_asset_ret['S&P 500']).cumprod() - 1
                    if 'KOSPI' in df_assets.columns: bm_cum['KOSPI'] = (1 + df_asset_ret['KOSPI']).cumprod() - 1
            
            # Tabs
            t1, t2, t3, t4, t5 = st.tabs(["📈 Chart", "📊 Analysis", "🔗 Correlation", "🌍 Cross Asset", "🧪 Simulation"])
            
            with t1:
                strat = st.selectbox("Select Strategy", df_user_ret.columns)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_user_ret.index, y=df_user_ret[strat], name=strat, line=dict(width=2)))
                bm_name = 'SPX' if any(k in strat for k in ['해외', 'Global', 'US']) else 'KOSPI'
                if bm_name in bm_cum.columns:
                    fig.add_trace(go.Scatter(x=df_user_ret.index, y=bm_cum[bm_name], name=bm_name, line=dict(color='grey', dash='dash')))
                st.plotly_chart(fig, use_container_width=True)
                
            with t2:
                stats = pd.DataFrame(index=df_daily_ret.columns)
                stats['Volatility'] = df_daily_ret.std() * np.sqrt(252)
                stats['Sharpe'] = (df_daily_ret.mean() / df_daily_ret.std() * np.sqrt(252)).fillna(0)
                nav = (1 + df_daily_ret).cumprod()
                stats['MDD'] = ((nav - nav.cummax()) / nav.cummax()).min()
                stats['Total Return'] = df_user_ret.iloc[-1]
                
                disp = stats.copy()
                for c in disp.columns:
                    if c == 'Sharpe': disp[c] = disp[c].apply(lambda x: f"{x:.2f}")
                    else: disp[c] = disp[c].apply(lambda x: f"{x:.2%}")
                
                disp.insert(0, 'Strategy', disp.index)
                disp['Strategy'] = disp['Strategy'].apply(lambda x: x.split('_')[0])
                st.markdown(create_manual_html_table(disp), unsafe_allow_html=True)

            with t3:
                corr = df_daily_ret.corr()
                fig_corr = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale='RdBu', zmin=-1, zmax=1))
                fig_corr.update_layout(height=700)
                st.plotly_chart(fig_corr)
                
            with t4:
                if not df_assets.empty:
                    comb = pd.concat([df_daily_ret, df_asset_ret], axis=1).corr()
                    sub_corr = comb.loc[df_daily_ret.columns, df_asset_ret.columns]
                    fig_cross = go.Figure(data=go.Heatmap(z=sub_corr.values, x=sub_corr.columns, y=sub_corr.index, colorscale='RdBu', zmin=-1, zmax=1))
                    st.plotly_chart(fig_cross)

            with t5:
                st.subheader("Simulation")
                c_in, c_out = st.columns([1,3])
                with c_in:
                    weights = {}
                    for col in df_daily_ret.columns:
                        weights[col] = st.slider(col, 0.0, 1.0, 1.0/len(df_daily_ret.columns), 0.05)
                with c_out:
                    sim_daily = df_daily_ret.mul(pd.Series(weights), axis=1).sum(axis=1)
                    sim_cum = (1 + sim_daily).cumprod() - 1
                    fig_sim = go.Figure()
                    fig_sim.add_trace(go.Scatter(x=sim_cum.index, y=sim_cum, name="Simulated", line=dict(color='red')))
                    st.plotly_chart(fig_sim, use_container_width=True)
        else: st.error(err)

# ------------------------------------------------------------------------------
# MENU 2: Cash Equity Analysis
# ------------------------------------------------------------------------------
elif menu == "Cash Equity Analysis":
    st.subheader("📈 Cash Equity Portfolio Analysis")
    uploaded_file_ce = st.sidebar.file_uploader("Upload 'Holdings3.xlsx'", type=['xlsx'], key="ce")
    
    if uploaded_file_ce:
        with st.spinner("Processing..."):
            res = load_cash_equity_data(uploaded_file_ce)
            df_perf, df_last, logs, err = res
        
        if err: st.error(err)
        elif df_perf is not None:
            view_opt = st.radio("Currency View", ["KRW (Unhedged / Hedged)", "Local Currency (Price Return Only)"], horizontal=True)
            
            last_day = df_perf.iloc[-1]
            curr_aum = df_perf.iloc[-1]['원화평가금액']
            
            # Summary
            c1, c2, c3, c4 = st.columns(4)
            if view_opt.startswith("KRW"):
                c1.metric("Total Return (Hedged)", f"{last_day['Cum_Total_KRW']:.2%}")
                c2.metric("Equity Return (KRW)", f"{last_day['Cum_Equity_KRW']:.2%}")
                c3.metric("Hedge Impact", f"{(last_day['Cum_Total_KRW'] - last_day['Cum_Equity_KRW']):.2%}")
                y_col = 'Cum_Total_KRW'; sub_col = 'Cum_Equity_KRW'; name_main = 'Total (Hedged)'; name_sub = 'Equity (KRW)'
            else:
                c1.metric("Local Return", f"{last_day['Cum_Equity_Local']:.2%}")
                c2.metric("Equity Return (KRW)", f"{last_day['Cum_Equity_KRW']:.2%}")
                c3.metric("FX Impact", f"{(last_day['Cum_Equity_KRW'] - last_day['Cum_Equity_Local']):.2%}")
                y_col = 'Cum_Equity_Local'; sub_col = None; name_main = 'Equity (Local)'; name_sub = None
            c4.metric("Current AUM", f"{curr_aum:,.0f} KRW")

            # Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_perf.index, y=df_perf[y_col], name=name_main, line=dict(color='#2563eb', width=3)))
            if sub_col: fig.add_trace(go.Scatter(x=df_perf.index, y=df_perf[sub_col], name=name_sub, line=dict(color='#60a5fa', dash='dot')))
            
            bm_df = download_benchmark(df_perf.index.min(), df_perf.index.max())
            if not bm_df.empty:
                bm_cum = (1 + bm_df.reindex(df_perf.index, method='ffill').pct_change().fillna(0)).cumprod() - 1
                if '^GSPC' in bm_cum.columns: fig.add_trace(go.Scatter(x=bm_cum.index, y=bm_cum['^GSPC'], name='S&P 500', line=dict(color='grey', dash='dash')))
                if '^KS11' in bm_cum.columns: fig.add_trace(go.Scatter(x=bm_cum.index, y=bm_cum['^KS11'], name='KOSPI', line=dict(color='silver', dash='dash')))
            st.plotly_chart(fig, use_container_width=True)
            
            # Details
            t1, t2 = st.tabs(["Sector Allocation", "Top Movers"])
            with t1:
                # 중요: df_last는 전체 기간의 마지막 스냅샷이 아니라, 개별 종목의 마지막 데이터임.
                # 현재 보유중인 것만 걸러야 함 (잔고수량 > 0)
                # 하지만 df_last는 '전체 종목'의 마지막 상태임. 
                # 현재 날짜 기준 보유 종목을 보려면? df_perf의 마지막 날짜와 일치하는 holdings를 찾아야 하는데,
                # load 함수에서 원본 eq를 리턴하지 않음.
                # 대안: df_last에서 '기준일자'가 max_date인 것만 필터링
                max_date = df_perf.index.max()
                curr_hold = df_last[(df_last['기준일자'] == max_date) & (df_last['잔고수량'] > 0)]
                
                if not curr_hold.empty:
                    # reset_index() to avoid ambiguity if index is named '섹터'
                    sec_grp = curr_hold.groupby('섹터')['원화평가금액'].sum().reset_index()
                    pie = px.pie(sec_grp, values='원화평가금액', names='섹터', title="Current Sector Exposure")
                    st.plotly_chart(pie)
                else:
                    st.write("현재 보유 종목이 없습니다.")
            
            with t2:
                cols = ['종목명', '섹터', 'Final_PnL']
                # df_last 전체 사용 (매도된 종목 포함)
                pnl_rank = df_last.sort_values('Final_PnL', ascending=False)[cols]
                cw, cl = st.columns(2)
                cw.success("Top 5 Winners")
                cw.dataframe(pnl_rank.head(5).style.format({'Final_PnL': '{:,.0f}'}))
                cl.error("Top 5 Losers")
                cl.dataframe(pnl_rank.tail(5).style.format({'Final_PnL': '{:,.0f}'}))