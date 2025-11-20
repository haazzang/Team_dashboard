import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import datetime

# --- 페이지 설정 ---
st.set_page_config(page_title="Portfolio Dashboard", layout="wide")
st.title("🚀 Team Portfolio Analysis Dashboard")

# ==============================================================================
# [Helper Functions] 공통 함수 모음
# ==============================================================================

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

# [HTML Table Generator] 표 밀림 방지용 (Team PNL 탭에서 사용)
def create_manual_html_table(df, title=None):
    html = ''
    if title:
        html += f'<h5 style="margin-top:20px; margin-bottom:10px;">{title}</h5>'
    html += '<div style="overflow-x:auto;"><table style="width:100%; border-collapse: collapse; font-size: 0.9rem;">'
    
    # Header
    html += '<thead style="background-color: #f8f9fa; border-bottom: 2px solid #dee2e6;"><tr>'
    for col in df.columns:
        html += f'<th style="padding: 12px; text-align: center; white-space: nowrap; color: #212529;">{col}</th>'
    html += '</tr></thead>'
    
    # Body
    html += '<tbody>'
    for _, row in df.iterrows():
        html += '<tr style="border-bottom: 1px solid #dee2e6;">'
        for i, val in enumerate(row):
            align = 'left' if i == 0 else 'right'
            color = 'inherit'
            weight = 'normal'
            
            val_str = str(val)
            if '%' in val_str:
                if '-' in val_str: color = '#dc3545' # Red
                else: color = '#198754' # Green
            
            html += f'<td style="padding: 10px; text-align: {align}; color: {color}; font-weight: {weight}; white-space: nowrap;">{val}</td>'
        html += '</tr>'
    html += '</tbody></table></div>'
    return html

# ==============================================================================
# [PART 1] Total Portfolio (Team PNL) 데이터 로드
# ==============================================================================
@st.cache_data
def load_team_pnl_data(file):
    try:
        # PNL Sheet
        df_pnl_raw = pd.read_excel(file, sheet_name='PNL', header=None, engine='openpyxl')
        h_idx = -1
        for i in range(15):
            if '일자' in [str(x).strip() for x in df_pnl_raw.iloc[i].values]:
                h_idx = i; break
        if h_idx == -1: return None, None, "PNL 시트 헤더(일자)를 찾을 수 없습니다."
        
        # 컬럼명 설정 (중복 방지 및 순서 유지)
        raw_cols = df_pnl_raw.iloc[h_idx].tolist()
        new_cols = []
        seen = {}
        for c in raw_cols:
            c_str = str(c).strip()
            if c_str in ['nan', 'None', '']: continue
            if c_str in seen: seen[c_str] += 1; new_cols.append(f"{c_str}_{seen[c_str]}")
            else: seen[c_str] = 0; new_cols.append(c_str)
            
        df_pnl = df_pnl_raw.iloc[h_idx+1:].copy()
        # 데이터 컬럼 개수와 헤더 개수 맞추기 (빈 컬럼 제외)
        valid_indices = [i for i, c in enumerate(df_pnl_raw.iloc[h_idx]) if str(c).strip() not in ['nan', 'None', '']]
        df_pnl = df_pnl.iloc[:, valid_indices]
        df_pnl.columns = new_cols

        # Date Index
        date_col = [c for c in df_pnl.columns if '일자' in c][0]
        df_pnl = df_pnl.set_index(date_col)
        df_pnl.index = pd.to_datetime(df_pnl.index, errors='coerce')
        df_pnl = df_pnl.dropna(how='all')
        df_pnl = df_pnl.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Position Sheet (같은 로직)
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
        df_pos = df_pos.dropna(how='all')
        df_pos = df_pos.apply(pd.to_numeric, errors='coerce').fillna(0)

        return df_pnl, df_pos, None

    except Exception as e:
        return None, None, f"Team_PNL 로드 오류: {e}"

# ==============================================================================
# [PART 2] Cash Equity 데이터 로드 (수정된 로직)
# ==============================================================================
@st.cache_data
def load_cash_equity_data(file):
    debug_logs = []
    try:
        xls = pd.ExcelFile(file, engine='openpyxl')
        all_holdings = []
        df_hedge = pd.DataFrame()
        
        for sheet in xls.sheet_names:
            # Hedge Sheet
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
                            daily_hedge = df_h[col_cum].diff().fillna(0)
                            if df_hedge.empty: df_hedge = daily_hedge.to_frame(name='Hedge_PnL_KRW')
                            else: df_hedge = df_hedge.add(daily_hedge.to_frame(name='Hedge_PnL_KRW'), fill_value=0)
                except: pass
            
            # Equity Sheet
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

        if not all_holdings: return None, None, None, "Holdings 데이터 없음"

        eq = pd.concat(all_holdings, ignore_index=True)
        eq['기준일자'] = pd.to_datetime(eq['기준일자'], errors='coerce')
        eq = eq.dropna(subset=['기준일자'])
        
        # 숫자 변환
        for c in ['원화평가금액', '원화총평가손익', '원화총매매손익', '잔고수량', 'Market Price']:
            if c in eq.columns: eq[c] = pd.to_numeric(eq[c], errors='coerce').fillna(0)

        # 섹터 매핑 (최우선)
        id_col = '심볼' if '심볼' in eq.columns else '종목코드'
        if '섹터' not in eq.columns:
            if '심볼' in eq.columns:
                uniques = eq['심볼'].dropna().unique()
                sec_map = fetch_sectors_cached(tuple(uniques))
                eq['섹터'] = eq['심볼'].map(sec_map).fillna('Unknown')
            else: eq['섹터'] = 'Unknown'

        # 정렬 및 전일 데이터 계산
        eq = eq.sort_values([id_col, '기준일자'])
        eq['Prev_Qty'] = eq.groupby(id_col)['잔고수량'].shift(1).fillna(0)
        eq['Prev_MV'] = eq.groupby(id_col)['원화평가금액'].shift(1).fillna(0)
        
        # 일별 PnL 집계 (Total Cum PnL Diff 방식)
        eq['Total_Cum_PnL_Stock'] = eq['원화총평가손익'] + eq['원화총매매손익']
        
        daily_agg = eq.groupby('기준일자').agg({
            'Total_Cum_PnL_Stock': 'sum',
            '원화평가금액': 'sum'
        })
        daily_agg['Daily_PnL_KRW'] = daily_agg['Total_Cum_PnL_Stock'].diff().fillna(0)
        daily_agg['Prev_MV'] = daily_agg['원화평가금액'].shift(1)
        
        # Hedge 병합
        df_perf = daily_agg.join(df_hedge, how='outer').fillna(0)
        df_perf['Total_PnL_KRW'] = df_perf['Daily_PnL_KRW'] + df_perf['Hedge_PnL_KRW']
        
        # 수익률
        df_perf['Ret_Equity'] = np.where(df_perf['Prev_MV'] > 0, df_perf['Daily_PnL_KRW'] / df_perf['Prev_MV'], 0)
        df_perf['Ret_Total'] = np.where(df_perf['Prev_MV'] > 0, df_perf['Total_PnL_KRW'] / df_perf['Prev_MV'], 0)
        
        df_perf = df_perf.iloc[1:] # 첫날 제거
        
        df_perf['Cum_Equity'] = (1 + df_perf['Ret_Equity']).cumprod() - 1
        df_perf['Cum_Total'] = (1 + df_perf['Ret_Total']).cumprod() - 1

        # 마지막 보유 현황
        df_last = eq.sort_values('기준일자').groupby(id_col).tail(1)
        df_last['Final_PnL'] = df_last['원화총평가손익'] + df_last['원화총매매손익']

        return df_perf, df_last, debug_logs, None

    except Exception as e:
        return None, None, None, f"Load Error: {e}"

# ==============================================================================
# [MAIN UI] 사이드바 및 탭 구성
# ==============================================================================

menu = st.sidebar.radio("Dashboard Menu", ["Total Portfolio (Team PNL)", "Cash Equity Analysis"])

# ------------------------------------------------------------------------------
# [MENU 1] Total Portfolio (Team PNL) - 복구됨
# ------------------------------------------------------------------------------
if menu == "Total Portfolio (Team PNL)":
    st.subheader("📊 Total Team Portfolio Dashboard")
    uploaded_file = st.sidebar.file_uploader("Upload 'Team_PNL.xlsx'", type=['xlsx'], key="pnl")
    
    if uploaded_file:
        df_pnl, df_pos, err = load_team_pnl_data(uploaded_file)
        if df_pnl is not None:
            # 공통 데이터
            common_idx = df_pnl.index.intersection(df_pos.index)
            common_cols = [c for c in df_pnl.columns if c in df_pos.columns] # PNL 기준 순서
            
            df_pnl = df_pnl.loc[common_idx, common_cols]
            df_pos = df_pos.loc[common_idx, common_cols]
            
            # 지표 계산
            df_cum_pnl = df_pnl.cumsum()
            df_user_ret = df_cum_pnl.div(df_pos.replace(0, np.nan)).fillna(0)
            df_daily_ret = df_pnl.div(df_pos.replace(0, np.nan)).fillna(0)
            
            # 벤치마크
            with st.spinner("Fetching Cross Assets..."):
                df_assets = download_cross_assets(df_pnl.index.min(), df_pnl.index.max())
                bm_cum = pd.DataFrame(index=df_user_ret.index)
                if not df_assets.empty:
                    df_assets = df_assets.reindex(df_user_ret.index, method='ffill')
                    df_asset_ret = df_assets.pct_change().fillna(0)
                    if 'S&P 500' in df_assets.columns: bm_cum['SPX'] = (1 + df_asset_ret['S&P 500']).cumprod() - 1
                    if 'KOSPI' in df_assets.columns: bm_cum['KOSPI'] = (1 + df_asset_ret['KOSPI']).cumprod() - 1

            # TABS
            t1, t2, t3, t4, t5 = st.tabs(["📈 Chart", "📊 Analysis", "🔗 Correlation", "🌍 Cross Asset", "🧪 Simulation"])
            
            # 1. Chart
            with t1:
                strat = st.selectbox("Select Strategy", df_user_ret.columns)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_user_ret.index, y=df_user_ret[strat], name=strat, line=dict(width=2)))
                
                # BM Auto-Match
                bm_name = 'SPX' if any(k in strat for k in ['해외', 'Global', 'US']) else 'KOSPI'
                if bm_name in bm_cum.columns:
                    fig.add_trace(go.Scatter(x=df_user_ret.index, y=bm_cum[bm_name], name=bm_name, line=dict(color='grey', dash='dash')))
                
                st.plotly_chart(fig, use_container_width=True)
                
            # 2. Analysis (Manual Table)
            with t2:
                # Stats Calculation
                stats = pd.DataFrame(index=df_daily_ret.columns)
                stats['Volatility'] = df_daily_ret.std() * np.sqrt(252)
                stats['Sharpe'] = (df_daily_ret.mean() / df_daily_ret.std() * np.sqrt(252)).fillna(0)
                nav = (1 + df_daily_ret).cumprod()
                stats['MDD'] = ((nav - nav.cummax()) / nav.cummax()).min()
                stats['Total Return'] = df_user_ret.iloc[-1]
                
                # Formatting
                disp = stats.copy()
                for c in disp.columns:
                    if c == 'Sharpe': disp[c] = disp[c].apply(lambda x: f"{x:.2f}")
                    else: disp[c] = disp[c].apply(lambda x: f"{x:.2%}")
                
                disp.insert(0, 'Strategy', disp.index)
                disp['Strategy'] = disp['Strategy'].apply(lambda x: x.split('_')[0])
                
                st.markdown(create_manual_html_table(disp), unsafe_allow_html=True)

            # 3. Correlation
            with t3:
                corr = df_daily_ret.corr()
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr.values, x=corr.columns, y=corr.index,
                    colorscale='RdBu', zmin=-1, zmax=1, text=np.round(corr.values, 2), texttemplate="%{text}"
                ))
                fig_corr.update_layout(height=700)
                st.plotly_chart(fig_corr)
                
            # 4. Cross Asset
            with t4:
                if not df_assets.empty:
                    comb = pd.concat([df_daily_ret, df_asset_ret], axis=1).corr()
                    sub_corr = comb.loc[df_daily_ret.columns, df_asset_ret.columns]
                    fig_cross = go.Figure(data=go.Heatmap(
                        z=sub_corr.values, x=sub_corr.columns, y=sub_corr.index,
                        colorscale='RdBu', zmin=-1, zmax=1, text=np.round(sub_corr.values, 2), texttemplate="%{text}"
                    ))
                    st.plotly_chart(fig_cross)

            # 5. Simulation
            with t5:
                st.subheader("Portfolio Simulation")
                c_input, c_out = st.columns([1, 3])
                with c_input:
                    weights = {}
                    for col in df_daily_ret.columns:
                        weights[col] = st.slider(col, 0.0, 1.0, 1.0/len(df_daily_ret.columns), 0.05)
                
                with c_out:
                    w_series = pd.Series(weights)
                    sim_daily = df_daily_ret.mul(w_series, axis=1).sum(axis=1)
                    sim_cum = (1 + sim_daily).cumprod() - 1
                    
                    fig_sim = go.Figure()
                    fig_sim.add_trace(go.Scatter(x=sim_cum.index, y=sim_cum, name="Simulated", line=dict(color='red')))
                    # Actual Portfolio (Sum of PnL / Sum of Pos)
                    act_daily = df_pnl.sum(axis=1).div(df_pos.sum(axis=1)).fillna(0)
                    act_cum = (1 + act_daily).cumprod() - 1
                    fig_sim.add_trace(go.Scatter(x=act_cum.index, y=act_cum, name="Actual", line=dict(color='black', dash='dot')))
                    
                    st.plotly_chart(fig_sim, use_container_width=True)

        else:
            st.error(err)

# ------------------------------------------------------------------------------
# [MENU 2] Cash Equity Analysis - 수정된 로직
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
            last_row = df_perf.iloc[-1]
            
            # Summary
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Return (Hedged)", f"{last_row['Cum_Total']:.2%}")
            c2.metric("Equity Return", f"{last_row['Cum_Equity']:.2%}")
            c3.metric("Hedge Effect", f"{(last_row['Cum_Total'] - last_row['Cum_Equity']):.2%}")
            c4.metric("Current Equity AUM", f"{last_row['원화평가금액']:,.0f} KRW")
            
            # Chart
            st.subheader("Cumulative Return")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_perf.index, y=df_perf['Cum_Total'], name='Total (Hedged)', line=dict(color='#2563eb', width=3)))
            fig.add_trace(go.Scatter(x=df_perf.index, y=df_perf['Cum_Equity'], name='Equity Only', line=dict(color='#60a5fa', dash='dot')))
            
            bm_df = download_benchmark(df_perf.index.min(), df_perf.index.max())
            if not bm_df.empty:
                bm_cum = (1 + bm_df.reindex(df_perf.index, method='ffill').pct_change().fillna(0)).cumprod() - 1
                if '^GSPC' in bm_cum.columns: fig.add_trace(go.Scatter(x=bm_cum.index, y=bm_cum['^GSPC'], name='S&P 500', line=dict(color='grey', dash='dash')))
                if '^KS11' in bm_cum.columns: fig.add_trace(go.Scatter(x=bm_cum.index, y=bm_cum['^KS11'], name='KOSPI', line=dict(color='silver', dash='dash')))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Details
            t1, t2 = st.tabs(["Sector Allocation", "Top Movers"])
            with t1:
                curr_hold = df_last[df_last['잔고수량'] > 0]
                if not curr_hold.empty:
                    pie = px.pie(curr_hold, values='원화평가금액', names='섹터', title="Current Sector Exposure", hole=0.4)
                    st.plotly_chart(pie)
            with t2:
                cols = ['종목명', '섹터', 'Final_PnL']
                df_pnl = df_last[cols].sort_values('Final_PnL', ascending=False)
                cw, cl = st.columns(2)
                cw.success("Top 5 Winners")
                cw.dataframe(df_pnl.head(5).style.format({'Final_PnL': '{:,.0f}'}))
                cl.error("Top 5 Losers")
                cl.dataframe(df_pnl.tail(5).sort_values('Final_PnL').style.format({'Final_PnL': '{:,.0f}'}))

            with st.expander("Daily Data"):
                st.dataframe(df_perf.style.format("{:.4%}", subset=['Ret_Equity', 'Ret_Total', 'Cum_Equity', 'Cum_Total']))