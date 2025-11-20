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
# [Helper Functions]
# ==============================================================================

@st.cache_data
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
# [PART 1] Team PNL Load (ORIGINAL CODE - UNTOUCHED)
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
# [PART 2] Cash Equity Load (FIXED LOGIC: Full Grid + Weighted Local Return)
# ==============================================================================
@st.cache_data
def load_cash_equity_data(file):
    debug_logs = []
    try:
        xls = pd.ExcelFile(file, engine='openpyxl')
        all_holdings = []
        df_hedge = pd.DataFrame()
        
        # 1. Load All Sheets
        for sheet in xls.sheet_names:
            # [A] Hedge
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
            
            # [B] Equity
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

        eq = pd.concat(all_holdings, ignore_index=True)
        eq['기준일자'] = pd.to_datetime(eq['기준일자'], errors='coerce')
        eq = eq.dropna(subset=['기준일자'])
        
        # Rename
        rename_map = {
            '평가단가': 'Market Price',
            '종목코드': 'Ticker', '심볼': 'Symbol'
        }
        eq.rename(columns=rename_map, inplace=True)
        
        # Numeric Conversion
        cols_num = ['원화평가금액', '원화총평가손익', '원화총매매손익', '잔고수량', 'Market Price', 'Market Value', '외화평가손익', '외화총매매손익', '평가환율', '환손익']
        for c in cols_num:
            if c in eq.columns: eq[c] = pd.to_numeric(eq[c], errors='coerce').fillna(0)

        # ID
        id_col = 'Ticker' if 'Ticker' in eq.columns else 'Symbol'
        if id_col not in eq.columns: id_col = '종목명'
        eq['Ticker_ID'] = eq[id_col].fillna('Unknown')

        # Sector
        if '섹터' not in eq.columns:
            if 'Symbol' in eq.columns:
                uniques = eq['Symbol'].dropna().unique()
                sec_map = fetch_sectors_cached(tuple(uniques))
                eq['섹터'] = eq['Symbol'].map(sec_map).fillna('Unknown')
            else: eq['섹터'] = 'Unknown'
        else: eq['섹터'] = eq['섹터'].fillna('Unknown')

        # -----------------------------------------------------------
        # [FULL GRID LOGIC] Fix missing rows after Sell
        # -----------------------------------------------------------
        all_dates = sorted(eq['기준일자'].unique())
        all_tickers = eq['Ticker_ID'].unique()
        
        # Create Cartesian Product (Date x Ticker)
        idx = pd.MultiIndex.from_product([all_dates, all_tickers], names=['기준일자', 'Ticker_ID'])
        grid = pd.DataFrame(index=idx).reset_index()
        
        # Merge Actual Data
        # Drop duplicates to ensure 1 row per date-ticker
        eq_dedup = eq.drop_duplicates(subset=['기준일자', 'Ticker_ID'])
        cols_to_keep = ['원화평가금액', '원화총평가손익', '원화총매매손익', '외화평가금액', '외화평가손익', '외화총매매손익', '환손익', '통화', '섹터', '종목명']
        cols_to_keep = [c for c in cols_to_keep if c in eq_dedup.columns]
        
        merged = pd.merge(grid, eq_dedup[['기준일자', 'Ticker_ID'] + cols_to_keep], on=['기준일자', 'Ticker_ID'], how='left')
        merged = merged.sort_values(['Ticker_ID', '기준일자'])

        # --- FILLING LOGIC ---
        # 1. Realized PnL: Forward Fill (Maintain cumulative profit after exit)
        for c in ['원화총매매손익', '외화총매매손익']:
            if c in merged.columns:
                merged[c] = merged.groupby('Ticker_ID')[c].ffill().fillna(0)
        
        # 2. Unrealized PnL & MV: Fill 0 (No position = No unrealized)
        for c in ['원화평가손익', '외화평가손익', '원화평가금액', '외화평가금액', '원화총평가손익', '환손익']:
            if c in merged.columns:
                merged[c] = merged[c].fillna(0)
        
        # 3. Static Info: Forward Fill (Currency, Sector, Name)
        for c in ['통화', '섹터', '종목명']:
            if c in merged.columns:
                merged[c] = merged.groupby('Ticker_ID')[c].ffill().fillna('Unknown')

        # -----------------------------------------------------------
        # [RETURN CALCULATION]
        # -----------------------------------------------------------

        # 1. Ticker-level KRW cumulative and daily PnL
        merged['Cum_PnL_KRW'] = merged['원화총평가손익'] + merged['원화총매매손익']
        merged['Daily_PnL_KRW'] = merged.groupby('Ticker_ID')['Cum_PnL_KRW'].diff().fillna(0)

        # 2. Ticker-level FX cumulative and daily PnL (in KRW)
        if '환손익' in merged.columns:
            merged['Cum_FX_KRW'] = merged.groupby('Ticker_ID')['환손익'].ffill().fillna(0)
            merged['Daily_FX_KRW'] = merged.groupby('Ticker_ID')['Cum_FX_KRW'].diff().fillna(0)
        else:
            merged['Daily_FX_KRW'] = 0.0

        # 3. Portfolio-level aggregates per day
        daily_agg = merged.groupby('기준일자').agg({
            'Daily_PnL_KRW': 'sum',
            'Daily_FX_KRW': 'sum',
            '원화평가금액': 'sum'
        }).rename(columns={
            '원화평가금액': 'Total_MV_KRW'
        }).sort_index()

        # Previous-day portfolio MV in KRW
        daily_agg['Total_Prev_MV_KRW'] = daily_agg['Total_MV_KRW'].shift(1)

        # 4. Join hedge PnL (already in KRW)
        if not df_hedge.empty:
            tmp_hedge = df_hedge.copy()
            tmp_hedge.index = pd.to_datetime(tmp_hedge.index)
            daily_agg = daily_agg.join(tmp_hedge, how='left')
        else:
            daily_agg['Hedge_PnL_KRW'] = 0.0

        daily_agg['Hedge_PnL_KRW'] = daily_agg['Hedge_PnL_KRW'].fillna(0)

        # 5. Build performance DataFrame
        df_perf = daily_agg.copy()

        # Total PnL in KRW (Equity + Hedge)
        df_perf['Total_PnL_KRW'] = df_perf['Daily_PnL_KRW'] + df_perf['Hedge_PnL_KRW']

        # 6. Returns (denominator = previous day MV in KRW)
        denom = df_perf['Total_Prev_MV_KRW'].replace(0, np.nan)

        # Equity return in KRW
        df_perf['Ret_Equity_KRW'] = df_perf['Daily_PnL_KRW'] / denom

        # Total return in KRW (Equity + Hedge)
        df_perf['Ret_Total_KRW'] = df_perf['Total_PnL_KRW'] / denom

        # FX Impact (daily contribution from FX only)
        df_perf['FX_Impact_Daily'] = df_perf['Daily_FX_KRW'] / denom

        # Local equity (unhedged): KRW equity return minus FX contribution
        df_perf['Ret_Equity_Local_Unhedged'] = df_perf['Ret_Equity_KRW'] - df_perf['FX_Impact_Daily']

        # Local total (hedged): local equity + hedge PnL (hedge logic same as KRW)
        df_perf['Ret_Total_Local_Hedged'] = df_perf['Ret_Equity_Local_Unhedged'] + (df_perf['Hedge_PnL_KRW'] / denom)

        # Backward-compatible daily local equity return
        df_perf['Ret_Equity_Local'] = df_perf['Ret_Equity_Local_Unhedged']

        # 7. Drop first day (no previous MV) and fill NaN with 0
        df_perf = df_perf.iloc[1:].fillna(0)

        # 8. Cumulative returns
        df_perf['Cum_Equity_KRW'] = (1 + df_perf['Ret_Equity_KRW']).cumprod() - 1
        df_perf['Cum_Total_KRW'] = (1 + df_perf['Ret_Total_KRW']).cumprod() - 1
        df_perf['Cum_Equity_Local_Unhedged'] = (1 + df_perf['Ret_Equity_Local_Unhedged']).cumprod() - 1
        df_perf['Cum_Total_Local_Hedged'] = (1 + df_perf['Ret_Total_Local_Hedged']).cumprod() - 1
        # Backward-compatible alias so existing UI can use Cum_Equity_Local
        df_perf['Cum_Equity_Local'] = df_perf['Cum_Equity_Local_Unhedged']


        
        # Last Status (for details)
        df_last = eq.sort_values('기준일자').groupby('Ticker_ID').tail(1)
        df_last['Final_PnL'] = df_last['원화총평가손익'] + df_last['원화총매매손익']
        
        return df_perf, df_last, debug_logs, None

    except Exception as e:
        return None, None, None, f"Process Error: {e}"


# ==============================================================================
# [MAIN UI]
# ==============================================================================

menu = st.sidebar.radio("Dashboard Menu", ["Total Portfolio (Team PNL)", "Cash Equity Analysis"])

# --- MENU 1: Team PNL ---
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
            
            # BM
            with st.spinner("Fetching Market Data..."):
                df_assets = download_cross_assets(df_pnl.index.min(), df_pnl.index.max())
                bm_cum = pd.DataFrame(index=df_user_ret.index)
                if not df_assets.empty:
                    df_assets = df_assets.reindex(df_user_ret.index, method='ffill')
                    df_asset_ret = df_assets.pct_change().fillna(0)
                    if 'S&P 500' in df_assets.columns: bm_cum['SPX'] = (1 + df_asset_ret['S&P 500']).cumprod() - 1
                    if 'KOSPI' in df_assets.columns: bm_cum['KOSPI'] = (1 + df_asset_ret['KOSPI']).cumprod() - 1
            
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

# --- MENU 2: Cash Equity Analysis ---
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
            curr_aum = df_perf.iloc[-1]['Total_MV_KRW']
            
            c1, c2, c3, c4 = st.columns(4)
            if view_opt.startswith("KRW"):
                c1.metric("Total Return (Hedged)", f"{last_day['Cum_Total_KRW']:.2%}")
                c2.metric("Equity Return (KRW)", f"{last_day['Cum_Equity_KRW']:.2%}")
                c3.metric("Hedge Impact", f"{(last_day['Cum_Total_KRW'] - last_day['Cum_Equity_KRW']):.2%}")
                y_main, y_sub = 'Cum_Total_KRW', 'Cum_Equity_KRW'
                name_main, name_sub = 'Total (Hedged)', 'Equity (KRW)'
            else:
                c1.metric("Local Return", f"{last_day['Cum_Equity_Local']:.2%}")
                c2.metric("Equity Return (KRW)", f"{last_day['Cum_Equity_KRW']:.2%}")
                c3.metric("FX Impact", f"{(last_day['Cum_Equity_KRW'] - last_day['Cum_Equity_Local']):.2%}")
                y_main, y_sub = 'Cum_Equity_Local', None
                name_main, name_sub = 'Equity (Local)', None
            c4.metric("Current AUM", f"{curr_aum:,.0f} KRW")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_perf.index, y=df_perf[y_main], name=name_main, line=dict(color='#2563eb', width=3)))
            if y_sub: fig.add_trace(go.Scatter(x=df_perf.index, y=df_perf[y_sub], name=name_sub, line=dict(color='#60a5fa', dash='dot')))
            
            bm_df = download_benchmark(df_perf.index.min(), df_perf.index.max())
            if not bm_df.empty:
                bm_cum = (1 + bm_df.reindex(df_perf.index, method='ffill').pct_change().fillna(0)).cumprod() - 1
                if '^GSPC' in bm_cum.columns: fig.add_trace(go.Scatter(x=bm_cum.index, y=bm_cum['^GSPC'], name='S&P 500', line=dict(color='grey', dash='dash')))
                if '^KS11' in bm_cum.columns: fig.add_trace(go.Scatter(x=bm_cum.index, y=bm_cum['^KS11'], name='KOSPI', line=dict(color='silver', dash='dash')))
            st.plotly_chart(fig, use_container_width=True)

            t1, t2 = st.tabs(["Sector Allocation", "Top Movers"])
            with t1:
                max_date = df_perf.index.max()
                # df_last contains last record for all stocks.
                # Filter for those still held on max_date
                # Note: df_last is last record per ticker. If last record date < max_date, it means sold.
                # So checking date is enough.
                curr_hold = df_last[(df_last['기준일자'] == max_date) & (df_last['잔고수량'] > 0)]
                
                if not curr_hold.empty:
                    sec_grp = curr_hold.groupby('섹터')['원화평가금액'].sum().reset_index()
                    st.plotly_chart(px.pie(sec_grp, values='원화평가금액', names='섹터', title="Current Sector Exposure"))
                else: st.write("No current holdings.")
            with t2:
                pnl_df = df_last.sort_values('Final_PnL', ascending=False)[['종목명','섹터','Final_PnL']]
                cw, cl = st.columns(2)
                cw.success("Top Winners"); cw.dataframe(pnl_df.head(5).style.format({'Final_PnL':'{:,.0f}'}))
                cl.error("Top Losers"); cl.dataframe(pnl_df.tail(5).style.format({'Final_PnL':'{:,.0f}'}))
            
            with st.expander("Daily Data"):
                st.dataframe(df_perf)