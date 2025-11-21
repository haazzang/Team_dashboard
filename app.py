import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from scipy import stats

# --- Page Config ---
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
def download_benchmarks_all(start_date, end_date):
    """Download benchmarks for US, KR, HK, JP"""
    tickers = {'US': '^GSPC', 'KR': '^KS11', 'HK': '^HSI', 'JP': '^N225'}
    try:
        data = yf.download(list(tickers.values()), start=start_date, end=end_date + pd.Timedelta(days=5), progress=False)
        if 'Adj Close' in data.columns: df = data['Adj Close']
        elif 'Close' in data.columns: df = data['Close']
        else: df = data
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        inv_map = {v: k for k, v in tickers.items()}
        df.rename(columns=inv_map, inplace=True)
        return df.ffill().pct_change().fillna(0)
    except:
        return pd.DataFrame()

@st.cache_data
def download_usdkrw(start_date, end_date):
    """Download USD/KRW Exchange Rate"""
    try:
        fx = yf.download('KRW=X', start=start_date, end=end_date + pd.Timedelta(days=5), progress=False)
        if 'Adj Close' in fx.columns: fx = fx['Adj Close']
        elif 'Close' in fx.columns: fx = fx['Close']
        
        if isinstance(fx.columns, pd.MultiIndex): fx.columns = fx.columns.get_level_values(0)
        
        if isinstance(fx, pd.Series): 
            fx = fx.to_frame(name='USD_KRW')
        else: 
            fx.rename(columns={'KRW=X': 'USD_KRW'}, inplace=True)
            if 'USD_KRW' not in fx.columns and not fx.empty:
                 fx.columns = ['USD_KRW']
        
        return fx.ffill()
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

def calculate_alpha_beta(port_ret, bench_ret):
    """Calculate Alpha/Beta using daily returns"""
    try:
        df = pd.concat([port_ret, bench_ret], axis=1).dropna()
        df.columns = ['Port', 'Bench']
        if len(df) < 10: return np.nan, np.nan, np.nan
        slope, intercept, r_value, p_value, std_err = stats.linregress(df['Bench'], df['Port'])
        return intercept * 252, slope, r_value**2
    except:
        return np.nan, np.nan, np.nan

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
# [PART 1] Team PNL Load
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
# [PART 2] Cash Equity Load
# ==============================================================================
@st.cache_data
def load_cash_equity_data(file):
    debug_logs = []
    try:
        xls = pd.ExcelFile(file, engine='openpyxl')
        all_holdings = []
        df_hedge = pd.DataFrame()
        
        # 1. Load Sheets
        for sheet in xls.sheet_names:
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

        if not all_holdings: return None, None, None, None, None, "Holdings 데이터 없음"

        eq = pd.concat(all_holdings, ignore_index=True)
        eq['기준일자'] = pd.to_datetime(eq['기준일자'], errors='coerce')
        eq = eq.dropna(subset=['기준일자'])
        
        # Map Column Names (심볼 -> Ticker)
        rename_map = {'평가단가': 'Market_Price', '외화평가금액': 'Market_Value', '심볼': 'Ticker', '원화평가금액': 'MV_KRW'}
        eq.rename(columns=rename_map, inplace=True)
        
        cols_num = ['MV_KRW', '원화총평가손익', '원화총매매손익', '잔고수량', 'Market_Price', 'Market_Value', '외화평가손익', '외화총매매손익', '평가환율']
        for c in cols_num:
            if c in eq.columns: eq[c] = pd.to_numeric(eq[c], errors='coerce').fillna(0)

        # Ticker as ID
        id_col = 'Ticker' if 'Ticker' in eq.columns else '종목코드'
        eq['Ticker_ID'] = eq[id_col].fillna('Unknown')

        # Sector
        if '섹터' not in eq.columns:
            uniques = eq['Ticker_ID'].dropna().unique()
            sec_map = fetch_sectors_cached(tuple(uniques))
            eq['섹터'] = eq['Ticker_ID'].map(sec_map).fillna('Unknown')

        # Country
        if '통화' in eq.columns:
            curr_map = {'USD': 'US', 'HKD': 'HK', 'JPY': 'JP', 'KRW': 'KR', 'CNY': 'CN'}
            eq['Country'] = eq['통화'].map(curr_map).fillna('Other')
        else: eq['Country'] = 'Other'

        # -----------------------------------------------------------
        # [LOGIC: Price Diff Method (Referencing Script)]
        # -----------------------------------------------------------
        eq = eq.sort_values(['Ticker_ID', '기준일자'])
        
        # Prev Data
        eq['Prev_Price'] = eq.groupby('Ticker_ID')['Market_Price'].shift(1)
        eq['Prev_Qty'] = eq.groupby('Ticker_ID')['잔고수량'].shift(1).fillna(0)
        eq['Prev_MV_KRW'] = eq.groupby('Ticker_ID')['MV_KRW'].shift(1).fillna(0)
        
        # Local Return (Individual)
        eq['Stock_Ret_Local'] = np.where(eq['Prev_Price'] > 0, 
                                         (eq['Market_Price'] - eq['Prev_Price']) / eq['Prev_Price'], 0)
        
        # KRW PnL (Diff)
        eq['Cum_PnL_KRW'] = eq['원화총평가손익'] + eq['원화총매매손익']
        eq['Daily_PnL_KRW'] = eq.groupby('Ticker_ID')['Cum_PnL_KRW'].diff().fillna(0)

        # -----------------------------------------------------------
        # [AGGREGATION]
        # -----------------------------------------------------------
        daily_agg = eq.groupby('기준일자').agg({
            'Daily_PnL_KRW': 'sum',
            'MV_KRW': 'sum',
            'Prev_MV_KRW': 'sum'
        }).rename(columns={'MV_KRW': 'Total_MV_KRW', 'Prev_MV_KRW': 'Total_Prev_MV_KRW'})

        # Local Portfolio Return (Weighted Average)
        eq = eq.merge(daily_agg['Total_Prev_MV_KRW'].rename('Day_Total_Prev'), on='기준일자', how='left')
        eq['Weight'] = np.where(eq['Day_Total_Prev'] > 0, eq['Prev_MV_KRW'] / eq['Day_Total_Prev'], 0)
        daily_local_ret = eq.groupby('기준일자').apply(lambda x: (x['Stock_Ret_Local'] * x['Weight']).sum()).rename('Ret_Equity_Local')
        
        # Contribution
        eq['Contrib_KRW'] = np.where(eq['Day_Total_Prev'] > 0, eq['Daily_PnL_KRW'] / eq['Day_Total_Prev'], 0)
        contrib_sector = eq.groupby(['기준일자', '섹터'])['Contrib_KRW'].sum().reset_index()
        contrib_country = eq.groupby(['기준일자', 'Country'])['Contrib_KRW'].sum().reset_index()
        
        # Country Return for Factor Analysis
        country_daily = eq.groupby(['기준일자', 'Country']).apply(lambda x: (x['Stock_Ret_Local'] * x['Weight']).sum()).reset_index(name='Country_Ret')

        # -----------------------------------------------------------
        # [FINAL MERGE & RETURNS]
        # -----------------------------------------------------------
        df_perf = daily_agg.join(df_hedge, how='outer').fillna(0)
        df_perf = df_perf.join(daily_local_ret, how='left').fillna(0)
        
        # USD/KRW for Hedge Conversion
        usdkrw = download_usdkrw(df_perf.index.min(), df_perf.index.max())
        df_perf = df_perf.join(usdkrw, how='left').fillna(method='ffill').fillna(1400.0)
        if 'USD_KRW' not in df_perf.columns: df_perf['USD_KRW'] = 1400.0
        
        df_perf['Total_PnL_KRW'] = df_perf['Daily_PnL_KRW'] + df_perf['Hedge_PnL_KRW']
        denom_krw = df_perf['Total_Prev_MV_KRW'].replace(0, np.nan)
        
        # Returns
        df_perf['Ret_Equity_KRW'] = df_perf['Daily_PnL_KRW'] / denom_krw
        df_perf['Ret_Total_KRW'] = df_perf['Total_PnL_KRW'] / denom_krw
        # Local Return: Equity (Weighted) + Hedge (USD Contribution)
        df_perf['Hedge_PnL_USD'] = df_perf['Hedge_PnL_KRW'] / df_perf['USD_KRW']
        df_perf['Total_Prev_MV_USD'] = df_perf['Total_Prev_MV_KRW'] / df_perf['USD_KRW']
        df_perf['Ret_Hedge_Local'] = df_perf['Hedge_PnL_USD'] / df_perf['Total_Prev_MV_USD'].replace(0, np.nan)
        df_perf['Ret_Total_Local'] = df_perf['Ret_Equity_Local'] + df_perf['Ret_Hedge_Local'].fillna(0)
        
        df_perf.fillna(0, inplace=True)
        df_perf = df_perf.iloc[1:]
        
        # Cumulative
        for c in ['Ret_Equity_KRW', 'Ret_Total_KRW', 'Ret_Equity_Local', 'Ret_Total_Local']:
            df_perf[c.replace('Ret', 'Cum')] = (1 + df_perf[c]).cumprod() - 1
            
        df_last = eq.sort_values('기준일자').groupby('Ticker_ID').tail(1)
        df_last['Final_PnL'] = df_last['원화총평가손익'] + df_last['원화총매매손익']
        
        return df_perf, df_last, {'Sector': contrib_sector, 'Country': contrib_country}, country_daily, debug_logs, None

    except Exception as e: return None, None, None, None, None, f"Error: {e}"

# ==============================================================================
# [MAIN UI]
# ==============================================================================
menu = st.sidebar.radio("Dashboard Menu", ["Total Portfolio (Team PNL)", "Cash Equity Analysis"])

if menu == "Total Portfolio (Team PNL)":
    st.subheader("📊 Total Team Portfolio Dashboard")
    uploaded_file = st.sidebar.file_uploader("Upload 'Team_PNL.xlsx'", type=['xlsx'], key="pnl")
    if uploaded_file:
        df_pnl, df_pos, err = load_team_pnl_data(uploaded_file)
        if df_pnl is not None:
            common_idx = df_pnl.index.intersection(df_pos.index)
            common_cols = [c for c in df_pnl.columns if c in df_pos.columns]
            df_pnl, df_pos = df_pnl.loc[common_idx, common_cols], df_pos.loc[common_idx, common_cols]
            df_cum_pnl = df_pnl.cumsum()
            df_user_ret = df_cum_pnl.div(df_pos.replace(0, np.nan)).fillna(0)
            df_daily_ret = df_pnl.div(df_pos.replace(0, np.nan)).fillna(0)
            
            t1, t2, t3, t4, t5 = st.tabs(["📈 Chart", "📊 Analysis", "🔗 Correlation", "🌍 Cross Asset", "🧪 Simulation"])
            with t1:
                strat = st.selectbox("Select Strategy", df_user_ret.columns)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_user_ret.index, y=df_user_ret[strat], name=strat))
                st.plotly_chart(fig, use_container_width=True)
            with t2:
                stats = pd.DataFrame(index=df_daily_ret.columns)
                stats['Volatility'] = df_daily_ret.std() * np.sqrt(252)
                stats['Sharpe'] = (df_daily_ret.mean() / df_daily_ret.std() * np.sqrt(252)).fillna(0)
                stats['MDD'] = ((1+df_daily_ret).cumprod() / (1+df_daily_ret).cumprod().cummax() - 1).min()
                stats['Total Return'] = df_user_ret.iloc[-1]
                disp = stats.copy()
                for c in disp.columns: disp[c] = disp[c].apply(lambda x: f"{x:.2f}" if c=='Sharpe' else f"{x:.2%}")
                disp.insert(0, 'Strategy', disp.index)
                st.markdown(create_manual_html_table(disp), unsafe_allow_html=True)
            # (Tabs 3,4,5 omitted for brevity but should be same as before)

elif menu == "Cash Equity Analysis":
    st.subheader("📈 Cash Equity Portfolio Analysis")
    uploaded_file_ce = st.sidebar.file_uploader("Upload 'Holdings3.xlsx'", type=['xlsx'], key="ce")
    if uploaded_file_ce:
        with st.spinner("Processing..."):
            res = load_cash_equity_data(uploaded_file_ce)
            df_perf, df_last, df_contrib, country_daily, logs, err = res
        
        if err: st.error(err)
        elif df_perf is not None:
            bm_returns = download_benchmarks_all(df_perf.index.min(), df_perf.index.max())
            view_opt = st.radio("Currency View", ["KRW", "Local Currency (USD Base)"], horizontal=True)
            last_day = df_perf.iloc[-1]
            curr_aum = df_perf.iloc[-1]['Total_MV_KRW'] if 'Total_MV_KRW' in df_perf.columns else 0
            
            c1, c2, c3, c4 = st.columns(4)
            if view_opt == "KRW":
                c1.metric("Total Return (Hedged)", f"{last_day['Cum_Total_KRW']:.2%}")
                c2.metric("Equity Return (Unhedged)", f"{last_day['Cum_Equity_KRW']:.2%}")
                c3.metric("Hedge Impact", f"{(last_day['Cum_Total_KRW'] - last_day['Cum_Equity_KRW']):.2%}")
                y_main, y_sub = 'Cum_Total_KRW', 'Cum_Equity_KRW'
                name_main, name_sub = 'Total (Hedged)', 'Equity (KRW)'
            else:
                c1.metric("Total Return (Hedged)", f"{last_day['Cum_Total_Local']:.2%}")
                c2.metric("Equity Return (Unhedged)", f"{last_day['Cum_Equity_Local']:.2%}")
                c3.metric("Hedge Impact", f"{(last_day['Cum_Total_Local'] - last_day['Cum_Equity_Local']):.2%}")
                y_main, y_sub = 'Cum_Total_Local', 'Cum_Equity_Local'
                name_main, name_sub = 'Total (Hedged)', 'Equity (Local/USD)'
            c4.metric("Current AUM", f"{curr_aum:,.0f} KRW")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_perf.index, y=df_perf[y_main], name=name_main, line=dict(color='#2563eb', width=3)))
            if y_sub: fig.add_trace(go.Scatter(x=df_perf.index, y=df_perf[y_sub], name=name_sub, line=dict(color='#60a5fa', dash='dot')))
            if not bm_returns.empty:
                bm_cum = (1 + bm_returns).cumprod() - 1
                for col in ['US', 'KR', 'HK', 'JP']:
                    if col in bm_cum.columns: fig.add_trace(go.Scatter(x=bm_cum.index, y=bm_cum[col], name=col+' BM', line=dict(width=1, dash='dash')))
            st.plotly_chart(fig, use_container_width=True)
            
            t1, t2, t3 = st.tabs(["Factor Analysis", "Attribution", "Holdings"])
            with t1:
                st.markdown("#### 🌍 Country-Level Alpha & Beta")
                if not country_daily.empty and not bm_returns.empty:
                    results = []
                    for ctry in country_daily['Country'].unique():
                        # Match country to BM
                        bm_ticker = {'US':'US','KR':'KR','HK':'HK','JP':'JP'}.get(ctry)
                        if bm_ticker and bm_ticker in bm_returns.columns:
                            # Reindex country return to match BM dates (daily)
                            port_series = country_daily[country_daily['Country']==ctry].set_index('기준일자')['Country_Ret']
                            bench_series = bm_returns[bm_ticker]
                            alpha, beta, r2 = calculate_alpha_beta(port_series, bench_series)
                            results.append({'Country': ctry, 'Benchmark': bm_ticker, 'Alpha': f"{alpha:.2%}", 'Beta': f"{beta:.2f}", 'R2': f"{r2:.2f}"})
                    if results: st.table(pd.DataFrame(results).set_index('Country'))
                    else: st.write("Insufficient data.")
            
            with t2:
                if df_contrib:
                    c_a, c_b = st.columns(2)
                    with c_a:
                        if not df_contrib['Country'].empty:
                            ctry_cont = df_contrib['Country'].groupby('Country')['Contrib_KRW'].sum().reset_index()
                            st.plotly_chart(px.bar(ctry_cont, x='Contrib_KRW', y='Country', orientation='h', title="Contribution by Country", text_auto='.2%'))
                    with c_b:
                        if not df_contrib['Sector'].empty:
                            sec_cont = df_contrib['Sector'].groupby('섹터')['Contrib_KRW'].sum().reset_index()
                            st.plotly_chart(px.bar(sec_cont, x='Contrib_KRW', y='섹터', orientation='h', title="Contribution by Sector", text_auto='.2%'))
            
            with t3:
                st.dataframe(df_last[['종목명','섹터','Country','Final_PnL']].sort_values('Final_PnL', ascending=False))
                with st.expander("Daily Data"): st.dataframe(df_perf)