import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from scipy import stats
import datetime

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
    try:
        fx = yf.download('KRW=X', start=start_date, end=end_date + pd.Timedelta(days=5), progress=False)
        if 'Adj Close' in fx.columns: fx = fx['Adj Close']
        elif 'Close' in fx.columns: fx = fx['Close']
        if isinstance(fx.columns, pd.MultiIndex): fx.columns = fx.columns.get_level_values(0)
        if isinstance(fx, pd.Series): fx = fx.to_frame(name='USD_KRW')
        else: 
            fx.rename(columns={'KRW=X': 'USD_KRW'}, inplace=True)
            if 'USD_KRW' not in fx.columns and not fx.empty: fx.columns = ['USD_KRW']
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

# --- Factor Data Download ---
@st.cache_data
def download_factors(start_date, end_date):
    factors = {
        'Global Mkt': 'ACWI', 'Value': 'VLUE', 'Growth': 'IWF', 'Momentum': 'MTUM',
        'Quality': 'QUAL', 'Low Vol': 'USMV', 'Small Cap': 'IWM', 'Emerging': 'EEM', 
        'Bond': 'TLT', 'USD': 'UUP', 'Gold': 'GLD', 'Oil': 'USO',
        'High Beta': 'SPHB', 'Meme': 'MEME', 'Spec Tech': 'ARKK'
    }
    try:
        data = yf.download(list(factors.values()), start=start_date, end=end_date + pd.Timedelta(days=5), progress=False)
        if 'Adj Close' in data.columns: df = data['Adj Close']
        elif 'Close' in data.columns: df = data['Close']
        else: df = data
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        inv_map = {v: k for k, v in factors.items()}
        df.rename(columns=inv_map, inplace=True)
        return df.ffill().pct_change().fillna(0)
    except:
        return pd.DataFrame()

def perform_factor_regression(port_ret, factor_ret):
    try:
        df = pd.concat([port_ret, factor_ret], axis=1).dropna()
        if len(df) < 5: return None, None, None
        Y, X = df.iloc[:, 0].values, df.iloc[:, 1:].values
        X_w_const = np.column_stack([np.ones(len(X)), X])
        betas, residuals, rank, s = np.linalg.lstsq(X_w_const, Y, rcond=None)
        
        names = ['Alpha'] + list(df.columns[1:])
        exposures = pd.Series(betas, index=names)
        contrib = pd.DataFrame(index=df.index)
        for i, f in enumerate(names[1:]): contrib[f] = df.iloc[:, 1+i] * betas[i+1]
        contrib['Alpha'] = betas[0]
        contrib['Unexplained'] = Y - (X_w_const @ betas)
        
        ss_tot = np.sum((Y - np.mean(Y))**2)
        ss_res = np.sum(residuals) if len(residuals) > 0 else 0
        r2 = 1 - (ss_res / ss_tot)
        return exposures, contrib, r2
    except: return None, None, None

def calculate_alpha_beta(port_ret, bench_ret):
    try:
        df = pd.concat([port_ret, bench_ret], axis=1).dropna()
        df.columns = ['Port', 'Bench']
        if len(df) < 5: return np.nan, np.nan, np.nan
        slope, intercept, r_value, p_value, std_err = stats.linregress(df['Bench'], df['Port'])
        return intercept * 252, slope, r_value**2
    except: return np.nan, np.nan, np.nan

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
        
        for sheet in xls.sheet_names:
            # Hedge
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
            # Equity
            else:
                try:
                    df = pd.read_excel(file, sheet_name=sheet, header=None, engine='openpyxl')
                    h_idx = -1
                    for i in range(15):
                        row_vals = [str(x).strip() for x in df.iloc[i].values]
                        if '기준일자' in row_vals and ('종목명' in row_vals or '종목코드' in row_vals):
                            h_idx = i; break
                    if h_idx != -1:
                        raw_cols = [str(c).strip() for c in df.iloc[h_idx]]
                        new_cols = []
                        seen = {}
                        for c in raw_cols:
                            if c in seen: seen[c] += 1; new_cols.append(f"{c}_{seen[c]}")
                            else: seen[c] = 0; new_cols.append(c)
                        df.columns = new_cols
                        df = df.iloc[h_idx+1:].copy()
                        if '기준일자' in df.columns: all_holdings.append(df)
                except: pass

        if not all_holdings: return None, None, None, None, None, None, "Holdings 데이터 없음"

        eq = pd.concat(all_holdings, ignore_index=True)
        eq['기준일자'] = pd.to_datetime(eq['기준일자'], errors='coerce')
        eq = eq.dropna(subset=['기준일자'])
        
        rename_map = {'평가단가': 'Market_Price', '외화평가금액': 'Market_Value', '종목코드': 'Ticker', '심볼': 'Symbol'}
        eq.rename(columns=rename_map, inplace=True)
        
        cols_num = ['원화평가금액', '원화총평가손익', '원화총매매손익', '잔고수량', 'Market_Price', 'Market_Value', '외화평가손익', '외화총매매손익', '평가환율']
        for c in cols_num:
            if c in eq.columns: eq[c] = pd.to_numeric(eq[c], errors='coerce').fillna(0)

        id_col = 'Ticker' if 'Ticker' in eq.columns else 'Symbol'
        if id_col not in eq.columns: id_col = '종목명'
        eq['Ticker_ID'] = eq[id_col].fillna('Unknown')

        if '섹터' not in eq.columns:
            if 'Symbol' in eq.columns:
                uniques = eq['Symbol'].dropna().unique()
                sec_map = fetch_sectors_cached(tuple(uniques))
                eq['섹터'] = eq['Symbol'].map(sec_map).fillna('Unknown')
            else: eq['섹터'] = 'Unknown'
        else: eq['섹터'] = eq['섹터'].fillna('Unknown')

        if '통화' in eq.columns:
            curr_map = {'USD': 'US', 'HKD': 'HK', 'JPY': 'JP', 'KRW': 'KR', 'CNY': 'CN'}
            eq['Country'] = eq['통화'].map(curr_map).fillna('Other')
        else: eq['Country'] = 'Other'

        # Full Grid
        all_dates = sorted(eq['기준일자'].unique())
        all_tickers = eq['Ticker_ID'].unique()
        idx = pd.MultiIndex.from_product([all_dates, all_tickers], names=['기준일자', 'Ticker_ID'])
        grid = pd.DataFrame(index=idx).reset_index()
        
        eq_dedup = eq.drop_duplicates(subset=['기준일자', 'Ticker_ID'])
        cols_keep = ['원화평가금액', '원화총평가손익', '원화총매매손익', '외화평가손익', '외화총매매손익', '통화', '섹터', 'Country', '종목명', 'Market_Value']
        cols_keep = [c for c in cols_keep if c in eq_dedup.columns]
        
        merged = pd.merge(grid, eq_dedup[['기준일자', 'Ticker_ID'] + cols_keep], on=['기준일자', 'Ticker_ID'], how='left')
        merged = merged.sort_values(['Ticker_ID', '기준일자'])

        for c in ['원화총매매손익', '외화총매매손익']:
            if c in merged.columns: merged[c] = merged.groupby('Ticker_ID')[c].ffill().fillna(0)
        for c in ['원화평가손익', '외화평가손익', '원화평가금액', 'Market_Value']:
            if c in merged.columns: merged[c] = merged[c].fillna(0)
        for c in ['통화', '섹터', 'Country', '종목명']:
            if c in merged.columns: merged[c] = merged.groupby('Ticker_ID')[c].ffill().fillna('Unknown')

        # Returns
        merged['Cum_PnL_KRW'] = merged['원화총평가손익'] + merged['원화총매매손익']
        merged['Daily_PnL_KRW'] = merged.groupby('Ticker_ID')['Cum_PnL_KRW'].diff().fillna(0)
        
        # FX
        if 'Market_Value' in merged.columns:
            merged['Implied_FX'] = np.where((merged['원화평가금액']!=0) & (merged['Market_Value']!=0),
                                            merged['원화평가금액']/merged['Market_Value'], 0)
            if '통화' in merged.columns:
                merged.loc[merged['통화']=='KRW', 'Implied_FX'] = 1.0
        else: merged['Implied_FX'] = 0
        
        fx_daily = merged[merged['Implied_FX']>0].groupby(['기준일자', '통화'])['Implied_FX'].median().reset_index().rename(columns={'Implied_FX': 'Daily_FX'})
        merged = pd.merge(merged, fx_daily, on=['기준일자', '통화'], how='left')
        merged['Daily_FX'] = merged.groupby('Ticker_ID')['Daily_FX'].ffill().fillna(1.0)
        merged['FX_Ret'] = merged.groupby('Ticker_ID')['Daily_FX'].pct_change().fillna(0)
        
        merged['Prev_MV_KRW'] = merged.groupby('Ticker_ID')['원화평가금액'].shift(1).fillna(0)
        
        # Aggregation
        daily_agg = merged.groupby('기준일자').agg({
            'Daily_PnL_KRW': 'sum',
            '원화평가금액': 'sum',
            'Prev_MV_KRW': 'sum'
        }).rename(columns={'원화평가금액': 'Total_MV_KRW', 'Prev_MV_KRW': 'Total_Prev_MV_KRW'})
        
        # Contribution Data (Safe Division)
        merged = merged.merge(daily_agg['Total_Prev_MV_KRW'].rename('Day_Total_Prev'), on='기준일자', how='left')
        
        # [FIXED] Split into multiple lines to avoid SyntaxError
        cond = merged['Day_Total_Prev'] > 0
        val = merged['Daily_PnL_KRW'] / merged['Day_Total_Prev']
        merged['Contrib_KRW'] = np.where(cond, val, 0)
        
        contrib_sector = merged.groupby(['기준일자', '섹터'])['Contrib_KRW'].sum().reset_index()
        contrib_country = merged.groupby(['기준일자', 'Country'])['Contrib_KRW'].sum().reset_index()
        
        # Portfolio FX Return
        merged['Weight'] = np.where(merged['Day_Total_Prev']>0, merged['Prev_MV_KRW']/merged['Day_Total_Prev'], 0)
        
        # [FIXED] Split lambda to avoid syntax issues
        def weighted_sum(x):
            return (x['FX_Ret'] * x['Weight']).sum()
            
        daily_fx_ret = merged.groupby('기준일자').apply(weighted_sum).rename('Port_FX_Ret')
        
        # Country Returns
        country_agg = merged.groupby(['기준일자', 'Country']).agg({
            'Daily_PnL_KRW': 'sum', 'Prev_MV_KRW': 'sum'
        }).reset_index()
        
        # [FIXED] Safe Division
        c_cond = country_agg['Prev_MV_KRW'] > 0
        c_val = country_agg['Daily_PnL_KRW'] / country_agg['Prev_MV_KRW']
        country_agg['Country_Ret'] = np.where(c_cond, c_val, 0)

        # Final Merge
        df_perf = daily_agg.join(df_hedge, how='outer').fillna(0)
        df_perf = df_perf.join(daily_fx_ret, how='left').fillna(0)
        
        min_d, max_d = df_perf.index.min(), df_perf.index.max()
        usdkrw = download_usdkrw(min_d, max_d)
        df_perf = df_perf.join(usdkrw, how='left').fillna(method='ffill').fillna(1400.0)
        if 'USD_KRW' not in df_perf.columns: df_perf['USD_KRW'] = 1400.0
        
        df_perf['Hedge_PnL_USD'] = df_perf['Hedge_PnL_KRW'] / df_perf['USD_KRW']
        df_perf['Total_Prev_MV_USD'] = df_perf['Total_Prev_MV_KRW'] / df_perf['USD_KRW']
        
        df_perf['Total_PnL_KRW'] = df_perf['Daily_PnL_KRW'] + df_perf['Hedge_PnL_KRW']
        
        denom_krw = df_perf['Total_Prev_MV_KRW'].replace(0, np.nan)
        denom_usd = df_perf['Total_Prev_MV_USD'].replace(0, np.nan)
        
        df_perf['Ret_Equity_KRW'] = df_perf['Daily_PnL_KRW'] / denom_krw
        df_perf['Ret_Total_KRW'] = df_perf['Total_PnL_KRW'] / denom_krw
        
        df_perf['Ret_Equity_Local'] = (1 + df_perf['Ret_Equity_KRW'].fillna(0)) / (1 + df_perf['Port_FX_Ret'].fillna(0)) - 1
        df_perf['Ret_Hedge_Local'] = df_perf['Hedge_PnL_USD'] / denom_usd
        df_perf['Ret_Total_Local'] = df_perf['Ret_Equity_Local'] + df_perf['Ret_Hedge_Local'].fillna(0)
        
        df_perf.fillna(0, inplace=True)
        df_perf = df_perf.iloc[1:]
        
        for c in ['Ret_Equity_KRW', 'Ret_Total_KRW', 'Ret_Equity_Local', 'Ret_Total_Local']:
            df_perf[c.replace('Ret', 'Cum')] = (1 + df_perf[c]).cumprod() - 1
            
        df_last = eq.sort_values('기준일자').groupby('Ticker_ID').tail(1)
        df_last['Final_PnL'] = df_last['원화총평가손익'] + df_last['원화총매매손익']
        
        # Return 7 Values
        return df_perf, df_last, {'Sector':contrib_sector, 'Country':contrib_country}, country_agg, merged, debug_logs, None

    except Exception as e:
        return None, None, None, None, None, None, f"Process Error: {e}"


# ==============================================================================
# [MAIN UI]
# ==============================================================================

menu = st.sidebar.radio("Dashboard Menu", ["Total Portfolio (Team PNL)", "Cash Equity Analysis", "📑 Weekly Report Generator"])

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
            
            t1, t2, t3, t4, t5 = st.tabs(["📈 Chart", "📊 Analysis", "🔗 Correlation", "🌍 Cross Asset", "🧪 Simulation"])
            
            with t1:
                strat = st.selectbox("Select Strategy", df_user_ret.columns)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_user_ret.index, y=df_user_ret[strat], name=strat, line=dict(width=2)))
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
                if not download_cross_assets(df_pnl.index.min(), df_pnl.index.max()).empty:
                    df_assets = download_cross_assets(df_pnl.index.min(), df_pnl.index.max())
                    df_assets = df_assets.reindex(df_user_ret.index, method='ffill')
                    df_asset_ret = df_assets.pct_change().fillna(0)
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

elif menu == "Cash Equity Analysis":
    st.subheader("📈 Cash Equity Portfolio Analysis")
    uploaded_file_ce = st.sidebar.file_uploader("Upload 'Holdings3.xlsx'", type=['xlsx'], key="ce")
    
    if uploaded_file_ce:
        with st.spinner("Processing Data & Fetching Factors..."):
            res = load_cash_equity_data(uploaded_file_ce)
            df_perf, df_last, df_contrib, country_daily, logs, err, _ = res
        
        if err: st.error(err)
        elif df_perf is not None:
            start_dt, end_dt = df_perf.index.min(), df_perf.index.max()
            bm_returns = download_benchmarks_all(start_dt, end_dt)
            factor_returns = download_factors(start_dt, end_dt)
            
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
                target_ret = df_perf['Ret_Total_KRW']
            else:
                c1.metric("Total Return (Hedged)", f"{last_day['Cum_Total_Local']:.2%}")
                c2.metric("Equity Return (Unhedged)", f"{last_day['Cum_Equity_Local']:.2%}")
                c3.metric("Hedge Impact", f"{(last_day['Cum_Total_Local'] - last_day['Cum_Equity_Local']):.2%}")
                y_main, y_sub = 'Cum_Total_Local', 'Cum_Equity_Local'
                name_main, name_sub = 'Total (Hedged)', 'Equity (Local/USD)'
                target_ret = df_perf['Ret_Total_Local']
            c4.metric("Current AUM", f"{curr_aum:,.0f} KRW")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_perf.index, y=df_perf[y_main], name=name_main, line=dict(color='#2563eb', width=3)))
            if y_sub: fig.add_trace(go.Scatter(x=df_perf.index, y=df_perf[y_sub], name=name_sub, line=dict(color='#60a5fa', dash='dot')))
            
            if not bm_returns.empty:
                bm_cum = (1 + bm_returns).cumprod() - 1
                for col in ['US', 'KR', 'HK', 'JP']:
                    if col in bm_cum.columns:
                        fig.add_trace(go.Scatter(x=bm_cum.index, y=bm_cum[col], name=col+' BM', line=dict(width=1, dash='dash')))
            st.plotly_chart(fig, use_container_width=True)

            t1, t2, t3 = st.tabs(["Factor Risk & Attribution", "Selection Effect", "Holdings"])
            
            with t1:
                st.markdown("#### 🧪 12-Factor Analysis (Risk & Attribution)")
                if not factor_returns.empty:
                    exposures, contrib, r2 = perform_factor_regression(target_ret, factor_returns)
                    
                    if exposures is not None:
                        st.write(f"**R-Squared:** {r2:.2f} (Explained by Factors)")
                        c_exp, c_attr = st.columns(2)
                        with c_exp:
                            st.markdown("**Factor Exposures (Beta)**")
                            fig_exp = px.bar(exposures, orientation='h', labels={'value':'Beta', 'index':'Factor'})
                            fig_exp.update_layout(showlegend=False)
                            st.plotly_chart(fig_exp, use_container_width=True)
                        with c_attr:
                            st.markdown("**Cumulative Factor Attribution**")
                            if not contrib.empty:
                                contrib_cum = (1 + contrib).cumprod() - 1
                                fig_attr = go.Figure()
                                for col in contrib_cum.columns:
                                    if col != 'Alpha' and col != 'Unexplained':
                                        fig_attr.add_trace(go.Scatter(x=contrib_cum.index, y=contrib_cum[col], name=col))
                                st.plotly_chart(fig_attr, use_container_width=True)
                        
                        st.markdown("#### 📅 Monthly Factor Attribution")
                        m_contrib = contrib.resample('ME').apply(lambda x: (1+x).prod()-1)
                        m_contrib.index = m_contrib.index.strftime('%Y-%m')
                        fig_heat = go.Figure(data=go.Heatmap(
                            z=m_contrib.T.values, x=m_contrib.index, y=m_contrib.columns,
                            colorscale='RdBu', zmin=-0.03, zmax=0.03
                        ))
                        fig_heat.update_layout(height=500)
                        st.plotly_chart(fig_heat, use_container_width=True)
                else: st.warning("Factor data download failed.")

            with t2:
                st.markdown("#### 💹 Return Contribution")
                if df_contrib:
                    c_a, c_b = st.columns(2)
                    with c_a:
                        if not df_contrib['Country'].empty:
                            ctry_cont = df_contrib['Country'].groupby('Country')['Contrib_KRW'].sum().sort_values(ascending=False).reset_index()
                            st.plotly_chart(px.bar(ctry_cont, x='Contrib_KRW', y='Country', orientation='h', title="Contribution by Country", text_auto='.2%'))
                    with c_b:
                        if not df_contrib['Sector'].empty:
                            sec_cont = df_contrib['Sector'].groupby('섹터')['Contrib_KRW'].sum().sort_values(ascending=False).reset_index()
                            st.plotly_chart(px.bar(sec_cont, x='Contrib_KRW', y='섹터', orientation='h', title="Contribution by Sector", text_auto='.2%'))
                    
                    st.markdown("---")
                    st.markdown("#### 🥧 Current Allocation Breakdown")
                    max_date = df_perf.index.max()
                    curr_hold = df_last[(df_last['기준일자'] == max_date) & (df_last['잔고수량'] > 0)]
                    
                    if not curr_hold.empty:
                        st.plotly_chart(px.pie(curr_hold, values='원화평가금액', names='섹터', title="Sector Allocation", hole=0.4), use_container_width=True)
                        st.plotly_chart(px.pie(curr_hold, values='원화평가금액', names='Country', title="Country Allocation", hole=0.4), use_container_width=True)

            with t3:
                pnl_df = df_last.sort_values('Final_PnL', ascending=False)[['종목명','섹터','Country','Final_PnL']]
                cw, cl = st.columns(2)
                cw.success("Top Winners"); cw.dataframe(pnl_df.head(5).style.format({'Final_PnL':'{:,.0f}'}))
                cl.error("Top Losers"); cl.dataframe(pnl_df.tail(5).style.format({'Final_PnL':'{:,.0f}'}))
                with st.expander("Daily Data"): st.dataframe(df_perf)

elif menu == "📑 Weekly Report Generator":
    st.subheader("📑 Weekly Meeting Report Generator")
    uploaded_file_ce = st.sidebar.file_uploader("Upload 'Holdings3.xlsx' for Report", type=['xlsx'], key="rep")
    
    if uploaded_file_ce:
        with st.spinner("Generating Report Data..."):
            res = load_cash_equity_data(uploaded_file_ce)
            df_perf, df_last, df_contrib, country_daily, df_daily_stock, logs, err, _ = res
            
        if err: st.error(err)
        elif df_perf is not None:
            max_date = df_perf.index.max()
            report_date = st.date_input("Report Date", max_date)
            report_date = pd.to_datetime(report_date)
            
            factor_returns = download_factors(df_perf.index.min(), report_date)
            _, factor_contrib, _ = perform_factor_regression(df_perf['Ret_Total_KRW'], factor_returns)
            
            dates = {
                'WTD': df_perf.index[df_perf.index <= report_date][-1] - pd.to_timedelta(df_perf.index[df_perf.index <= report_date][-1].weekday(), unit='D'),
                'MTD': report_date.replace(day=1),
                'QTD': report_date.replace(month=((report_date.month-1)//3)*3+1, day=1),
                'YTD': report_date.replace(month=1, day=1)
            }
            
            df_perf_cut = df_perf[df_perf.index <= report_date]
            df_stock_cut = df_daily_stock[df_daily_stock['기준일자'] <= report_date]
            if factor_contrib is not None:
                factor_contrib_cut = factor_contrib[factor_contrib.index <= report_date]
            else: factor_contrib_cut = None
            
            def calc_period_stats(start_dt, label):
                sub_perf = df_perf_cut[df_perf_cut.index >= start_dt]
                if sub_perf.empty: return None
                cum_ret = (1 + sub_perf['Ret_Total_KRW']).prod() - 1
                abs_pnl = sub_perf['Total_PnL_KRW'].sum()
                sub_stock = df_stock_cut[df_stock_cut['기준일자'] >= start_dt]
                stock_contrib = sub_stock.groupby(['종목명', 'Ticker_ID'])['Contrib_KRW'].sum().reset_index()
                top5 = stock_contrib.sort_values('Contrib_KRW', ascending=False).head(5)
                bot5 = stock_contrib.sort_values('Contrib_KRW', ascending=True).head(5)
                ctry_contrib = sub_stock.groupby('Country')['Contrib_KRW'].sum().sort_values(ascending=False)
                sect_contrib = sub_stock.groupby('섹터')['Contrib_KRW'].sum().sort_values(ascending=False)
                f_cont = pd.Series()
                if factor_contrib_cut is not None:
                    sub_f = factor_contrib_cut[factor_contrib_cut.index >= start_dt]
                    if not sub_f.empty:
                        f_cont = sub_f.apply(lambda x: (1+x).prod()-1).sort_values(ascending=False)
                return {'label': label, 'ret': cum_ret, 'pnl': abs_pnl, 'top5': top5, 'bot5': bot5, 'ctry': ctry_contrib, 'sect': sect_contrib, 'factor': f_cont}

            tabs = st.tabs(["Summary Report", "WTD", "MTD", "QTD", "YTD"])
            stats_res = {}
            for p in ['WTD', 'MTD', 'QTD', 'YTD']:
                stats_res[p] = calc_period_stats(dates[p], p)
                with tabs[list(dates.keys()).index(p) + 1]:
                    if stats_res[p]:
                        st.markdown(f"### {p} Performance ({dates[p].date()} ~ {report_date.date()})")
                        c1, c2 = st.columns(2)
                        c1.metric("Return", f"{stats_res[p]['ret']:.2%}")
                        c2.metric("PnL (KRW)", f"{stats_res[p]['pnl']:,.0f}")
                        st.markdown("#### Top Contributors")
                        c3, c4 = st.columns(2)
                        with c3: st.table(stats_res[p]['top5'][['종목명', 'Contrib_KRW']].style.format({'Contrib_KRW': '{:.2%}'}))
                        with c4: st.table(stats_res[p]['bot5'][['종목명', 'Contrib_KRW']].style.format({'Contrib_KRW': '{:.2%}'}))
                        st.markdown("#### Attribution Analysis")
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.markdown("**Country**")
                            st.dataframe(stats_res[p]['ctry'].to_frame().style.format('{:.2%}'))
                        with col_b:
                            st.markdown("**Sector**")
                            st.dataframe(stats_res[p]['sect'].to_frame().style.format('{:.2%}'))
                        with col_c:
                            st.markdown("**Factor**")
                            if not stats_res[p]['factor'].empty:
                                st.dataframe(stats_res[p]['factor'].to_frame(name='Contrib').style.format('{:.2%}'))
                            else: st.write("No factor data")
                    else: st.write("No data.")

            with tabs[0]:
                st.markdown("### 📝 Weekly Meeting Commentary")
                txt = f"**[Portfolio Weekly Update - {report_date.date()}]**\n\n"
                
                wtd = stats_res.get('WTD')
                if wtd:
                    txt += f"**1. WTD Performance:** {wtd['ret']:.2%} ({wtd['pnl']:,.0f} KRW)\n"
                    if not wtd['top5'].empty: txt += f"   - **Lead:** {wtd['top5'].iloc[0]['종목명']} (+{wtd['top5'].iloc[0]['Contrib_KRW']:.2%})\n"
                    if not wtd['factor'].empty: txt += f"   - **Factor:** Driven by {wtd['factor'].idxmax()} (+{wtd['factor'].max():.2%})\n"
                
                mtd = stats_res.get('MTD')
                if mtd: 
                    txt += f"**2. MTD:** {mtd['ret']:.2%}. Best Sector: {mtd['sect'].idxmax()}.\n"
                    if not mtd['factor'].empty: txt += f"   - Factor: {mtd['factor'].idxmax()} style worked well.\n"
                
                ytd = stats_res.get('YTD')
                if ytd: txt += f"**3. YTD:** {ytd['ret']:.2%}, Total PnL {ytd['pnl']:,.0f} KRW.\n"
                
                st.text_area("Copy this:", txt, height=300)