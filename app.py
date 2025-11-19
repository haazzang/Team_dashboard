import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import datetime

# --- 페이지 설정 ---
st.set_page_config(page_title="Portfolio Dashboard", layout="wide")
st.title("🚀 Team Portfolio Analysis Dashboard")

# --- 1. 데이터 로드 함수 ---
@st.cache_data
def load_data(file):
    try:
        # PNL 시트 로드
        df_pnl_raw = pd.read_excel(file, sheet_name='PNL', header=None, engine='openpyxl')
        header_idx = -1
        for i in range(10):
            if '일자' in df_pnl_raw.iloc[i].astype(str).values:
                header_idx = i
                break
        if header_idx == -1: return None, None, "PNL 시트에서 '일자'를 찾을 수 없습니다."
        
        pnl_cols = df_pnl_raw.iloc[header_idx].tolist()
        df_pnl_raw.columns = pnl_cols
        df_pnl = df_pnl_raw.iloc[header_idx+1:].copy()
        
        # Position 시트 로드
        df_pos_raw = pd.read_excel(file, sheet_name='Position', header=None, engine='openpyxl')
        header_idx_pos = -1
        for i in range(10):
            if '일자' in df_pos_raw.iloc[i].astype(str).values:
                header_idx_pos = i
                break
        if header_idx_pos == -1: return None, None, "Position 시트에서 '일자'를 찾을 수 없습니다."
        
        pos_cols = df_pos_raw.iloc[header_idx_pos].tolist()
        df_pos_raw.columns = pos_cols
        df_pos = df_pos_raw.iloc[header_idx_pos+1:].copy()

        # 공통 정제 함수
        def clean_df(df):
            date_col = next((c for c in df.columns if str(c).strip() == '일자'), None)
            if not date_col: return None
            df.set_index(date_col, inplace=True)
            df.index = pd.to_datetime(df.index, errors='coerce')
            df = df.dropna(how='all')
            df = df[df.index.notnull()]
            df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # 중복 컬럼 처리
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
        return None, None, f"파일 처리 중 오류 발생: {e}"

# --- 2. Cross Asset 데이터 로드 함수 ---
@st.cache_data
def download_cross_assets(start_date, end_date):
    assets = {
        'S&P 500': '^GSPC', 'Nasdaq': '^IXIC', 'KOSPI': '^KS11', 'KOSDAQ': '^KQ11',
        'Nikkei 225': '^N225', 'Hang Seng': '^HSI', 'USD/KRW': 'KRW=X',
        'US 10Y Yield': '^TNX', 'Gold': 'GC=F', 'Crude Oil': 'CL=F'
    }
    try:
        data = yf.download(list(assets.values()), start=start_date, end=end_date + pd.Timedelta(days=5), progress=False)
        if 'Adj Close' in data.columns: df_close = data['Adj Close']
        elif 'Close' in data.columns: df_close = data['Close']
        else: df_close = data
            
        if isinstance(df_close.columns, pd.MultiIndex):
            df_close.columns = df_close.columns.get_level_values(0)
            
        inv_assets = {v: k for k, v in assets.items()}
        df_close.rename(columns=inv_assets, inplace=True)
        return df_close
    except Exception as e:
        return pd.DataFrame()

# --- 3. 월별 수익률 계산 함수 ---
def calculate_monthly_returns(df_daily):
    df_monthly = df_daily.resample('M').apply(lambda x: (1 + x).prod() - 1)
    return df_monthly

# --- 사이드바: 파일 업로드 ---
st.sidebar.header("📂 데이터 업로드")
uploaded_file = st.sidebar.file_uploader("Team_PNL.xlsx 파일을 올려주세요", type=['xlsx'])

if uploaded_file is not None:
    df_pnl, df_pos, pnl_cols = load_data(uploaded_file)
    
    if df_pnl is not None:
        # 공통 데이터 처리
        common_idx = df_pnl.index.intersection(df_pos.index)
        common_cols = [c for c in pnl_cols if c in df_pos.columns]
        
        df_pnl = df_pnl.loc[common_idx, common_cols]
        df_pos = df_pos.loc[common_idx, common_cols]
        
        # 지표 계산
        df_cum_pnl = df_pnl.cumsum()
        df_user_ret = df_cum_pnl.div(df_pos.replace(0, np.nan)).fillna(0)
        df_daily_ret = df_pnl.div(df_pos.replace(0, np.nan)).fillna(0)
        
        # 실제 전체 포트폴리오 수익률 (Actual Portfolio Return)
        # Total PnL / Total Position
        total_pnl_series = df_pnl.sum(axis=1)
        total_pos_series = df_pos.sum(axis=1)
        actual_port_daily_ret = total_pnl_series.div(total_pos_series.replace(0, np.nan)).fillna(0)
        actual_port_cum = (1 + actual_port_daily_ret).cumprod() - 1
        
        # 월별 수익률 계산
        df_monthly_ret = calculate_monthly_returns(df_daily_ret)
        
        # 벤치마크 및 Cross Asset 다운로드
        start, end = df_user_ret.index.min(), df_user_ret.index.max()
        with st.spinner('글로벌 자산 데이터 다운로드 중...'):
            df_assets = download_cross_assets(start, end)
            
            if not df_assets.empty:
                df_assets.index = pd.to_datetime(df_assets.index).tz_localize(None)
                df_assets = df_assets.reindex(df_user_ret.index, method='ffill')
                df_asset_ret = df_assets.pct_change().fillna(0)
                
                bm_cum = pd.DataFrame(index=df_user_ret.index)
                if 'S&P 500' in df_assets.columns:
                    bm_cum['SPX'] = (1 + df_asset_ret['S&P 500']).cumprod() - 1
                if 'KOSPI' in df_assets.columns:
                    bm_cum['KOSPI'] = (1 + df_asset_ret['KOSPI']).cumprod() - 1

        # --- 탭 구성 ---
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Chart", "📊 Analysis", "📅 Monthly Returns", "🔗 Strategy Corr", "🌍 Cross Asset Corr", "🧪 Simulation"
        ])

        # Tab 1: Chart
        with tab1:
            st.subheader("Cumulative Return Over Time")
            strategies = df_user_ret.columns.tolist()
            selected_strat = st.selectbox("Select Strategy:", strategies, key='chart_select')
            
            is_overseas = any(k in selected_strat for k in ['해외', 'Global', 'US'])
            bm_name = 'SPX' if is_overseas else 'KOSPI'
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_user_ret.index, y=df_user_ret[selected_strat], name=selected_strat, line=dict(color='#2563eb', width=2)))
            if not df_assets.empty and bm_name in bm_cum.columns:
                fig.add_trace(go.Scatter(x=df_user_ret.index, y=bm_cum[bm_name], name=f"BM: {bm_name}", line=dict(color='grey', dash='dash')))
            fig.update_layout(template="plotly_white", height=500, yaxis_tickformat=".2%")
            st.plotly_chart(fig, use_container_width=True)

        # Tab 2: Analysis
        with tab2:
            st.subheader("Detailed Performance Metrics")
            stats = pd.DataFrame(index=df_daily_ret.columns)
            stats['Volatility'] = df_daily_ret.std() * np.sqrt(252)
            stats['Sharpe'] = (df_daily_ret.mean() / df_daily_ret.std() * np.sqrt(252)).fillna(0)
            nav = (1 + df_daily_ret).cumprod()
            stats['MDD'] = ((nav - nav.cummax()) / nav.cummax()).min()
            stats['Total Return'] = df_user_ret.iloc[-1]
            
            win_days = (df_daily_ret > 0).sum()
            total_days = (df_daily_ret != 0).sum()
            stats['Win Rate'] = (win_days / total_days).fillna(0)
            
            avg_gain = df_daily_ret[df_daily_ret > 0].mean()
            avg_loss = df_daily_ret[df_daily_ret < 0].mean().abs()
            stats['Profit Factor'] = (avg_gain / avg_loss).fillna(0)
            
            stats_disp = stats.copy()
            stats_disp['Volatility'] = stats_disp['Volatility'].apply(lambda x: f"{x:.2%}")
            stats_disp['MDD'] = stats_disp['MDD'].apply(lambda x: f"{x:.2%}")
            stats_disp['Sharpe'] = stats_disp['Sharpe'].apply(lambda x: f"{x:.2f}")
            stats_disp['Total Return'] = stats_disp['Total Return'].apply(lambda x: f"{x:.2%}")
            stats_disp['Win Rate'] = stats_disp['Win Rate'].apply(lambda x: f"{x:.1%}" if isinstance(x, (int, float)) else x)
            stats_disp['Profit Factor'] = stats_disp['Profit Factor'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
            
            st.dataframe(stats_disp.style.applymap(lambda x: 'color: red' if '-' in str(x) and '%' in str(x) else 'color: green' if '%' in str(x) else '', subset=['Total Return', 'MDD']), use_container_width=True)

        # Tab 3: Monthly Returns
        with tab3:
            st.subheader("📅 Monthly Returns Analysis")
            view_type = st.radio("View Type:", ["All Strategies Overview", "Single Strategy Details"], horizontal=True)
            if view_type == "All Strategies Overview":
                st.markdown("##### 최근 월별 전략 성과 (Strategies vs Months)")
                df_m_display = df_monthly_ret.T
                df_m_display.columns = [d.strftime('%Y-%m') for d in df_m_display.columns]
                st.dataframe(df_m_display.style.format("{:.2%}").background_gradient(cmap='RdYlGn', vmin=-0.1, vmax=0.1), use_container_width=True, height=600)
            else:
                st.markdown("##### 개별 전략 연도별/월별 성과표 (Year vs Month)")
                sel_strat_m = st.selectbox("Select Strategy:", df_monthly_ret.columns, key='monthly_select')
                strat_m_data = df_monthly_ret[sel_strat_m].to_frame(name='Return')
                strat_m_data['Year'] = strat_m_data.index.year
                strat_m_data['Month'] = strat_m_data.index.month
                pivot_table = strat_m_data.pivot(index='Year', columns='Month', values='Return')
                import calendar
                pivot_table.columns = [calendar.month_abbr[i] for i in pivot_table.columns]
                ytds = []
                for year in pivot_table.index:
                    daily_year = df_daily_ret[df_daily_ret.index.year == year][sel_strat_m]
                    ytd = (1 + daily_year).prod() - 1
                    ytds.append(ytd)
                pivot_table['YTD'] = ytds
                st.dataframe(pivot_table.style.format("{:.2%}", na_rep="-").background_gradient(cmap='RdYlGn', vmin=-0.1, vmax=0.1, subset=pivot_table.columns[:-1]).applymap(lambda x: 'font-weight: bold', subset=['YTD']), use_container_width=True)

        # Tab 4: Strategy Correlation
        with tab4:
            st.subheader("Correlation Matrix (Strategies)")
            corr = df_daily_ret.corr()
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.index,
                colorscale='RdBu', zmin=-1, zmax=1, text=np.round(corr.values, 2),
                texttemplate="%{text}", textfont={"size":10}
            ))
            fig_corr.update_layout(height=700)
            st.plotly_chart(fig_corr, use_container_width=True)

        # Tab 5: Cross Asset Correlation
        with tab5:
            st.subheader("🌍 Cross Asset Correlation")
            if not df_asset_ret.empty:
                combined_ret = pd.concat([df_daily_ret, df_asset_ret], axis=1)
                full_corr = combined_ret.corr()
                cross_corr = full_corr.loc[df_daily_ret.columns, df_asset_ret.columns]
                fig_cross = go.Figure(data=go.Heatmap(
                    z=cross_corr.values, x=cross_corr.columns, y=cross_corr.index,
                    colorscale='RdBu', zmin=-1, zmax=1, text=np.round(cross_corr.values, 2),
                    texttemplate="%{text}", textfont={"size":10}
                ))
                fig_cross.update_layout(height=800, xaxis_title="Global Assets", yaxis_title="My Strategies")
                st.plotly_chart(fig_cross, use_container_width=True)
            else:
                st.warning("자산 데이터를 불러오지 못했습니다.")

        # Tab 6: Simulation (New!)
        with tab6:
            st.subheader("🧪 Portfolio Allocation Simulation")
            st.markdown("각 전략의 비중을 조절하여 **가상의 포트폴리오** 성과를 시뮬레이션 해보세요.")
            
            col_sim_input, col_sim_output = st.columns([1, 3])
            
            with col_sim_input:
                st.markdown("#### Weight Adjustment")
                sim_weights = {}
                total_weight = 0.0
                
                # 1/N Weight 계산
                n_strats = len(df_daily_ret.columns)
                default_w = 1.0 / n_strats
                
                if st.button("Reset to Equal Weights"):
                    st.session_state['reset_weights'] = True
                
                for strat in df_daily_ret.columns:
                    # 세션 스테이트 초기화
                    key = f"weight_{strat}"
                    if 'reset_weights' in st.session_state and st.session_state['reset_weights']:
                         st.session_state[key] = default_w

                    # 슬라이더 생성
                    val = st.slider(f"{strat}", 0.0, 1.0, st.session_state.get(key, default_w), 0.01, key=key)
                    sim_weights[strat] = val
                    total_weight += val
                
                # Reset 플래그 끄기
                if 'reset_weights' in st.session_state:
                    st.session_state['reset_weights'] = False

                # 합계 표시
                st.markdown("---")
                sum_color = "green" if 0.99 <= total_weight <= 1.01 else "red"
                st.markdown(f"**Total Weight:** <span style='color:{sum_color}'>{total_weight:.0%}</span>", unsafe_allow_html=True)
                if sum_color == "red":
                    st.warning("합계가 100%가 아닙니다. (레버리지 or 현금 보유 가정)")

            with col_sim_output:
                # 시뮬레이션 수익률 계산 (가중 평균)
                # Daily Return * Weight
                weighted_ret = df_daily_ret.mul(pd.Series(sim_weights), axis=1)
                sim_daily_ret = weighted_ret.sum(axis=1)
                sim_cum_ret = (1 + sim_daily_ret).cumprod() - 1
                
                # 차트 비교 (Actual vs Simulated vs Benchmark)
                fig_sim = go.Figure()
                
                # 1. Actual
                fig_sim.add_trace(go.Scatter(x=actual_port_cum.index, y=actual_port_cum, name="Actual Portfolio", line=dict(color='black', width=2)))
                # 2. Simulated
                fig_sim.add_trace(go.Scatter(x=sim_cum_ret.index, y=sim_cum_ret, name="Simulated Portfolio", line=dict(color='#e11d48', width=2, dash='dot')))
                # 3. Benchmark
                if 'SPX' in bm_cum.columns:
                    fig_sim.add_trace(go.Scatter(x=bm_cum.index, y=bm_cum['SPX'], name="SPX", line=dict(color='grey', width=1, dash='dash')))
                if 'KOSPI' in bm_cum.columns:
                    fig_sim.add_trace(go.Scatter(x=bm_cum.index, y=bm_cum['KOSPI'], name="KOSPI", line=dict(color='lightgrey', width=1, dash='dash')))
                
                fig_sim.update_layout(title="Simulation Result: Cumulative Return", template="plotly_white", height=500, yaxis_tickformat=".2%")
                st.plotly_chart(fig_sim, use_container_width=True)
                
                # 통계 비교 표
                st.markdown("#### Performance Comparison")
                
                # 지표 계산 함수
                def calc_stats_series(series, name):
                    vol = series.std() * np.sqrt(252)
                    sharpe = (series.mean() / series.std() * np.sqrt(252)) if series.std() != 0 else 0
                    nav = (1 + series).cumprod()
                    mdd = ((nav - nav.cummax()) / nav.cummax()).min()
                    tot_ret = (1 + series).prod() - 1
                    return pd.Series([f"{tot_ret:.2%}", f"{vol:.2%}", f"{sharpe:.2f}", f"{mdd:.2%}"], 
                                     index=['Total Return', 'Volatility', 'Sharpe', 'MDD'], name=name)

                stats_act = calc_stats_series(actual_port_daily_ret, "Actual")
                stats_sim = calc_stats_series(sim_daily_ret, "Simulated")
                
                comp_df = pd.concat([stats_act, stats_sim], axis=1)
                st.table(comp_df)

    else:
        st.error(f"오류: {pnl_cols}") 
else:
    st.info("👈 왼쪽 사이드바에서 엑셀 파일을 업로드해주세요.")