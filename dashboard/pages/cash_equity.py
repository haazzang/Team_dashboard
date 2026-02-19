from pathlib import Path

from dashboard.core import *  # noqa: F401,F403

ROOT_DIR = Path(__file__).resolve().parents[2]

def render_cash_equity_page():
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
            factor_prices = download_factors(start_dt, end_dt, return_prices=True)
            
            view_opt = st.radio("Currency View", ["KRW", "Local Currency (USD Base)"], horizontal=True)

            max_perf_date = df_perf.index.max() if not df_perf.empty else pd.NaT
            max_hold_date = df_last['기준일자'].max() if df_last is not None else pd.NaT
            max_date = max_perf_date if pd.notna(max_perf_date) else max_hold_date
            if pd.notna(max_date):
                curr_hold = df_last[(df_last['기준일자'] == max_date) & (df_last['잔고수량'] > 0)]
            else:
                curr_hold = df_last[df_last['잔고수량'] > 0] if df_last is not None else pd.DataFrame()

            if 'Total_MV_KRW' in df_perf.columns and not df_perf.empty:
                curr_aum = df_perf.iloc[-1]['Total_MV_KRW']
            else:
                curr_aum = curr_hold['원화평가금액'].sum() if not curr_hold.empty else 0
            
            c1, c2, c3, c4 = st.columns(4)
            if df_perf.empty:
                c1.metric("Total Return (Hedged)", "N/A")
                c2.metric("Equity Return (Unhedged)", "N/A")
                c3.metric("Hedge Impact", "N/A")
                y_main = y_sub = None
                name_main = name_sub = None
                target_ret = pd.Series(dtype=float)
            elif view_opt == "KRW":
                last_day = df_perf.iloc[-1]
                c1.metric("Total Return (Hedged)", f"{last_day['Cum_Total_KRW']:.2%}")
                c2.metric("Equity Return (Unhedged)", f"{last_day['Cum_Equity_KRW']:.2%}")
                c3.metric("Hedge Impact", f"{(last_day['Cum_Total_KRW'] - last_day['Cum_Equity_KRW']):.2%}")
                y_main, y_sub = 'Cum_Total_KRW', 'Cum_Equity_KRW'
                name_main, name_sub = 'Total (Hedged)', 'Equity (KRW)'
                target_ret = df_perf['Ret_Total_KRW']
            else:
                last_day = df_perf.iloc[-1]
                c1.metric("Total Return (Hedged)", f"{last_day['Cum_Total_Local']:.2%}")
                c2.metric("Equity Return (Unhedged)", f"{last_day['Cum_Equity_Local']:.2%}")
                c3.metric("Hedge Impact", f"{(last_day['Cum_Total_Local'] - last_day['Cum_Equity_Local']):.2%}")
                y_main, y_sub = 'Cum_Total_Local', 'Cum_Equity_Local'
                name_main, name_sub = 'Total (Hedged)', 'Equity (Local/USD)'
                target_ret = df_perf['Ret_Total_Local']
            c4.metric("Current AUM", f"{curr_aum:,.0f} KRW")

            if not df_perf.empty and y_main:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_perf.index, y=df_perf[y_main], name=name_main, line=dict(color='#2563eb', width=3)))
                if y_sub: fig.add_trace(go.Scatter(x=df_perf.index, y=df_perf[y_sub], name=name_sub, line=dict(color='#60a5fa', dash='dot')))
                
                if not bm_returns.empty:
                    bm_cum = (1 + bm_returns).cumprod() - 1
                    for col in ['US', 'KR', 'HK', 'JP']:
                        if col in bm_cum.columns:
                            fig.add_trace(go.Scatter(x=bm_cum.index, y=bm_cum[col], name=col+' BM', line=dict(width=1, dash='dash')))
                st.plotly_chart(fig, use_container_width=True)

            factor_returns = align_factor_returns(target_ret.index, factor_prices)

            st.markdown("#### 📊 Risk Metrics (Hedged Total Returns)")
            rows = []
            if view_opt == "KRW":
                port_label = "Portfolio (Hedged, KRW)"
            else:
                port_label = "Portfolio (Hedged, Local/USD)"
                st.caption("Local currency view uses total hedged returns (includes hedge PnL).")
            port_metrics = _calc_perf_metrics(target_ret)
            if port_metrics:
                rows.append({"Asset": port_label, **port_metrics})
            if not bm_returns.empty:
                bm_aligned = bm_returns.reindex(df_perf.index).dropna(how='all')
                for col in bm_aligned.columns:
                    metrics = _calc_perf_metrics(bm_aligned[col].dropna())
                    if metrics:
                        rows.append({"Asset": f"Benchmark {col}", **metrics})
            if rows:
                metrics_df = pd.DataFrame(rows)
                metric_order = [
                    "Total Return",
                    "CAGR",
                    "Annualized Volatility",
                    "Sharpe Ratio",
                    "Sortino Ratio",
                    "Max Drawdown",
                    "Calmar Ratio",
                    "Win Rate",
                ]
                metrics_df = metrics_df[["Asset"] + metric_order]
                disp = metrics_df.copy()
                percent_cols = ["Total Return", "CAGR", "Annualized Volatility", "Max Drawdown", "Win Rate"]
                ratio_cols = ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"]
                for col in percent_cols:
                    disp[col] = disp[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")
                for col in ratio_cols:
                    disp[col] = disp[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
                st.markdown(create_manual_html_table(disp, title="Risk Metrics vs Benchmarks"), unsafe_allow_html=True)
            else:
                st.write("Risk metrics not available.")

            t1, t2, t3, t4 = st.tabs(["Factor Risk & Attribution", "Selection Effect", "Holdings", "Beta Trend"])
            
            with t1:
                st.markdown(f"#### 🧪 {FACTOR_TARGET_COUNT}-Factor Analysis (Risk & Attribution)")
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
                    else:
                        st.write("Insufficient data to compute factor regression.")
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
                    if not curr_hold.empty:
                        st.plotly_chart(px.pie(curr_hold, values='원화평가금액', names='섹터', title="Sector Allocation", hole=0.4), use_container_width=True)
                        st.plotly_chart(px.pie(curr_hold, values='원화평가금액', names='Country', title="Country Allocation", hole=0.4), use_container_width=True)

            with t3:
                pnl_df = df_last.sort_values('Final_PnL', ascending=False)[['종목명','섹터','Country','Final_PnL']]
                cw, cl = st.columns(2)
                cw.success("Top Winners"); cw.dataframe(pnl_df.head(5).style.format({'Final_PnL':'{:,.0f}'}))
                cl.error("Top Losers"); cl.dataframe(pnl_df.tail(5).style.format({'Final_PnL':'{:,.0f}'}))
                with st.expander("Daily Data"): st.dataframe(df_perf)

            with t4:
                st.markdown("#### 📈 Rolling Beta Trend vs Benchmarks")
                if bm_returns.empty:
                    st.warning("Benchmark data download failed.")
                else:
                    beta_window = st.slider(
                        "Rolling window (trading days)",
                        min_value=20,
                        max_value=252,
                        value=60,
                        step=5,
                        key="beta_window",
                    )
                    beta_fig = go.Figure()
                    bench_map = {"SPX": "US", "Hang Seng": "HK", "Nikkei 225": "JP"}
                    for label, col in bench_map.items():
                        if col in bm_returns.columns:
                            beta_series = calculate_rolling_beta(target_ret, bm_returns[col], window=beta_window)
                            if not beta_series.empty:
                                beta_fig.add_trace(go.Scatter(x=beta_series.index, y=beta_series, name=f"{label} Beta"))
                    if beta_fig.data:
                        beta_fig.update_layout(yaxis_title="Beta", xaxis_title="Date")
                        st.plotly_chart(beta_fig, use_container_width=True)
                    else:
                        st.write("Insufficient data to compute rolling beta.")

                st.markdown("#### 🧮 Holdings-Weighted Beta (Latest)")
                bench_yf_map = {"S&P 500": "^GSPC", "Hang Seng": "^HSI", "Nikkei 225": "^N225", "KOSPI": "^KS11"}
                holdings_beta = calculate_holdings_beta(curr_hold, bench_yf_map, end_date=max_date)
                if holdings_beta:
                    beta_df = pd.Series(holdings_beta, name="Beta").sort_values().reset_index()
                    beta_df.columns = ["Benchmark", "Beta"]
                    fig_beta = px.bar(beta_df, x="Beta", y="Benchmark", orientation="h", title="Holdings-Weighted Beta")
                    fig_beta.update_layout(showlegend=False)
                    st.plotly_chart(fig_beta, use_container_width=True)
                else:
                    st.write("Insufficient data to compute holdings-weighted beta.")
