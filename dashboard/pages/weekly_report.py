from pathlib import Path

from dashboard.core import *  # noqa: F401,F403

ROOT_DIR = Path(__file__).resolve().parents[2]

def render_weekly_report_page():
    st.subheader("📑 Weekly Meeting Report Generator")
    uploaded_file_ce = st.sidebar.file_uploader("Upload 'Holdings3.xlsx' for Report", type=['xlsx'], key="rep")

    if uploaded_file_ce:
        with st.spinner("Generating Report Data..."):
            res = load_cash_equity_data(uploaded_file_ce)
            df_perf, df_last, df_contrib, country_daily, df_daily_stock, logs, err = res
            
        if err: st.error(err)
        elif df_perf is not None:
            view_opt = st.radio("Currency View", ["KRW (Hedged)", "Local (Hedged, USD base)"], horizontal=True, key="weekly_view_opt")
            if "Local" in view_opt:
                ret_col = "Ret_Total_Local"
                view_label = "Total Return (Local, hedged)"
            else:
                ret_col = "Ret_Total_KRW"
                view_label = "Total Return (KRW, hedged)"

            max_date = df_perf.index.max()
            report_date = st.date_input("Report Date", max_date)
            report_date = pd.to_datetime(report_date)
            
            factor_prices = download_factors(df_perf.index.min(), report_date, return_prices=True)
            factor_returns = align_factor_returns(df_perf[ret_col].index, factor_prices)
            _, factor_contrib, _ = perform_factor_regression(df_perf[ret_col], factor_returns)
            
            dates = {
                'WTD': df_perf.index[df_perf.index <= report_date][-1] - pd.to_timedelta(df_perf.index[df_perf.index <= report_date][-1].weekday(), unit='D'),
                'MTD': report_date.replace(day=1),
                'QTD': report_date.replace(month=((report_date.month-1)//3)*3+1, day=1),
                'YTD': report_date.replace(month=1, day=1)
            }
            
            global_px = download_global_indices(min(dates.values()), report_date)
            df_perf_cut = df_perf[df_perf.index <= report_date]
            df_stock_cut = df_daily_stock[df_daily_stock['기준일자'] <= report_date]
            if factor_contrib is not None:
                factor_contrib_cut = factor_contrib[factor_contrib.index <= report_date]
            else: factor_contrib_cut = None
            
            def calc_period_stats(start_dt, label, global_px):
                sub_perf = df_perf_cut[df_perf_cut.index >= start_dt]
                if sub_perf.empty: return None
                cum_ret = (1 + sub_perf[ret_col]).prod() - 1
                abs_pnl = sub_perf['Total_PnL_KRW'].sum()  # already includes hedge PnL in KRW
                sub_stock = df_stock_cut[df_stock_cut['기준일자'] >= start_dt]
                stock_contrib = sub_stock.groupby(['종목명', 'Ticker_ID'])['Contrib_KRW'].sum().reset_index()
                top5 = stock_contrib.sort_values('Contrib_KRW', ascending=False).head(5)
                bot5 = stock_contrib.sort_values('Contrib_KRW', ascending=True).head(5)
                ctry_contrib = sub_stock.groupby('Country')['Contrib_KRW'].sum().sort_values(ascending=False)
                sect_contrib = sub_stock.groupby('섹터')['Contrib_KRW'].sum().sort_values(ascending=False)
                f_cont = pd.Series(dtype=float)
                if factor_contrib_cut is not None:
                    sub_f = factor_contrib_cut[factor_contrib_cut.index >= start_dt]
                    if not sub_f.empty:
                        f_cont = sub_f.apply(lambda x: (1+x).prod()-1).sort_values(ascending=False)
                idx_ret = pd.Series(dtype=float)
                sub_px = None
                if global_px is not None and not global_px.empty:
                    sub_px = global_px[(global_px.index >= start_dt) & (global_px.index <= report_date)]
                    sub_px = sub_px.ffill().bfill()
                    if not sub_px.empty:
                        idx_ret = sub_px.iloc[-1] / sub_px.iloc[0] - 1
                hedge_contrib = None
                hedge_pnl_krw = None
                if 'Hedge_PnL_KRW' in sub_perf.columns:
                    hedge_pnl_krw = sub_perf['Hedge_PnL_KRW'].sum()
                if ret_col == "Ret_Total_Local":
                    if 'Ret_Hedge_Local' in sub_perf.columns:
                        hedge_contrib = sub_perf['Ret_Hedge_Local'].fillna(0).sum()
                else:
                    if 'Hedge_PnL_KRW' in sub_perf.columns and 'Total_Prev_MV_KRW' in sub_perf.columns:
                        denom = sub_perf['Total_Prev_MV_KRW'].replace(0, np.nan)
                        hedge_contrib = (sub_perf['Hedge_PnL_KRW'] / denom).fillna(0).sum()
                portfolio_risk = _calc_risk_metrics(sub_perf[ret_col])
                benchmark_risk = {}
                if sub_px is not None and not sub_px.empty:
                    bench_ret = sub_px.pct_change().dropna(how='all')
                    for col in bench_ret.columns:
                        metrics = _calc_risk_metrics(bench_ret[col])
                        if metrics:
                            benchmark_risk[col] = metrics
                return {'label': label, 'ret': cum_ret, 'pnl': abs_pnl, 'top5': top5, 'bot5': bot5, 
                        'ctry': ctry_contrib, 'sect': sect_contrib, 'factor': f_cont, 'indices': idx_ret,
                        'hedge_contrib': hedge_contrib, 'hedge_pnl_krw': hedge_pnl_krw,
                        'risk': {'portfolio': portfolio_risk, 'benchmarks': benchmark_risk}}

            tabs = st.tabs(["Summary Report", "WTD", "MTD", "QTD", "YTD"])
            stats_res = {}
            for p in ['WTD', 'MTD', 'QTD', 'YTD']:
                stats_res[p] = calc_period_stats(dates[p], p, global_px)
                with tabs[list(dates.keys()).index(p) + 1]:
                    if stats_res[p]:
                        st.markdown(f"### {p} Performance ({dates[p].date()} ~ {report_date.date()})")
                        if not stats_res[p]['indices'].empty:
                            idx_df = stats_res[p]['indices'].sort_values(ascending=False).reset_index()
                            idx_df.columns = ['Index', 'Return']
                            idx_df['Return'] = idx_df['Return'].apply(lambda x: f"{x:.2%}")
                            st.markdown(create_manual_html_table(idx_df, title="Global Index Returns"), unsafe_allow_html=True)
                        c1, c2 = st.columns(2)
                        c1.metric(view_label, f"{stats_res[p]['ret']:.2%}")
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
                        hedge_val = stats_res[p].get('hedge_contrib')
                        hedge_display = f"{hedge_val:.2%}" if hedge_val is not None else "N/A"
                        st.markdown("**Hedge Contribution**")
                        st.metric("Hedge Contribution (Return)", hedge_display)
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

                st.markdown("#### 🤖 AI-Generated Weekly Report")
                llm_choice = st.radio(
                    "Select LLM",
                    ["OpenAI (gpt-4o-mini)", "DeepSeek (deepseek-chat)"],
                    horizontal=True,
                    key="llm_choice_radio",
                )
                lang_choice = st.radio(
                    "언어 / Language",
                    ["English", "한국어"],
                    horizontal=True,
                    key="ai_lang_choice_radio",
                )
                user_comment = st.text_area(
                    "Optional: Add your own market/positioning comments",
                    height=120,
                    key="ai_comment_input",
                )
                if st.button("Generate AI Report", key="ai_report_btn"):
                    provider = "deepseek" if "DeepSeek" in llm_choice else "openai"
                    try:
                        ai_text = generate_ai_weekly_report(
                            stats_res,
                            report_date,
                            user_comment,
                            provider=provider,
                            language=lang_choice,
                        )
                        st.text_area("AI Report", ai_text, height=400, key="ai_report_text")
                    except Exception as e:
                        st.error(f"Failed to generate AI report: {e}")
