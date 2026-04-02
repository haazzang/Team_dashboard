from pathlib import Path

import dashboard.core as core
from dashboard.core import *  # noqa: F401,F403

ROOT_DIR = Path(__file__).resolve().parents[2]
WEEKLY_REPORT_DEFAULT_FILES = [
    "2026_멀티.xlsx",
    "Holdings3.xlsx",
]
WEEKLY_REPORT_BENCHMARKS = {
    "SPX": "^GSPC",
    "NDX": "^NDX",
    "Hang Seng": "^HSI",
    "Nikkei 225": "^N225",
    "ACWI": "ACWI",
}


def _download_weekly_report_benchmarks(start_date, end_date):
    prices = core.download_price_history(
        list(WEEKLY_REPORT_BENCHMARKS.values()),
        start_date,
        end_date,
    )
    if prices.empty:
        return pd.DataFrame()
    return prices.rename(columns={v: k for k, v in WEEKLY_REPORT_BENCHMARKS.items()}).ffill()


def _compute_total_return(ret_series):
    clean = pd.Series(ret_series).dropna()
    if clean.empty:
        return np.nan
    return float((1 + clean).prod() - 1)


def _build_benchmark_comparison_table(portfolio_returns, benchmark_prices):
    if portfolio_returns is None or benchmark_prices is None or benchmark_prices.empty:
        return pd.DataFrame()

    benchmark_returns = benchmark_prices.ffill().pct_change(fill_method=None).dropna(how="all")
    rows = []
    for benchmark_name in benchmark_returns.columns:
        aligned = pd.concat(
            [
                pd.Series(portfolio_returns, name="portfolio"),
                benchmark_returns[benchmark_name].rename("benchmark"),
            ],
            axis=1,
        ).dropna()
        if aligned.shape[0] < 2:
            continue

        port_ret = aligned["portfolio"]
        bench_ret = aligned["benchmark"]
        active_ret = port_ret - bench_ret

        port_metrics = core._calc_perf_metrics(port_ret)
        bench_metrics = core._calc_perf_metrics(bench_ret)
        if not port_metrics or not bench_metrics:
            continue

        active_total = port_metrics["Total Return"] - bench_metrics["Total Return"]
        active_std = active_ret.std()
        tracking_error = active_std * np.sqrt(252) if pd.notna(active_std) else np.nan
        information_ratio = (
            (active_ret.mean() / active_std) * np.sqrt(252)
            if active_std is not None and active_std > 0 and np.isfinite(active_std)
            else np.nan
        )
        non_zero_active = (active_ret != 0).sum()
        hit_ratio = (active_ret > 0).sum() / non_zero_active if non_zero_active > 0 else np.nan

        rows.append(
            {
                "Benchmark": benchmark_name,
                "Portfolio Return": port_metrics["Total Return"],
                "Benchmark Return": bench_metrics["Total Return"],
                "Active Return": active_total,
                "Portfolio Vol": port_metrics["Annualized Volatility"],
                "Benchmark Vol": bench_metrics["Annualized Volatility"],
                "Vol Spread": port_metrics["Annualized Volatility"] - bench_metrics["Annualized Volatility"],
                "Portfolio Sharpe": port_metrics["Sharpe Ratio"],
                "Benchmark Sharpe": bench_metrics["Sharpe Ratio"],
                "Portfolio Sortino": port_metrics["Sortino Ratio"],
                "Benchmark Sortino": bench_metrics["Sortino Ratio"],
                "Hit Ratio": hit_ratio,
                "Tracking Error": tracking_error,
                "Information Ratio": information_ratio,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Active Return", ascending=False).reset_index(drop=True)


def _build_monthly_relative_tables(portfolio_returns, benchmark_prices):
    if portfolio_returns is None or benchmark_prices is None or benchmark_prices.empty:
        return pd.DataFrame(), pd.DataFrame()

    port_ret = pd.Series(portfolio_returns).dropna()
    if port_ret.empty:
        return pd.DataFrame(), pd.DataFrame()
    port_ret.index = pd.to_datetime(port_ret.index, errors="coerce")
    port_ret = port_ret[~port_ret.index.isna()].sort_index()
    if port_ret.empty:
        return pd.DataFrame(), pd.DataFrame()

    benchmark_returns = benchmark_prices.ffill().pct_change(fill_method=None).dropna(how="all")
    if benchmark_returns.empty:
        return pd.DataFrame(), pd.DataFrame()
    benchmark_returns.index = pd.to_datetime(benchmark_returns.index, errors="coerce")
    benchmark_returns = benchmark_returns.loc[~benchmark_returns.index.isna()].sort_index()
    if benchmark_returns.empty:
        return pd.DataFrame(), pd.DataFrame()

    monthly_port = (1 + port_ret).resample("ME").prod() - 1
    monthly_bench = (1 + benchmark_returns).resample("ME").prod() - 1
    monthly = monthly_bench.copy()
    monthly.insert(0, "Portfolio", monthly_port.reindex(monthly_bench.index))
    monthly = monthly.dropna(how="all")
    if monthly.empty:
        return pd.DataFrame(), pd.DataFrame()

    active = pd.DataFrame(index=monthly.index)
    for benchmark_name in monthly_bench.columns:
        active[benchmark_name] = monthly["Portfolio"] - monthly[benchmark_name]

    monthly.index = monthly.index.strftime("%Y-%m")
    active.index = active.index.strftime("%Y-%m")
    return monthly, active


def _format_percent_or_na(value):
    return "N/A" if pd.isna(value) else f"{value:.2%}"


def _format_ratio_or_na(value):
    return "N/A" if pd.isna(value) else f"{value:.2f}"


def _render_benchmark_comparison_table(df, title):
    if df is None or df.empty:
        st.write("Benchmark comparison data not available.")
        return

    display_df = df.copy()
    pct_cols = [
        "Portfolio Return",
        "Benchmark Return",
        "Active Return",
        "Portfolio Vol",
        "Benchmark Vol",
        "Vol Spread",
        "Hit Ratio",
        "Tracking Error",
    ]
    ratio_cols = [
        "Portfolio Sharpe",
        "Benchmark Sharpe",
        "Portfolio Sortino",
        "Benchmark Sortino",
        "Information Ratio",
    ]
    for col in pct_cols:
        display_df[col] = display_df[col].apply(_format_percent_or_na)
    for col in ratio_cols:
        display_df[col] = display_df[col].apply(_format_ratio_or_na)
    st.markdown(create_manual_html_table(display_df, title=title), unsafe_allow_html=True)

def _resolve_weekly_report_source(uploaded_file):
    if uploaded_file is not None:
        return core._coerce_excel_input(uploaded_file), uploaded_file.name, "uploaded"

    base_dirs = [
        ROOT_DIR,
        Path.cwd(),
        Path.home() / "Desktop" / "Workspace" / "Team",
    ]
    candidates = []

    for key in ["WEEKLY_REPORT_XLSX_PATH", "HOLDINGS3_XLSX_PATH"]:
        env_path = os.getenv(key)
        if env_path:
            resolved_env = core._resolve_normalized_path(env_path)
            candidates.append(resolved_env if resolved_env else Path(env_path))

        secret_path = core._get_streamlit_secret(key)
        if secret_path:
            resolved_secret = core._resolve_normalized_path(secret_path)
            candidates.append(resolved_secret if resolved_secret else Path(secret_path))

    default_dirs = [
        ROOT_DIR,
        Path.cwd(),
        Path.home() / "Desktop" / "Workspace" / "Team",
    ]
    for file_name in WEEKLY_REPORT_DEFAULT_FILES:
        for folder in default_dirs:
            candidates.append(folder / file_name)

    data_path = next((p for p in candidates if p is not None and Path(p).exists()), None)
    if data_path is None:
        for file_name in WEEKLY_REPORT_DEFAULT_FILES:
            data_path = core._find_file_by_name(file_name, base_dirs)
            if data_path is not None:
                break

    if data_path is None:
        return None, None, None
    return data_path, data_path.name, "local"

def render_weekly_report_page():
    st.subheader("📑 Weekly Meeting Report Generator")
    uploaded_file_ce = st.sidebar.file_uploader("Optional override upload for report", type=['xlsx'], key="rep")
    data_source, source_label, source_type = _resolve_weekly_report_source(uploaded_file_ce)

    if data_source is None:
        st.info("기본 파일 `2026_멀티.xlsx`를 찾지 못했습니다. 필요하면 업로드하거나 `WEEKLY_REPORT_XLSX_PATH`를 지정하세요.")
        return

    if source_type == "local":
        st.sidebar.caption(f"Using default local file: {source_label}")

    with st.spinner("Generating Report Data..."):
        res = load_cash_equity_data(data_source)
        df_perf, df_last, df_contrib, country_daily, df_daily_stock, logs, err = res

    if err:
        st.error(err)
    elif df_perf is not None:
        st.caption(f"Data source: {source_label}")
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
        valid_report_dates = df_perf.index[df_perf.index <= report_date]
        if len(valid_report_dates) == 0:
            st.error("선택한 기준일 이전의 포트폴리오 수익률 데이터가 없습니다.")
            return
        effective_report_date = valid_report_dates[-1]
        if pd.Timestamp(report_date).normalize() != pd.Timestamp(effective_report_date).normalize():
            st.caption(f"Using latest available trading date: {effective_report_date.date()}")
        report_date = effective_report_date

        factor_prices = download_factors(df_perf.index.min(), report_date, return_prices=True)
        factor_returns = align_factor_returns(df_perf[ret_col].index, factor_prices)
        _, factor_contrib, _ = perform_factor_regression(df_perf[ret_col], factor_returns)
        
        dates = {
            'WTD': df_perf.index[df_perf.index <= report_date][-1] - pd.to_timedelta(df_perf.index[df_perf.index <= report_date][-1].weekday(), unit='D'),
            'MTD': report_date.replace(day=1),
            'QTD': report_date.replace(month=((report_date.month-1)//3)*3+1, day=1),
            'YTD': report_date.replace(month=1, day=1)
        }

        benchmark_prices = _download_weekly_report_benchmarks(min(dates.values()), report_date)
        df_perf_cut = df_perf[df_perf.index <= report_date]
        df_stock_cut = df_daily_stock[df_daily_stock['기준일자'] <= report_date]
        benchmark_prices_cut = benchmark_prices[benchmark_prices.index <= report_date] if not benchmark_prices.empty else pd.DataFrame()
        monthly_benchmark_returns, monthly_active_returns = _build_monthly_relative_tables(
            df_perf_cut[ret_col],
            benchmark_prices_cut,
        )
        if factor_contrib is not None:
            factor_contrib_cut = factor_contrib[factor_contrib.index <= report_date]
        else: factor_contrib_cut = None
        
        def calc_period_stats(start_dt, label, benchmark_prices):
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
            benchmark_comparison = pd.DataFrame()
            if benchmark_prices is not None and not benchmark_prices.empty:
                sub_px = benchmark_prices[(benchmark_prices.index >= start_dt) & (benchmark_prices.index <= report_date)]
                sub_px = sub_px.ffill().bfill()
                if not sub_px.empty:
                    idx_ret = sub_px.iloc[-1] / sub_px.iloc[0] - 1
                    benchmark_comparison = _build_benchmark_comparison_table(sub_perf[ret_col], sub_px)
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
            portfolio_risk = core._calc_risk_metrics(sub_perf[ret_col])
            benchmark_risk = {}
            if sub_px is not None and not sub_px.empty:
                bench_ret = sub_px.pct_change(fill_method=None).dropna(how='all')
                for col in bench_ret.columns:
                    metrics = core._calc_risk_metrics(bench_ret[col])
                    if metrics:
                        benchmark_risk[col] = metrics
            return {'label': label, 'ret': cum_ret, 'pnl': abs_pnl, 'top5': top5, 'bot5': bot5, 
                    'ctry': ctry_contrib, 'sect': sect_contrib, 'factor': f_cont, 'indices': idx_ret,
                    'hedge_contrib': hedge_contrib, 'hedge_pnl_krw': hedge_pnl_krw,
                    'risk': {'portfolio': portfolio_risk, 'benchmarks': benchmark_risk},
                    'benchmark_comparison': benchmark_comparison}

        tabs = st.tabs(["Summary Report", "WTD", "MTD", "QTD", "YTD"])
        stats_res = {}
        for p in ['WTD', 'MTD', 'QTD', 'YTD']:
            stats_res[p] = calc_period_stats(dates[p], p, benchmark_prices_cut)
            with tabs[list(dates.keys()).index(p) + 1]:
                if stats_res[p]:
                    st.markdown(f"### {p} Performance ({dates[p].date()} ~ {report_date.date()})")
                    if not stats_res[p]['indices'].empty:
                        idx_df = stats_res[p]['indices'].sort_values(ascending=False).reset_index()
                        idx_df.columns = ['Index', 'Return']
                        idx_df['Return'] = idx_df['Return'].apply(lambda x: f"{x:.2%}")
                        st.markdown(create_manual_html_table(idx_df, title="Global Index Returns"), unsafe_allow_html=True)
                    if not stats_res[p]['benchmark_comparison'].empty:
                        st.markdown("#### Benchmark Relative Performance")
                        _render_benchmark_comparison_table(
                            stats_res[p]['benchmark_comparison'],
                            title=f"{p} Relative Metrics vs Benchmarks",
                        )
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
            st.caption(
                "Benchmark and monthly relative tables use FMP daily prices "
                f"({', '.join(WEEKLY_REPORT_BENCHMARKS.keys())})."
            )

            if not monthly_active_returns.empty:
                st.markdown("#### Monthly Outperformance / Underperformance vs Benchmarks")
                heatmap = go.Figure(
                    data=go.Heatmap(
                        z=monthly_active_returns.T.values,
                        x=monthly_active_returns.index,
                        y=monthly_active_returns.columns,
                        colorscale="RdYlGn",
                        zmid=0,
                        colorbar=dict(title="Active Return"),
                    )
                )
                heatmap.update_layout(height=320, xaxis_title="Month", yaxis_title="Benchmark")
                st.plotly_chart(heatmap, use_container_width=True)

                monthly_display = pd.DataFrame({"Month": monthly_active_returns.index})
                monthly_display["Portfolio"] = monthly_benchmark_returns["Portfolio"].values
                for benchmark_name in monthly_active_returns.columns:
                    monthly_display[f"{benchmark_name} Return"] = monthly_benchmark_returns[benchmark_name].values
                    monthly_display[f"{benchmark_name} Active"] = monthly_active_returns[benchmark_name].values

                for col in monthly_display.columns[1:]:
                    monthly_display[col] = monthly_display[col].apply(_format_percent_or_na)
                st.markdown(
                    create_manual_html_table(monthly_display, title="Monthly Relative Performance vs Benchmarks"),
                    unsafe_allow_html=True,
                )

            summary_rows = []
            for period_name in ['WTD', 'MTD', 'QTD', 'YTD']:
                period_stats = stats_res.get(period_name)
                if period_stats is None:
                    continue
                comparison_df = period_stats.get('benchmark_comparison')
                if comparison_df is None or comparison_df.empty:
                    continue
                best_row = comparison_df.sort_values("Active Return", ascending=False).iloc[0]
                summary_rows.append(
                    {
                        "Period": period_name,
                        "Portfolio Return": period_stats["ret"],
                        "Best Relative Benchmark": best_row["Benchmark"],
                        "Active Return": best_row["Active Return"],
                        "Information Ratio": best_row["Information Ratio"],
                        "Hit Ratio": best_row["Hit Ratio"],
                    }
                )
            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                summary_df["Portfolio Return"] = summary_df["Portfolio Return"].apply(_format_percent_or_na)
                summary_df["Active Return"] = summary_df["Active Return"].apply(_format_percent_or_na)
                summary_df["Information Ratio"] = summary_df["Information Ratio"].apply(_format_ratio_or_na)
                summary_df["Hit Ratio"] = summary_df["Hit Ratio"].apply(_format_percent_or_na)
                st.markdown(create_manual_html_table(summary_df, title="Best Relative Benchmark by Period"), unsafe_allow_html=True)

            txt = f"**[Portfolio Weekly Update - {report_date.date()}]**\n\n"
            
            wtd = stats_res.get('WTD')
            if wtd:
                txt += f"**1. WTD Performance:** {wtd['ret']:.2%} ({wtd['pnl']:,.0f} KRW)\n"
                if not wtd['top5'].empty: txt += f"   - **Lead:** {wtd['top5'].iloc[0]['종목명']} (+{wtd['top5'].iloc[0]['Contrib_KRW']:.2%})\n"
                if not wtd['factor'].empty: txt += f"   - **Factor:** Driven by {wtd['factor'].idxmax()} (+{wtd['factor'].max():.2%})\n"
                if not wtd['benchmark_comparison'].empty:
                    best_wtd = wtd['benchmark_comparison'].sort_values("Active Return", ascending=False).iloc[0]
                    txt += f"   - **Relative:** Outperformed {best_wtd['Benchmark']} by {best_wtd['Active Return']:.2%} (IR {best_wtd['Information Ratio']:.2f}).\n"
            
            mtd = stats_res.get('MTD')
            if mtd: 
                txt += f"**2. MTD:** {mtd['ret']:.2%}."
                if not mtd['sect'].empty:
                    txt += f" Best Sector: {mtd['sect'].idxmax()}."
                txt += "\n"
                if not mtd['factor'].empty: txt += f"   - Factor: {mtd['factor'].idxmax()} style worked well.\n"
                if not mtd['benchmark_comparison'].empty:
                    best_mtd = mtd['benchmark_comparison'].sort_values("Active Return", ascending=False).iloc[0]
                    txt += f"   - Relative: Best vs {best_mtd['Benchmark']} ({best_mtd['Active Return']:.2%}, Hit Ratio {best_mtd['Hit Ratio']:.2%}).\n"
            
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
