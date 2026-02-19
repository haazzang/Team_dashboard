from dashboard.core import *  # noqa: F401,F403

# ==============================================================================
# [MAIN UI]
# ==============================================================================
menu = st.sidebar.radio(
    "Dashboard Menu",
    ["📌 Portfolio Snapshot", "Total Portfolio (Team PNL)", "Cash Equity Analysis", "📑 Weekly Report Generator", "📊 Swap Report Analysis"],
)

if menu == "📌 Portfolio Snapshot":
    st.subheader("📌 Portfolio Snapshot (2026_멀티.xlsx)")
    script_dir = Path(__file__).resolve().parent
    base_dirs = [
        script_dir,
        Path.cwd(),
        Path.home() / "Desktop" / "Workspace" / "Team",
    ]
    candidates = []
    env_path = os.getenv("PORTFOLIO_XLSX_PATH")
    if env_path:
        resolved_env = _resolve_normalized_path(env_path)
        candidates.append(resolved_env if resolved_env else Path(env_path))
    if hasattr(st, "secrets") and "PORTFOLIO_XLSX_PATH" in st.secrets:
        secret_path = st.secrets["PORTFOLIO_XLSX_PATH"]
        resolved_secret = _resolve_normalized_path(secret_path)
        candidates.append(resolved_secret if resolved_secret else Path(secret_path))
    candidates.extend([
        script_dir / "2026_멀티.xlsx",
        Path.cwd() / "2026_멀티.xlsx",
        Path.home() / "Desktop" / "Workspace" / "Team" / "2026_멀티.xlsx",
    ])
    data_path = next((p for p in candidates if p is not None and p.exists()), None)
    if data_path is None:
        data_path = _find_file_by_name("2026_멀티.xlsx", base_dirs)

    uploaded_snapshot = None
    if data_path is None:
        st.error("2026_멀티.xlsx 파일을 찾지 못했습니다.")
        st.caption("컨테이너/배포 환경에서는 로컬 파일이 보이지 않을 수 있습니다.")
        st.caption(
            "해결: 1) 파일을 앱 폴더에 복사하거나 2) PORTFOLIO_XLSX_PATH 환경변수/시크릿으로 경로를 지정하세요."
        )
        st.caption("검색 경로: " + " , ".join(str(p) for p in candidates if p is not None))
        uploaded_snapshot = st.file_uploader("Upload '2026_멀티.xlsx'", type=['xlsx'], key="snapshot_upload")
        if uploaded_snapshot is None:
            st.stop()

    with st.spinner("포트폴리오 현황 불러오는 중..."):
        if uploaded_snapshot is not None:
            df_snapshot, err = load_portfolio_snapshot_upload(uploaded_snapshot)
        else:
            df_snapshot, err = load_portfolio_snapshot(str(data_path), data_path.stat().st_mtime)

    if err or df_snapshot is None or df_snapshot.empty:
        st.error(f"데이터 로드 실패: {err}")
    else:
        latest_date = df_snapshot["기준일자"].max()
        latest_all = df_snapshot[df_snapshot["기준일자"] == latest_date].copy()

        if "원화평가금액" not in latest_all.columns and {"외화평가금액", "평가환율"}.issubset(latest_all.columns):
            latest_all["원화평가금액"] = latest_all["외화평가금액"] * latest_all["평가환율"]

        id_col = "심볼" if "심볼" in latest_all.columns else ("종목코드" if "종목코드" in latest_all.columns else "종목명")
        latest_all["Ticker_ID"] = latest_all[id_col].fillna(latest_all.get("종목명", latest_all[id_col]))
        if "종목명" not in latest_all.columns:
            latest_all["종목명"] = latest_all["Ticker_ID"]
        if "통화" not in latest_all.columns:
            latest_all["통화"] = "N/A"

        def _resolve_symbol(row):
            candidates = [
                row.get(id_col),
                row.get("Ticker_ID"),
                row.get("심볼") if "심볼" in latest_all.columns else None,
                row.get("종목코드") if "종목코드" in latest_all.columns else None,
            ]
            for base in candidates:
                sym = normalize_yf_ticker(base, row.get("통화"))
                if sym:
                    return sym
            return None
        latest_all["YF_Symbol"] = latest_all.apply(_resolve_symbol, axis=1)

        if "섹터" not in latest_all.columns:
            tickers = tuple(sorted(latest_all["YF_Symbol"].dropna().unique()))
            sector_map = fetch_sectors_cached(tickers)
            latest_all["섹터"] = latest_all["YF_Symbol"].map(sector_map).fillna("Unknown")
        else:
            latest_all["섹터"] = latest_all["섹터"].fillna("Unknown")
            unknown_mask = (
                latest_all["섹터"].astype(str).str.strip().str.upper().isin(["", "UNKNOWN", "NAN", "NONE"])
            )
            unknown_tickers = tuple(sorted(latest_all.loc[unknown_mask, "YF_Symbol"].dropna().unique()))
            if unknown_tickers:
                sector_map = fetch_sectors_cached(unknown_tickers)
                refilled = latest_all.loc[unknown_mask, "YF_Symbol"].map(sector_map)
                latest_all.loc[unknown_mask, "섹터"] = refilled.fillna(latest_all.loc[unknown_mask, "섹터"])
            latest_all["섹터"] = latest_all["섹터"].replace("", "Unknown").fillna("Unknown")

        etf_mask = pd.Series(False, index=latest_all.index)
        if "상품구분" in latest_all.columns:
            etf_mask |= latest_all["상품구분"].apply(is_etf_product_type)
        if "종목명" in latest_all.columns:
            etf_mask |= latest_all["종목명"].apply(is_etf_value)
        etf_tickers = tuple(sorted(latest_all["YF_Symbol"].dropna().unique()))
        if etf_tickers:
            etf_symbol_map = fetch_etf_flags_cached(etf_tickers)
            etf_mask |= latest_all["YF_Symbol"].map(etf_symbol_map).fillna(False)
        latest_all.loc[etf_mask, "섹터"] = "ETF"
        latest_all["Is_ETF"] = etf_mask

        if "원화평가금액" not in latest_all.columns:
            st.error("원화평가금액 컬럼이 없어 비중 계산이 불가능합니다.")
            latest_all = pd.DataFrame()

        if latest_all.empty:
            st.warning("최신일 데이터가 없습니다.")
            st.stop()

        latest_for_weights = latest_all[latest_all["원화평가금액"] != 0].copy()
        if latest_for_weights.empty:
            latest_for_weights = latest_all.copy()

        latest_for_weights["Group_ID"] = latest_for_weights["YF_Symbol"].fillna(latest_for_weights["Ticker_ID"])
        holdings = latest_for_weights.groupby("Group_ID", dropna=False).agg(
            원화평가금액=("원화평가금액", "sum"),
            종목명=("종목명", "first"),
            섹터=("섹터", "first"),
            통화=("통화", "first"),
            Ticker_ID=("Ticker_ID", "first"),
            Is_ETF=("Is_ETF", "first"),
        ).reset_index()
        total_mv = holdings["원화평가금액"].sum()
        holdings["Weight"] = np.where(total_mv > 0, holdings["원화평가금액"] / total_mv, 0)
        holdings["Label"] = holdings["종목명"].astype(str) + " (" + holdings["Group_ID"].astype(str) + ")"

        etf_weight = holdings.loc[holdings["섹터"] == "ETF", "Weight"].sum() if not holdings.empty else 0
        holdings_non_etf = holdings[holdings["섹터"] != "ETF"].copy()
        total_mv_non_etf = holdings_non_etf["원화평가금액"].sum()
        sector_weights = holdings_non_etf.groupby("섹터")["원화평가금액"].sum().sort_values(ascending=False)
        sector_weights_pct = sector_weights / total_mv_non_etf if total_mv_non_etf else sector_weights * 0

        currency_weights = holdings.groupby("통화")["원화평가금액"].sum().sort_values(ascending=False)
        currency_weights_pct = currency_weights / total_mv if total_mv else currency_weights * 0

        total_pnl = latest_all.get("원화총평가손익", pd.Series(0, index=latest_all.index)).sum() + \
                    latest_all.get("원화총매매손익", pd.Series(0, index=latest_all.index)).sum()
        fx_pnl = latest_all.get("환손익", pd.Series(0, index=latest_all.index)).sum()
        local_pnl = total_pnl - fx_pnl

        hhi = (holdings["Weight"] ** 2).sum() if not holdings.empty else 0
        eff_n = (1 / hhi) if hhi > 0 else 0
        top5_weight = holdings["Weight"].nlargest(5).sum() if not holdings.empty else 0

        # 시뮬레이션용 holdings 데이터 준비
        holdings["YF_Symbol"] = holdings["Group_ID"]

        # 탭 생성: 현황 / 전일 등락률 / 시뮬레이션
        tab_snapshot, tab_heatmap, tab_simulation = st.tabs([
            "📊 포트폴리오 현황",
            "🟩 전일 등락률 Heatmap",
            "🔬 포트폴리오 시뮬레이션",
        ])

        with tab_snapshot:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("기준일자", latest_date.strftime("%Y-%m-%d"))
            c2.metric("총 AUM (KRW)", f"{total_mv:,.0f}")
            c3.metric("Total PnL (KRW)", f"{total_pnl:,.0f}")
            c4.metric("Local PnL (KRW)", f"{local_pnl:,.0f}")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("보유 종목 수", f"{len(holdings):,}")
            c6.metric("Top 5 비중", f"{top5_weight:.2%}")
            c7.metric("HHI", f"{hhi:.4f}")
            c8.metric("유효 보유 종목 수", f"{eff_n:.1f}")

            # 포트폴리오 변동성 계산
            current_weights = holdings.set_index("YF_Symbol")["Weight"].to_dict()
            with st.spinner("변동성 계산 중..."):
                vol_metrics = calculate_portfolio_volatility(current_weights, lookback_days=30)

                if vol_metrics:
                    c9, c10, c11, c12 = st.columns(4)
                    c9.metric("30일 변동성 (연율)", f"{vol_metrics['annual_volatility']:.2%}")
                    c10.metric("30일 MDD", f"{vol_metrics['max_drawdown']:.2%}")
                    c11.metric("VaR 95%", f"{vol_metrics['var_95']:.2%}")
                    c12.metric("VaR 99%", f"{vol_metrics['var_99']:.2%}")

                st.caption(f"ETF 비중: {etf_weight:.2%} (섹터 비중/비교는 ETF 제외 기준)")

                st.markdown("#### 🧬 지수 복제율 (Holdings-based)")
                st.caption("보유 비중 기준 최근 수익률로 계산한 SPX/NDX 복제율(R²)입니다.")
                rep_lookback = st.slider(
                    "Lookback window (trading days)",
                    min_value=20,
                    max_value=252,
                    value=120,
                    step=5,
                    key="rep_snapshot_lookback",
                )
                with st.spinner("복제율 계산 중..."):
                    port_ret = calculate_portfolio_returns(current_weights, lookback_days=rep_lookback)

                if port_ret.empty:
                    st.warning("복제율을 계산할 수 없습니다. (가격 데이터 부족)")
                else:
                    rep_bm = download_replication_benchmarks(port_ret.index.min(), port_ret.index.max())
                    if rep_bm.empty:
                        st.warning("Replication benchmark data download failed.")
                    else:
                        spx_ret = rep_bm['SPX'].reindex(port_ret.index) if 'SPX' in rep_bm.columns else pd.Series(dtype=float)
                        ndx_ret = rep_bm['NDX'].reindex(port_ret.index) if 'NDX' in rep_bm.columns else pd.Series(dtype=float)

                        spx_r2 = calculate_alpha_beta(port_ret, spx_ret)[2] if not spx_ret.empty else np.nan
                        ndx_r2 = calculate_alpha_beta(port_ret, ndx_ret)[2] if not ndx_ret.empty else np.nan

                        c_rep1, c_rep2 = st.columns(2)
                        spx_disp = f"{spx_r2:.2%}" if pd.notnull(spx_r2) else "N/A"
                        ndx_disp = f"{ndx_r2:.2%}" if pd.notnull(ndx_r2) else "N/A"
                        c_rep1.metric("SPX Replication (R²)", spx_disp)
                        c_rep2.metric("NDX Replication (R²)", ndx_disp)

                        if len(port_ret) >= 20:
                            rep_window = st.slider(
                                "Rolling window (trading days)",
                                min_value=20,
                                max_value=min(252, len(port_ret)),
                                value=min(60, len(port_ret)),
                                step=5,
                                key="rep_snapshot_window",
                            )
                            fig_rep = go.Figure()
                            if not spx_ret.empty:
                                spx_series = calculate_rolling_r2(port_ret, spx_ret, window=rep_window)
                                if not spx_series.empty:
                                    fig_rep.add_trace(go.Scatter(x=spx_series.index, y=spx_series, name="SPX R²"))
                            if not ndx_ret.empty:
                                ndx_series = calculate_rolling_r2(port_ret, ndx_ret, window=rep_window)
                                if not ndx_series.empty:
                                    fig_rep.add_trace(go.Scatter(x=ndx_series.index, y=ndx_series, name="NDX R²"))

                            if fig_rep.data:
                                fig_rep.update_layout(yaxis_title="R²", xaxis_title="Date", yaxis=dict(range=[0, 1]))
                                st.plotly_chart(fig_rep, use_container_width=True)
                            else:
                                st.write("Insufficient data to compute rolling replication.")
                        else:
                            st.write("Not enough data for rolling replication (need 20+ data points).")

                st.markdown("#### 🔎 보유 종목 비중")
                top_holdings = holdings.sort_values("Weight", ascending=False).head(15)
                fig_hold = go.Figure(
                    data=go.Bar(
                    x=top_holdings["Label"],
                    y=top_holdings["Weight"],
                    text=[f"{w:.2%}" for w in top_holdings["Weight"]],
                    textposition="auto",
                )
            )
            fig_hold.update_layout(yaxis_tickformat=".1%", xaxis_title="", yaxis_title="Weight")
            st.plotly_chart(fig_hold, use_container_width=True)

            st.markdown("#### 🧭 섹터 비중")
            fig_sector = go.Figure(
                data=go.Pie(labels=sector_weights_pct.index, values=sector_weights_pct.values, hole=0.45)
            )
            fig_sector.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_sector, use_container_width=True)

            st.markdown("#### 💱 통화 비중")
            fig_fx = go.Figure(
                data=go.Bar(
                    x=currency_weights_pct.index.astype(str),
                    y=currency_weights_pct.values,
                    text=[f"{w:.2%}" for w in currency_weights_pct.values],
                    textposition="auto",
                )
            )
            fig_fx.update_layout(yaxis_tickformat=".1%", xaxis_title="", yaxis_title="Weight")
            st.plotly_chart(fig_fx, use_container_width=True)

            st.markdown("#### 🆚 S&P 500 섹터 Weight 차이 (Portfolio - SP500)")
            with st.spinner("S&P 500 섹터 가중치 계산 중..."):
                sp_sector = fetch_sp500_sector_weights()
            if sp_sector.empty:
                st.warning("S&P 500 섹터 데이터를 불러오지 못했습니다.")
            else:
                port_sector = sector_weights_pct.copy()
                if "Unknown" in port_sector.index:
                    port_sector = port_sector.drop("Unknown")
                sp_sector = sp_sector.drop("Unknown", errors="ignore")
                if port_sector.sum() > 0:
                    port_sector = port_sector / port_sector.sum()
                if sp_sector.sum() > 0:
                    sp_sector = sp_sector / sp_sector.sum()

                all_sectors = sorted(set(port_sector.index) | set(sp_sector.index))
                diff = port_sector.reindex(all_sectors, fill_value=0) - sp_sector.reindex(all_sectors, fill_value=0)
                colors = np.where(diff.values >= 0, "#16a34a", "#dc2626")
                fig_diff = go.Figure(
                    data=go.Bar(x=diff.index, y=diff.values, marker_color=colors)
                )
                fig_diff.update_layout(yaxis_tickformat=".1%", xaxis_title="", yaxis_title="Weight Difference")
                st.plotly_chart(fig_diff, use_container_width=True)

                comp = pd.DataFrame({
                    "Portfolio": port_sector.reindex(all_sectors, fill_value=0),
                    "S&P 500": sp_sector.reindex(all_sectors, fill_value=0),
                })
                comp["Diff"] = comp["Portfolio"] - comp["S&P 500"]
                st.dataframe(comp.style.format("{:.2%}"))

            # 포트폴리오 베타 (30/60/90일, 국가별)
            st.markdown("#### 📊 포트폴리오 베타 (국가별 벤치마크)")
            st.caption("각 국가 벤치마크 대비 포트폴리오 베타입니다. 베타 > 1이면 벤치마크보다 변동성이 큽니다.")

            with st.spinner("베타 계산 중..."):
                beta_results = calculate_portfolio_beta_multi_period(current_weights, [30, 60, 90])

            if beta_results:
                # 베타 데이터 정리
                beta_data = []
                for period, benchmarks in beta_results.items():
                    for bench_name, beta_val in benchmarks.items():
                        beta_data.append({
                            "기간": period,
                            "벤치마크": bench_name,
                            "베타": beta_val
                        })

                if beta_data:
                    df_beta = pd.DataFrame(beta_data)

                    # 베타 차트 (그룹 바 차트)
                    fig_beta = go.Figure()

                    periods = ["30D", "60D", "90D"]
                    colors = {"30D": "#3b82f6", "60D": "#8b5cf6", "90D": "#ec4899"}

                    for period in periods:
                        period_data = df_beta[df_beta["기간"] == period]
                        if not period_data.empty:
                            fig_beta.add_trace(go.Bar(
                                name=period,
                                x=period_data["벤치마크"],
                                y=period_data["베타"],
                                text=[f"{v:.2f}" for v in period_data["베타"]],
                                textposition="auto",
                                marker_color=colors.get(period, "#6366f1")
                            ))

                    fig_beta.add_hline(y=1.0, line_dash="dash", line_color="red",
                                      annotation_text="Beta = 1", annotation_position="right")
                    fig_beta.update_layout(
                        barmode="group",
                        xaxis_title="",
                        yaxis_title="Beta",
                        legend_title="기간",
                        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
                    )
                    st.plotly_chart(fig_beta, use_container_width=True)

                    # 베타 테이블
                    df_beta_pivot = df_beta.pivot(index="벤치마크", columns="기간", values="베타")
                    df_beta_pivot = df_beta_pivot.reindex(columns=["30D", "60D", "90D"])
                    st.dataframe(df_beta_pivot.style.format("{:.3f}").background_gradient(cmap="RdYlGn_r", vmin=0.5, vmax=1.5))
            else:
                st.warning("베타를 계산할 수 없습니다.")

            # 팩터 익스포저
            st.markdown("#### 📈 팩터 익스포저 (Factor Exposure)")
            st.caption("팩터 ETF 대비 베타로 측정한 익스포저입니다. (60일 기준)")

            with st.spinner("팩터 익스포저 계산 중..."):
                factor_exposures = calculate_portfolio_factor_exposure(current_weights, lookback_days=60)

            if factor_exposures:
                # 팩터 익스포저 차트
                factors = list(factor_exposures.keys())
                values = list(factor_exposures.values())

                colors_factor = ["#16a34a" if v >= 0 else "#dc2626" for v in values]

                fig_factor = go.Figure(data=go.Bar(
                    x=factors,
                    y=values,
                    text=[f"{v:.2f}" for v in values],
                    textposition="auto",
                    marker_color=colors_factor
                ))
                fig_factor.add_hline(y=1.0, line_dash="dash", line_color="gray",
                                    annotation_text="Exposure = 1", annotation_position="right")
                fig_factor.update_layout(
                    xaxis_title="",
                    yaxis_title="Factor Beta",
                )
                st.plotly_chart(fig_factor, use_container_width=True)

                # 팩터 익스포저 테이블
                df_factor = pd.DataFrame({
                    "팩터": factors,
                    "익스포저": values
                })
                df_factor = df_factor.sort_values("익스포저", ascending=False)
                st.dataframe(df_factor.style.format({"익스포저": "{:.3f}"}).background_gradient(
                    subset=["익스포저"], cmap="RdYlGn", vmin=-0.5, vmax=1.5
                ))
            else:
                st.warning("팩터 익스포저를 계산할 수 없습니다.")

            st.markdown("#### 📋 보유 종목 상세")
            show_cols = ["Group_ID", "종목명", "섹터", "통화", "원화평가금액", "Weight"]
            show_cols = [c for c in show_cols if c in holdings.columns]
            st.dataframe(holdings.sort_values("Weight", ascending=False)[show_cols].style.format({
                "원화평가금액": "{:,.0f}",
                "Weight": "{:.2%}",
            }))

        with tab_heatmap:
            st.markdown("### 🟩 보유 종목 전일 등락률 Heatmap")
            st.caption("사이즈는 원화평가금액, 색상은 최근 거래일 기준 전일 등락률입니다.")

            with st.spinner("전일 등락률 계산 중..."):
                prev_ret = fetch_prev_day_returns(tuple(holdings["YF_Symbol"].dropna().unique()))

            heatmap_df = holdings.copy()
            heatmap_df = heatmap_df[heatmap_df["YF_Symbol"].notna()].copy()
            heatmap_df = heatmap_df.merge(prev_ret, on="YF_Symbol", how="left")
            heatmap_df["Heatmap_Label"] = (
                heatmap_df["종목명"].astype(str) + " (" + heatmap_df["YF_Symbol"].astype(str) + ")"
            )
            heatmap_df["최근거래일_문자열"] = pd.to_datetime(heatmap_df["최근거래일"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("-")
            heatmap_df["직전거래일_문자열"] = pd.to_datetime(heatmap_df["직전거래일"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("-")

            if heatmap_df.empty:
                st.warning("Heatmap을 표시할 보유 종목이 없습니다.")
            else:
                plot_df = heatmap_df.dropna(subset=["전일등락률"]).copy()
                plot_df = plot_df[plot_df["원화평가금액"] > 0].copy()

                if plot_df.empty:
                    st.warning("최근 2개 거래일 가격 데이터가 없어 heatmap을 표시할 수 없습니다.")
                else:
                    max_abs = float(np.nanmax(np.abs(plot_df["전일등락률"].values)))
                    if not np.isfinite(max_abs) or max_abs == 0:
                        max_abs = 0.01

                    fig_daily_heatmap = px.treemap(
                        plot_df,
                        path=[px.Constant("Portfolio"), "섹터", "Heatmap_Label"],
                        values="원화평가금액",
                        color="전일등락률",
                        color_continuous_scale=[(0.0, "#b91c1c"), (0.5, "#f8fafc"), (1.0, "#15803d")],
                        color_continuous_midpoint=0.0,
                        custom_data=["YF_Symbol", "Weight", "전일등락률", "최근거래일_문자열", "직전거래일_문자열"],
                    )
                    fig_daily_heatmap.update_traces(
                        texttemplate="%{label}<br>%{customdata[2]:+.2%}",
                        hovertemplate=(
                            "<b>%{label}</b><br>"
                            "Ticker: %{customdata[0]}<br>"
                            "Weight: %{customdata[1]:.2%}<br>"
                            "MV: %{value:,.0f} KRW<br>"
                            "전일 등락률: %{customdata[2]:+.2%}<br>"
                            "최근 거래일: %{customdata[3]}<br>"
                            "직전 거래일: %{customdata[4]}<extra></extra>"
                        ),
                    )
                    fig_daily_heatmap.update_coloraxes(
                        cmin=-max_abs,
                        cmax=max_abs,
                        colorbar=dict(title="전일 등락률", tickformat=".2%"),
                    )
                    fig_daily_heatmap.update_layout(margin=dict(t=30, l=10, r=10, b=10))
                    st.plotly_chart(fig_daily_heatmap, use_container_width=True)

                ranked = heatmap_df.dropna(subset=["전일등락률"]).sort_values("전일등락률")
                if not ranked.empty:
                    top_loser = ranked.iloc[0]
                    top_gainer = ranked.iloc[-1]
                    coverage = len(ranked) / len(heatmap_df) if len(heatmap_df) > 0 else 0
                    c_gain, c_loss, c_cov = st.columns(3)
                    c_gain.metric("Top Gainer", str(top_gainer["종목명"]), f"{top_gainer['전일등락률']:+.2%}")
                    c_loss.metric("Top Loser", str(top_loser["종목명"]), f"{top_loser['전일등락률']:+.2%}")
                    c_cov.metric("가격 커버리지", f"{coverage:.1%}")

                missing_count = int(heatmap_df["전일등락률"].isna().sum())
                if missing_count > 0:
                    st.info(f"{missing_count}개 종목은 가격 데이터 부족으로 전일 등락률이 표시되지 않습니다.")

                st.markdown("#### 📋 전일 등락률 상세")
                detail_cols = [
                    "YF_Symbol", "종목명", "섹터", "Weight", "원화평가금액",
                    "전일등락률", "최근거래일", "직전거래일", "최근종가", "직전종가",
                ]
                detail_cols = [c for c in detail_cols if c in heatmap_df.columns]
                detail_df = heatmap_df.sort_values("전일등락률", ascending=False)
                st.dataframe(
                    detail_df[detail_cols].style.format({
                        "Weight": "{:.2%}",
                        "원화평가금액": "{:,.0f}",
                        "전일등락률": "{:+.2%}",
                        "최근종가": "{:,.2f}",
                        "직전종가": "{:,.2f}",
                    }).format({
                        "최근거래일": lambda x: x.strftime("%Y-%m-%d") if pd.notnull(x) else "-",
                        "직전거래일": lambda x: x.strftime("%Y-%m-%d") if pd.notnull(x) else "-",
                    })
                )

        with tab_simulation:
            st.markdown("### 🔬 포트폴리오 비중 시뮬레이션")
            st.caption("기존 종목의 비중을 조절하거나 신규 종목을 추가하여 NAV 변화를 시뮬레이션합니다. (전일 종가 기준)")

            # 시뮬레이션 설정
            col_sim_settings1, col_sim_settings2 = st.columns(2)

            with col_sim_settings1:
                sim_days = st.slider("시뮬레이션 기간 (일)", min_value=5, max_value=90, value=30, step=5)

            with col_sim_settings2:
                # 추가 현금 투입 옵션
                use_additional_cash = st.checkbox("💰 추가 현금 투입", value=False,
                                                  help="비중 상향 시 기존 NAV를 유지하면서 추가 자금을 투입합니다.")

            additional_cash_krw = 0
            if use_additional_cash:
                st.markdown("#### 💵 추가 현금 투입 설정")

                cash_input_col1, cash_input_col2 = st.columns(2)
                with cash_input_col1:
                    additional_cash_krw = st.number_input(
                        "추가 투입 금액 (KRW)",
                        min_value=0,
                        max_value=100_000_000_000,  # 1000억
                        value=0,
                        step=100_000_000,  # 1억 단위
                        format="%d",
                        help="추가로 투입할 현금 (원화)"
                    )
                with cash_input_col2:
                    if additional_cash_krw > 0:
                        new_total_nav = total_mv + additional_cash_krw
                        st.metric("새로운 총 NAV", f"₩{new_total_nav:,.0f}")
                        st.caption(f"기존 NAV: ₩{total_mv:,.0f} + 추가: ₩{additional_cash_krw:,.0f}")

            st.markdown("---")

            # 두 개의 컬럼으로 나누기
            col_existing, col_new = st.columns(2)

            with col_existing:
                st.markdown("#### 📈 기존 종목 비중 조절")
                st.caption("비중을 조절할 종목을 선택하고 새로운 비중(%)을 입력하세요.")

                # 기존 종목 리스트 (상위 20개)
                top_20 = holdings.sort_values("Weight", ascending=False).head(20)

                # 세션 상태 초기화
                if "weight_adjustments" not in st.session_state:
                    st.session_state.weight_adjustments = {}

                # 종목별 슬라이더
                weight_adjustments = {}
                for idx, row in top_20.iterrows():
                    ticker = row["YF_Symbol"]
                    current_weight = row["Weight"] * 100  # %로 변환
                    label = f"{row['종목명']} ({ticker})"

                    new_weight = st.number_input(
                        label,
                        min_value=0.0,
                        max_value=100.0,
                        value=float(current_weight),
                        step=0.5,
                        format="%.2f",
                        key=f"weight_{ticker}",
                        help=f"현재 비중: {current_weight:.2f}%"
                    )
                    if abs(new_weight - current_weight) > 0.01:
                        weight_adjustments[ticker] = new_weight / 100  # 비율로 변환

            with col_new:
                st.markdown("#### ➕ 신규 종목 추가")
                st.caption("추가할 종목 티커, 마켓, 비중(%)을 입력하세요.")

                # 마켓 옵션
                market_options = {
                    "US": "미국 (기본)",
                    "JP": "일본 (.T)",
                    "HK": "홍콩 (.HK)",
                    "KR": "한국 (.KS)"
                }

                # 신규 종목 입력 (최대 5개)
                new_positions = []
                for i in range(5):
                    c1, c2, c3 = st.columns([2, 1, 1])
                    with c1:
                        new_ticker_raw = st.text_input(
                            f"티커 {i+1}",
                            value="",
                            placeholder="예: AAPL, 7203, 0700",
                            key=f"new_ticker_{i}"
                        ).upper().strip()
                    with c2:
                        new_market = st.selectbox(
                            f"마켓 {i+1}",
                            options=list(market_options.keys()),
                            format_func=lambda x: market_options[x],
                            key=f"new_market_{i}"
                        )
                    with c3:
                        new_weight_pct = st.number_input(
                            f"비중 % {i+1}",
                            min_value=0.0,
                            max_value=50.0,
                            value=0.0,
                            step=0.5,
                            format="%.2f",
                            key=f"new_weight_{i}"
                        )

                    # 티커 변환 (마켓에 따라 suffix 추가)
                    if new_ticker_raw and new_weight_pct > 0:
                        if new_market == "JP":
                            final_ticker = f"{new_ticker_raw}.T" if not new_ticker_raw.endswith(".T") else new_ticker_raw
                        elif new_market == "HK":
                            # 홍콩은 4자리 숫자로 패딩
                            if new_ticker_raw.isdigit():
                                final_ticker = f"{new_ticker_raw.zfill(4)}.HK"
                            elif not new_ticker_raw.endswith(".HK"):
                                final_ticker = f"{new_ticker_raw}.HK"
                            else:
                                final_ticker = new_ticker_raw
                        elif new_market == "KR":
                            final_ticker = f"{new_ticker_raw}.KS" if not new_ticker_raw.endswith(".KS") else new_ticker_raw
                        else:
                            final_ticker = new_ticker_raw

                        new_positions.append({
                            "ticker": final_ticker,
                            "weight": new_weight_pct / 100,
                            "market": new_market
                        })

                if new_positions:
                    st.caption("**추가될 종목:**")
                    for pos in new_positions:
                        st.caption(f"  • {pos['ticker']} ({pos['weight']*100:.1f}%)")

            st.markdown("---")

            # 시뮬레이션 실행 버튼
            if st.button("🚀 시뮬레이션 실행", type="primary", use_container_width=True):
                if not weight_adjustments and not new_positions and additional_cash_krw == 0:
                    st.warning("비중을 조절하거나 신규 종목을 추가하거나 추가 현금을 투입해주세요.")
                else:
                    # 시뮬레이션 NAV 결정 (추가 현금 포함 여부)
                    sim_base_nav = total_mv + additional_cash_krw if use_additional_cash else total_mv

                    with st.spinner("시뮬레이션 실행 중..."):
                        result = simulate_portfolio_nav(
                            holdings_df=holdings,
                            weight_adjustments=weight_adjustments,
                            new_positions=new_positions,
                            base_nav=sim_base_nav,
                            simulation_days=sim_days,
                            additional_cash=additional_cash_krw if use_additional_cash else 0,
                            original_nav=total_mv
                        )

                    if result is None:
                        st.error("시뮬레이션 실행 실패. 가격 데이터를 가져올 수 없습니다.")
                    else:
                        st.success("시뮬레이션 완료!")

                        # 추가 현금 투입 시 안내 메시지
                        if use_additional_cash and additional_cash_krw > 0:
                            st.info(f"💰 **추가 현금 투입 모드**: 기존 NAV ₩{total_mv:,.0f} + 추가 현금 ₩{additional_cash_krw:,.0f} = 새 NAV ₩{sim_base_nav:,.0f}")

                        # 결과 표시
                        st.markdown("### 📊 시뮬레이션 결과")

                        # NAV 비교 차트
                        fig_nav = go.Figure()
                        fig_nav.add_trace(go.Scatter(
                            x=result["original_nav"].index,
                            y=result["original_nav"].values,
                            mode="lines",
                            name="원래 포트폴리오",
                            line=dict(color="#6366f1", width=2)
                        ))
                        fig_nav.add_trace(go.Scatter(
                            x=result["sim_nav"].index,
                            y=result["sim_nav"].values,
                            mode="lines",
                            name=f"시뮬레이션 포트폴리오{' (추가 현금)' if additional_cash_krw > 0 else ''}",
                            line=dict(color="#f97316", width=2, dash="dash")
                        ))
                        fig_nav.update_layout(
                            title="포트폴리오 NAV 비교",
                            xaxis_title="날짜",
                            yaxis_title="NAV (KRW)",
                            yaxis_tickformat=",",
                            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig_nav, use_container_width=True)

                        # 성과 비교 메트릭
                        orig_final = result["original_nav"].iloc[-1]
                        sim_final = result["sim_nav"].iloc[-1]
                        orig_return = (orig_final / total_mv - 1) * 100
                        sim_return = (sim_final / sim_base_nav - 1) * 100
                        nav_diff = sim_final - orig_final
                        return_diff = sim_return - orig_return

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("원래 NAV", f"{orig_final:,.0f}")
                        m2.metric("시뮬레이션 NAV", f"{sim_final:,.0f}", delta=f"{nav_diff:,.0f}")
                        m3.metric("원래 수익률", f"{orig_return:.2f}%")
                        m4.metric("시뮬레이션 수익률", f"{sim_return:.2f}%", delta=f"{return_diff:+.2f}%")

                        # 변동성 비교
                        st.markdown("### 📉 변동성 비교 (30일 기준)")

                        with st.spinner("변동성 계산 중..."):
                            orig_vol = calculate_portfolio_volatility(result["original_weights"], lookback_days=30)
                            sim_vol = calculate_portfolio_volatility(result["sim_weights"], lookback_days=30)

                        if orig_vol and sim_vol:
                            vol_diff = sim_vol["annual_volatility"] - orig_vol["annual_volatility"]
                            mdd_diff = sim_vol["max_drawdown"] - orig_vol["max_drawdown"]
                            var95_diff = sim_vol["var_95"] - orig_vol["var_95"]

                            v1, v2, v3, v4 = st.columns(4)
                            v1.metric("원래 변동성 (연율)", f"{orig_vol['annual_volatility']:.2%}")
                            v2.metric("시뮬레이션 변동성", f"{sim_vol['annual_volatility']:.2%}",
                                     delta=f"{vol_diff:+.2%}",
                                     delta_color="inverse")  # 변동성 증가는 빨간색
                            v3.metric("원래 VaR 95%", f"{orig_vol['var_95']:.2%}")
                            v4.metric("시뮬레이션 VaR 95%", f"{sim_vol['var_95']:.2%}",
                                     delta=f"{var95_diff:+.2%}",
                                     delta_color="inverse")

                            v5, v6, v7, v8 = st.columns(4)
                            v5.metric("원래 MDD", f"{orig_vol['max_drawdown']:.2%}")
                            v6.metric("시뮬레이션 MDD", f"{sim_vol['max_drawdown']:.2%}",
                                     delta=f"{mdd_diff:+.2%}",
                                     delta_color="inverse")
                            v7.metric("원래 VaR 99%", f"{orig_vol['var_99']:.2%}")
                            v8.metric("시뮬레이션 VaR 99%", f"{sim_vol['var_99']:.2%}")

                            # 리스크/리턴 요약
                            st.markdown("#### 리스크-리턴 요약")
                            orig_sharpe = orig_return / (orig_vol['annual_volatility'] * 100) if orig_vol['annual_volatility'] > 0 else 0
                            sim_sharpe = sim_return / (sim_vol['annual_volatility'] * 100) if sim_vol['annual_volatility'] > 0 else 0
                            sharpe_diff = sim_sharpe - orig_sharpe

                            rs1, rs2, rs3 = st.columns(3)
                            rs1.metric("원래 샤프비율", f"{orig_sharpe:.3f}")
                            rs2.metric("시뮬레이션 샤프비율", f"{sim_sharpe:.3f}", delta=f"{sharpe_diff:+.3f}")
                            rs3.metric("리스크 조정 효과",
                                      "개선" if sharpe_diff > 0 else "악화" if sharpe_diff < 0 else "동일",
                                      delta=f"{sharpe_diff:+.3f}")
                        else:
                            st.warning("변동성을 계산할 수 없습니다.")

                        # 비중 변경 요약
                        st.markdown("### 📋 비중 변경 요약")

                        # 변경된 비중 테이블
                        changes = []
                        for ticker, new_w in result["sim_weights"].items():
                            orig_w = result["original_weights"].get(ticker, 0)
                            if abs(new_w - orig_w) > 0.0001:
                                # 종목명 찾기
                                name_row = holdings[holdings["YF_Symbol"] == ticker]
                                name = name_row["종목명"].values[0] if len(name_row) > 0 else ticker
                                changes.append({
                                    "티커": ticker,
                                    "종목명": name,
                                    "원래 비중": orig_w,
                                    "변경 비중": new_w,
                                    "변경폭": new_w - orig_w
                                })

                        if changes:
                            df_changes = pd.DataFrame(changes)
                            df_changes = df_changes.sort_values("변경폭", ascending=False)
                            st.dataframe(
                                df_changes.style.format({
                                    "원래 비중": "{:.2%}",
                                    "변경 비중": "{:.2%}",
                                    "변경폭": "{:+.2%}"
                                }).background_gradient(subset=["변경폭"], cmap="RdYlGn", vmin=-0.1, vmax=0.1),
                                use_container_width=True
                            )
                        else:
                            st.info("비중 변경 사항이 없습니다.")

                        # 매매 주수 계산
                        st.markdown("### 🛒 매매 주문 (Trade Orders)")
                        if use_additional_cash and additional_cash_krw > 0:
                            st.caption(f"목표 비중 달성을 위해 매매해야 하는 주수입니다. (새 NAV ₩{sim_base_nav:,.0f} 기준, 각 국가별 최종 영업일 종가)")
                        else:
                            st.caption("목표 비중 달성을 위해 매매해야 하는 주수입니다. (각 국가별 최종 영업일 종가 기준)")

                        with st.spinner("매매 주수 계산 중..."):
                            trades = calculate_trade_shares(
                                result["original_weights"],
                                result["sim_weights"],
                                sim_base_nav,  # 추가 현금 포함된 NAV 사용
                                holdings,
                                new_positions,
                                original_nav=total_mv,
                                additional_cash=additional_cash_krw if use_additional_cash else 0
                            )

                        if trades:
                            df_trades = pd.DataFrame(trades)

                            # 매수/매도 분리
                            buy_trades = df_trades[df_trades["매매"] == "매수"].copy()
                            sell_trades = df_trades[df_trades["매매"] == "매도"].copy()

                            col_buy, col_sell = st.columns(2)

                            with col_buy:
                                st.markdown("#### 🟢 매수 주문")
                                if not buy_trades.empty:
                                    buy_display = buy_trades[["티커", "종목명", "주수", "현지통화가격", "통화", "매매금액(KRW)"]].copy()
                                    buy_display = buy_display.sort_values("매매금액(KRW)", ascending=False)
                                    st.dataframe(
                                        buy_display.style.format({
                                            "주수": "{:,.0f}",
                                            "현지통화가격": "{:,.2f}",
                                            "매매금액(KRW)": "{:,.0f}"
                                        }),
                                        use_container_width=True
                                    )
                                    total_buy_krw = buy_trades["매매금액(KRW)"].sum()
                                    st.metric("총 매수 금액 (KRW)", f"{total_buy_krw:,.0f}")
                                else:
                                    st.info("매수할 종목이 없습니다.")

                            with col_sell:
                                st.markdown("#### 🔴 매도 주문")
                                if not sell_trades.empty:
                                    sell_display = sell_trades[["티커", "종목명", "주수", "현지통화가격", "통화", "매매금액(KRW)"]].copy()
                                    sell_display = sell_display.sort_values("매매금액(KRW)", ascending=False)
                                    st.dataframe(
                                        sell_display.style.format({
                                            "주수": "{:,.0f}",
                                            "현지통화가격": "{:,.2f}",
                                            "매매금액(KRW)": "{:,.0f}"
                                        }),
                                        use_container_width=True
                                    )
                                    total_sell_krw = sell_trades["매매금액(KRW)"].sum()
                                    st.metric("총 매도 금액 (KRW)", f"{total_sell_krw:,.0f}")
                                else:
                                    st.info("매도할 종목이 없습니다.")

                            # 전체 매매 상세 테이블
                            with st.expander("📊 전체 매매 상세 보기"):
                                df_trades_display = df_trades[[
                                    "티커", "종목명", "매매", "주수", "현지통화가격", "통화",
                                    "원래비중", "목표비중", "비중변화", "매매금액(현지)", "매매금액(KRW)"
                                ]].copy()
                                df_trades_display = df_trades_display.sort_values("매매금액(KRW)", ascending=False)

                                st.dataframe(
                                    df_trades_display.style.format({
                                        "주수": "{:,.0f}",
                                        "현지통화가격": "{:,.2f}",
                                        "원래비중": "{:.2%}",
                                        "목표비중": "{:.2%}",
                                        "비중변화": "{:+.2%}",
                                        "매매금액(현지)": "{:,.2f}",
                                        "매매금액(KRW)": "{:,.0f}"
                                    }),
                                    use_container_width=True
                                )

                                # 순 현금 흐름
                                total_buy = buy_trades["매매금액(KRW)"].sum() if not buy_trades.empty else 0
                                total_sell = sell_trades["매매금액(KRW)"].sum() if not sell_trades.empty else 0
                                net_cash = total_sell - total_buy

                                st.markdown("---")
                                nc1, nc2, nc3 = st.columns(3)
                                nc1.metric("총 매수", f"₩{total_buy:,.0f}")
                                nc2.metric("총 매도", f"₩{total_sell:,.0f}")
                                nc3.metric("순 현금 흐름", f"₩{net_cash:,.0f}",
                                          delta="현금 유입" if net_cash > 0 else "현금 유출" if net_cash < 0 else "균형")
                        else:
                            st.info("매매할 종목이 없습니다.")

                        # 섹터 비중 비교
                        st.markdown("### 🧭 섹터 비중 비교")

                        sector_map = result.get("sector_map", {})

                        # 원래 포트폴리오 섹터 비중
                        orig_sector_weights = {}
                        for ticker, weight in result["original_weights"].items():
                            sector = sector_map.get(ticker, "Unknown")
                            orig_sector_weights[sector] = orig_sector_weights.get(sector, 0) + weight

                        # 시뮬레이션 포트폴리오 섹터 비중
                        sim_sector_weights = {}
                        for ticker, weight in result["sim_weights"].items():
                            sector = sector_map.get(ticker, "Unknown")
                            sim_sector_weights[sector] = sim_sector_weights.get(sector, 0) + weight

                        # 모든 섹터 합치기
                        all_sectors_sim = sorted(set(orig_sector_weights.keys()) | set(sim_sector_weights.keys()))

                        sector_comparison = []
                        for sector in all_sectors_sim:
                            orig_w = orig_sector_weights.get(sector, 0)
                            sim_w = sim_sector_weights.get(sector, 0)
                            sector_comparison.append({
                                "섹터": sector,
                                "원래 비중": orig_w,
                                "시뮬레이션 비중": sim_w,
                                "변경폭": sim_w - orig_w
                            })

                        df_sector_comp = pd.DataFrame(sector_comparison)
                        df_sector_comp = df_sector_comp.sort_values("시뮬레이션 비중", ascending=False)

                        # 섹터 비중 차트
                        col_sector1, col_sector2 = st.columns(2)

                        with col_sector1:
                            fig_sector_orig = go.Figure(data=go.Pie(
                                labels=list(orig_sector_weights.keys()),
                                values=list(orig_sector_weights.values()),
                                hole=0.4,
                                title="원래 포트폴리오"
                            ))
                            fig_sector_orig.update_traces(textinfo="percent+label")
                            st.plotly_chart(fig_sector_orig, use_container_width=True)

                        with col_sector2:
                            fig_sector_sim = go.Figure(data=go.Pie(
                                labels=list(sim_sector_weights.keys()),
                                values=list(sim_sector_weights.values()),
                                hole=0.4,
                                title="시뮬레이션 포트폴리오"
                            ))
                            fig_sector_sim.update_traces(textinfo="percent+label")
                            st.plotly_chart(fig_sector_sim, use_container_width=True)

                        # 섹터 비중 변화 바 차트
                        df_sector_diff = df_sector_comp[df_sector_comp["변경폭"].abs() > 0.0001].copy()
                        if not df_sector_diff.empty:
                            colors_sector = np.where(df_sector_diff["변경폭"].values >= 0, "#16a34a", "#dc2626")
                            fig_sector_diff = go.Figure(data=go.Bar(
                                x=df_sector_diff["섹터"],
                                y=df_sector_diff["변경폭"],
                                marker_color=colors_sector,
                                text=[f"{v:+.1%}" for v in df_sector_diff["변경폭"]],
                                textposition="auto"
                            ))
                            fig_sector_diff.update_layout(
                                title="섹터 비중 변화",
                                yaxis_tickformat=".1%",
                                xaxis_title="",
                                yaxis_title="비중 변화"
                            )
                            st.plotly_chart(fig_sector_diff, use_container_width=True)

                        # 섹터 비중 테이블
                        st.dataframe(
                            df_sector_comp.style.format({
                                "원래 비중": "{:.2%}",
                                "시뮬레이션 비중": "{:.2%}",
                                "변경폭": "{:+.2%}"
                            }).background_gradient(subset=["변경폭"], cmap="RdYlGn", vmin=-0.05, vmax=0.05),
                            use_container_width=True
                        )

                        # 팩터 익스포저
                        st.markdown("### 📈 팩터 익스포저 (Factor Exposure)")
                        st.caption("팩터 ETF 대비 베타로 측정한 익스포저입니다.")

                        with st.spinner("팩터 익스포저 계산 중..."):
                            orig_exposure = calculate_factor_exposure(
                                result["original_weights"],
                                result["returns"],
                                sim_days
                            )
                            sim_exposure = calculate_factor_exposure(
                                result["sim_weights"],
                                result["returns"],
                                sim_days
                            )

                        if orig_exposure or sim_exposure:
                            all_factors = sorted(set(orig_exposure.keys()) | set(sim_exposure.keys()))

                            factor_comparison = []
                            for factor in all_factors:
                                orig_exp = orig_exposure.get(factor, 0)
                                sim_exp = sim_exposure.get(factor, 0)
                                factor_comparison.append({
                                    "팩터": factor,
                                    "원래 익스포저": orig_exp,
                                    "시뮬레이션 익스포저": sim_exp,
                                    "변경폭": sim_exp - orig_exp
                                })

                            df_factor = pd.DataFrame(factor_comparison)

                            # 팩터 익스포저 비교 차트
                            fig_factor = go.Figure()
                            fig_factor.add_trace(go.Bar(
                                name="원래 포트폴리오",
                                x=df_factor["팩터"],
                                y=df_factor["원래 익스포저"],
                                marker_color="#6366f1"
                            ))
                            fig_factor.add_trace(go.Bar(
                                name="시뮬레이션 포트폴리오",
                                x=df_factor["팩터"],
                                y=df_factor["시뮬레이션 익스포저"],
                                marker_color="#f97316"
                            ))
                            fig_factor.update_layout(
                                title="팩터 익스포저 비교 (베타)",
                                barmode="group",
                                xaxis_title="",
                                yaxis_title="베타",
                                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
                            )
                            st.plotly_chart(fig_factor, use_container_width=True)

                            # 팩터 익스포저 테이블
                            st.dataframe(
                                df_factor.style.format({
                                    "원래 익스포저": "{:.3f}",
                                    "시뮬레이션 익스포저": "{:.3f}",
                                    "변경폭": "{:+.3f}"
                                }).background_gradient(subset=["변경폭"], cmap="RdYlGn", vmin=-0.2, vmax=0.2),
                                use_container_width=True
                            )
                        else:
                            st.warning("팩터 익스포저를 계산할 수 없습니다.")

elif menu == "Total Portfolio (Team PNL)":
    st.subheader("📊 Total Team Portfolio Dashboard")
    uploaded_file = st.sidebar.file_uploader("Upload 'Team_PNL.xlsx'", type=['xlsx'], key="pnl")

    pnl_file = uploaded_file
    pnl_data_path = None
    if pnl_file is None:
        script_dir = Path(__file__).resolve().parent
        pnl_base_dirs = [
            script_dir,
            Path.cwd(),
            Path.home() / "Desktop" / "Workspace" / "Team",
        ]
        pnl_candidates = []
        env_path = os.getenv("TEAM_PNL_XLSX_PATH")
        if env_path:
            resolved_env = _resolve_normalized_path(env_path)
            pnl_candidates.append(resolved_env if resolved_env else Path(env_path))
        if hasattr(st, "secrets") and "TEAM_PNL_XLSX_PATH" in st.secrets:
            secret_path = st.secrets["TEAM_PNL_XLSX_PATH"]
            resolved_secret = _resolve_normalized_path(secret_path)
            pnl_candidates.append(resolved_secret if resolved_secret else Path(secret_path))
        pnl_candidates.extend([
            script_dir / "Team_PNL.xlsx",
            Path.cwd() / "Team_PNL.xlsx",
            Path.home() / "Desktop" / "Workspace" / "Team" / "Team_PNL.xlsx",
        ])
        pnl_data_path = next((p for p in pnl_candidates if p is not None and p.exists()), None)
        if pnl_data_path is None:
            pnl_data_path = _find_file_by_name("Team_PNL.xlsx", pnl_base_dirs)
        if pnl_data_path is not None:
            pnl_file = pnl_data_path
            st.sidebar.caption(f"Using local file: {pnl_data_path.name}")

    if pnl_file:
        df_pnl, df_pos, err = load_team_pnl_data(pnl_file)
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
                
                # Add Benchmarks
                bm_returns = download_benchmarks_all(df_pnl.index.min(), df_pnl.index.max())
                if not bm_returns.empty:
                    bm_cum = (1 + bm_returns).cumprod() - 1
                    for col in ['US', 'KR']:
                         if col in bm_cum.columns:
                            fig.add_trace(go.Scatter(x=bm_cum.index, y=bm_cum[col], name=f"{col} BM", line=dict(width=1, dash='dash')))
                st.plotly_chart(fig, use_container_width=True)
            
            with t2:
                stats = pd.DataFrame(index=df_daily_ret.columns)
                stats['Volatility'] = df_daily_ret.std() * np.sqrt(252)
                stats['Sharpe'] = (df_daily_ret.mean() / df_daily_ret.std() * np.sqrt(252)).fillna(0)
                stats['Win Rate'] = (df_daily_ret > 0).sum() / (df_daily_ret != 0).sum()
                
                # Profit Factor
                gp = df_daily_ret[df_daily_ret > 0].sum()
                gl = df_daily_ret[df_daily_ret < 0].sum().abs()
                stats['Profit Factor'] = (gp / gl).fillna(0)
                
                stats['MDD'] = ((1+df_daily_ret).cumprod() / (1+df_daily_ret).cumprod().cummax() - 1).min()
                stats['Total Return'] = df_user_ret.iloc[-1]
                
                disp = stats.copy()
                for c in disp.columns:
                    if c in ['Sharpe', 'Profit Factor']: disp[c] = disp[c].apply(lambda x: f"{x:.2f}")
                    else: disp[c] = disp[c].apply(lambda x: f"{x:.2%}")
                
                disp.insert(0, 'Strategy', disp.index)
                disp['Strategy'] = disp['Strategy'].apply(lambda x: x.split('_')[0])
                st.markdown(create_manual_html_table(disp), unsafe_allow_html=True)

            with t3:
                corr = df_daily_ret.corr()
                fig_corr = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale='RdBu', zmin=-1, zmax=1))
                fig_corr.update_layout(height=700)
                st.plotly_chart(fig_corr, use_container_width=True)
                
            with t4:
                df_assets = download_cross_assets(df_pnl.index.min(), df_pnl.index.max())
                if not df_assets.empty:
                    df_assets = df_assets.reindex(df_user_ret.index, method='ffill').pct_change().fillna(0)
                    comb = pd.concat([df_daily_ret, df_assets], axis=1).corr()
                    sub_corr = comb.loc[df_daily_ret.columns, df_assets.columns]
                    fig_cross = go.Figure(data=go.Heatmap(z=sub_corr.values, x=sub_corr.columns, y=sub_corr.index, colorscale='RdBu', zmin=-1, zmax=1))
                    st.plotly_chart(fig_cross, use_container_width=True)
                else: st.write("Data not available.")
            
            with t5:
                st.subheader("Simulation")
                c_in, c_out = st.columns([1,3])
                with c_in:
                    weights = {}
                    for col in df_daily_ret.columns:
                        weights[col] = st.slider(col, 0.0, 1.0, 1.0/len(df_daily_ret.columns), 0.05, key=f"sim_{col}")
                with c_out:
                    w_series = pd.Series(weights)
                    sim_daily = df_daily_ret.mul(w_series, axis=1).sum(axis=1)
                    sim_cum = (1 + sim_daily).cumprod() - 1
                    
                    fig_sim = go.Figure()
                    fig_sim.add_trace(go.Scatter(x=sim_cum.index, y=sim_cum, name="Simulated", line=dict(color='red')))
                    act_daily = df_pnl.sum(axis=1).div(df_pos.sum(axis=1)).fillna(0)
                    act_cum = (1 + act_daily).cumprod() - 1
                    fig_sim.add_trace(go.Scatter(x=act_cum.index, y=act_cum, name="Actual", line=dict(color='grey', dash='dot')))
                    st.plotly_chart(fig_sim, use_container_width=True)
        else: st.error(err)
    else:
        st.info("Team_PNL.xlsx 파일을 업로드하거나 TEAM_PNL_XLSX_PATH 환경변수/시크릿으로 경로를 지정하세요.")

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

elif menu == "📑 Weekly Report Generator":
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

elif menu == "📊 Swap Report Analysis":
    st.subheader("📊 Swap Report Analysis (JMLNKWGE)")

    # SQLite DB 경로 - 여러 경로 시도
    possible_paths = [
        Path(__file__).resolve().parent / 'swap_reports.db',
        Path('/Users/hyejinha/Desktop/Workspace/Team/swap_reports.db'),
        Path.cwd() / 'swap_reports.db'
    ]

    SWAP_DB_FILE = None
    for p in possible_paths:
        if p.exists():
            SWAP_DB_FILE = p
            break

    def load_swap_data():
        """SQLite DB에서 Swap Report 데이터 로드"""
        if SWAP_DB_FILE is None or not SWAP_DB_FILE.exists():
            return None, None, None, None

        conn = sqlite3.connect(SWAP_DB_FILE)

        # 리포트 목록
        df_reports = pd.read_sql_query('''
            SELECT * FROM reports ORDER BY report_date DESC
        ''', conn)

        # Underlying 데이터
        df_underlying = pd.read_sql_query('''
            SELECT u.*, r.report_date
            FROM underlying u
            JOIN reports r ON u.report_id = r.id
            ORDER BY r.report_date DESC, u.market_value_usd DESC
        ''', conn)

        # Overview 데이터
        df_overview = pd.read_sql_query('''
            SELECT o.*, r.report_date
            FROM overview o
            JOIN reports r ON o.report_id = r.id
            ORDER BY r.report_date DESC
        ''', conn)

        # Und Summary 데이터
        df_und = pd.read_sql_query('''
            SELECT us.*, r.report_date
            FROM und_summary us
            JOIN reports r ON us.report_id = r.id
            ORDER BY r.report_date DESC
        ''', conn)

        conn.close()
        return df_reports, df_underlying, df_overview, df_und

    # 데이터 로드
    df_reports, df_underlying, df_overview, df_und = load_swap_data()

    if df_reports is None or df_reports.empty:
        st.warning("Swap Report 데이터가 없습니다.")
        st.info("""
        **데이터를 가져오려면:**
        1. Google Cloud Console에서 Gmail API 설정
        2. credentials.json 파일을 이 폴더에 저장
        3. 터미널에서 실행: `python automation/swap/swap_report_fetcher.py`
        """)

        # 수동 업로드 옵션
        st.markdown("---")
        st.markdown("### 📤 수동 업로드")
        uploaded_file = st.file_uploader("Swap Report Excel 파일 업로드", type=['xlsx'])

        if uploaded_file:
            try:
                xlsx = pd.ExcelFile(uploaded_file)
                st.success(f"파일 로드 성공! 시트: {xlsx.sheet_names}")

                # 시트 선택
                selected_sheet = st.selectbox("분석할 시트 선택", xlsx.sheet_names)
                df_preview = pd.read_excel(xlsx, sheet_name=selected_sheet)
                st.dataframe(df_preview)
            except Exception as e:
                st.error(f"파일 로드 실패: {e}")
    else:
        # 데이터가 있는 경우
        st.success(f"총 {len(df_reports)}개 리포트 로드됨")

        # 날짜 범위
        df_reports['report_date'] = pd.to_datetime(df_reports['report_date'])
        min_date = df_reports['report_date'].min()
        max_date = df_reports['report_date'].max()
        st.caption(f"데이터 기간: {min_date.strftime('%Y-%m-%d')} ~ {max_date.strftime('%Y-%m-%d')}")

        # 탭 생성
        tab_overview, tab_holdings, tab_pnl, tab_attribution = st.tabs([
            "📈 Overview", "📋 Holdings", "💰 P&L Analysis", "🎯 Attribution"
        ])

        with tab_overview:
            st.markdown("### 포트폴리오 Overview")

            # 날짜 선택
            available_dates = sorted(df_reports['report_date'].unique(), reverse=True)
            selected_date = st.selectbox(
                "리포트 날짜 선택",
                available_dates,
                format_func=lambda x: pd.Timestamp(x).strftime('%Y-%m-%d')
            )

            # 선택된 날짜의 데이터
            df_date_underlying = df_underlying[
                pd.to_datetime(df_underlying['report_date']) == pd.Timestamp(selected_date)
            ].copy()

            if not df_date_underlying.empty:
                # 주요 지표
                total_mv = df_date_underlying['market_value_usd'].sum()
                total_pnl = df_date_underlying['pnl_usd'].sum()
                total_return = (df_date_underlying['pnl_usd'].sum() / total_mv * 100) if total_mv > 0 else 0

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total NAV (USD)", f"${total_mv:,.0f}")
                c2.metric("Daily P&L (USD)", f"${total_pnl:,.0f}",
                         delta=f"{total_return:.2f}%")
                c3.metric("# Holdings", f"{len(df_date_underlying)}")
                c4.metric("Report Date", pd.Timestamp(selected_date).strftime('%Y-%m-%d'))

                # Top/Bottom Performers
                col_top, col_bottom = st.columns(2)

                with col_top:
                    st.markdown("#### 🟢 Top 5 Performers")
                    top5 = df_date_underlying.nlargest(5, 'pnl_usd')[['ticker', 'name', 'pnl_usd', 'pnl_pct', 'contribution']]
                    st.dataframe(top5.style.format({
                        'pnl_usd': '${:,.0f}',
                        'pnl_pct': '{:.2f}%',
                        'contribution': '{:.2f}%'
                    }))

                with col_bottom:
                    st.markdown("#### 🔴 Bottom 5 Performers")
                    bottom5 = df_date_underlying.nsmallest(5, 'pnl_usd')[['ticker', 'name', 'pnl_usd', 'pnl_pct', 'contribution']]
                    st.dataframe(bottom5.style.format({
                        'pnl_usd': '${:,.0f}',
                        'pnl_pct': '{:.2f}%',
                        'contribution': '{:.2f}%'
                    }))

        with tab_holdings:
            st.markdown("### 보유 종목 상세")

            # 날짜 선택
            selected_date_holdings = st.selectbox(
                "날짜 선택",
                available_dates,
                format_func=lambda x: pd.Timestamp(x).strftime('%Y-%m-%d'),
                key="holdings_date"
            )

            df_holdings = df_underlying[
                pd.to_datetime(df_underlying['report_date']) == pd.Timestamp(selected_date_holdings)
            ].copy()

            if not df_holdings.empty:
                # 비중 파이 차트
                col_chart, col_table = st.columns([1, 1])

                with col_chart:
                    # 상위 15개 + 기타
                    top_15 = df_holdings.nlargest(15, 'weight')
                    others_weight = df_holdings[~df_holdings['ticker'].isin(top_15['ticker'])]['weight'].sum()

                    labels = list(top_15['ticker']) + (['Others'] if others_weight > 0 else [])
                    values = list(top_15['weight']) + ([others_weight] if others_weight > 0 else [])

                    fig_pie = go.Figure(data=go.Pie(labels=labels, values=values, hole=0.4))
                    fig_pie.update_traces(textinfo='percent+label')
                    fig_pie.update_layout(title="Portfolio Weights")
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col_table:
                    # 섹터별 비중
                    if 'sector' in df_holdings.columns and df_holdings['sector'].notna().any():
                        sector_weights = df_holdings.groupby('sector')['weight'].sum().sort_values(ascending=False)
                        fig_sector = go.Figure(data=go.Bar(
                            x=sector_weights.index,
                            y=sector_weights.values,
                            text=[f"{v:.1f}%" for v in sector_weights.values],
                            textposition='auto'
                        ))
                        fig_sector.update_layout(title="Sector Allocation", yaxis_tickformat=".1%")
                        st.plotly_chart(fig_sector, use_container_width=True)

                # 전체 Holdings 테이블
                st.markdown("#### 전체 보유 종목")
                display_cols = ['ticker', 'name', 'quantity', 'price', 'market_value_usd', 'weight', 'pnl_usd', 'pnl_pct', 'sector']
                display_cols = [c for c in display_cols if c in df_holdings.columns]
                st.dataframe(
                    df_holdings[display_cols].sort_values('weight', ascending=False).style.format({
                        'market_value_usd': '${:,.0f}',
                        'weight': '{:.2f}%',
                        'pnl_usd': '${:,.0f}',
                        'pnl_pct': '{:.2f}%',
                        'price': '${:,.2f}',
                        'quantity': '{:,.0f}'
                    }),
                    use_container_width=True
                )

        with tab_pnl:
            st.markdown("### P&L 분석")

            # 일별 P&L 계산
            daily_pnl = df_underlying.groupby('report_date').agg({
                'market_value_usd': 'sum',
                'pnl_usd': 'sum'
            }).reset_index()
            daily_pnl['report_date'] = pd.to_datetime(daily_pnl['report_date'])
            daily_pnl = daily_pnl.sort_values('report_date')
            daily_pnl['daily_return'] = daily_pnl['pnl_usd'] / daily_pnl['market_value_usd'].shift(1)
            daily_pnl['cumulative_pnl'] = daily_pnl['pnl_usd'].cumsum()
            daily_pnl['cumulative_return'] = (1 + daily_pnl['daily_return'].fillna(0)).cumprod() - 1

            # P&L 차트
            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Bar(
                x=daily_pnl['report_date'],
                y=daily_pnl['pnl_usd'],
                name='Daily P&L',
                marker_color=np.where(daily_pnl['pnl_usd'] >= 0, '#16a34a', '#dc2626')
            ))
            fig_pnl.update_layout(
                title="Daily P&L (USD)",
                xaxis_title="Date",
                yaxis_title="P&L ($)",
                yaxis_tickformat="$,.0f"
            )
            st.plotly_chart(fig_pnl, use_container_width=True)

            # 누적 수익률 차트
            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(
                x=daily_pnl['report_date'],
                y=daily_pnl['cumulative_return'],
                mode='lines+markers',
                name='Cumulative Return',
                line=dict(color='#6366f1', width=2)
            ))
            fig_cum.update_layout(
                title="Cumulative Return",
                xaxis_title="Date",
                yaxis_title="Return",
                yaxis_tickformat=".2%"
            )
            st.plotly_chart(fig_cum, use_container_width=True)

            # P&L 통계
            st.markdown("#### P&L 통계")
            total_pnl_all = daily_pnl['pnl_usd'].sum()
            avg_daily_pnl = daily_pnl['pnl_usd'].mean()
            win_rate = (daily_pnl['pnl_usd'] > 0).sum() / len(daily_pnl) * 100 if len(daily_pnl) > 0 else 0
            max_pnl = daily_pnl['pnl_usd'].max()
            min_pnl = daily_pnl['pnl_usd'].min()

            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Total P&L", f"${total_pnl_all:,.0f}")
            s2.metric("Avg Daily P&L", f"${avg_daily_pnl:,.0f}")
            s3.metric("Win Rate", f"{win_rate:.1f}%")
            s4.metric("Best Day", f"${max_pnl:,.0f}")
            s5.metric("Worst Day", f"${min_pnl:,.0f}")

            # P&L 테이블
            st.markdown("#### 일별 P&L 상세")
            st.dataframe(
                daily_pnl[['report_date', 'market_value_usd', 'pnl_usd', 'daily_return', 'cumulative_pnl']].sort_values('report_date', ascending=False).style.format({
                    'report_date': lambda x: x.strftime('%Y-%m-%d'),
                    'market_value_usd': '${:,.0f}',
                    'pnl_usd': '${:,.0f}',
                    'daily_return': '{:.2%}',
                    'cumulative_pnl': '${:,.0f}'
                }),
                use_container_width=True
            )

        with tab_attribution:
            st.markdown("### Contribution 분석")

            # 기간 선택
            col_start, col_end = st.columns(2)
            with col_start:
                start_date = st.date_input("시작일", value=min_date.date(), key="attr_start")
            with col_end:
                end_date = st.date_input("종료일", value=max_date.date(), key="attr_end")

            # 기간 내 데이터
            df_period = df_underlying[
                (pd.to_datetime(df_underlying['report_date']) >= pd.Timestamp(start_date)) &
                (pd.to_datetime(df_underlying['report_date']) <= pd.Timestamp(end_date))
            ].copy()

            if not df_period.empty:
                # 종목별 Contribution 합계
                ticker_contrib = df_period.groupby(['ticker', 'name']).agg({
                    'pnl_usd': 'sum',
                    'contribution': 'sum',
                    'market_value_usd': 'last'
                }).reset_index()
                ticker_contrib = ticker_contrib.sort_values('pnl_usd', ascending=False)

                # Contribution 바 차트 (Top 20)
                top_20_contrib = ticker_contrib.head(20)
                colors = np.where(top_20_contrib['pnl_usd'] >= 0, '#16a34a', '#dc2626')

                fig_contrib = go.Figure(data=go.Bar(
                    x=top_20_contrib['ticker'],
                    y=top_20_contrib['pnl_usd'],
                    text=[f"${v:,.0f}" for v in top_20_contrib['pnl_usd']],
                    textposition='auto',
                    marker_color=colors
                ))
                fig_contrib.update_layout(
                    title="Top 20 Contributors (P&L)",
                    xaxis_title="",
                    yaxis_title="P&L ($)",
                    yaxis_tickformat="$,.0f"
                )
                st.plotly_chart(fig_contrib, use_container_width=True)

                # 섹터별 Contribution
                if 'sector' in df_period.columns and df_period['sector'].notna().any():
                    sector_contrib = df_period.groupby('sector').agg({
                        'pnl_usd': 'sum',
                        'contribution': 'sum'
                    }).reset_index()
                    sector_contrib = sector_contrib.sort_values('pnl_usd', ascending=False)

                    colors_sector = np.where(sector_contrib['pnl_usd'] >= 0, '#16a34a', '#dc2626')
                    fig_sector_contrib = go.Figure(data=go.Bar(
                        x=sector_contrib['sector'],
                        y=sector_contrib['pnl_usd'],
                        text=[f"${v:,.0f}" for v in sector_contrib['pnl_usd']],
                        textposition='auto',
                        marker_color=colors_sector
                    ))
                    fig_sector_contrib.update_layout(
                        title="Sector Contribution",
                        xaxis_title="",
                        yaxis_title="P&L ($)",
                        yaxis_tickformat="$,.0f"
                    )
                    st.plotly_chart(fig_sector_contrib, use_container_width=True)

                # Contribution 테이블
                st.markdown("#### 종목별 Contribution 상세")
                st.dataframe(
                    ticker_contrib.style.format({
                        'pnl_usd': '${:,.0f}',
                        'contribution': '{:.2f}%',
                        'market_value_usd': '${:,.0f}'
                    }),
                    use_container_width=True
                )
