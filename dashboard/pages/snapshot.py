from pathlib import Path

from dashboard.core import *  # noqa: F401,F403

ROOT_DIR = Path(__file__).resolve().parents[2]

def render_snapshot_page():
    st.subheader("📌 Portfolio Snapshot (2026_멀티.xlsx)")
    script_dir = ROOT_DIR
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
