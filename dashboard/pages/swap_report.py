import os
import sys
from pathlib import Path

from dashboard.core import *  # noqa: F401,F403

ROOT_DIR = Path(__file__).resolve().parents[2]


def _resolve_pair_pnl_root() -> Path:
    """Find the pnl_report module folder, falling back across deploy/local paths."""
    candidates = [
        ROOT_DIR / "pnl_report",
        Path("/Users/hyejinha/Desktop/Workspace/PnL Report"),
        Path(os.environ.get("PNL_REPORT_DIR", "")).expanduser() if os.environ.get("PNL_REPORT_DIR") else None,
    ]
    for cand in candidates:
        if cand and cand.exists() and (cand / "pair_pnl.py").exists():
            return cand
    return ROOT_DIR / "pnl_report"


PAIR_PNL_ROOT = _resolve_pair_pnl_root()
PAIR_PNL_REPORTS_DIR = PAIR_PNL_ROOT / "reports"
SMT_NAV_CACHE = PAIR_PNL_ROOT / "data" / "smt_nav_history.json"

# Default SpaceX weight in SMT NAV (as at 30-Apr-2026 filings ≈ 18-20%).
DEFAULT_SPACEX_WEIGHT = 0.18


def _get_secret(name: str) -> str | None:
    """Read a secret from env var first, then Streamlit secrets manager."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        return st.secrets.get(name)
    except Exception:
        return None


def _load_pair_pnl_module():
    if str(PAIR_PNL_ROOT) not in sys.path:
        sys.path.insert(0, str(PAIR_PNL_ROOT))
    import importlib
    import pair_pnl as _pp
    return importlib.reload(_pp)


def _load_smt_nav_module():
    if str(PAIR_PNL_ROOT) not in sys.path:
        sys.path.insert(0, str(PAIR_PNL_ROOT))
    import importlib
    import smt_nav as _sn
    return importlib.reload(_sn)


@st.cache_data(show_spinner=False)
def _compute_pair_pnl_results(report_paths: tuple[str, ...], require_smt: bool):
    pp = _load_pair_pnl_module()
    config_path = PAIR_PNL_ROOT / "config.json"
    config = pp.load_config(config_path)
    if not require_smt:
        config.setdefault("market_data", {})["required"] = False

    results = []
    skipped = []
    for p in report_paths:
        try:
            results.append(pp.calculate_report(Path(p), config))
        except SystemExit as exc:
            skipped.append((Path(p).name, str(exc)))
    return results, skipped


def _auto_fetch_if_stale(pp, config, max_age_hours: int = 12) -> None:
    """If the latest report on disk is older than today (KST), try a Gmail pull.

    Runs at most once per Streamlit session. Silently no-ops if creds are missing.
    """
    if st.session_state.get("_pair_pnl_autofetch_done"):
        return
    st.session_state["_pair_pnl_autofetch_done"] = True

    user_env = config.get("email", {}).get("imap_user_env", "GMAIL_USER")
    pwd_env = config.get("email", {}).get("imap_password_env", "GMAIL_APP_PASSWORD")
    user_val = _get_secret(user_env)
    pwd_val = _get_secret(pwd_env)
    if not user_val or not pwd_val:
        return

    try:
        existing = [p for p in PAIR_PNL_REPORTS_DIR.iterdir() if p.suffix.lower() in {".xlsx", ".xlsm"}] if PAIR_PNL_REPORTS_DIR.exists() else []
    except Exception:
        existing = []
    if existing:
        latest_mtime = max(p.stat().st_mtime for p in existing)
        age_hours = (datetime.now().timestamp() - latest_mtime) / 3600
        if age_hours < max_age_hours:
            return

    os.environ[user_env] = user_val
    os.environ[pwd_env] = pwd_val
    try:
        PAIR_PNL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        cfg_for_fetch = {**config, "email": {**config["email"], "download_dir": str(PAIR_PNL_REPORTS_DIR)}}
        prev_cwd = os.getcwd()
        try:
            os.chdir(PAIR_PNL_ROOT)
            saved = pp.fetch_gmail_attachments(cfg_for_fetch)
        finally:
            os.chdir(prev_cwd)
        if saved:
            st.toast(f"Auto-fetched {len(saved)} report(s) from Gmail.", icon="📥")
    except Exception:
        pass  # Silent fallback — user can press the manual button


@st.cache_data(show_spinner=False)
def _load_nav_cache() -> pd.DataFrame:
    sn = _load_smt_nav_module()
    cache = sn.load_cache(SMT_NAV_CACHE)
    if not cache:
        return pd.DataFrame(columns=[
            "valuation_date", "announcement_date", "cum_par", "cum_fair",
            "ex_par", "ex_fair", "source_url",
        ])
    df = pd.DataFrame(sn.history_as_records(cache))
    df["valuation_date"] = pd.to_datetime(df["valuation_date"])
    return df.sort_values("valuation_date").reset_index(drop=True)


def _render_long_short_pair_section():
    st.markdown("## Long/Short Pair PnL — SMT LN × US Short Basket")
    st.caption(
        "Long: SMT LN @ 1,433.5082 × 154,699 shares (USD quanto)  |  "
        "Short: USD-denominated underlyings with negative quantity in the daily "
        "JMLNKWGE Synthetic Portfolio EOD Report."
    )

    pp = _load_pair_pnl_module()
    config_path = PAIR_PNL_ROOT / "config.json"
    if not config_path.exists():
        st.error(f"Missing config.json at {config_path}")
        return
    config = pp.load_config(config_path)

    fmp_env = config.get("market_data", {}).get("fmp_api_key_env", "FMP_API_KEY")
    fmp_key_present = bool(_get_secret(fmp_env))
    if fmp_key_present and not os.environ.get(fmp_env):
        # Surface st.secrets value into env so pair_pnl module reads it
        os.environ[fmp_env] = _get_secret(fmp_env)

    _auto_fetch_if_stale(pp, config)

    c_fetch, c_recalc, c_status = st.columns([1.2, 1, 3])
    with c_fetch:
        if st.button("Fetch latest from Gmail", key="pair_pnl_fetch_gmail"):
            user_env = config.get("email", {}).get("imap_user_env", "GMAIL_USER")
            pwd_env = config.get("email", {}).get("imap_password_env", "GMAIL_APP_PASSWORD")
            user_val = _get_secret(user_env)
            pwd_val = _get_secret(pwd_env)
            if not user_val or not pwd_val:
                st.error(
                    f"Set {user_env} and {pwd_env} in environment or Streamlit secrets to fetch from Gmail."
                )
            else:
                os.environ[user_env] = user_val
                os.environ[pwd_env] = pwd_val
                with st.spinner("Connecting to Gmail and downloading new reports..."):
                    try:
                        PAIR_PNL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
                        cfg_for_fetch = {**config}
                        cfg_for_fetch["email"] = {**config["email"], "download_dir": str(PAIR_PNL_REPORTS_DIR)}
                        prev_cwd = os.getcwd()
                        try:
                            os.chdir(PAIR_PNL_ROOT)
                            saved = pp.fetch_gmail_attachments(cfg_for_fetch)
                        finally:
                            os.chdir(prev_cwd)
                        st.success(f"Downloaded / verified {len(saved)} attachment(s).")
                        st.cache_data.clear()
                    except SystemExit as exc:
                        st.error(str(exc))
                    except Exception as exc:
                        st.error(f"Fetch failed: {type(exc).__name__}: {exc}")
    with c_recalc:
        if st.button("Recalculate", key="pair_pnl_recalc"):
            st.cache_data.clear()
            st.rerun()
    with c_status:
        if not fmp_key_present:
            st.warning(
                f"`{fmp_env}` is not set — SMT LN current price unavailable. "
                "Short basket PnL still computed; total pair PnL will exclude SMT leg."
            )

    PAIR_PNL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    user_env = config.get("email", {}).get("imap_user_env", "GMAIL_USER")
    pwd_env = config.get("email", {}).get("imap_password_env", "GMAIL_APP_PASSWORD")
    gmail_present = bool(_get_secret(user_env) and _get_secret(pwd_env))

    with st.expander("Setup status", expanded=not gmail_present):
        s1, s2, s3 = st.columns(3)
        s1.markdown(f"**Gmail IMAP**  \n{'OK' if gmail_present else 'NOT SET'}")
        s2.markdown(f"**FMP API key**  \n{'OK' if fmp_key_present else 'NOT SET (Yahoo fallback active)'}")
        s3.markdown(f"**Reports dir**  \n`{PAIR_PNL_REPORTS_DIR}`")
        if not gmail_present:
            st.markdown(
                "Set `GMAIL_USER` and `GMAIL_APP_PASSWORD` in **Streamlit Cloud → Manage app → "
                "Settings → Secrets** to enable auto-fetch. Until then, upload JMLNKWGE `.xlsx` "
                "files directly below."
            )

    uploads = st.file_uploader(
        "Upload JMLNKWGE swap report(s) (.xlsx)",
        type=["xlsx", "xlsm"],
        accept_multiple_files=True,
        key="pair_pnl_uploader",
        help="Upload one or more daily JMLNKWGE EOD reports. They are saved to the reports folder for this session.",
    )
    if uploads:
        saved_names = []
        for uf in uploads:
            dest = PAIR_PNL_REPORTS_DIR / uf.name
            dest.write_bytes(uf.getvalue())
            saved_names.append(uf.name)
        st.success(f"Saved {len(saved_names)} file(s): {', '.join(saved_names)}")
        st.cache_data.clear()
        st.rerun()

    report_files = sorted(
        p for p in PAIR_PNL_REPORTS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".xlsx", ".xlsm", ".csv"}
    )
    if not report_files:
        st.info(
            "No swap reports on disk yet. Use the uploader above, the **Fetch latest from Gmail** "
            "button (requires Gmail secrets), or drop `.xlsx` files into "
            f"`{PAIR_PNL_REPORTS_DIR}` if running locally."
        )
        return

    st.caption(f"{len(report_files)} report file(s) loaded.")

    results, skipped = _compute_pair_pnl_results(
        tuple(str(p) for p in report_files),
        require_smt=fmp_key_present,
    )
    if skipped:
        with st.expander(f"{len(skipped)} report(s) skipped"):
            for name, reason in skipped:
                st.write(f"- **{name}**: {reason}")
    if not results:
        st.error("No PnL could be calculated from available reports.")
        return

    df = pp.results_to_frame(results)
    df_view = df.copy()
    df_view["report_date"] = pd.to_datetime(df_view["report_date"])
    df_view = df_view.sort_values("report_date").drop_duplicates("report_date", keep="last").reset_index(drop=True)
    df_view["daily_pair_pnl_change"] = df_view["pair_pnl"].diff().fillna(df_view["pair_pnl"])

    # Daily exposure metrics.
    otc_cfg = config.get("pair", {}).get("smt_otc_trs", {})
    otc_shares_cfg = float(otc_cfg.get("shares", 0))
    df_view["smt_otc_market_value_usd"] = (
        df_view["smt_otc_current_usd_price"].astype(float) * otc_shares_cfg
    )
    df_view["smt_long_mv_total"] = (
        df_view["smt_otc_market_value_usd"].fillna(0.0)
        + df_view["smt_swap_long_market_value_usd"].fillna(0.0)
    )
    df_view["short_mv_abs"] = df_view["short_market_value"].abs()
    df_view["gross_exposure"] = df_view["smt_long_mv_total"] + df_view["short_mv_abs"]
    df_view["net_exposure"] = df_view["smt_long_mv_total"] - df_view["short_mv_abs"]
    df_view["hedge_ratio"] = df_view.apply(
        lambda r: (r["short_mv_abs"] / r["smt_long_mv_total"])
        if r["smt_long_mv_total"] and r["smt_long_mv_total"] > 0 else float("nan"),
        axis=1,
    )

    latest = df_view.iloc[-1]
    with st.expander("SMT OTC TRS — term sheet"):
        cols = st.columns(3)
        cols[0].markdown(
            f"**Counterparty**  {otc_cfg.get('counterparty', 'n/a')}  \n"
            f"**Account**  {otc_cfg.get('account', 'n/a')}  \n"
            f"**Trade Date**  {otc_cfg.get('trade_date', 'n/a')}  \n"
            f"**Effective**  {otc_cfg.get('effective_date', 'n/a')}  \n"
            f"**Termination**  {otc_cfg.get('termination_date', 'n/a')}"
        )
        cols[1].markdown(
            f"**Shares**  {int(otc_cfg.get('shares', 0)):,}  \n"
            f"**Initial Gross**  {otc_cfg.get('initial_gross_price_pence', 0):,.4f}p  \n"
            f"**Initial FX**  {otc_cfg.get('initial_fx_gbpusd', 0):.4f}  \n"
            f"**Initial Net USD**  ${otc_cfg.get('initial_net_price_usd', 0):.4f}  \n"
            f"**Notional**  ${otc_cfg.get('target_notional_usd', 0):,.0f}"
        )
        cols[2].markdown(
            f"**Financing**  {otc_cfg.get('financing_base', 'n/a')} "
            f"+ {otc_cfg.get('financing_spread_bps', 0)}bps  \n"
            f"**Day Count**  {otc_cfg.get('financing_day_count', 360)}  \n"
            f"**Commission In/Out**  "
            f"{otc_cfg.get('commission_in_bps', 0)}bps / {otc_cfg.get('commission_out_bps', 0)}bps  \n"
        )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Report Date", latest["report_date"].strftime("%Y-%m-%d"))
    m2.metric("Short Basket PnL", f"${latest['short_pnl']:,.0f}")
    smt_pnl_value = latest["smt_pnl"]
    m3.metric(
        "SMT LN PnL (combined)",
        f"${smt_pnl_value:,.0f}" if pd.notna(smt_pnl_value) else "n/a",
    )
    m4.metric("Pair PnL", f"${latest['pair_pnl']:,.0f}")
    m5.metric("Daily Change", f"${latest['daily_pair_pnl_change']:,.0f}")

    s1, s2, s3 = st.columns(3)
    s1.metric(
        "→ OTC TRS leg",
        f"${latest['smt_otc_pnl']:,.0f}" if pd.notna(latest["smt_otc_pnl"]) else "n/a",
        help="Long SMT LN via JPM CBNA — 154,699 shares, FX-linked TRS.",
    )
    s2.metric(
        "→ Swap-report SMT long",
        f"${latest['smt_swap_long_pnl']:,.0f}",
        delta=f"{int(latest['smt_swap_long_quantity']):,} shares",
        help="Long SMT LN positions sitting inside the JMLNKWGE swap report (positive quantity).",
    )
    s3.metric(
        "→ FX (GBPUSD)",
        f"{latest['smt_otc_current_fx']:.4f}" if pd.notna(latest["smt_otc_current_fx"]) else "n/a",
        delta=(
            f"{(latest['smt_otc_current_fx'] - otc_cfg.get('initial_fx_gbpusd', 0)):.4f} vs initial"
            if pd.notna(latest["smt_otc_current_fx"]) and otc_cfg.get("initial_fx_gbpusd") else None
        ),
    )

    st.markdown("#### Daily Exposure")
    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric(
        "SMT Long MV (combined)",
        f"${latest['smt_long_mv_total']:,.0f}",
        help="OTC TRS current notional + SMT positions inside the swap report (USD).",
    )
    e2.metric(
        "Short Basket MV",
        f"${latest['short_mv_abs']:,.0f}",
        help="Absolute notional of the US negative-quantity rows in the swap report.",
    )
    e3.metric(
        "Gross Exposure",
        f"${latest['gross_exposure']:,.0f}",
        help="|Long MV| + |Short MV| — total capital at risk.",
    )
    net_delta_sign = "Net long" if latest["net_exposure"] > 0 else ("Net short" if latest["net_exposure"] < 0 else "Flat")
    e4.metric(
        "Net Exposure",
        f"${latest['net_exposure']:,.0f}",
        delta=net_delta_sign,
        help="Long MV − |Short MV|. Positive = net long the pair.",
    )
    hr = latest["hedge_ratio"]
    hr_delta = None
    if pd.notna(hr):
        if hr < 0.95:
            hr_delta = f"Under-hedged ({(1-hr)*100:.0f}% net long)"
        elif hr > 1.05:
            hr_delta = f"Over-hedged ({(hr-1)*100:.0f}% net short)"
        else:
            hr_delta = "Near balanced"
    e5.metric(
        "Hedge Ratio",
        f"{hr:.1%}" if pd.notna(hr) else "n/a",
        delta=hr_delta,
        delta_color="off",
        help="|Short MV| / Long MV. 100% = fully delta-matched.",
    )

    smt_price = latest.get("smt_current_price")
    smt_source = latest.get("smt_price_source")
    smt_usd = latest.get("smt_otc_current_usd_price")
    st.caption(
        f"Short names: {int(latest['short_count'])}  |  "
        f"SMT GBp: {f'{smt_price:,.2f}' if pd.notna(smt_price) else 'n/a'} "
        f"({smt_source if pd.notna(smt_source) else 'n/a'})  |  "
        f"SMT USD/share: ${smt_usd:,.4f}" if pd.notna(smt_usd) else "SMT USD/share: n/a"
    )

    if len(df_view) > 1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_view["report_date"], y=df_view["short_pnl"],
            name="Short Basket PnL", marker_color="#dc2626",
        ))
        fig.add_trace(go.Bar(
            x=df_view["report_date"], y=df_view["smt_otc_pnl"].fillna(0),
            name="SMT OTC TRS PnL", marker_color="#0f4c81",
        ))
        fig.add_trace(go.Bar(
            x=df_view["report_date"], y=df_view["smt_swap_long_pnl"].fillna(0),
            name="SMT Swap-report Long PnL", marker_color="#7c3aed",
        ))
        fig.add_trace(go.Scatter(
            x=df_view["report_date"], y=df_view["pair_pnl"],
            name="Pair PnL (total)", mode="lines+markers",
            line=dict(color="#16a34a", width=3),
        ))
        fig.update_layout(
            title="Daily Pair PnL Breakdown — Short Basket + SMT (OTC TRS + Swap Report Long)",
            xaxis_title="Report Date",
            yaxis_title="PnL (USD)",
            yaxis_tickformat="$,.0f",
            barmode="relative",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    if len(df_view) > 1:
        exp_chart = go.Figure()
        exp_chart.add_trace(go.Bar(
            x=df_view["report_date"], y=df_view["smt_long_mv_total"],
            name="SMT Long MV", marker_color="#0f4c81",
        ))
        exp_chart.add_trace(go.Bar(
            x=df_view["report_date"], y=-df_view["short_mv_abs"],
            name="Short Basket MV (shown negative)", marker_color="#dc2626",
        ))
        exp_chart.add_trace(go.Scatter(
            x=df_view["report_date"], y=df_view["net_exposure"],
            name="Net Exposure", mode="lines+markers",
            line=dict(color="#16a34a", width=2),
        ))
        exp_chart.add_trace(go.Scatter(
            x=df_view["report_date"], y=df_view["hedge_ratio"],
            name="Hedge Ratio (right axis)", mode="lines+markers",
            line=dict(color="#7c3aed", width=2, dash="dot"),
            yaxis="y2",
        ))
        exp_chart.update_layout(
            title="Daily Exposure: Long MV vs Short MV with Net Exposure & Hedge Ratio",
            barmode="relative",
            yaxis=dict(title="USD", tickformat="$,.0f"),
            yaxis2=dict(title="Hedge Ratio", overlaying="y", side="right", tickformat=".0%"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(exp_chart, use_container_width=True)

    st.markdown("### Daily PnL & Exposure Table")
    table = df_view[[
        "report_date", "short_count", "short_pnl",
        "smt_current_price", "smt_otc_current_fx", "smt_otc_current_usd_price",
        "smt_otc_pnl", "smt_swap_long_quantity", "smt_swap_long_pnl",
        "smt_pnl", "pair_pnl", "daily_pair_pnl_change",
        "smt_long_mv_total", "short_mv_abs",
        "gross_exposure", "net_exposure", "hedge_ratio",
    ]].copy()
    table["report_date"] = table["report_date"].dt.strftime("%Y-%m-%d")
    st.dataframe(
        table.sort_values("report_date", ascending=False).style.format({
            "short_pnl": "${:,.0f}",
            "smt_current_price": "{:,.2f}p",
            "smt_otc_current_fx": "{:,.4f}",
            "smt_otc_current_usd_price": "${:,.4f}",
            "smt_otc_pnl": "${:,.0f}",
            "smt_swap_long_quantity": "{:,.0f}",
            "smt_swap_long_pnl": "${:,.0f}",
            "smt_pnl": "${:,.0f}",
            "pair_pnl": "${:,.0f}",
            "daily_pair_pnl_change": "${:,.0f}",
            "smt_long_mv_total": "${:,.0f}",
            "short_mv_abs": "${:,.0f}",
            "gross_exposure": "${:,.0f}",
            "net_exposure": "${:,.0f}",
            "hedge_ratio": "{:.1%}",
        }, na_rep="n/a"),
        use_container_width=True,
        hide_index=True,
    )

    latest_result = next((r for r in results if r.report_date == latest["report_date"].date()), None)
    detail_cols = st.columns(2)
    with detail_cols[0]:
        st.markdown("### Latest Short Basket Detail")
        if latest_result and latest_result.short_basket_detail:
            detail_df = pd.DataFrame(latest_result.short_basket_detail).sort_values("pnl_usd")
            st.dataframe(
                detail_df.style.format({
                    "quantity": "{:,.0f}",
                    "trade_price": "{:,.4f}",
                    "current_price": "{:,.4f}",
                    "market_value_usd": "${:,.2f}",
                    "cost_market_value_usd": "${:,.2f}",
                    "pnl_usd": "${:,.2f}",
                }, na_rep="n/a"),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No short basket detail available for the latest report.")
    with detail_cols[1]:
        st.markdown("### Latest SMT Swap-report Long Detail")
        if latest_result and latest_result.smt_swap_long_detail:
            sl_df = pd.DataFrame(latest_result.smt_swap_long_detail)
            st.dataframe(
                sl_df.style.format({
                    "quantity": "{:,.0f}",
                    "trade_price_gbp": "{:,.4f}",
                    "current_price_gbp": "{:,.4f}",
                    "market_value_usd": "${:,.2f}",
                    "cost_market_value_usd": "${:,.2f}",
                    "pnl_usd": "${:,.2f}",
                }, na_rep="n/a"),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No SMT LN long position in this report's Component Underlyings.")

    st.markdown("---")
    _render_nav_and_spacex_section(df_view, results, pp, config)


def _build_nav_view(
    df_view: pd.DataFrame, nav_df: pd.DataFrame, pair_cfg: dict, nav_field: str
) -> pd.DataFrame:
    """Build NAV-based PnL view for the FX-linked OTC TRS leg.

    NAV-basis PnL for OTC TRS = (NAV_today_pence/100 × current_FX
                                  - NAV_at_trade_date_pence/100 × initial_FX) × shares.
    The swap-report SMT long leg is already mark-to-market in the report — its PnL
    is carried through unchanged, and only the OTC leg is re-priced on NAV.
    """
    otc_cfg = pair_cfg.get("smt_otc_trs", {}) or {}
    otc_shares = float(otc_cfg.get("shares", pair_cfg.get("smt_shares", 0)))
    initial_fx = float(otc_cfg.get("initial_fx_gbpusd", 0.0)) or None
    trade_date_raw = otc_cfg.get("trade_date")
    trade_date = (
        pd.Timestamp(trade_date_raw) if isinstance(trade_date_raw, str) and trade_date_raw else None
    )

    merged = df_view.merge(
        nav_df[["valuation_date", nav_field]].rename(columns={
            "valuation_date": "report_date", nav_field: "smt_nav_pence",
        }),
        on="report_date",
        how="left",
    )

    # Anchor NAV: NAV as of the trade date (closest <= trade_date if exact not in cache).
    initial_nav_pence = None
    if trade_date is not None and not nav_df.empty:
        eligible = nav_df[nav_df["valuation_date"] <= trade_date]
        if not eligible.empty:
            initial_nav_pence = float(eligible.iloc[-1][nav_field])

    if initial_nav_pence and initial_fx:
        nav_initial_usd = (initial_nav_pence / 100.0) * initial_fx
        merged["smt_nav_usd_per_share"] = (
            merged["smt_nav_pence"] / 100.0 * merged["smt_otc_current_fx"]
        )
        merged["smt_otc_nav_pnl"] = (merged["smt_nav_usd_per_share"] - nav_initial_usd) * otc_shares
    else:
        merged["smt_nav_usd_per_share"] = pd.NA
        merged["smt_otc_nav_pnl"] = pd.NA

    # Combined NAV-basis SMT PnL = OTC NAV PnL + swap-report SMT long PnL (mark-to-market).
    merged["smt_nav_pnl"] = merged["smt_otc_nav_pnl"].astype(float) + merged["smt_swap_long_pnl"].fillna(0.0)
    merged["nav_minus_price_pnl"] = merged["smt_nav_pnl"] - merged["smt_pnl"]
    merged["pair_pnl_nav_basis"] = merged["short_pnl"] + merged["smt_nav_pnl"]
    merged.attrs["initial_nav_pence"] = initial_nav_pence
    merged.attrs["nav_initial_usd"] = (
        (initial_nav_pence / 100.0) * initial_fx if (initial_nav_pence and initial_fx) else None
    )
    return merged


def _render_nav_and_spacex_section(df_view, results, pp, config):
    st.markdown("## NAV-based Decomposition & SpaceX Proxy Analysis")

    sn = _load_smt_nav_module()
    nav_df = _load_nav_cache()

    nav_c1, nav_c2, nav_c3 = st.columns([1, 1, 3])
    with nav_c1:
        if st.button("Refresh SMT NAV from RNS", key="refresh_nav"):
            with st.spinner("Scraping Investegate RNS feed..."):
                try:
                    cache, added = sn.refresh_nav_history(SMT_NAV_CACHE, max_new=30)
                    st.success(f"Added {added} new NAV record(s); total {len(cache)}.")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as exc:
                    st.error(f"NAV refresh failed: {type(exc).__name__}: {exc}")
    with nav_c2:
        nav_field = st.selectbox(
            "NAV basis",
            options=["cum_fair", "cum_par", "ex_fair", "ex_par"],
            index=0,
            key="nav_field_select",
            help="cum/ex income × debt at par/fair value (IPEV-style).",
        )
    with nav_c3:
        if nav_df.empty:
            st.warning("No NAV history cached yet — click 'Refresh SMT NAV from RNS' to seed.")
        else:
            st.caption(
                f"NAV cache: {len(nav_df)} record(s) "
                f"({nav_df['valuation_date'].min().date()} → {nav_df['valuation_date'].max().date()}). "
                f"Source: Investegate RNS feed."
            )

    if nav_df.empty:
        return

    pair_cfg = config["pair"]
    nav_view = _build_nav_view(df_view, nav_df, pair_cfg, nav_field)

    nav_tab, contrib_tab, spacex_tab = st.tabs([
        "NAV vs Price PnL", "Daily Contribution", "SpaceX Proxy",
    ])

    with nav_tab:
        initial_nav_pence = nav_view.attrs.get("initial_nav_pence")
        nav_initial_usd = nav_view.attrs.get("nav_initial_usd")
        otc_cfg_local = config.get("pair", {}).get("smt_otc_trs", {})
        st.markdown(
            "**SMT LN PnL decomposed (OTC TRS leg).**  Price-basis values the OTC TRS "
            f"position at `current_GBP × current_FX − $19.3925` (term-sheet net cost). "
            f"NAV-basis values it at `NAV_GBp/100 × current_FX − NAV_at_trade_date/100 × {otc_cfg_local.get('initial_fx_gbpusd', 0):.4f}`. "
            "The gap = discount/premium drift relative to the trade entry.  \n"
            f"Anchor NAV (`{nav_field}` on/before "
            f"{otc_cfg_local.get('trade_date', 'trade date')}): "
            f"{initial_nav_pence:.2f}p" if initial_nav_pence else
            "Anchor NAV: not yet resolved (need NAV history covering the trade date)."
        )
        st.caption(
            f"Anchor NAV USD/share = {nav_initial_usd:.4f}  |  "
            f"Combined NAV PnL = OTC NAV PnL + swap-report SMT long PnL (mark-to-market)."
            if nav_initial_usd else
            ""
        )

        nav_available = nav_view.dropna(subset=["smt_nav_pence", "smt_otc_nav_pnl"])
        if nav_available.empty:
            st.info("No swap-report dates overlap the NAV cache yet.")
        else:
            latest = nav_available.iloc[-1]
            tag = f" (as of {latest['report_date'].strftime('%Y-%m-%d')})"
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("SMT Price PnL (combined)" + tag, _money(latest["smt_pnl"]))
            c2.metric("SMT NAV PnL (combined)" + tag, _money(latest["smt_nav_pnl"]))
            c3.metric("Discount Drift PnL", _money(latest["nav_minus_price_pnl"]))
            c4.metric("Pair PnL (NAV basis)", _money(latest["pair_pnl_nav_basis"]))
            d1, d2, d3 = st.columns(3)
            d1.metric("→ OTC TRS NAV PnL", _money(latest["smt_otc_nav_pnl"]))
            d2.metric("→ Swap-report SMT long PnL", _money(latest["smt_swap_long_pnl"]))
            d3.metric("NAV USD/share", f"${latest['smt_nav_usd_per_share']:.4f}" if pd.notna(latest["smt_nav_usd_per_share"]) else "n/a")
            if len(nav_view) > len(nav_available):
                missing = nav_view[nav_view["smt_nav_pence"].isna()]["report_date"].dt.strftime("%Y-%m-%d").tolist()
                st.caption(f"NAV not yet published for: {', '.join(missing)} (RNS releases T+1 ~11:30 BST).")

        nav_chart = go.Figure()
        nav_chart.add_trace(go.Scatter(
            x=nav_view["report_date"], y=nav_view["smt_pnl"],
            name="SMT Price PnL", mode="lines+markers",
            line=dict(color="#0f4c81", width=2),
        ))
        nav_chart.add_trace(go.Scatter(
            x=nav_view["report_date"], y=nav_view["smt_nav_pnl"],
            name="SMT NAV PnL", mode="lines+markers",
            line=dict(color="#16a34a", width=2, dash="dash"),
        ))
        nav_chart.add_trace(go.Bar(
            x=nav_view["report_date"], y=nav_view["nav_minus_price_pnl"],
            name="Discount drift (NAV - Price)", marker_color="#9ca3af", opacity=0.5,
        ))
        nav_chart.update_layout(
            title="SMT LN: NAV-based vs Price-based PnL",
            yaxis_tickformat="$,.0f", yaxis_title="PnL (USD)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(nav_chart, use_container_width=True)

        show = nav_view[[
            "report_date", "smt_current_price", "smt_nav_pence",
            "smt_pnl", "smt_nav_pnl", "nav_minus_price_pnl",
            "short_pnl", "pair_pnl", "pair_pnl_nav_basis",
        ]].copy()
        show["report_date"] = show["report_date"].dt.strftime("%Y-%m-%d")
        st.dataframe(
            show.sort_values("report_date", ascending=False).style.format({
                "smt_current_price": "{:,.2f}p",
                "smt_nav_pence": "{:,.2f}p",
                "smt_pnl": "${:,.0f}",
                "smt_nav_pnl": "${:,.0f}",
                "nav_minus_price_pnl": "${:,.0f}",
                "short_pnl": "${:,.0f}",
                "pair_pnl": "${:,.0f}",
                "pair_pnl_nav_basis": "${:,.0f}",
            }, na_rep="n/a"),
            use_container_width=True, hide_index=True,
        )

    with contrib_tab:
        st.markdown(
            "**Day-over-day contribution to pair PnL.**  Decomposes the daily change "
            "into SMT (NAV-basis) and Short Basket components. Positive bars = pair PnL added; "
            "negative = pair PnL lost."
        )
        contrib = nav_view.copy()
        contrib["smt_nav_pnl_change"] = contrib["smt_nav_pnl"].diff()
        contrib["short_pnl_change"] = contrib["short_pnl"].diff()
        contrib["pair_pnl_nav_change"] = contrib["pair_pnl_nav_basis"].diff()
        contrib.loc[contrib.index[0], "smt_nav_pnl_change"] = contrib["smt_nav_pnl"].iloc[0]
        contrib.loc[contrib.index[0], "short_pnl_change"] = contrib["short_pnl"].iloc[0]
        contrib.loc[contrib.index[0], "pair_pnl_nav_change"] = contrib["pair_pnl_nav_basis"].iloc[0]

        contrib_chart = go.Figure()
        contrib_chart.add_trace(go.Bar(
            x=contrib["report_date"], y=contrib["smt_nav_pnl_change"],
            name="SMT NAV (Long)", marker_color="#16a34a",
        ))
        contrib_chart.add_trace(go.Bar(
            x=contrib["report_date"], y=contrib["short_pnl_change"],
            name="Short Basket", marker_color="#dc2626",
        ))
        contrib_chart.add_trace(go.Scatter(
            x=contrib["report_date"], y=contrib["pair_pnl_nav_change"],
            name="Net Pair PnL Δ", mode="lines+markers",
            line=dict(color="#18242f", width=2),
        ))
        contrib_chart.update_layout(
            title="Daily PnL Contribution (NAV basis)",
            barmode="relative", yaxis_tickformat="$,.0f",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(contrib_chart, use_container_width=True)

        show = contrib[[
            "report_date", "smt_nav_pence", "smt_nav_pnl_change",
            "short_pnl_change", "pair_pnl_nav_change",
        ]].copy()
        show["report_date"] = show["report_date"].dt.strftime("%Y-%m-%d")
        st.dataframe(
            show.sort_values("report_date", ascending=False).style.format({
                "smt_nav_pence": "{:,.2f}p",
                "smt_nav_pnl_change": "${:,.0f}",
                "short_pnl_change": "${:,.0f}",
                "pair_pnl_nav_change": "${:,.0f}",
            }, na_rep="n/a"),
            use_container_width=True, hide_index=True,
        )

    with spacex_tab:
        st.markdown(
            "**The pair-trade thesis: long SMT vs short US-growth basket isolates SMT's "
            "private holdings, of which SpaceX (~18-20% of NAV, valued at $1.25T as of "
            "31-Mar-26) is the largest. The 'residual' below ≈ SpaceX-implied PnL.**"
        )
        cfg_c1, cfg_c2 = st.columns([1, 3])
        with cfg_c1:
            spacex_weight = st.slider(
                "SpaceX % of SMT NAV",
                min_value=0.05, max_value=0.30, value=DEFAULT_SPACEX_WEIGHT, step=0.005,
                key="spacex_weight",
                help="Adjust based on latest SMT factsheet. Default 18% as at 30-Apr-2026.",
            )
        with cfg_c2:
            st.caption(
                "**Method:** SpaceX-implied PnL = SMT NAV PnL × SpaceX weight. "
                "Listed-hedge residual = NAV PnL × (1 - weight) + Short Basket PnL. "
                "When the listed-hedge residual is near zero, the short basket cleanly hedges "
                "SMT's listed book and the pair PnL collapses onto SpaceX (+ other privates)."
            )

        spacex_view = nav_view.copy()
        spacex_view["spacex_implied_pnl"] = spacex_view["smt_nav_pnl"] * spacex_weight
        spacex_view["listed_part_pnl"] = spacex_view["smt_nav_pnl"] * (1 - spacex_weight)
        spacex_view["listed_hedge_residual"] = spacex_view["listed_part_pnl"] + spacex_view["short_pnl"]
        spacex_view["pair_pnl_nav_basis"] = spacex_view["short_pnl"] + spacex_view["smt_nav_pnl"]

        spacex_available = spacex_view.dropna(subset=["smt_nav_pence"])
        if spacex_available.empty:
            st.info("No NAV-aligned rows yet — refresh the NAV cache after the RNS release.")
            return
        latest = spacex_available.iloc[-1]
        tag = f" (as of {latest['report_date'].strftime('%Y-%m-%d')})"
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("SMT NAV PnL" + tag, _money(latest["smt_nav_pnl"]))
        c2.metric(
            f"SpaceX-implied PnL ({spacex_weight:.1%})",
            _money(latest["spacex_implied_pnl"]),
        )
        c3.metric("Listed hedge residual", _money(latest["listed_hedge_residual"]))
        c4.metric("Pair PnL (NAV basis)", _money(latest["pair_pnl_nav_basis"]))

        spacex_chart = go.Figure()
        spacex_chart.add_trace(go.Bar(
            x=spacex_view["report_date"], y=spacex_view["spacex_implied_pnl"],
            name=f"SpaceX-implied ({spacex_weight:.0%})", marker_color="#7c3aed",
        ))
        spacex_chart.add_trace(go.Bar(
            x=spacex_view["report_date"], y=spacex_view["listed_hedge_residual"],
            name="Listed hedge residual", marker_color="#f97316",
        ))
        spacex_chart.add_trace(go.Scatter(
            x=spacex_view["report_date"], y=spacex_view["pair_pnl_nav_basis"],
            name="Pair PnL (NAV)", mode="lines+markers",
            line=dict(color="#18242f", width=2.5),
        ))
        spacex_chart.update_layout(
            title="Pair PnL Attribution: SpaceX-implied vs Listed Hedge Residual",
            yaxis_tickformat="$,.0f", barmode="relative",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(spacex_chart, use_container_width=True)

        show = spacex_view[[
            "report_date", "smt_nav_pence", "smt_nav_pnl",
            "spacex_implied_pnl", "listed_part_pnl", "short_pnl",
            "listed_hedge_residual", "pair_pnl_nav_basis",
        ]].copy()
        show["report_date"] = show["report_date"].dt.strftime("%Y-%m-%d")
        st.dataframe(
            show.sort_values("report_date", ascending=False).style.format({
                "smt_nav_pence": "{:,.2f}p",
                "smt_nav_pnl": "${:,.0f}",
                "spacex_implied_pnl": "${:,.0f}",
                "listed_part_pnl": "${:,.0f}",
                "short_pnl": "${:,.0f}",
                "listed_hedge_residual": "${:,.0f}",
                "pair_pnl_nav_basis": "${:,.0f}",
            }, na_rep="n/a"),
            use_container_width=True, hide_index=True,
        )

        if not spacex_available.empty:
            avg_spacex = float(spacex_available["spacex_implied_pnl"].mean())
            avg_residual = float(spacex_available["listed_hedge_residual"].mean())
            st.info(
                f"**Average daily attribution over {len(spacex_available)} NAV-aligned day(s):** "
                f"SpaceX-implied = {_money(avg_spacex)}  |  "
                f"Listed hedge residual = {_money(avg_residual)}. "
                "A consistently large residual means the short basket is under- or over-hedging "
                "SMT's listed book — re-tune weights to flatten the residual and let SpaceX/private "
                "marks dominate the pair PnL."
            )


def _money(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"${float(value):,.0f}"


def render_swap_report_page():
    st.subheader("Swap Report Analysis (JMLNKWGE)")

    _render_long_short_pair_section()

    # SQLite DB 경로 - 여러 경로 시도
    possible_paths = [
        ROOT_DIR / 'swap_reports.db',
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
        st.markdown("### 수동 업로드")
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
            "Overview", "Holdings", "P&L Analysis", "Attribution"
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
                    st.markdown("#### Top 5 Performers")
                    top5 = df_date_underlying.nlargest(5, 'pnl_usd')[['ticker', 'name', 'pnl_usd', 'pnl_pct', 'contribution']]
                    st.dataframe(top5.style.format({
                        'pnl_usd': '${:,.0f}',
                        'pnl_pct': '{:.2f}%',
                        'contribution': '{:.2f}%'
                    }))

                with col_bottom:
                    st.markdown("#### Bottom 5 Performers")
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
