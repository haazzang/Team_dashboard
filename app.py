import streamlit as st

from dashboard.pages.cash_equity import render_cash_equity_page
from dashboard.pages.snapshot import render_snapshot_page
from dashboard.pages.swap_report import render_swap_report_page
from dashboard.pages.team_pnl import render_team_pnl_page
from dashboard.pages.weekly_report import render_weekly_report_page

MENU_RENDERERS = {
    "📌 Portfolio Snapshot": render_snapshot_page,
    "Total Portfolio (Team PNL)": render_team_pnl_page,
    "Cash Equity Analysis": render_cash_equity_page,
    "📑 Weekly Report Generator": render_weekly_report_page,
    "📊 Swap Report Analysis": render_swap_report_page,
}

menu = st.sidebar.radio("Dashboard Menu", list(MENU_RENDERERS.keys()))
MENU_RENDERERS[menu]()
