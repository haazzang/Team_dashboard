import streamlit as st

from dashboard.pages.cash_equity import render_cash_equity_page
from dashboard.pages.snapshot import render_snapshot_page
from dashboard.pages.swap_report import render_swap_report_page
from dashboard.pages.team_pnl import render_team_pnl_page
from dashboard.pages.weekly_report import render_weekly_report_page

def _apply_global_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

        :root {
            --app-bg-top: #edf2f6;
            --app-bg-bottom: #f7f9fb;
            --panel-bg: #ffffff;
            --panel-soft: #f5f8fb;
            --border: #d7e0e8;
            --text: #18242f;
            --muted: #5f7182;
            --accent: #0f4c81;
            --accent-strong: #0b385d;
            --sidebar-top: #111821;
            --sidebar-bottom: #172635;
            --shadow: 0 12px 30px rgba(15, 23, 42, 0.07);
        }

        html, body, [class*="css"] {
            font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
            color: var(--text);
        }

        .stApp {
            background: linear-gradient(180deg, var(--app-bg-top) 0%, var(--app-bg-bottom) 100%);
        }

        [data-testid="stAppViewContainer"] > .main {
            background: transparent;
        }

        .block-container {
            max-width: 1440px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, var(--sidebar-top) 0%, var(--sidebar-bottom) 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] * {
            color: #e8edf3;
        }

        h1, h2, h3 {
            color: var(--text);
            letter-spacing: -0.02em;
        }

        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.94);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            box-shadow: var(--shadow);
        }

        div[data-testid="stMetricLabel"] {
            color: var(--muted);
        }

        [data-baseweb="tab-list"] {
            gap: 0.45rem;
        }

        button[data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid var(--border);
            border-radius: 999px;
            color: var(--muted);
            padding: 0.5rem 0.95rem;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            background: var(--accent);
            border-color: var(--accent);
            color: #ffffff;
        }

        div.stButton > button {
            background: var(--accent);
            border: 1px solid var(--accent);
            border-radius: 10px;
            color: #ffffff;
            font-weight: 600;
        }

        div.stButton > button:hover {
            background: var(--accent-strong);
            border-color: var(--accent-strong);
        }

        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {
            background: var(--panel-bg);
            border: 1px solid var(--border);
            border-radius: 14px;
            overflow: hidden;
            box-shadow: var(--shadow);
        }

        details {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid var(--border);
            border-radius: 12px;
        }

        [data-testid="stAlertContainer"] {
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

MENU_RENDERERS = {
    "Portfolio Snapshot": render_snapshot_page,
    "Total Portfolio (Team PNL)": render_team_pnl_page,
    "Cash Equity Analysis": render_cash_equity_page,
    "Weekly Report Generator": render_weekly_report_page,
    "Swap Report Analysis": render_swap_report_page,
}

_apply_global_styles()

menu = st.sidebar.radio("Navigation", list(MENU_RENDERERS.keys()))
MENU_RENDERERS[menu]()
