from pathlib import Path

from dashboard.core import *  # noqa: F401,F403

ROOT_DIR = Path(__file__).resolve().parents[2]

def render_swap_report_page():
    st.subheader("Swap Report Analysis (JMLNKWGE)")

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
