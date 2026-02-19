from pathlib import Path

from dashboard.core import *  # noqa: F401,F403

ROOT_DIR = Path(__file__).resolve().parents[2]

def render_team_pnl_page():
    st.subheader("📊 Total Team Portfolio Dashboard")
    uploaded_file = st.sidebar.file_uploader("Upload 'Team_PNL.xlsx'", type=['xlsx'], key="pnl")

    pnl_file = uploaded_file
    pnl_data_path = None
    if pnl_file is None:
        script_dir = ROOT_DIR
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
