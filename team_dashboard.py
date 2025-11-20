import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.io as pio
from plotly.subplots import make_subplots
import yfinance as yf

# --- 1. 데이터 로드 함수 ---
def load_data_preserve_order(filename, sheet_name):
    print(f"--- [{sheet_name}] 시트 로딩 중 ---")
    try:
        df_raw = pd.read_excel(filename, sheet_name=sheet_name, header=None, engine='openpyxl')
    except Exception as e:
        print(f"파일 읽기 실패: {e}")
        return pd.DataFrame(), []

    header_idx = -1
    for i in range(10):
        if '일자' in df_raw.iloc[i].astype(str).values:
            header_idx = i
            break
    
    if header_idx == -1: return pd.DataFrame(), []

    raw_cols = df_raw.iloc[header_idx].tolist()
    df_raw.columns = raw_cols
    df_data = df_raw.iloc[header_idx+1:].copy()

    date_col = next((c for c in df_data.columns if str(c).strip() == '일자'), None)
    if not date_col: return pd.DataFrame(), []

    df_data.set_index(date_col, inplace=True)
    df_data.index = pd.to_datetime(df_data.index, errors='coerce')
    df_data = df_data.dropna(how='all')
    df_data = df_data[df_data.index.notnull()]
    
    df_data = df_data.apply(pd.to_numeric, errors='coerce').fillna(0)

    new_columns = []
    seen = {}
    for col in df_data.columns:
        col_str = str(col).strip()
        if col_str in ['nan', 'None', '', 'NaT']: continue
        
        if col_str in seen:
            seen[col_str] += 1
            new_name = f"{col_str}_{seen[col_str]}"
        else:
            seen[col_str] = 0
            new_name = col_str
        new_columns.append(new_name)

    valid_indices = [i for i, c in enumerate(df_raw.iloc[header_idx]) if str(c).strip() not in ['nan', 'None', '', 'NaT', '일자']]
    df_final = df_data.iloc[:, [i-1 for i in valid_indices]]
    df_final.columns = new_columns

    return df_final, new_columns

# --- 2. HTML 표 생성 함수 ---
def create_manual_html_table(df, title=None):
    html = ''
    if title:
        html += f'<h5 class="mt-4 mb-2">{title}</h5>'
    html += '<table class="table table-striped table-hover text-center align-middle table-sm">'
    
    html += '<thead class="table-light" style="color: black; border-bottom: 2px solid #333;"><tr>'
    for col in df.columns:
        html += f'<th scope="col" style="white-space: nowrap;">{col}</th>'
    html += '</tr></thead>'
    
    html += '<tbody>'
    for _, row in df.iterrows():
        html += '<tr>'
        for i, val in enumerate(row):
            align = 'text-start' if i == 0 else 'text-end'
            style = ''
            val_str = str(val)
            if '%' in val_str and '-' in val_str: style = 'color: #dc3545; font-weight: 500;'
            elif '%' in val_str: style = 'color: #198754; font-weight: 500;'
            elif ('Win Rate' in str(df.columns[i]) or 'Profit Factor' in str(df.columns[i])) and val != 'N/A':
                 style = 'font-weight: bold;'
                
            html += f'<td class="{align}" style="{style}">{val}</td>'
        html += '</tr>'
    html += '</tbody></table>'
    return html

# --- 메인 실행 ---
filename = 'Team_PNL.xlsx'

try:
    print("--- 분석 시작 ---")
    df_pnl, cols_pnl = load_data_preserve_order(filename, 'PNL')
    df_pos, cols_pos = load_data_preserve_order(filename, 'Position')

    common_idx = df_pnl.index.intersection(df_pos.index)
    common_cols = [c for c in cols_pnl if c in df_pos.columns]
    
    df_pnl = df_pnl.loc[common_idx, common_cols]
    df_pos = df_pos.loc[common_idx, common_cols]

    # --- 지표 계산 ---
    print("--- 지표 계산 중 ---")
    df_cum_pnl = df_pnl.cumsum()
    df_user_ret = df_cum_pnl.div(df_pos.replace(0, np.nan)).fillna(0)
    df_daily_ret = df_pnl.div(df_pos.replace(0, np.nan)).fillna(0)

    # 벤치마크
    start, end = df_user_ret.index.min(), df_user_ret.index.max()
    bm_stats = pd.DataFrame()
    try:
        print("벤치마크 다운로드...")
        bm_data = yf.download(['^GSPC', '^KS11'], start=start, end=end + pd.Timedelta(days=5), progress=False)
        if 'Adj Close' in bm_data.columns: bm_close = bm_data['Adj Close']
        elif 'Close' in bm_data.columns: bm_close = bm_data['Close']
        else: bm_close = bm_data
        if isinstance(bm_close.columns, pd.MultiIndex): bm_close.columns = bm_close.columns.get_level_values(0)
        bm_close.rename(columns={'^GSPC': 'SPX', '^KS11': 'KOSPI'}, inplace=True)
        bm_close = bm_close.reindex(df_user_ret.index, method='ffill')
        bm_ret = bm_close.pct_change().fillna(0)
        bm_cum = (1 + bm_ret).cumprod() - 1
        
        # BM Stats
        bm_stats = pd.DataFrame(index=['KOSPI', 'SPX'])
        bm_stats['Volatility'] = bm_ret.std() * np.sqrt(252)
        bm_stats['Sharpe'] = (bm_ret.mean() / bm_ret.std() * np.sqrt(252)).fillna(0)
        nav = (1 + bm_ret).cumprod()
        bm_stats['MDD'] = ((nav - nav.cummax()) / nav.cummax()).min()
        bm_stats['Total Return'] = bm_cum.iloc[-1]
        # Add Dummy columns for advanced stats
        bm_stats['Win Rate'] = 'N/A'
        bm_stats['Profit Factor'] = 'N/A'
        
    except:
        bm_cum = pd.DataFrame(0, index=df_user_ret.index, columns=['SPX', 'KOSPI'])

    # --- 1. 기본 통계 (Basic Stats) ---
    stats = pd.DataFrame(index=df_daily_ret.columns)
    stats['Volatility'] = df_daily_ret.std() * np.sqrt(252)
    stats['Sharpe'] = (df_daily_ret.mean() / df_daily_ret.std() * np.sqrt(252)).fillna(0)
    nav = (1 + df_daily_ret).cumprod()
    stats['MDD'] = ((nav - nav.cummax()) / nav.cummax()).min()
    stats['Total Return'] = df_user_ret.iloc[-1]

    # --- 2. 심화 통계 (Advanced Stats: 승률, 손익비) ---
    # 승률: 수익 난 일수 / 전체 일수
    win_days = (df_daily_ret > 0).sum()
    total_days = (df_daily_ret != 0).sum() # 거래가 없는 날은 제외
    stats['Win Rate'] = (win_days / total_days).fillna(0)

    # 손익비: 평균 수익 / 평균 손실 (절대값)
    avg_gain = df_daily_ret[df_daily_ret > 0].mean()
    avg_loss = df_daily_ret[df_daily_ret < 0].mean().abs()
    stats['Profit Factor'] = (avg_gain / avg_loss).fillna(0)

    # 벤치마크 합치기
    if not bm_stats.empty:
        stats = pd.concat([stats, bm_stats])

    # 포맷팅
    disp = stats.copy()
    disp['Volatility'] = disp['Volatility'].apply(lambda x: f"{x:.2%}")
    disp['MDD'] = disp['MDD'].apply(lambda x: f"{x:.2%}")
    disp['Sharpe'] = disp['Sharpe'].apply(lambda x: f"{x:.2f}")
    disp['Total Return'] = disp['Total Return'].apply(lambda x: f"{x:.2%}")
    disp['Win Rate'] = disp['Win Rate'].apply(lambda x: f"{x:.1%}" if isinstance(x, (int, float)) else x)
    disp['Profit Factor'] = disp['Profit Factor'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)

    # 인덱스 정리
    disp.insert(0, 'Strategy', disp.index)
    disp.reset_index(drop=True, inplace=True)
    disp['Strategy'] = disp['Strategy'].apply(lambda x: x.split('_')[0] if '_' in x else x)
    
    table_html = create_manual_html_table(disp, "Comprehensive Performance Metrics")

    # --- 3. 상관관계 매트릭스 (Correlation Matrix) ---
    # 일별 수익률 기준 상관관계 계산
    corr_matrix = df_daily_ret.corr()
    
    # 히트맵 생성 (Plotly Heatmap)
    z_vals = corr_matrix.values
    x_labels = [c.split('_')[0] for c in corr_matrix.columns]
    y_labels = [c.split('_')[0] for c in corr_matrix.index]
    
    # 주석(Annotation) 텍스트 생성
    annotations = []
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            val = z_vals[i][j]
            text_color = "white" if abs(val) > 0.5 else "black"
            annotations.append(dict(
                x=x_labels[j], y=y_labels[i], text=f"{val:.2f}",
                font=dict(color=text_color), showarrow=False
            ))

    fig_corr = go.Figure(data=go.Heatmap(
        z=z_vals, x=x_labels, y=y_labels,
        colorscale='RdBu', zmin=-1, zmax=1,
        colorbar=dict(title="Correlation")
    ))
    fig_corr.update_layout(
        title="Strategy Correlation Matrix (Daily Returns)",
        height=700, width=900,
        xaxis=dict(tickangle=-45),
        annotations=annotations
    )

    # --- 차트 1: 누적 수익률 ---
    fig1 = go.Figure()
    buttons = []
    strategies = df_user_ret.columns.tolist()
    for i, col in enumerate(strategies):
        is_overseas = any(k in col for k in ['해외', 'Global', 'US'])
        bm_name = 'SPX' if is_overseas else 'KOSPI'
        display_name = col.split('_')[0]
        
        fig1.add_trace(go.Scatter(x=df_user_ret.index, y=df_user_ret[col], name=display_name, visible=(i==0),
                                line=dict(color='#2563eb', width=2), hovertemplate='%{y:.2%}'))
        bm_series = bm_cum[bm_name] if bm_name in bm_cum.columns else pd.Series(0, index=df_user_ret.index)
        fig1.add_trace(go.Scatter(x=df_user_ret.index, y=bm_series, name=f"BM: {bm_name}", visible=(i==0),
                                line=dict(color='#9ca3af', width=2, dash='dash'), hovertemplate='%{y:.2%}'))
        
        vis = [False] * (2 * len(strategies))
        vis[2*i] = True; vis[2*i+1] = True
        buttons.append(dict(label=display_name, method="update", args=[{"visible": vis}, {"title": f"{display_name} vs {bm_name}"}]))

    fig1.update_layout(title=f"{strategies[0].split('_')[0]} Performance", 
                       updatemenus=[dict(active=0, buttons=buttons, direction="down", pad={"r": 10, "t": 10}, showactive=True, x=1.05, y=1.15)],
                       template="plotly_white", height=500, yaxis_tickformat=".2%")

    # --- HTML 생성 ---
    div1 = pio.to_html(fig1, full_html=False, include_plotlyjs='cdn')
    div_corr = pio.to_html(fig_corr, full_html=False, include_plotlyjs=False)

    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>PM Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ background: #f8f9fa; padding: 20px; font-family: 'Segoe UI', sans-serif; }}
            .card {{ border: none; box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 20px; border-radius: 10px; }}
            .nav-tabs .nav-link.active {{ border-bottom: 3px solid #0d6efd; font-weight: bold; color: #0d6efd; }}
            .nav-tabs .nav-link {{ color: #495057; }}
            h3 {{ font-weight: 800; color: #333; }}
        </style>
    </head>
    <body>
        <div class="container-fluid" style="max-width: 1600px;">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h3>🚀 PM Portfolio Dashboard</h3>
                <span class="badge bg-primary">Updated: {df_user_ret.index.max().date()}</span>
            </div>

            <ul class="nav nav-tabs mb-3" id="myTab" role="tablist">
                <li class="nav-item"><button class="nav-link active" id="t1" data-bs-toggle="tab" data-bs-target="#v1">📈 Performance Chart</button></li>
                <li class="nav-item"><button class="nav-link" id="t2" data-bs-toggle="tab" data-bs-target="#v2">📊 Metrics Analysis</button></li>
                <li class="nav-item"><button class="nav-link" id="t3" data-bs-toggle="tab" data-bs-target="#v3">🔗 Correlation</button></li>
            </ul>

            <div class="tab-content">
                <div class="tab-pane fade show active" id="v1">
                    <div class="card"><div class="card-body">{div1}</div></div>
                </div>
                
                <div class="tab-pane fade" id="v2">
                    <div class="card">
                        <div class="card-header bg-white fw-bold">Strategy Statistics</div>
                        <div class="p-4">{table_html}</div>
                    </div>
                    <div class="alert alert-info">
                        <strong>Tip:</strong> 
                        <ul class="mb-0">
                            <li><strong>Win Rate (승률):</strong> 수익 난 일수 / 전체 거래 일수. 50% 이상이면 긍정적입니다.</li>
                            <li><strong>Profit Factor (손익비):</strong> 평균 수익금 / 평균 손실금. 1.5 이상이면 우수합니다.</li>
                            <li><strong>Sharpe Ratio:</strong> 1.0 이상이면 양호, 2.0 이상이면 매우 우수합니다.</li>
                        </ul>
                    </div>
                </div>

                <div class="tab-pane fade" id="v3">
                    <div class="card">
                        <div class="card-body d-flex justify-content-center">
                            {div_corr}
                        </div>
                    </div>
                    <div class="alert alert-warning">
                        <strong>Insight:</strong> 상관관계가 높은(붉은색, > 0.7) 전략들은 시장 충격 시 같이 손실이 날 위험이 있습니다. 분산 투자를 위해 상관관계가 낮은 전략들을 조합하세요.
                    </div>
                </div>
            </div>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """

    with open('portfolio_analysis_dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html_template)
    print("성공: PM 대시보드(상관관계, 승률, 손익비 포함) 생성 완료!")

except Exception as e:
    print(f"오류: {e}")