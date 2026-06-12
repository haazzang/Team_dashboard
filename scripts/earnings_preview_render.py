#!/usr/bin/env python3
"""Render Korean earnings preview Notion-Markdown from fetched FMP data.

Outputs a JSON file mapping ticker -> {title, content_markdown}.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

DATA_PATH = Path("/Users/hyejinha/Desktop/Workspace/Team/analysis_outputs/earnings_preview_data.json")
OUT_PATH = Path("/Users/hyejinha/Desktop/Workspace/Team/analysis_outputs/earnings_preview_pages.json")

# Known BMO/AMC defaults — calendar endpoint omits time field on free tier
TIME_OF_DAY = {
    "WMT": "BMO",
    "ZM": "AMC",
    "NIO": "BMO",
}

# Sector-specific watch points per ticker (analyst-grade qualitative bullets)
WATCH_POINTS = {
    "WMT": [
        "미국 동일점포 매출 (US comp sales ex-fuel) 성장률 — 컨센서스 +3~4% 부근, 트래픽 vs 평균 객단가 기여 분해",
        "E-commerce GMV 성장률 (글로벌 +20%대 유지 여부)과 Walmart Connect 광고 매출 기여",
        "Sam's Club 멤버십 매출과 회원 갱신율, 신규 가입자 트렌드",
        "관세(tariff) 영향과 식료품 vs 일반 상품 마진 mix — gross margin 가이던스 톤",
        "FY27 가이던스: 매출 +3~4%, EPS YoY 한 자릿수 후반 성장 유지 여부",
        "Capex 가이던스 (자동화·AI·물류센터)와 FCF conversion",
        "재고 회전 / inventory health — supply chain 안정화 메시지",
        "주가 모멘텀: $135 신고가 부근 valuation re-rating (NTM PE 35x+)에 대한 펀더멘털 정당화 여부",
    ],
    "ZM": [
        "Enterprise 매출 성장률 (전체 매출 60% 이상 비중) — 두 자릿수 성장 회복 여부",
        "Online (SMB) 매출 churn 안정화 — 12개월 연속 감소 추세 brake 여부",
        "Net dollar expansion rate (>$100K customers) — 100% 이상 유지 가능성",
        "AI Companion 유료화 (Zoom AI Companion 2.0 / Custom AI) 매출 기여 가시화",
        "Contact Center / Workvivo / Phone 비-Meetings 제품 mix 비중 (>20% 목표)",
        "Operating margin (Non-GAAP) — 39~40% 가이던스 상회 여부, 자사주 매입 페이스",
        "FY27 가이던스: 매출 ~$5.0~5.1B, 한 자릿수 후반 EPS 성장",
        "현금 $7B+ 대비 시총 $29B → 밸류에이션 매력 vs 성장 정체 디스카운트",
    ],
    "NIO": [
        "분기 차량 인도 대수 (4Q25 실제 ~72k vs 1Q26 가이던스 41k~43k) 달성 여부",
        "차량 매출 총이익률(vehicle margin) — 2H25 13~15% 회복 추세의 1Q 지속성",
        "ONVO (서브-브랜드) 인도량 ramp — 월 1만대 도달 시점",
        "Firefly 유럽 출시 진척, ET9 플래그쉽 인도 본격화",
        "Battery Swap 네트워크 확장 vs CapEx burn (분기 ~RMB10B 영업현금흐름 적자)",
        "신규 자금조달 / 정부 지원 (Hefei) — 유동성 12개월 runway 확보 여부",
        "중국 NEV 시장 점유율과 BYD·Xiaomi와의 가격 경쟁 강도",
        "2026 BEP(손익분기) 가이던스 유지 여부 — 컨센은 FY26 적자 지속 전제",
    ],
}


CURRENCY_PREFIX = {"USD": "$", "CNY": "¥", "EUR": "€", "JPY": "¥", "KRW": "₩", "HKD": "HK$", "GBP": "£"}


def fmt_num(v, suffix="", sig=2, currency="USD"):
    if v is None:
        return "n/a"
    try:
        x = float(v)
    except Exception:
        return str(v)
    p = CURRENCY_PREFIX.get(currency, "$")
    if abs(x) >= 1e12:
        return f"{p}{x/1e12:.{sig}f}T{suffix}"
    if abs(x) >= 1e9:
        return f"{p}{x/1e9:.{sig}f}B{suffix}"
    if abs(x) >= 1e6:
        return f"{p}{x/1e6:.{sig}f}M{suffix}"
    if abs(x) >= 1e3:
        return f"{p}{x/1e3:.{sig}f}K{suffix}"
    return f"{p}{x:.{sig}f}{suffix}"


def fmt_pct(v, sig=1):
    if v is None:
        return "n/a"
    try:
        x = float(v) * (100 if abs(float(v)) < 5 else 1)
    except Exception:
        return str(v)
    return f"{x:.{sig}f}%"


def fmt_pct_raw(v, sig=1):
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.{sig}f}%"
    except Exception:
        return str(v)


def fmt_mult(v, sig=2):
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.{sig}f}x"
    except Exception:
        return str(v)


def fmt_price(v, sig=2):
    if v is None:
        return "n/a"
    try:
        return f"${float(v):.{sig}f}"
    except Exception:
        return str(v)


def yoy(curr, prev):
    if curr is None or prev is None:
        return None
    try:
        c = float(curr); p = float(prev)
        if p == 0:
            return None
        return (c / p - 1) * 100
    except Exception:
        return None


def get_consensus_korean(c):
    mapping = {
        "Strong Buy": "Strong Buy",
        "Buy": "Buy",
        "Hold": "Hold",
        "Sell": "Sell",
        "Strong Sell": "Strong Sell",
    }
    return mapping.get(c, c or "n/a")


def next_quarter_label(symbol, last_period, last_fiscal_year):
    """Compute upcoming quarter label given last reported quarter."""
    # last_period like 'Q4', last_fiscal_year like '2026' (string)
    try:
        ly = int(last_fiscal_year)
    except Exception:
        ly = datetime.now().year
    seq = {"Q1": "Q2", "Q2": "Q3", "Q3": "Q4", "Q4": "Q1"}
    nxt = seq.get(last_period, "Q1")
    nxt_year = ly + 1 if last_period == "Q4" else ly
    # Calendar vs fiscal label
    if symbol in ("WMT", "ZM"):
        return f"{nxt} FY{nxt_year}"
    return f"{nxt} {nxt_year}"


def render_one(item):
    sym = item["symbol"]
    q = item.get("quote") or {}
    prof = item.get("profile") or {}
    ratios = item.get("ratios_ttm") or {}
    km = item.get("key_metrics_ttm") or {}
    pt = item.get("price_target") or {}
    grades = item.get("grades") or {}
    ntm = item.get("ntm") or {}
    iq = item.get("income_quarterly") or []
    cal = item.get("calendar") or {}

    # --- Snapshot fields ---
    price = q.get("price")
    mcap = q.get("marketCap") or prof.get("marketCap")
    rng = prof.get("range") or f"{q.get('yearLow')}-{q.get('yearHigh')}"
    ma50 = q.get("priceAvg50")
    ma200 = q.get("priceAvg200")
    pt_cons = pt.get("targetConsensus")
    pt_high = pt.get("targetHigh")
    pt_low = pt.get("targetLow")
    pt_med = pt.get("targetMedian")

    cons_text = get_consensus_korean(grades.get("consensus"))
    sb = grades.get("strongBuy") or 0
    bb = grades.get("buy") or 0
    hh = grades.get("hold") or 0
    ss = grades.get("sell") or 0
    sss = grades.get("strongSell") or 0
    total = sb + bb + hh + ss + sss

    upside = None
    if price and pt_cons:
        upside = (float(pt_cons) / float(price) - 1) * 100

    # --- Quarter labels ---
    most_recent = iq[0] if iq else {}
    last_period = most_recent.get("period", "Q4")
    last_fy = most_recent.get("fiscalYear") or str(datetime.now().year)
    upcoming_q_label = next_quarter_label(sym, last_period, last_fy)
    report_ccy = most_recent.get("reportedCurrency") or "USD"
    # Rough FX to USD for YoY comparison only (track record uses local ccy)
    FX_TO_USD = {"USD": 1.0, "CNY": 1 / 7.2, "JPY": 1 / 155.0, "EUR": 1.08, "GBP": 1.27, "HKD": 1 / 7.8, "KRW": 1 / 1380.0}
    fx = FX_TO_USD.get(report_ccy, 1.0)

    earnings_date = cal.get("date") or "TBD"
    tod = TIME_OF_DAY.get(sym, "TBD")

    # --- Consensus estimates this quarter ---
    eps_est = cal.get("epsEstimated")
    rev_est = cal.get("revenueEstimated")
    # YoY using same-quarter year ago (index 3 in 5-quarter list = same Q a year ago: i=3 since [Q4,Q3,Q2,Q1,Q4y-1])
    # Actually upcoming is next quarter after most recent; YoY comparator = the quarter 4 reports back = iq[3]
    yoy_comp = iq[3] if len(iq) > 3 else None
    # Convert prior-Q to USD for YoY against USD consensus
    yoy_comp_rev_usd = (yoy_comp.get("revenue") * fx) if yoy_comp and yoy_comp.get("revenue") else None
    yoy_comp_eps_usd = ((yoy_comp.get("epsdiluted") or yoy_comp.get("eps") or 0) * fx) if yoy_comp else None
    yoy_rev = yoy(rev_est, yoy_comp_rev_usd) if yoy_comp_rev_usd else None
    yoy_eps = yoy(eps_est, yoy_comp_eps_usd) if yoy_comp_eps_usd else None

    # --- Track record (last 5 reported quarters: actuals + YoY) ---
    track_rows = []
    for i, r in enumerate(iq[:5]):
        # YoY comp = i+4 (same quarter previous year)
        rev = r.get("revenue")
        eps = r.get("epsdiluted") or r.get("eps")
        # Find same-quarter prior year by date matching month
        same_q_prev = None
        d = r.get("date", "")
        if d and len(d) >= 7:
            month = d[5:7]
            for r2 in iq[i + 1:]:
                if (r2.get("date") or "")[5:7] == month and (r2.get("date") or "")[:4] != d[:4]:
                    same_q_prev = r2; break
        rev_yoy = yoy(rev, same_q_prev.get("revenue") if same_q_prev else None)
        eps_yoy = yoy(eps, (same_q_prev.get("epsdiluted") or same_q_prev.get("eps")) if same_q_prev else None)
        track_rows.append({
            "period": f"{r.get('period','')} FY{r.get('fiscalYear','')}",
            "date": d,
            "revenue": rev,
            "rev_yoy": rev_yoy,
            "eps": eps,
            "eps_yoy": eps_yoy,
            "gross_margin": (r.get("grossProfit") or 0) / rev * 100 if rev else None,
            "op_margin": (r.get("operatingIncome") or 0) / rev * 100 if rev else None,
        })

    # --- Valuation: TTM vs NTM ---
    ttm_pe = ratios.get("priceToEarningsRatioTTM") or ratios.get("peRatioTTM")
    ttm_ev_sales = km.get("evToSalesTTM")
    ttm_ev_ebitda = km.get("enterpriseValueOverEBITDATTM") or km.get("evToEbitdaTTM")
    ev = km.get("enterpriseValueTTM")

    ntm_rev_local = ntm.get("ntm_revenue")
    ntm_eps_local = ntm.get("ntm_eps")
    ntm_ebitda_local = ntm.get("ntm_ebitda")
    # Estimates are in company's reporting currency.
    # EV from key-metrics-ttm is also in reporting currency -> EV/Sales, EV/EBITDA stay in local.
    # Price (USD) requires EPS to be converted to USD for the P/E multiple.
    ntm_eps_usd = ntm_eps_local * fx if ntm_eps_local is not None else None

    ntm_pe = (price / ntm_eps_usd) if (price and ntm_eps_usd and ntm_eps_usd != 0) else None
    ntm_ev_sales = (ev / ntm_rev_local) if (ev and ntm_rev_local) else None
    ntm_ev_ebitda = (ev / ntm_ebitda_local) if (ev and ntm_ebitda_local) else None

    # --- Bull / Base / Bear ---
    base_target = pt_cons
    bull_target = pt_high or (pt_cons * 1.15 if pt_cons else None)
    bear_target = pt_low or (pt_cons * 0.80 if pt_cons else None)

    def upside_pct(target):
        if not target or not price:
            return None
        return (float(target) / float(price) - 1) * 100

    # ---------- Build Markdown ----------
    L = []  # lines

    # Section 1: Snapshot
    L.append("## 스냅샷")
    L.append("")
    L.append("| 항목 | 값 |")
    L.append("| --- | --- |")
    L.append(f"| 종목 / 회사명 | **{sym}** — {prof.get('companyName','')} |")
    L.append(f"| 섹터 / 산업 | {prof.get('sector','n/a')} / {prof.get('industry','n/a')} |")
    L.append(f"| 발표 일자 / 시점 | **{earnings_date} ({tod})** |")
    L.append(f"| 대상 분기 | **{upcoming_q_label}** |")
    L.append(f"| 현재가 | {fmt_price(price)} |")
    L.append(f"| 시가총액 | {fmt_num(mcap)} |")
    L.append(f"| 52주 범위 | {rng} |")
    L.append(f"| 50D MA / 200D MA | {fmt_price(ma50)} / {fmt_price(ma200)} |")
    L.append(f"| 애널리스트 컨센서스 | **{cons_text}** (총 {total}명 — SB:{sb} B:{bb} H:{hh} S:{ss} SS:{sss}) |")
    L.append(f"| Price Target (Cons / Median / High / Low) | {fmt_price(pt_cons)} / {fmt_price(pt_med)} / {fmt_price(pt_high)} / {fmt_price(pt_low)} |")
    L.append(f"| 컨센서스 대비 상승여력 | **{fmt_pct_raw(upside)}** |")
    L.append("")

    # Section 2: Quarter consensus
    L.append(f"## {upcoming_q_label} 컨센서스 예상치")
    L.append("")
    L.append("| 지표 | 컨센서스 | 전년 동기 (실제) | YoY |")
    L.append("| --- | --- | --- | --- |")
    # Show USD-equivalent comparator if foreign currency
    L.append(f"| 매출 (Revenue, USD) | {fmt_num(rev_est)} | {fmt_num(yoy_comp_rev_usd)} | {fmt_pct_raw(yoy_rev)} |")
    if eps_est is not None and yoy_comp_eps_usd is not None:
        L.append(f"| 주당순이익 (EPS, USD) | ${eps_est:.2f} | ${yoy_comp_eps_usd:.2f} | {fmt_pct_raw(yoy_eps)} |")
    else:
        L.append(f"| 주당순이익 (EPS, USD) | {eps_est} | n/a | {fmt_pct_raw(yoy_eps)} |")
    L.append("")
    ccy_note = "" if report_ccy == "USD" else f" · 회사 보고통화 {report_ccy} → USD 환산 (FX≈1 USD = {1/fx:.2f} {report_ccy})"
    L.append(f"_컨센서스 출처: FMP analyst-estimates (집계일 {cal.get('lastUpdated','n/a')}){ccy_note}_")
    L.append("")

    # Section 3: Track record
    L.append(f"## 실적 트랙 레코드 (최근 5분기, 보고통화 {report_ccy})")
    L.append("")
    L.append(f"| 분기 | 결산일 | 매출 ({report_ccy}) | 매출 YoY | EPS (diluted, {report_ccy}) | EPS YoY | Gross M | Op M |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    eps_prefix = CURRENCY_PREFIX.get(report_ccy, "$")
    for r in track_rows:
        eps_str = f"{eps_prefix}{r['eps']:.2f}" if r['eps'] is not None else "n/a"
        L.append(
            f"| {r['period']} | {r['date']} | {fmt_num(r['revenue'], currency=report_ccy)} | "
            f"{fmt_pct_raw(r['rev_yoy'])} | {eps_str} | {fmt_pct_raw(r['eps_yoy'])} | "
            f"{fmt_pct_raw(r['gross_margin'])} | {fmt_pct_raw(r['op_margin'])} |"
        )
    L.append("")
    L.append("_주: 컨센서스 beat/miss 이력은 FMP 무료 등급 제한으로 본 페이지에서는 제공 불가. 위 표는 실제 발표 수치 기준._")
    L.append("")

    # Section 4: Watch points
    L.append("## 주요 관전 포인트")
    L.append("")
    for bp in WATCH_POINTS.get(sym, ["섹터 일반 펀더멘털 모니터링 항목 — 데이터 보강 필요"]):
        L.append(f"- {bp}")
    L.append("")

    # Section 5: Valuation
    L.append("## Valuation (TTM vs NTM)")
    L.append("")
    L.append("| 지표 | TTM | NTM (0.75·FY0E + 0.25·FY1E) |")
    L.append("| --- | --- | --- |")
    L.append(f"| P/E | {fmt_mult(ttm_pe)} | {fmt_mult(ntm_pe)} |")
    L.append(f"| EV / Sales | {fmt_mult(ttm_ev_sales)} | {fmt_mult(ntm_ev_sales)} |")
    L.append(f"| EV / EBITDA | {fmt_mult(ttm_ev_ebitda)} | {fmt_mult(ntm_ev_ebitda)} |")
    L.append("")
    if ntm.get("fy0_year") and ntm.get("fy1_year") and ntm_eps_local is not None:
        eps_p = CURRENCY_PREFIX.get(report_ccy, "$")
        ccy_tail = f" · EPS for P/E는 USD 환산 ({eps_p}{ntm_eps_local:.2f} → ${ntm_eps_usd:.2f})" if report_ccy != "USD" else ""
        L.append(
            f"_NTM blend: FY{ntm['fy0_year']} 75% + FY{ntm['fy1_year']} 25% — "
            f"NTM Rev {fmt_num(ntm_rev_local, currency=report_ccy)} | "
            f"NTM EPS {eps_p}{ntm_eps_local:.2f} | "
            f"NTM EBITDA {fmt_num(ntm_ebitda_local, currency=report_ccy)}{ccy_tail}_"
        )
    L.append("")

    # Section 6: Bull / Base / Bear
    L.append("## Bull / Base / Bear 시나리오")
    L.append("")
    L.append("| 시나리오 | 타겟 가격 | 현재가 대비 | 트리거 |")
    L.append("| --- | --- | --- | --- |")
    L.append(f"| Bull | {fmt_price(bull_target)} | {fmt_pct_raw(upside_pct(bull_target))} | 컨센 매출·EPS 동시 beat + 가이던스 상향 |")
    L.append(f"| Base | {fmt_price(base_target)} | {fmt_pct_raw(upside_pct(base_target))} | In-line 분기, 가이던스 유지 |")
    L.append(f"| Bear | {fmt_price(bear_target)} | {fmt_pct_raw(upside_pct(bear_target))} | EPS miss 또는 가이던스 하향 |")
    L.append("")
    L.append("_타겟: 애널리스트 PT High/Consensus/Low 사용. PT가 결측인 경우 ±15%/±20% 휴리스틱._")
    L.append("")

    # Section 7: Recent news
    L.append("## 최근 뉴스")
    L.append("")
    news = item.get("news") or []
    if news:
        for n in news[:5]:
            t = n.get("title") or n.get("text") or ""
            url = n.get("url") or n.get("link") or ""
            pub = (n.get("publishedDate") or n.get("date") or "")[:10]
            src = n.get("site") or n.get("publisher") or n.get("source") or ""
            L.append(f"- **{pub}** [{src}]: [{t}]({url})")
    else:
        L.append("- 본 데이터 등급에서는 stock news 엔드포인트 접근 제한 (HTTP 402). 별도 단말 (Bloomberg / Reuters / 회사 IR 발표) 확인 필요.")
    L.append("")

    # Section 8: Trade setup
    L.append("## 트레이드 셋업")
    L.append("")
    L.append("**기술적 레벨**")
    try:
        lo, hi = (rng or "0-0").split("-")
        lo, hi = float(lo), float(hi)
        from_low = (float(price) / lo - 1) * 100 if price else None
        from_high = (float(price) / hi - 1) * 100 if price else None
        L.append(f"- 52주 저점 {lo:.2f} / 고점 {hi:.2f} — 현재가 저점 대비 {fmt_pct_raw(from_low)}, 고점 대비 {fmt_pct_raw(from_high)}")
    except Exception:
        L.append(f"- 52주 범위: {rng}")
    L.append(f"- 50일 / 200일 이평선 ({fmt_price(ma50)} / {fmt_price(ma200)}) — 현재가 위치 확인: "
             f"50D {'위' if (price and ma50 and float(price) > float(ma50)) else '아래'}, "
             f"200D {'위' if (price and ma200 and float(price) > float(ma200)) else '아래'}")
    L.append("")
    L.append("**옵션 / 내재변동성**")
    L.append("- 옵션 IV 데이터는 본 FMP 등급에서 제공되지 않음. CBOE/OPRA 또는 브로커 단말로 별도 확인 권장.")
    L.append(f"- 참고 휴리스틱: 대형 발표 직전 ATM straddle 가격이 통상 분기 평균 무브의 1.0~1.2배 수준에서 형성됨.")
    L.append("")
    L.append("**포지셔닝 / 그레이드 모멘텀**")
    L.append(f"- 애널리스트 컨센서스 **{cons_text}** (총 {total}명). PT consensus {fmt_price(pt_cons)} — 현재가 대비 {fmt_pct_raw(upside)}.")
    if grades.get("strongBuy") is not None and total > 0:
        bull_share = (sb + bb) / total * 100
        L.append(f"- Buy 비중 {bull_share:.0f}% (SB+B={sb+bb} / 총 {total})")
    L.append("")
    L.append("---")
    L.append(f"_데이터: FMP stable API · 생성 시각 KST {datetime.now().isoformat(timespec='minutes')} · 본 자료는 정보 제공 목적이며 매매 권유가 아님._")

    content = "\n".join(L)

    title = f"{sym} — {upcoming_q_label} Earnings Preview ({earnings_date} {tod})"
    return {"symbol": sym, "title": title, "content": content}


def main():
    data = json.loads(DATA_PATH.read_text())
    pages = [render_one(it) for it in data["items"]]
    OUT_PATH.write_text(json.dumps({"pages": pages}, ensure_ascii=False, indent=2))
    for p in pages:
        print(p["symbol"], "->", p["title"], "(", len(p["content"]), "chars )")
    print(f"[ok] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
