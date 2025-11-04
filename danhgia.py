import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
import warnings
import datetime

warnings.filterwarnings("ignore")


# ----------------------- # Lấy tên công ty # -----------------------
def get_company_name(stock_code):
    stock_code = stock_code.strip().upper()
    try:
        t = yf.Ticker(stock_code + ".VN")
        info = {}
        try:
            info = t.info or {}
        except Exception:
            try:
                t2 = yf.Ticker(stock_code)
                info = t2.info or {}
            except Exception:
                info = {}
        if info:
            name = info.get("longName") or info.get("shortName") or info.get("companyShortName")
            if name and isinstance(name, str) and len(name.strip()) > 1:
                return name.strip()
    except Exception:
        pass
    try:
        url = f"https://www.hnx.vn/vi-vn/co-phieu-{stock_code}.html"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        r = requests.get(url, headers=headers, timeout=8, verify=False)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            h1 = soup.find("h1")
            if h1 and h1.text.strip():
                return h1.text.strip()
            title = soup.find("title")
            if title and title.text.strip():
                return title.text.split("|")[0].strip()
    except Exception:
        pass
    return "Không tìm thấy tên công ty"


# ----------------------- # Tính MACD, RSI, EMA và mua tốt # -----------------------
def calculate_indicators(df, short=12, long=26, signal=9, rsi_period=14):
    df["EMA_short"] = df["Close"].ewm(span=short, adjust=False).mean()
    df["EMA_long"] = df["Close"].ewm(span=long, adjust=False).mean()
    df["MACD"] = df["EMA_short"] - df["EMA_long"]
    df["Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["Histogram"] = df["MACD"] - df["Signal"]
    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=rsi_period).mean()
    avg_loss = pd.Series(loss).rolling(window=rsi_period).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Golden / Death Cross
    df["GC"] = (df["EMA_short"] > df["EMA_long"]) & (df["EMA_short"].shift(1) <= df["EMA_long"].shift(1))
    df["DC"] = (df["EMA_short"] < df["EMA_long"]) & (df["EMA_short"].shift(1) >= df["EMA_long"].shift(1))

    # Mua tốt: Golden Cross + RSI < 30
    df["BuySignal"] = df["GC"] & (df["RSI"] < 30)

    return df


# ----------------------- # Load dữ liệu giá # -----------------------
@st.cache_data(ttl=1800)
def load_stock_data(stock_code, period="6mo"):
    try:
        df = yf.download(f"{stock_code}.VN", period=period, progress=False)
        if df.empty:
            df = yf.download(stock_code, period=period, progress=False)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    if "Date" not in df.columns or "Close" not in df.columns:
        return None
    df = df[["Date", "Close"]].dropna().reset_index(drop=True)
    df = calculate_indicators(df)
    return df


# ----------------------- # Lấy chỉ số cơ bản từ yfinance # -----------------------
@st.cache_data(ttl=3600)
def get_fundamentals(stock_code):
    """Trả về dict các chỉ số: valuation, profitability, financial health, growth"""
    code_vn = stock_code + ".VN"
    t = yf.Ticker(code_vn)
    info = {}
    try:
        info = t.info or {}
    except Exception:
        try:
            t2 = yf.Ticker(stock_code)
            info = t2.info or {}
        except Exception:
            info = {}

    # Try to extract main fields with safe-get
    def safe(key, default=np.nan):
        return info.get(key, default)

    # Basic valuation
    trailingPE = safe("trailingPE", np.nan)
    forwardPE = safe("forwardPE", np.nan)
    priceToBook = safe("priceToBook", np.nan)
    priceToSales = safe("priceToSales", np.nan)
    pegRatio = safe("pegRatio", np.nan)
    marketCap = safe("marketCap", np.nan)
    enterpriseValue = safe("enterpriseValue", np.nan)
    evToEbitda = safe("enterpriseToEbitda", np.nan) or safe("evToEbitda", np.nan) or np.nan

    # Profitability
    roa = safe("returnOnAssets", np.nan)
    roe = safe("returnOnEquity", np.nan)
    netMargin = safe("profitMargins", np.nan)
    grossMargins = safe("grossMargins", np.nan)

    # Financial health
    debtToEquity = safe("debtToEquity", np.nan)
    currentRatio = safe("currentRatio", np.nan)
    quickRatio = safe("quickRatio", np.nan)
    interestCoverage = safe("interestCoverage", np.nan)

    # Growth - try to compute revenue/eps growth from financials if possible
    revenue_growth = np.nan
    eps_growth = np.nan
    try:
        # annual financials (may be empty)
        fin = t.financials  # columns are periods
        if isinstance(fin, pd.DataFrame) and not fin.empty:
            revenues = fin.loc["Total Revenue"] if "Total Revenue" in fin.index else (fin.iloc[0] if fin.shape[0] > 0 else None)
            if revenues is not None and revenues.shape[0] >= 2:
                revs = revenues.values
                # calculate YoY growth using last two periods
                revenue_growth = (revs[0] - revs[1]) / abs(revs[1]) if revs[1] != 0 else np.nan
    except Exception:
        revenue_growth = np.nan

    try:
        eps = t.earnings  # DataFrame with Year, Earnings
        # BUT better: use info fields
        eps_growth = safe("earningsQuarterlyGrowth", np.nan)
    except Exception:
        eps_growth = np.nan

    # Normalize some fields (convert to percentages where applicable)
    # Return as friendly dict
    fundamentals = {
        "trailingPE": trailingPE,
        "forwardPE": forwardPE,
        "pegRatio": pegRatio,
        "priceToBook": priceToBook,
        "priceToSales": priceToSales,
        "evToEbitda": evToEbitda,
        "marketCap": marketCap,
        "enterpriseValue": enterpriseValue,
        "ROA": roa,
        "ROE": roe,
        "netMargin": netMargin,
        "grossMargin": grossMargins,
        "debtToEquity": debtToEquity,
        "currentRatio": currentRatio,
        "quickRatio": quickRatio,
        "interestCoverage": interestCoverage,
        "revenueGrowth": revenue_growth,
        "epsGrowth": eps_growth,
        "info": info,
    }
    return fundamentals


# ----------------------- # Hàm chấm điểm các nhóm chỉ số # -----------------------
def score_valuation(f):
    """Higher score = attractive valuation. Returns 0-1"""
    scores = []
    # P/E: lower better (but too low maybe trouble) - we map typical range (5..50)
    pe = f.get("trailingPE", np.nan)
    if not np.isnan(pe):
        # clamp and invert
        pe_clamped = min(max(pe, 5), 50)
        scores.append((50 - (pe_clamped - 5)) / 50)  # 1 when pe=5, 0 when pe=50
    # P/B: lower better
    pb = f.get("priceToBook", np.nan)
    if not np.isnan(pb):
        pb_clamped = min(max(pb, 0.1), 10)
        scores.append((10 - pb_clamped) / 10)
    # PEG: closer to 1 is good, <1 excellent
    peg = f.get("pegRatio", np.nan)
    if not np.isnan(peg):
        peg_clamped = min(max(peg, 0.1), 5)
        scores.append((5 - (peg_clamped - 0.1)) / 5)
    # EV/EBITDA: lower better, typical 3..20
    ev = f.get("evToEbitda", np.nan)
    if not np.isnan(ev):
        ev_clamped = min(max(ev, 1), 30)
        scores.append((30 - (ev_clamped - 1)) / 30)
    if len(scores) == 0:
        return np.nan
    return np.nanmean(scores)


def score_profitability(f):
    """ROE, ROA, net margin"""
    scores = []
    roe = f.get("ROE", np.nan)
    if not np.isnan(roe):
        # expect roe in decimal (0.15 -> 15%)
        roe_pct = roe if abs(roe) <= 1 else roe / 100.0
        # good if >15%
        scores.append(min(max((roe_pct - 0.0) / 0.25, 0), 1))  # scale 0..25%
    roa = f.get("ROA", np.nan)
    if not np.isnan(roa):
        roa_pct = roa if abs(roa) <= 1 else roa / 100.0
        scores.append(min(max((roa_pct) / 0.15, 0), 1))  # target 15%
    netm = f.get("netMargin", np.nan)
    if not np.isnan(netm):
        net_pct = netm if abs(netm) <= 1 else netm / 100.0
        scores.append(min(max(net_pct / 0.2, 0), 1))  # target 20%
    if len(scores) == 0:
        return np.nan
    return np.nanmean(scores)


def score_financial_health(f):
    """D/E low, current ratio decent, interest coverage"""
    scores = []
    de = f.get("debtToEquity", np.nan)
    if not np.isnan(de):
        # lower is better; expect 0..3
        de_clamped = min(max(de, 0), 5)
        scores.append((5 - de_clamped) / 5)
    cr = f.get("currentRatio", np.nan)
    if not np.isnan(cr):
        cr_clamped = min(max(cr, 0), 3)
        scores.append(min(cr_clamped / 2, 1))
    ic = f.get("interestCoverage", np.nan)
    if not np.isnan(ic):
        ic_clamped = min(max(ic, 0), 20)
        scores.append(min(ic_clamped / 5, 1))
    if len(scores) == 0:
        return np.nan
    return np.nanmean(scores)


def score_growth(f):
    """Revenue & EPS growth"""
    scores = []
    rev = f.get("revenueGrowth", np.nan)
    if not np.isnan(rev):
        # assume rev ~ decimal
        rev_pct = rev if abs(rev) <= 1 else rev / 100.0
        scores.append(min(max(rev_pct / 0.25, 0), 1))  # target 25%
    epsg = f.get("epsGrowth", np.nan)
    if not np.isnan(epsg):
        eps_pct = epsg if abs(epsg) <= 1 else epsg / 100.0
        scores.append(min(max(eps_pct / 0.25, 0), 1))
    if len(scores) == 0:
        return np.nan
    return np.nanmean(scores)


def compute_overall_score(f):
    """Trả về điểm 0..100 theo trọng số"""
    weights = {"valuation": 0.30, "profit": 0.25, "health": 0.20, "growth": 0.25}
    s_val = score_valuation(f)
    s_prof = score_profitability(f)
    s_health = score_financial_health(f)
    s_growth = score_growth(f)

    # if any is nan, reduce weights proportionally
    parts = {"valuation": s_val, "profit": s_prof, "health": s_health, "growth": s_growth}
    available = {k: v for k, v in parts.items() if not np.isnan(v)}
    if len(available) == 0:
        return np.nan, parts
    total_w = sum(weights[k] for k in available.keys())
    score = 0.0
    for k, v in available.items():
        score += (weights[k] / total_w) * v
    return score * 100, parts  # scale 0..100


# ----------------------- # Dự báo Linear Regression # -----------------------
def linear_forecast(df, days_ahead=14):
    df_local = df.copy().reset_index(drop=True)
    df_local["t"] = np.arange(len(df_local))
    X = df_local[["t"]].values.reshape(-1, 1)
    y = df_local["Close"].values.ravel()
    model = LinearRegression()
    model.fit(X, y)
    future_t = np.arange(len(df_local), len(df_local) + days_ahead).reshape(-1, 1)
    preds = model.predict(future_t)
    future_dates = pd.date_range(df_local["Date"].iloc[-1] + pd.Timedelta(days=1), periods=days_ahead, freq="B")
    forecast = pd.DataFrame({"Date": future_dates, "Predicted": preds})
    return forecast, model


# ----------------------- # Streamlit UI # -----------------------
st.set_page_config(page_title="Định giá, Lợi nhuận, Tài chính, Tăng trưởng", layout="wide")
st.title("📈 Định giá, Lợi nhuận, Tài chính, Tăng trưởng")
# --------------------------------------------------
# Hiển thị chú thích tiếng Việt về các chỉ số tài chính & kỹ thuật
# --------------------------------------------------
with st.expander("📘 Giải thích các chỉ số (bấm để xem)"):
    st.markdown("""
    ### 💹 **Chỉ số kỹ thuật**
    - **EMA (Exponential Moving Average)** — Đường trung bình động hàm mũ:  
      → EMA12 phản ứng nhanh (ngắn hạn), EMA26 phản ứng chậm (dài hạn)  
      → Khi EMA12 cắt EMA26 từ dưới lên ⇒ *Golden Cross* (tín hiệu MUA)  
      → Khi EMA12 cắt EMA26 từ trên xuống ⇒ *Death Cross* (tín hiệu BÁN)

    - **MACD (Moving Average Convergence Divergence)** — Đo độ mạnh/yếu của xu hướng:  
      → *MACD > Signal* ⇒ Xu hướng tăng (Bullish)  
      → *MACD < Signal* ⇒ Xu hướng giảm (Bearish)  
      → *Histogram* thể hiện động lượng tăng/giảm

    - **RSI (Relative Strength Index)** — Chỉ số sức mạnh tương đối:  
      → RSI > 70 ⇒ Quá mua (*Overbought*)  
      → RSI < 30 ⇒ Quá bán (*Oversold*)  
      → RSI ≈ 50 ⇒ Trung tính

    ---

    ### 🧾 **Chỉ số định giá**
    - **P/E (Price to Earnings)** — Hệ số giá/lợi nhuận: thấp ⇒ có thể đang bị định giá thấp  
    - **P/B (Price to Book)** — Giá/thặng dư vốn: < 1 ⇒ rẻ so với giá trị sổ sách  
    - **EPS (Earnings Per Share)** — Lợi nhuận trên mỗi cổ phiếu: cao ⇒ sinh lời tốt  
    - **Dividend Yield** — Tỷ suất cổ tức: cao ⇒ cổ tức ổn định  
    - **PEG (Price/Earnings to Growth)** — P/E chia cho tăng trưởng lợi nhuận: < 1 ⇒ định giá thấp so với tốc độ tăng trưởng

    ---

    ### 💼 **Hiệu quả kinh doanh**
    - **ROE (Return on Equity)** — Lợi nhuận trên vốn chủ sở hữu: > 15% ⇒ tốt  
    - **ROA (Return on Assets)** — Lợi nhuận trên tổng tài sản: cao ⇒ sử dụng tài sản hiệu quả  
    - **Profit Margin** — Biên lợi nhuận ròng: cao ⇒ hoạt động hiệu quả

    ---

    ### 🧮 **An toàn tài chính**
    - **Debt to Equity (D/E)** — Tỷ lệ nợ/vốn chủ: < 1 ⇒ an toàn, > 2 ⇒ rủi ro cao  
    - **Current Ratio** — Hệ số thanh toán hiện hành: > 1.5 ⇒ tốt  
    - **Quick Ratio** — Hệ số thanh toán nhanh: > 1 ⇒ khả năng trả nợ ngắn hạn tốt

    ---

    ### 🚀 **Tăng trưởng tương lai**
    - **Revenue Growth** — Tăng trưởng doanh thu: cao, ổn định ⇒ doanh nghiệp phát triển tốt  
    - **EPS Growth** — Tăng trưởng lợi nhuận trên cổ phiếu: phản ánh triển vọng dài hạn  
    - **Cash Flow Growth** — Tăng trưởng dòng tiền: ổn định ⇒ tài chính lành mạnh

    ---

    ### 📊 **Tổng hợp đánh giá**
    - **Technical Rating** — Đánh giá kỹ thuật (RSI, MACD, EMA)  
    - **Fundamental Rating** — Đánh giá cơ bản (P/E, ROE, D/E, EPS, tăng trưởng)  
    - **Financial Safety** — An toàn tài chính  
    - **Growth Potential** — Tiềm năng tăng trưởng  
    - **Final Recommendation** — Kết luận đầu tư (MUA / GIỮ / BÁN / THEO DÕI)
    """)

stock_code = st.text_input("Nhập mã cổ phiếu:", "MSN").strip().upper()
period = st.selectbox("Khoảng thời gian dữ liệu:", ["3mo", "6mo", "1y", "2y"], index=1)
days_to_predict = st.slider("Số ngày dự đoán (ngày làm việc):", 5, 60, 14)

if st.button("🚀 Phân tích"):
    with st.spinner("Đang tải dữ liệu..."):
        df = load_stock_data(stock_code, period)
        fundamentals = get_fundamentals(stock_code)

    if df is None:
        st.error("Không tìm thấy dữ liệu giá cổ phiếu.")
    else:
        company_name = get_company_name(stock_code)
        st.subheader(f"{stock_code} — {company_name}")

        # ------------ Biểu đồ giá và chỉ báo ------------
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close"))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA_short"], mode="lines", name="EMA12"))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA_long"], mode="lines", name="EMA26"))

        gc_dates = df.loc[df["GC"], "Date"]
        gc_prices = df.loc[df["GC"], "Close"]
        dc_dates = df.loc[df["DC"], "Date"]
        dc_prices = df.loc[df["DC"], "Close"]
        fig.add_trace(go.Scatter(x=gc_dates, y=gc_prices, mode="markers", name="Golden Cross",
                                 marker=dict(symbol="triangle-up", color="green", size=12)))
        fig.add_trace(go.Scatter(x=dc_dates, y=dc_prices, mode="markers", name="Death Cross",
                                 marker=dict(symbol="triangle-down", color="red", size=12)))

        buy_dates = df.loc[df["BuySignal"], "Date"]
        buy_prices = df.loc[df["BuySignal"], "Close"]
        if len(buy_dates) > 0:
            fig.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode="markers+text", name="Mua tốt",
                                     marker=dict(symbol="star", color="blue", size=15),
                                     text=["Mua tốt"] * len(buy_dates), textposition="top center"))
        fig.update_layout(title="Giá + EMA12/26 + Golden/Death Cross + Mua tốt", xaxis_title="Ngày", yaxis_title="Giá")
        st.plotly_chart(fig, use_container_width=True)

        # MACD
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], name="MACD"))
        fig_macd.add_trace(go.Scatter(x=df["Date"], y=df["Signal"], name="Signal"))
        fig_macd.add_trace(go.Bar(x=df["Date"], y=df["Histogram"], name="Histogram"))
        fig_macd.update_layout(title="MACD")
        st.plotly_chart(fig_macd, use_container_width=True)

        # RSI
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], mode="lines", name="RSI"))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="blue")
        fig_rsi.update_layout(title="RSI (70=quá mua, 30=quá bán)")
        st.plotly_chart(fig_rsi, use_container_width=True)

        # ------------ Dự báo giá ------------
        forecast, _ = linear_forecast(df, days_ahead=days_to_predict)
        last_price = df["Close"].iloc[-1]
        avg_pred = forecast["Predicted"].mean()
        diff_pct = (avg_pred - last_price) / last_price * 100
        trend = "📈 TĂNG" if diff_pct > 0 else "📉 GIẢM"
        st.markdown(f"""
        ### 🔍 Xu hướng dự báo (dựa trên Linear Regression)
        - Giá hiện tại: **{last_price:,.2f}**
        - Giá trung bình {days_to_predict} ngày: **{avg_pred:,.2f}**
        - Chênh lệch: **{diff_pct:+.2f}%**
        - Xu hướng tổng thể: **{trend}**
        """)

        # ------------ Phân tích cơ bản ------------
        st.markdown("## 🧾 Phân tích cơ bản (Fundamentals)")
        # Hiển thị bảng cơ bản
        fun = fundamentals.copy()
        display = {
            "P/E (trailing)": fun.get("trailingPE", np.nan),
            "P/E (forward)": fun.get("forwardPE", np.nan),
            "PEG": fun.get("pegRatio", np.nan),
            "P/B": fun.get("priceToBook", np.nan),
            "P/S": fun.get("priceToSales", np.nan),
            "EV/EBITDA": fun.get("evToEbitda", np.nan),
            "Market Cap": fun.get("marketCap", np.nan),
            "ROE": fun.get("ROE", np.nan),
            "ROA": fun.get("ROA", np.nan),
            "Gross Margin": fun.get("grossMargin", np.nan),
            "Net Margin": fun.get("netMargin", np.nan),
            "Debt/Equity": fun.get("debtToEquity", np.nan),
            "Current Ratio": fun.get("currentRatio", np.nan),
            "Interest Coverage": fun.get("interestCoverage", np.nan),
            "Revenue Growth (est)": fun.get("revenueGrowth", np.nan),
            "EPS Growth (est)": fun.get("epsGrowth", np.nan),
        }
        df_fund = pd.DataFrame.from_dict(display, orient="index", columns=["Value"])
        # Format percentages and big numbers
        def fmt(x):
            if pd.isna(x):
                return "-"
            if abs(x) > 1e9:
                return f"{x:,.0f}"
            if abs(x) >= 1:
                return f"{x:.2f}"
            return f"{x:.2%}"
        df_fund["Formatted"] = df_fund["Value"].apply(fmt)
        st.table(df_fund[["Formatted"]])

        # ------------ Tính điểm tổng hợp ------------
        overall_score, parts = compute_overall_score(fundamentals)
        if np.isnan(overall_score):
            st.warning("Không đủ dữ liệu cơ bản để tính điểm tổng hợp.")
        else:
            st.metric("Điểm cơ bản tổng hợp (0-100)", f"{overall_score:.1f}")
            st.write("Chi tiết điểm từng nhóm (0-1):")
            st.write({
                "Valuation (P/E, P/B, PEG, EV/EBITDA)": parts.get("valuation"),
                "Profitability (ROE, ROA, Margin)": parts.get("profit"),
                "Financial Health (D/E, Current Ratio, Interest Coverage)": parts.get("health"),
                "Growth (Revenue/EPS growth)": parts.get("growth"),
            })

        # ------------ Kết luận kết hợp KT + CB ------------
        latest_macd = df["MACD"].iloc[-1]
        latest_signal = df["Signal"].iloc[-1]
        latest_rsi = df["RSI"].iloc[-1]
        macd_bullish = latest_macd > latest_signal
        buy_signal_today = df["BuySignal"].iloc[-1]

        # Decision từ cơ bản
        dec_basic = "Không rõ"
        if not np.isnan(overall_score):
            if overall_score >= 70:
                dec_basic = "FUNDAMENTALS: TỐT (ƯU TIÊN MUA / GIỮ dài hạn)"
            elif overall_score >= 50:
                dec_basic = "FUNDAMENTALS: TRUNG TÍNH (GIỮ / THEO DÕI)"
            else:
                dec_basic = "FUNDAMENTALS: YẾU (XEM XÉT BÁN HOẶC TRÁNH MUA)"
        else:
            dec_basic = "FUNDAMENTALS: DỮ LIỆU KHÔNG ĐỦ"

        # Logic tổng hợp: gộp tín hiệu kỹ thuật và cơ bản
        # Score kỹ thuật sơ bộ
        tech_points = 0
        tech_points += 1 if macd_bullish else 0
        tech_points += 1 if latest_rsi < 70 else 0
        tech_points += 1 if latest_rsi < 50 else 0
        tech_points += 1 if buy_signal_today else 0
        # tech_points in 0..4

        final_recommendation = ""
        # if fundamentals very strong and tech ok -> MUA
        if not np.isnan(overall_score):
            if overall_score >= 70 and tech_points >= 2:
                final_recommendation = "🚀 **MUA / TĂNG TỈ LỆ NẮM GIỮ** (Cơ bản mạnh + Kỹ thuật ủng hộ)"
            elif overall_score >= 70 and tech_points < 2:
                final_recommendation = "✅ **NÊN GIỮ** (Cơ bản mạnh, chờ tín hiệu kỹ thuật để mua thêm)"
            elif 50 <= overall_score < 70 and tech_points >= 3:
                final_recommendation = "⚠️ **CÂN NHẮC MUA** (Cơ bản trung tính, kỹ thuật mạnh — rủi ro ngắn hạn)"
            elif 50 <= overall_score < 70 and tech_points < 3:
                final_recommendation = "⏸️ **THEO DÕI** (Cơ bản trung tính, chờ xác nhận kỹ thuật)"
            elif overall_score < 50 and tech_points >= 3:
                final_recommendation = "⚠️ **RỦI RO CAO** (Kỹ thuật ủng hộ ngắn hạn nhưng cơ bản yếu — cân nhắc chốt lời/ngắn hạn)"
            else:
                final_recommendation = "❌ **NÊN BÁN / TRÁNH MUA** (Cơ bản yếu và kỹ thuật không ủng hộ)"
        else:
            # Không có điểm cơ bản -> chỉ dùng kỹ thuật
            if buy_signal_today:
                decision = "🚀 **MUA TỐT** (Golden Cross + RSI thấp, xu hướng tăng mạnh)"
            elif macd_bullish and latest_rsi < 70:
                decision = "✅ **GIỮ HOẶC MUA THÊM** (Xu hướng tăng, chưa quá mua)"
            elif macd_bullish and latest_rsi >= 70:
                decision = "⚠️ **THEO DÕI** (Giá có thể điều chỉnh sau khi quá mua)"
            elif not macd_bullish and latest_rsi > 70:
                decision = "💰 **CHỐT LỜI** (Xu hướng giảm sau quá mua)"
            elif not macd_bullish and latest_rsi < 30:
                decision = "🕐 **THEO DÕI MUA** (RSI thấp, có thể tạo đáy)"
            else:
                decision = "⏸️ **KHÔNG MUA MỚI** (Xu hướng giảm, chưa có tín hiệu hồi phục)"

        st.markdown("## 💡 Kết luận tổng hợp")
        st.markdown(f"- **Tín hiệu kỹ thuật (MACD bullish?)**: {'TĂNG' if macd_bullish else 'GIẢM'}")
        st.markdown(f"- **RSI hiện tại**: **{latest_rsi:.2f}**")
        st.markdown(f"- **Đánh giá cơ bản**: {dec_basic}")
        st.markdown(f"### 🔔 **Đề xuất cuối**: {final_recommendation}")

        # Thêm phần export / lưu kết quả
        export_df = pd.DataFrame({
            "Date": [datetime.datetime.now()],
            "Ticker": [stock_code],
            "Company": [company_name],
            "LastPrice": [last_price],
            "ForecastAvg": [avg_pred],
            "ForecastDiffPct": [diff_pct],
            "FundScore": [None if np.isnan(overall_score) else round(overall_score, 2)],
            "Tech_MACD_Bullish": [macd_bullish],
            "Tech_RSI": [round(latest_rsi, 2)],
            "FinalRec": [final_recommendation],
        })
        csv = export_df.to_csv(index=False).encode("utf-8")
        st.download_button("Tải kết quả phân tích (.csv)", data=csv, file_name=f"{stock_code}_analysis.csv", mime="text/csv")

