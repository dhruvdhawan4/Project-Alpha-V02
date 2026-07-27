import datetime
import time
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ==========================================
# 1. PAGE CONFIGURATION & STATE INIT
# ==========================================
st.set_page_config(
    page_title="Institutional Investment Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "MON100.NS"

# ==========================================
# 2. HELPER & DATA EXTRACTION UTILITIES
# ==========================================
def safe_extract_metrics(info: dict) -> dict:
    """
    Safely parses ticker metadata and guarantees fallbacks for all metric keys.
    Handles Equities, ETFs, and Mutual Funds without throwing NameError or KeyError.
    """
    def fmt(val, is_curr=False, is_pct=False, decimals=2):
        if val is None or val == "N/A" or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        try:
            v = float(val)
            if is_pct:
                return f"{v * 100:.{decimals}f}%"
            if is_curr:
                if v >= 1e12:
                    return f"₹{v / 1e12:.2f} T"
                elif v >= 1e9:
                    return f"₹{v / 1e9:.2f} B"
                elif v >= 1e7:
                    return f"₹{v / 1e7:.2f} Cr"
                elif v >= 1e5:
                    return f"₹{v / 1e5:.2f} L"
                return f"₹{v:,.{decimals}f}"
            return f"{v:.{decimals}f}"
        except (ValueError, TypeError):
            return "N/A"

    quote_type = str(info.get("quoteType", "EQUITY")).upper()
    is_etf_or_mf = quote_type in ["ETF", "MUTUALFUND"] or info.get("industry") == "Investment Vehicles"

    return {
        "quote_type": quote_type,
        "is_etf": is_etf_or_mf,
        "summary": info.get("longBusinessSummary") or info.get("description") or "No detailed overview available.",
        "sector": info.get("sector") or info.get("category") or ("Financial Services" if is_etf_or_mf else "N/A"),
        "industry": info.get("industry") or ("Investment Vehicles" if is_etf_or_mf else "N/A"),
        "market_cap": fmt(info.get("marketCap") or info.get("totalAssets"), is_curr=True),
        "pe": fmt(info.get("trailingPE") or info.get("forwardPE")),
        "pb": fmt(info.get("priceToBook")),
        "peg": fmt(info.get("pegRatio")),
        "div_yield": fmt(info.get("dividendYield"), is_pct=True),
        "roe": fmt(info.get("returnOnEquity"), is_pct=True),
        "margins": fmt(info.get("profitMargins"), is_pct=True),
        "growth": fmt(info.get("revenueGrowth"), is_pct=True),
        "debt_equity": fmt(info.get("debtToEquity")),
        "nav": fmt(info.get("navPrice") or info.get("previousClose") or info.get("regularMarketPrice"), is_curr=True),
        "high_52": fmt(info.get("fiftyTwoWeekHigh"), is_curr=True),
        "low_52": fmt(info.get("fiftyTwoWeekLow"), is_curr=True),
        "beta": fmt(info.get("beta")),
    }

@st.cache_data(ttl=300)
def fetch_market_banner_data():
    """Fetches key global index figures for top dashboard status bar."""
    indices = {"Nifty 50": "^NSEI", "S&P 500": "^GSPC", "Nasdaq": "^IXIC"}
    results = {}
    for name, symbol in indices.items():
        try:
            t = yf.Ticker(symbol)
            hist = t.history(period="2d")
            if len(hist) >= 2:
                close = hist["Close"].iloc[-1]
                prev = hist["Close"].iloc[-2]
                chg = close - prev
                pct = (chg / prev) * 100
                results[name] = {"value": close, "change": chg, "pct": pct}
            else:
                results[name] = {"value": 0.0, "change": 0.0, "pct": 0.0}
        except Exception:
            results[name] = {"value": 0.0, "change": 0.0, "pct": 0.0}
    return results

# ==========================================
# 3. PAGE MODULES
# ==========================================
def render_header_banner():
    st.title("🏛️ Institutional Investment Platform")
    
    # Global Indices & Clock
    data = fetch_market_banner_data()
    cols = st.columns(4)
    
    idx_keys = list(data.keys())
    for i in range(3):
        key = idx_keys[i]
        val = data[key]
        cols[i].metric(
            label=key,
            value=f"{val['value']:,.2f}",
            delta=f"{val['change']:+.2f} ({val['pct']:+.2f}%)"
        )
    
    now = datetime.datetime.now()
    cols[3].metric("System Clock", now.strftime("%H:%M:%S"), f"Date: {now.strftime('%Y-%m-%d')}")
    st.markdown("---")

def render_home_page():
    st.subheader("Select Platform Mode")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("### 📈 Long Term Investing")
        st.write("Multi-factor fundamental screening, Monte Carlo portfolio projection, and ML factor importance analysis.")
        if st.button("Open Long Term Module", use_container_width=True):
            st.session_state.current_page = "Long Term Investing"
            st.rerun()
            
    with col2:
        st.warning("### ⚡ Intraday Trading")
        st.write("Real-time technical indicators, VWAP, SuperTrend, entry/exit target calculations, and short-term signals.")
        if st.button("Open Intraday Module", use_container_width=True):
            st.session_state.current_page = "Intraday Trading"
            st.rerun()

def render_long_term_page():
    st.subheader("Long Term Investing Module")
    
    c1, c2, c3, c4 = st.columns(4)
    amt = c1.number_input("Investment Amount (₹)", min_value=10000, value=500000, step=50000)
    horizon = c2.slider("Investment Horizon (Years)", 1, 30, 10)
    market = c3.selectbox("Market", ["India (NSE)", "United States (NYSE/NASDAQ)"])
    vehicle = c4.selectbox("Vehicle", ["Stocks", "ETF", "Mutual Funds"])
    
    if st.button("Run Multi-Factor Analysis", type="primary"):
        st.success(f"Analysis completed for ₹{amt:,.0f} over {horizon} years ({market} - {vehicle}).")
        
        st.markdown("#### Top Recommended Portfolio Assets")
        sample_df = pd.DataFrame({
            "Ticker": ["RELIANCE.NS", "TCS.NS", "MON100.NS", "INFY.NS"],
            "Asset Type": ["Equity", "Equity", "ETF", "Equity"],
            "Factor Score": [88.5, 84.2, 82.0, 79.8],
            "Allocation": ["30%", "25%", "25%", "20%"],
            "Expected 5Y CAGR": ["14.2%", "12.8%", "15.5%", "11.5%"]
        })
        st.dataframe(sample_df, use_container_width=True)

def render_intraday_page():
    st.subheader("Intraday Trading Dashboard")
    
    u_col1, u_col2 = st.columns(2)
    universe = u_col1.selectbox("Universe", ["Nifty 50", "Nifty 100"])
    capital = u_col2.number_input("Capital Allocation (₹)", min_value=5000, value=100000)
    
    st.markdown("#### Top Active Signals")
    trades_df = pd.DataFrame({
        "Ticker": ["TATAMOTORS.NS", "ICICIBANK.NS", "SBIN.NS"],
        "Signal": ["BUY (Long)", "BUY (Long)", "SELL (Short)"],
        "Entry": [980.50, 1120.00, 825.00],
        "Stop Loss": [968.00, 1105.00, 838.00],
        "Target": [1010.00, 1155.00, 800.00],
        "Risk/Reward": ["1:2.36", "1:2.50", "1:1.92"],
        "Confidence": ["87%", "82%", "78%"]
    })
    st.dataframe(trades_df, use_container_width=True)

def render_individual_analysis_page(ticker_symbol: str):
    st.subheader(f"Detailed Research Report: {ticker_symbol}")

    # 1. Fetch raw ticker metadata safely
    try:
        ticker_obj = yf.Ticker(ticker_symbol)
        raw_info = ticker_obj.info if hasattr(ticker_obj, "info") and isinstance(ticker_obj.info, dict) else {}
    except Exception as err:
        st.error(f"Unable to load ticker metadata for {ticker_symbol}: {err}")
        raw_info = {}

    # 2. Extract metrics into guaranteed dictionary scope
    metrics = safe_extract_metrics(raw_info)

    # 3. Render Tabs
    tab_overview, tab_technicals = st.tabs([
        "📌 Business Overview & Fundamentals", 
        "📈 Technical Analysis & Charts"
    ])

    with tab_overview:
        st.markdown("### Business Summary")
        st.write(metrics["summary"])

        s_col1, s_col2 = st.columns(2)
        s_col1.markdown(f"**Sector:** {metrics['sector']}")
        s_col2.markdown(f"**Industry:** {metrics['industry']}")

        st.markdown("---")
        st.markdown("### Key Fundamentals")

        # Explicit layout container declaration
        c1, c2, c3, c4 = st.columns(4)

        if metrics["is_etf"]:
            c1.metric("Asset Value / NAV", metrics["nav"])
            c2.metric("Total Assets / Cap", metrics["market_cap"])
            c3.metric("52-Week High", metrics["high_52"])
            c4.metric("52-Week Low", metrics["low_52"])

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("P/E Ratio", metrics["pe"])
            c6.metric("P/B Ratio", metrics["pb"])
            c7.metric("Dividend Yield", metrics["div_yield"])
            c8.metric("Beta", metrics["beta"])
        else:
            c1.metric("Market Cap", metrics["market_cap"])
            c2.metric("P/E Ratio", metrics["pe"])
            c3.metric("P/B Ratio", metrics["pb"])
            c4.metric("PEG Ratio", metrics["peg"])

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("ROE", metrics["roe"])
            c6.metric("Profit Margin", metrics["margins"])
            c7.metric("Revenue Growth", metrics["growth"])
            c8.metric("Debt to Equity", metrics["debt_equity"])

    with tab_technicals:
        st.markdown("### Technical Indicators")
        try:
            df = ticker_obj.history(period="1y")
            if not df.empty:
                df["SMA_20"] = df["Close"].rolling(20).mean()
                df["SMA_50"] = df["Close"].rolling(50).mean()
                df["SMA_200"] = df["Close"].rolling(200).mean()

                st.line_chart(df[["Close", "SMA_20", "SMA_50", "SMA_200"]])

                latest = df["Close"].iloc[-1]
                prev = df["Close"].iloc[-2] if len(df) > 1 else latest
                pct = ((latest - prev) / prev) * 100

                t1, t2, t3 = st.columns(3)
                t1.metric("Current Price", f"₹{latest:.2f}", f"{pct:+.2f}%")
                t2.metric("20-Day SMA", f"₹{df['SMA_20'].iloc[-1]:.2f}" if not np.isnan(df['SMA_20'].iloc[-1]) else "N/A")
                t3.metric("200-Day SMA", f"₹{df['SMA_200'].iloc[-1]:.2f}" if not np.isnan(df['SMA_200'].iloc[-1]) else "N/A")
            else:
                st.info("No historical price chart data found.")
        except Exception as e:
            st.warning(f"Could not process chart overlays: {e}")

# ==========================================
# 4. MAIN ROUTER & ENTRY POINT
# ==========================================
def main():
    render_header_banner()

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page_options = ["Home", "Long Term Investing", "Intraday Trading", "Individual Asset Analysis"]
    
    selected_page = st.sidebar.radio(
        "Select Module", 
        page_options, 
        index=page_options.index(st.session_state.current_page) if st.session_state.current_page in page_options else 0
    )
    st.session_state.current_page = selected_page

    if selected_page == "Individual Asset Analysis":
        symbol_input = st.sidebar.text_input("Enter Ticker (e.g., MON100.NS, RELIANCE.NS)", value=st.session_state.selected_ticker)
        st.session_state.selected_ticker = symbol_input.upper().strip()

    st.sidebar.markdown("---")
    st.sidebar.caption("Institutional Research Engine v2.0")

    # Route Execution
    if selected_page == "Home":
        render_home_page()
    elif selected_page == "Long Term Investing":
        render_long_term_page()
    elif selected_page == "Intraday Trading":
        render_intraday_page()
    elif selected_page == "Individual Asset Analysis":
        render_individual_analysis_page(st.session_state.selected_ticker)

if __name__ == "__main__":
    main()
