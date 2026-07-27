import streamlit as st
import numpy as np
import yfinance as yf
from typing import Dict, Any

def safe_extract_metrics(info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safely parses ticker info and guarantees default fallbacks for all metric keys.
    Prevents KeyError and NameError regardless of whether the ticker is a Stock, ETF, or MF.
    """
    def fmt(val: Any, is_curr: bool = False, is_pct: bool = False, decimals: int = 2) -> str:
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


def render_individual_analysis_page(ticker_symbol: str):
    """
    Renders institutional detail report safely for any stock, ETF, or Mutual Fund.
    """
    st.header(f"Research Report: {ticker_symbol}")

    # 1. Fetch raw info safely
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
        st.subheader("Business Summary")
        st.write(metrics["summary"])

        s_col1, s_col2 = st.columns(2)
        s_col1.markdown(f"**Sector:** {metrics['sector']}")
        s_col2.markdown(f"**Industry:** {metrics['industry']}")

        st.markdown("---")
        st.subheader("Key Fundamentals")

        # Guarantee column container initialization in outer scope
        c1, c2, c3, c4 = st.columns(4)

        if metrics["is_etf"]:
            # Custom layout for ETFs & Mutual Funds (e.g. MON100)
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
            # Standard Stock / Equity layout
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
        st.subheader("Technical Trends")
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
                st.info("No historical price charts available for this symbol.")
        except Exception as e:
            st.warning(f"Could not calculate technical overlays: {e}")
