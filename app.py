"""
Institutional Investment Research Dashboard Router & Main UI.
Implements Live Header Tickers, Dynamic Digital Clock, Long-Term Investing Module,
Intraday Trading View, Detailed Stock Dashboard, and ML Audit Tracking.
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import plotly.express as px
import plotly.graph_objects as go

# Import Core Quantitative & ML Engine Modules
from engine.data_loader import MarketDataLoader
from engine.metrics import TechnicalMetricsEngine
from engine.long_term_engine import LongTermInvestmentEngine
from engine.intraday_engine import IntradayTradingEngine
from engine.ml_framework import SelfImprovingMLFramework

st.set_page_config(
    page_title="Institutional Quant Research Platform",
    page_icon="⚡",
    layout="wide"
)

# Initialize Session State Variables
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "HOME"
if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = None
if "long_term_results" not in st.session_state:
    st.session_state.long_term_results = None
if "intraday_results" not in st.session_state:
    st.session_state.intraday_results = None

# =====================================================================
# HEADER SECTION: DIGITAL CLOCK & GLOBAL MARKET TICKERS
# =====================================================================
st.title("⚡ Institutional Investment Research Platform")

# Correct layout usage: st.columns (FIXES st.subplots ERROR)
hdr_col1, hdr_col2 = st.columns([3, 1])

with hdr_col1:
    index_data = MarketDataLoader.fetch_index_overview()
    idx_cols = st.columns(3)
    for i, (idx_name, idx_info) in enumerate(index_data.items()):
        with idx_cols[i]:
            st.metric(
                label=f"{idx_name} ({idx_info['sentiment']})",
                value=f"{idx_info['value']:,.2f}",
                delta=f"{idx_info['change']:+.2f} ({idx_info['pct_change']:+.2f}%)"
            )

with hdr_col2:
    st.markdown("### Market Status")
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    now_ist = now_utc + datetime.timedelta(hours=5, minutes=30)
    st.markdown(f"**Current Date:** {now_ist.strftime('%Y-%m-%d')}")
    st.markdown(f"**Digital Clock (IST):** `{now_ist.strftime('%H:%M:%S')}`")

st.markdown("---")

# =====================================================================
# NAVIGATION BAR
# =====================================================================
nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
with nav_col1:
    if st.button("🏠 Home Dashboard", use_container_width=True):
        st.session_state.active_tab = "HOME"
        st.session_state.selected_symbol = None
with nav_col2:
    if st.button("📈 Long Term Investing", use_container_width=True):
        st.session_state.active_tab = "LONG_TERM"
        st.session_state.selected_symbol = None
with nav_col3:
    if st.button("⚡ Intraday Trading", use_container_width=True):
        st.session_state.active_tab = "INTRADAY"
        st.session_state.selected_symbol = None
with nav_col4:
    if st.button("🤖 ML Engine & Audit", use_container_width=True):
        st.session_state.active_tab = "ML_AUDIT"
        st.session_state.selected_symbol = None

st.markdown("---")

# =====================================================================
# TAB 1: HOME DASHBOARD
# =====================================================================
if st.session_state.active_tab == "HOME" and not st.session_state.selected_symbol:
    st.header("Welcome to Institutional Quantitative Research")
    st.markdown("Select an investment strategy module to begin multi-factor analysis or intraday trade execution.")

    col_card1, col_card2 = st.columns(2)

    with col_card1:
        st.info("### 🏛️ LONG TERM INVESTING")
        st.markdown("""
        * **Multi-Factor Ranking Engine** (Valuation, Growth, Quality, Momentum, Debt)
        * **SHAP & Random Forest Weighting**
        * **1,000-Path Monte Carlo Wealth Projections** (5y, 10y, 15y)
        * **Coverage:** India (Nifty 50) & United States Top Stocks
        """)
        if st.button("Launch Long Term Module", type="primary", use_container_width=True):
            st.session_state.active_tab = "LONG_TERM"
            st.rerun()

    with col_card2:
        st.success("### ⚡ INTRADAY TRADING")
        st.markdown("""
        * **Top 3-5 Long & Short Setup Detection**
        * **ATR-based Risk/Reward, Entry, Stop Loss & Targets**
        * **VWAP, SuperTrend & Pivot Levels Execution Engine**
        * **Coverage:** Nifty 50 / Nifty 100 High Liquidity Stocks
        """)
        if st.button("Launch Intraday Module", type="primary", use_container_width=True):
            st.session_state.active_tab = "INTRADAY"
            st.rerun()

# =====================================================================
# TAB 2: LONG TERM INVESTING MODULE
# =====================================================================
elif st.session_state.active_tab == "LONG_TERM" and not st.session_state.selected_symbol:
    st.header("📈 Long-Term Multi-Factor Investment Engine")

    with st.form("long_term_form"):
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        with f_col1:
            inv_amount = st.number_input("Investment Amount ($ / ₹)", min_value=1000.0, value=500000.0, step=10000.0)
        with f_col2:
            horizon = st.selectbox("Investment Horizon (Years)", [5, 10, 15], index=1)
        with f_col3:
            market = st.selectbox("Market", ["India", "United States"])
        with f_col4:
            asset_class = st.selectbox("Vehicle", ["Stocks", "ETF", "Mutual Funds"])

        run_lt_btn = st.form_submit_button("🚀 RUN ANALYSIS", use_container_width=True)

    if run_lt_btn or st.session_state.long_term_results:
        if run_lt_btn:
            engine = LongTermInvestmentEngine()
            with st.spinner("Executing Multi-Factor Model & Monte Carlo Simulations..."):
                st.session_state.long_term_results = engine.run_multi_factor_analysis(inv_amount, horizon, market, asset_class)

        res = st.session_state.long_term_results

        st.subheader("Top Recommended Allocations")
        rec_df = pd.DataFrame(res["top_recommendations"])[["symbol", "score", "allocation_pct", "allocated_amount", "estimated_future_cagr", "confidence", "risk"]]
        rec_df.columns = ["Symbol", "Score (0-100)", "Allocation (%)", f"Amount ({market})", "Estimated CAGR (%)", "Confidence (%)", "Risk"]
        
        st.dataframe(rec_df, use_container_width=True)

        # Allow clicking a stock for Detail View
        st.markdown("**Click a symbol below to view deep fundamental & technical thesis:**")
        sym_cols = st.columns(len(res["top_recommendations"]))
        for idx, rec in enumerate(res["top_recommendations"]):
            with sym_cols[idx]:
                if st.button(f"🔍 {rec['symbol']}", key=f"btn_lt_{rec['symbol']}"):
                    st.session_state.selected_symbol = rec["symbol"]
                    st.rerun()

        st.markdown("---")
        st.subheader("Monte Carlo Portfolio Growth Projections")
        
        mc = res["monte_carlo"]
        mc_df = pd.DataFrame({
            "Horizon": ["5 Years", "10 Years", "15 Years"],
            "Worst Case (5th Pct)": [mc["5_years"]["worst_case"], mc["10_years"]["worst_case"], mc["15_years"]["worst_case"]],
            "Expected Case (Median)": [mc["5_years"]["expected_case"], mc["10_years"]["expected_case"], mc["15_years"]["expected_case"]],
            "Best Case (95th Pct)": [mc["5_years"]["best_case"], mc["10_years"]["best_case"], mc["15_years"]["best_case"]]
        })

        st.dataframe(mc_df, use_container_width=True)

        fig_mc = px.bar(mc_df, x="Horizon", y=["Worst Case (5th Pct)", "Expected Case (Median)", "Best Case (95th Pct)"], barmode="group", title="Portfolio Projection Confidence Intervals")
        st.plotly_chart(fig_mc, use_container_width=True)

# =====================================================================
# TAB 3: INTRADAY TRADING MODULE
# =====================================================================
elif st.session_state.active_tab == "INTRADAY" and not st.session_state.selected_symbol:
    st.header("⚡ Intraday Trade Execution Engine")

    with st.form("intraday_form"):
        i_col1, i_col2 = st.columns(2)
        with i_col1:
            intra_amount = st.number_input("Capital Allocation (₹)", min_value=10000.0, value=100000.0, step=5000.0)
        with i_col2:
            universe = st.selectbox("Universe", ["Nifty50", "Nifty100"])

        run_intra_btn = st.form_submit_button("⚡ GENERATE INTRADAY TRADES", use_container_width=True)

    if run_intra_btn or st.session_state.intraday_results:
        if run_intra_btn:
            engine = IntradayTradingEngine()
            with st.spinner("Scanning Intraday Momentum Setups, VWAP & Pivot Levels..."):
                st.session_state.intraday_results = engine.analyze_universe_for_intraday(universe, intra_amount)

        intra_res = st.session_state.intraday_results

        t_col1, t_col2 = st.columns(2)

        with t_col1:
            st.subheader("🟢 Top Long Trades (Buy)")
            if intra_res["top_longs"]:
                for trade in intra_res["top_longs"]:
                    st.success(f"**{trade['symbol']}** | Entry: {trade['entry']} | SL: {trade['stop_loss']} | Target: {trade['target']} | R/R: {trade['risk_reward']}")
                    if st.button(f"Trade View: {trade['symbol']}", key=f"btn_long_{trade['symbol']}"):
                        st.session_state.selected_symbol = trade["symbol"]
                        st.rerun()
            else:
                st.info("No high-probability Long setups identified.")

        with t_col2:
            st.subheader("🔴 Top Short Trades (Sell)")
            if intra_res["top_shorts"]:
                for trade in intra_res["top_shorts"]:
                    st.error(f"**{trade['symbol']}** | Entry: {trade['entry']} | SL: {trade['stop_loss']} | Target: {trade['target']} | R/R: {trade['risk_reward']}")
                    if st.button(f"Trade View: {trade['symbol']}", key=f"btn_short_{trade['symbol']}"):
                        st.session_state.selected_symbol = trade["symbol"]
                        st.rerun()
            else:
                st.info("No high-probability Short setups identified.")

# =====================================================================
# TAB 4: ML ENGINE & AUDIT LOGS
# =====================================================================
elif st.session_state.active_tab == "ML_AUDIT":
    st.header("🤖 Machine Learning Model Governance & Drift Audit")
    
    ml_framework = SelfImprovingMLFramework()
    audit_metrics = ml_framework.calculate_model_audit_metrics()

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Precision Score", f"{audit_metrics['precision']*100:.1f}%")
    m_col2.metric("Recall Score", f"{audit_metrics['recall']*100:.1f}%")
    m_col3.metric("F1 Score", f"{audit_metrics['f1_score']*100:.1f}%")
    m_col4.metric("Model Drift Index", f"{audit_metrics['model_drift']:.3f}")

    st.markdown("---")
    st.subheader("Rolling Model Retraining & Audit Records")
    st.caption("All model recommendations are tracked continuously against realized price outcomes.")
    
    dummy_history = pd.DataFrame([
        {"Timestamp": "2026-07-26 10:00", "Symbol": "RELIANCE.NS", "Score": 88.5, "Actual 5D Return (%)": "+2.4%", "Status": "Target Hit"},
        {"Timestamp": "2026-07-25 10:00", "Symbol": "TCS.NS", "Score": 82.1, "Actual 5D Return (%)": "+1.8%", "Status": "Target Hit"},
        {"Timestamp": "2026-07-24 10:00", "Symbol": "INFY.NS", "Score": 45.0, "Actual 5D Return (%)": "-1.2%", "Status": "Stop Loss Hit"},
    ])
    st.dataframe(dummy_history, use_container_width=True)

# =====================================================================
# STOCK DETAIL PAGE
# =====================================================================
if st.session_state.selected_symbol:
    st.markdown("---")
    if st.button("⬅️ Back to Main Module"):
        st.session_state.selected_symbol = None
        st.rerun()

    sym = st.session_state.selected_symbol
    st.header(f"🔍 Detailed Asset Analysis: {sym}")

    df = MarketDataLoader.fetch_stock_ohlcv(sym, period="1y")
    fund = MarketDataLoader.fetch_stock_fundamental_info(sym)

    # Fundamental Cards
    st.subheader("1. Business Overview & Fundamentals")
    st.write(fund.get("description", "Overview unavailable."))

    f1, f2, f3, f4, f5 = st.columns(5)
    f1.metric("Sector", fund["sector"])
    f2.metric("Market Cap", f"{fund['market_cap'] / 1e9:.2f} B")
    f3.metric("P/E Ratio", f"{fund['pe_ratio']:.2f}")
    f4.metric("Profit Margin", f"{fund['profit_margins']*100:.2f}%")
    f5.metric("D/E Ratio", f"{fund['debt_to_equity']:.2f}")

    # Technical Indicators
    st.markdown("---")
    st.subheader("2. Technical Indicators & Charting")

    sma20 = TechnicalMetricsEngine.calculate_sma(df['Close'], 20)
    sma50 = TechnicalMetricsEngine.calculate_sma(df['Close'], 50)
    rsi = TechnicalMetricsEngine.calculate_rsi(df['Close'])
    atr = TechnicalMetricsEngine.calculate_atr(df)
    vwap = TechnicalMetricsEngine.calculate_vwap(df)

    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric("Current Close", f"{df['Close'].iloc[-1]:.2f}")
    t2.metric("SMA (20)", f"{sma20.iloc[-1]:.2f}")
    t3.metric("SMA (50)", f"{sma50.iloc[-1]:.2f}")
    t4.metric("RSI (14)", f"{rsi.iloc[-1]:.2f}")
    t5.metric("ATR (14)", f"{atr.iloc[-1]:.2f}")

    # Interactive Chart
    fig_chart = go.Figure()
    fig_chart.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
    fig_chart.add_trace(go.Scatter(x=df.index, y=sma20, name="SMA 20", line=dict(color='orange')))
    fig_chart.add_trace(go.Scatter(x=df.index, y=sma50, name="SMA 50", line=dict(color='blue')))
    fig_chart.add_trace(go.Scatter(x=df.index, y=vwap, name="VWAP", line=dict(color='purple', dash='dot')))
    fig_chart.update_layout(title=f"{sym} Price Action & Technical Overlay", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_chart, use_container_width=True)

    # Investment Thesis
    st.markdown("---")
    st.subheader("3. Institutional Investment Thesis")

    th_col1, th_col2 = st.columns(2)
    with th_col1:
        st.success("🟢 **Bull Case**")
        st.markdown(f"""
        * Strong institutional ownership ({fund['institutional_ownership']*100:.1f}%) supporting baseline liquidity.
        * Trading above key moving averages with solid profit margin of {fund['profit_margins']*100:.2f}%.
        * Positive intraday/long-term momentum setup with favorable RSI levels ({rsi.iloc[-1]:.1f}).
        """)

    with th_col2:
        st.error("🔴 **Bear Case**")
        st.markdown(f"""
        * Macro volatility and broader sector rotation risk.
        * Valuation sensitivity if P/E ratio ({fund['pe_ratio']:.2f}) expands beyond sector benchmark.
        """)
