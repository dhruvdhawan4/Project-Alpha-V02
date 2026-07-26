import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Import our quant engine
from institutional_quant_engine import run_institutional_pipeline

st.set_page_config(
    page_title="Institutional Quant Portfolio Engine",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Institutional Portfolio Research & Optimization Engine")
st.markdown("""
This application runs **GARCH(1,1) volatility forecasting**, **Ledoit-Wolf shrinkage correlation**, 
**Black-Litterman views blending**, and **Friction-Aware SLSQP optimization**.
""")

# =====================================================================
# SIDEBAR CONTROLS
# =====================================================================
st.sidebar.header("⚙️ Market & Risk Settings")

rf_rate = st.sidebar.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=15.0, value=4.5, step=0.25) / 100
gamma = st.sidebar.slider("Risk Aversion (Gamma)", min_value=0.5, max_value=10.0, value=2.5, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("🛡️ Operational Constraints")

max_stock_weight = st.sidebar.slider("Max Stock Weight Cap", min_value=0.05, max_value=0.50, value=0.25, step=0.05)
max_turnover = st.sidebar.slider("Turnover Constraint Cap", min_value=0.05, max_value=1.00, value=0.20, step=0.05)
tx_cost_bps = st.sidebar.number_input("Transaction Cost (Bps)", min_value=0, max_value=100, value=15, step=5) / 10000

# =====================================================================
# DATA & ASSET CONFIGURATION
# =====================================================================
@st.cache_data
def generate_sample_prices():
    np.random.seed(42)
    assets = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
    dates = pd.date_range(start="2025-01-01", periods=252, freq="B")
    
    returns_matrix = np.random.multivariate_normal(
        mean=[0.0005, 0.0004, 0.0003, 0.00045, 0.00035],
        cov=[
            [0.0004, 0.0001, 0.00015, 0.0001, 0.00012],
            [0.0001, 0.0003, 0.00008, 0.0002, 0.00009],
            [0.00015, 0.00008, 0.00025, 0.00009, 0.00018],
            [0.0001, 0.0002, 0.00009, 0.00035, 0.0001],
            [0.00012, 0.00009, 0.00018, 0.0001, 0.00028]
        ],
        size=252
    )
    df = pd.DataFrame(100 * np.exp(np.cumsum(returns_matrix, axis=0)), index=dates, columns=assets)
    return df

prices_df = generate_sample_prices()
assets = prices_df.columns.tolist()

st.subheader("1. Asset Universe & Base Weights")
col1, col2 = st.subplots([2, 1])

with col1:
    st.line_chart(prices_df)

with col2:
    st.markdown("**Benchmark Market Cap Weights**")
    mkt_w1 = st.number_input("RELIANCE Weight", 0.0, 1.0, 0.30)
    mkt_w2 = st.number_input("TCS Weight", 0.0, 1.0, 0.20)
    mkt_w3 = st.number_input("HDFCBANK Weight", 0.0, 1.0, 0.25)
    mkt_w4 = st.number_input("INFY Weight", 0.0, 1.0, 0.15)
    mkt_w5 = st.number_input("ICICIBANK Weight", 0.0, 1.0, 0.10)
    
    mkt_weights = np.array([mkt_w1, mkt_w2, mkt_w3, mkt_w4, mkt_w5])
    mkt_weights /= np.sum(mkt_weights)  # Normalize

prev_weights = np.array([0.20, 0.20, 0.20, 0.20, 0.20])

# =====================================================================
# BLACK-LITTERMAN TACTICAL VIEWS
# =====================================================================
st.subheader("2. Black-Litterman Tactical Views")
st.caption("Add absolute or relative views to override market equilibrium returns.")

use_views = st.checkbox("Enable Tactical Views", value=True)

if use_views:
    vcol1, vcol2 = st.subplots(2)
    with vcol1:
        view1_return = st.slider("View 1: RELIANCE Target Expected Return (%)", 0.0, 30.0, 14.0, 0.5) / 100
    with vcol2:
        view2_spread = st.slider("View 2: INFY Outperforms TCS by (%)", -10.0, 10.0, 3.0, 0.5) / 100

    P_views = np.array([
        [1, 0, 0, 0, 0],
        [0, -1, 0, 1, 0]
    ])
    Q_views = np.array([view1_return, view2_spread])
else:
    P_views = None
    Q_views = None

# Sector Constraints
sector_mapping = {
    "Technology": [1, 3],  # TCS, INFY
    "Financials": [2, 4]   # HDFCBANK, ICICIBANK
}
sector_caps = {
    "Technology": 0.35,
    "Financials": 0.40
}

# =====================================================================
# RUN PIPELINE & RENDER RESULTS
# =====================================================================
if st.button("🚀 Execute Optimization Pipeline", type="primary", use_container_width=True):
    with st.spinner("Calculating GARCH Volatilities, BL Returns, and Solving SLSQP..."):
        results = run_institutional_pipeline(
            prices_df=prices_df,
            market_cap_weights=mkt_weights,
            previous_weights=prev_weights,
            tactical_views_P=P_views,
            tactical_views_Q=Q_views,
            sector_mapping=sector_mapping,
            sector_caps=sector_caps,
            risk_free_rate=rf_rate
        )

    mvo = results["institutional_mvo_portfolio"]
    rp = results["risk_parity_portfolio"]

    st.success("Optimization Complete!")

    # Top KPI Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("MVO Expected Return (Net)", f"{mvo['net_expected_return']*100:.2f}%")
    kpi2.metric("MVO Volatility", f"{mvo['volatility']*100:.2f}%")
    kpi3.metric("MVO Sharpe Ratio", f"{mvo['sharpe_ratio']:.2f}")
    kpi4.metric("Executed Turnover", f"{mvo['turnover']*100:.2f}%")

    st.markdown("---")

    # Allocation Breakdown Plot
    st.subheader("3. Allocation Strategy Comparison")
    
    df_alloc = pd.DataFrame({
        "Asset": assets,
        "Benchmark (Mkt Cap)": mkt_weights,
        "Friction-Aware MVO": mvo["weights"],
        "Risk Parity (ERC)": rp["weights"]
    })

    fig_alloc = px.bar(
        df_alloc, 
        x="Asset", 
        y=["Benchmark (Mkt Cap)", "Friction-Aware MVO", "Risk Parity (ERC)"],
        barmode="group",
        title="Portfolio Weights Allocation Across Frameworks"
    )
    st.plotly_chart(fig_alloc, use_container_width=True)

    col_a, col_b = st.subplots(2)

    with col_a:
        st.markdown("### GARCH(1,1) Volatility Forecasts")
        garch_df = pd.DataFrame({
            "Asset": list(results["garch_volatilities"].keys()),
            "Annualized Volatility": [v * 100 for v in results["garch_volatilities"].values()]
        })
        fig_vol = px.bar(garch_df, x="Asset", y="Annualized Volatility", color="Asset", text_auto=".2f")
        st.plotly_chart(fig_vol, use_container_width=True)

    with col_b:
        st.markdown("### Black-Litterman Expected Returns")
        bl_df = pd.DataFrame({
            "Asset": list(results["black_litterman_returns"].keys()),
            "BL Expected Return (%)": [v * 100 for v in results["black_litterman_returns"].values()]
        })
        fig_bl = px.bar(bl_df, x="Asset", y="BL Expected Return (%)", color="Asset", text_auto=".2f")
        st.plotly_chart(fig_bl, use_container_width=True)
