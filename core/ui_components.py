"""
Institutional UI Components & Visualizations
"""

import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any, List


def render_macro_header(header_data: Dict[str, Any]) -> None:
    """Renders real-time global index ticker widgets with sentiment indicators."""
    st.markdown("""
        <style>
            .macro-card {
                background-color: #1E222D;
                border: 1px solid #2A2E39;
                border-radius: 8px;
                padding: 12px;
                text-align: center;
            }
            .metric-val { font-size: 22px; font-weight: bold; color: #FFFFFF; }
            .bullish { color: #089981; font-weight: bold; }
            .bearish { color: #F23645; font-weight: bold; }
            .neutral { color: #D1D4DC; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        nifty = header_data['nifty']
        color_cls = "bullish" if nifty['sentiment'] == "Bullish" else ("bearish" if nifty['sentiment'] == "Bearish" else "neutral")
        st.markdown(f"""
            <div class="macro-card">
                <div style="color: #848E9C;">Nifty 50 ({nifty['sentiment']})</div>
                <div class="metric-val">{nifty['price'] if not np.isnan(nifty['price']) else 'N/A'}</div>
                <div class="{color_cls}">{nifty['change']} ({nifty['pct_change']}%)</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        sp = header_data['sp500']
        color_cls = "bullish" if sp['sentiment'] == "Bullish" else ("bearish" if sp['sentiment'] == "Bearish" else "neutral")
        st.markdown(f"""
            <div class="macro-card">
                <div style="color: #848E9C;">S&P 500 ({sp['sentiment']})</div>
                <div class="metric-val">{sp['price'] if not np.isnan(sp['price']) else 'N/A'}</div>
                <div class="{color_cls}">{sp['change']} ({sp['pct_change']}%)</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        nasdaq = header_data['nasdaq']
        color_cls = "bullish" if nasdaq['sentiment'] == "Bullish" else ("bearish" if nasdaq['sentiment'] == "Bearish" else "neutral")
        st.markdown(f"""
            <div class="macro-card">
                <div style="color: #848E9C;">Nasdaq ({nasdaq['sentiment']})</div>
                <div class="metric-val">{nasdaq['price'] if not np.isnan(nasdaq['price']) else 'N/A'}</div>
                <div class="{color_cls}">{nasdaq['change']} ({nasdaq['pct_change']}%)</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
            <div class="macro-card">
                <div style="color: #848E9C;">Market Status</div>
                <div class="metric-val" style="font-size: 16px;">Date: {header_data['market_date']}</div>
                <div style="color: #2962FF; font-weight: bold;">{header_data['digital_clock']}</div>
            </div>
        """, unsafe_allow_html=True)


def plot_monte_carlo_paths(simulation_results: Dict[str, Any]) -> go.Figure:
    """Generates Plotly visual paths for Monte Carlo iterations."""
    fig = go.Figure()
    paths = simulation_results["simulation_paths"]
    
    for i in range(min(50, len(paths))):
        fig.add_trace(go.Scatter(y=paths[i], mode='lines', line=dict(width=1), opacity=0.15, showlegend=False))
        
    fig.update_layout(
        title="Monte Carlo Portfolio Simulation (50,000 Runs Sampled)",
        xaxis_title="Trading Days",
        yaxis_title="Portfolio Value ($ / ₹)",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig
