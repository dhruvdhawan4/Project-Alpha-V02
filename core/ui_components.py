"""
Institutional UI Components & Visualizations
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, List


def is_valid_number(val: Any) -> bool:
    """Safely validates if a value is a non-null, non-NaN numerical scalar."""
    if val is None:
        return False
    try:
        return not np.isnan(float(val))
    except (ValueError, TypeError):
        return False


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
                margin-bottom: 10px;
            }
            .metric-val { font-size: 22px; font-weight: bold; color: #FFFFFF; }
            .bullish { color: #089981; font-weight: bold; }
            .bearish { color: #F23645; font-weight: bold; }
            .neutral { color: #D1D4DC; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        nifty = header_data.get('nifty', {})
        nifty_price = nifty.get('price')
        price_disp = f"{nifty_price:,.2f}" if is_valid_number(nifty_price) else "N/A"
        chg_val = nifty.get('change', 0.0)
        pct_val = nifty.get('pct_change', 0.0)
        chg_disp = f"{chg_val:+.2f} ({pct_val:+.2f}%)" if is_valid_number(chg_val) else "N/A"
        
        sentiment = nifty.get('sentiment', 'Neutral')
        color_cls = "bullish" if sentiment == "Bullish" else ("bearish" if sentiment == "Bearish" else "neutral")
        
        st.markdown(f"""
            <div class="macro-card">
                <div style="color: #848E9C;">Nifty 50 ({sentiment})</div>
                <div class="metric-val">{price_disp}</div>
                <div class="{color_cls}">{chg_disp}</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        sp = header_data.get('sp500', {})
        sp_price = sp.get('price')
        sp_price_disp = f"{sp_price:,.2f}" if is_valid_number(sp_price) else "N/A"
        sp_chg_val = sp.get('change', 0.0)
        sp_pct_val = sp.get('pct_change', 0.0)
        sp_chg_disp = f"{sp_chg_val:+.2f} ({sp_pct_val:+.2f}%)" if is_valid_number(sp_chg_val) else "N/A"
        
        sp_sentiment = sp.get('sentiment', 'Neutral')
        sp_color_cls = "bullish" if sp_sentiment == "Bullish" else ("bearish" if sp_sentiment == "Bearish" else "neutral")
        
        st.markdown(f"""
            <div class="macro-card">
                <div style="color: #848E9C;">S&P 500 ({sp_sentiment})</div>
                <div class="metric-val">{sp_price_disp}</div>
                <div class="{sp_color_cls}">{sp_chg_disp}</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        nasdaq = header_data.get('nasdaq', {})
        nasdaq_price = nasdaq.get('price')
        nasdaq_price_disp = f"{nasdaq_price:,.2f}" if is_valid_number(nasdaq_price) else "N/A"
        nasdaq_chg_val = nasdaq.get('change', 0.0)
        nasdaq_pct_val = nasdaq.get('pct_change', 0.0)
        nasdaq_chg_disp = f"{nasdaq_chg_val:+.2f} ({nasdaq_pct_val:+.2f}%)" if is_valid_number(nasdaq_chg_val) else "N/A"
        
        nasdaq_sentiment = nasdaq.get('sentiment', 'Neutral')
        nasdaq_color_cls = "bullish" if nasdaq_sentiment == "Bullish" else ("bearish" if nasdaq_sentiment == "Bearish" else "neutral")
        
        st.markdown(f"""
            <div class="macro-card">
                <div style="color: #848E9C;">Nasdaq ({nasdaq_sentiment})</div>
                <div class="metric-val">{nasdaq_price_disp}</div>
                <div class="{nasdaq_color_cls}">{nasdaq_chg_disp}</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
            <div class="macro-card">
                <div style="color: #848E9C;">Market Status</div>
                <div class="metric-val" style="font-size: 16px;">Date: {header_data.get('market_date', 'N/A')}</div>
                <div style="color: #2962FF; font-weight: bold;">{header_data.get('digital_clock', 'N/A')}</div>
            </div>
        """, unsafe_allow_html=True)


def plot_monte_carlo_paths(simulation_results: Dict[str, Any]) -> go.Figure:
    """Generates Plotly visual paths for Monte Carlo iterations."""
    fig = go.Figure()
    paths = simulation_results.get("simulation_paths", [])
    
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
