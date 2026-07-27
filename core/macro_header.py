"""
Macro Header Data Engine
Handles global index retrieval with primary-to-proxy ticker failover, timezone calculation, and resilient market closure fallback.
"""

from datetime import datetime
import pytz
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Any, Tuple


def get_market_status(tz_name: str, open_hour: float, close_hour: float) -> Tuple[str, str]:
    """Calculates local market session status (Open/Closed)."""
    tz = pytz.timezone(tz_name)
    now = datetime.now(tz)
    
    if now.weekday() >= 5:
        return "Closed", now.strftime("%Y-%m-%d")
    
    current_decimal_hour = now.hour + (now.minute / 60.0)
    is_open = open_hour <= current_decimal_hour < close_hour
    status = "Open" if is_open else "Closed"
    return status, now.strftime("%Y-%m-%d")


def fetch_index_data(primary_ticker: str, fallback_ticker: str = None) -> Dict[str, Any]:
    """
    Fetches ticker price with explicit proxy failover.
    When market is closed or live feed is unavailable, dynamically evaluates based on the last closed market session data.
    """
    tickers_to_try = [primary_ticker]
    if fallback_ticker:
        tickers_to_try.append(fallback_ticker)
        
    for symbol in tickers_to_try:
        try:
            t = yf.Ticker(symbol)
            df = t.history(period="10d", interval="1d")
            
            if df is not None and not df.empty:
                df = df.dropna(subset=['Close'])
                
            if df is None or df.empty or len(df) < 1:
                continue
                
            if len(df) >= 2:
                last_price = float(df['Close'].iloc[-1])
                prev_price = float(df['Close'].iloc[-2])
            else:
                last_price = float(df['Close'].iloc[-1])
                prev_price = float(df['Open'].iloc[-1])
                
            if np.isnan(last_price) or last_price <= 0:
                continue
                
            change = last_price - prev_price
            pct_change = (change / prev_price) * 100.0 if prev_price > 0 else 0.0
            
            if pct_change >= 0.25:
                sentiment = "Bullish"
            elif pct_change <= -0.25:
                sentiment = "Bearish"
            else:
                sentiment = "Neutral"
                
            return {
                "price": round(last_price, 2),
                "change": round(change, 2),
                "pct_change": round(pct_change, 2),
                "sentiment": sentiment
            }
        except Exception:
            continue
            
    return {"price": np.nan, "change": np.nan, "pct_change": np.nan, "sentiment": "Neutral"}


def get_macro_header_data() -> Dict[str, Any]:
    """Fetches real-time dashboard macro index data with off-hours resilience."""
    nifty = fetch_index_data("^NSEI", "NIFTYBEES.NS")
    sp500 = fetch_index_data("^GSPC", "SPY")
    nasdaq = fetch_index_data("^IXIC", "QQQ")
    
    ist_status, ist_date = get_market_status("Asia/Kolkata", 9.25, 15.50)
    us_status, _ = get_market_status("America/New_York", 9.50, 16.00)
    
    tz_ist = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(tz_ist)
    
    return {
        "nifty": nifty,
        "sp500": sp500,
        "nasdaq": nasdaq,
        "market_date": ist_date,
        "digital_clock": now_ist.strftime("%H:%M:%S IST"),
        "india_open": ist_status == "Open",
        "us_open": us_status == "Open"
    }
