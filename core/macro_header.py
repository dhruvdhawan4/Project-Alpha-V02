"""
Macro Header Engine
Handles multi-market real-time index data, timezones, and trading session status.
"""

from datetime import datetime
import pytz
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Any, Tuple


def get_market_status(tz_name: str, open_hour: float, close_hour: float) -> Tuple[str, str]:
    """Calculates if a given market is currently open or closed based on local time."""
    tz = pytz.timezone(tz_name)
    now = datetime.now(tz)
    
    # Check if weekend
    if now.weekday() >= 5:
        return "Closed", now.strftime("%Y-%m-%d")
    
    current_decimal_hour = now.hour + (now.minute / 60.0)
    is_open = open_hour <= current_decimal_hour < close_hour
    status = "Open" if is_open else "Closed"
    return status, now.strftime("%Y-%m-%d")


def fetch_index_data(ticker_symbol: str, fallback_symbol: str = None) -> Dict[str, Any]:
    """
    Fetches market index data cleanly by using historical price series 
    to bypass yfinance metadata key omission/NaN bugs.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="5d", interval="1d")
        
        if df.empty or len(df) < 2:
            if fallback_symbol:
                ticker = yf.Ticker(fallback_symbol)
                df = ticker.history(period="5d", interval="1d")
        
        if df.empty or len(df) < 2:
            return {"price": np.nan, "change": np.nan, "pct_change": np.nan, "sentiment": "Neutral"}
        
        last_price = float(df['Close'].iloc[-1])
        prev_price = float(df['Close'].iloc[-2])
        change = last_price - prev_price
        pct_change = (change / prev_price) * 100.0
        
        if pct_change >= 0.35:
            sentiment = "Bullish"
        elif pct_change <= -0.35:
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
        return {"price": np.nan, "change": np.nan, "pct_change": np.nan, "sentiment": "Neutral"}


def get_macro_header_data() -> Dict[str, Any]:
    """Retrieves macro dashboard metrics for Nifty 50, S&P 500, and Nasdaq."""
    nifty = fetch_index_data("^NSEI", "^BSESN")
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
