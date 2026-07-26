"""
Production Market Data Loader.
Retrieves real-time market metrics for Global Benchmarks and Stock Universes using yfinance.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Any

# Universe Definition Lists
NIFTY_50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "LTIM.NS",
    "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS", "HCLTECH.NS", "ASIANPAINT.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "BAJFINANCE.NS", "ULTRACEMCO.NS"
]

US_TOP_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V"
]

class MarketDataLoader:

    @staticmethod
    def get_index_ticker_symbol(index_name: str) -> str:
        mapping = {
            "Nifty 50": "^NSEI",
            "S&P 500": "^GSPC",
            "Nasdaq": "^IXIC"
        }
        return mapping.get(index_name, "^NSEI")

    @staticmethod
    def fetch_index_overview() -> Dict[str, Dict[str, Any]]:
        indices = {
            "Nifty 50": "^NSEI",
            "S&P 500": "^GSPC",
            "Nasdaq": "^IXIC"
        }
        results = {}
        for name, ticker_str in indices.items():
            try:
                t = yf.Ticker(ticker_str)
                hist = t.history(period="5d")
                if len(hist) >= 2:
                    curr = hist['Close'].iloc[-1]
                    prev = hist['Close'].iloc[-2]
                    change = curr - prev
                    pct_change = (change / prev) * 100.0
                    sentiment = "Bullish" if pct_change > 0.1 else ("Bearish" if pct_change < -0.1 else "Neutral")
                    
                    results[name] = {
                        "value": round(float(curr), 2),
                        "change": round(float(change), 2),
                        "pct_change": round(float(pct_change), 2),
                        "sentiment": sentiment,
                        "status": "Market Open"
                    }
                else:
                    results[name] = {"value": 0.0, "change": 0.0, "pct_change": 0.0, "sentiment": "Neutral", "status": "Closed"}
            except Exception:
                results[name] = {"value": 0.0, "change": 0.0, "pct_change": 0.0, "sentiment": "Neutral", "status": "Closed"}
        return results

    @staticmethod
    def fetch_stock_ohlcv(symbol: str, period: str = "1y") -> pd.DataFrame:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            if df.empty:
                raise ValueError(f"No OHLCV data found for symbol: {symbol}")
            return df
        except Exception as e:
            raise RuntimeError(f"Error downloading data for {symbol}: {str(e)}")

    @staticmethod
    def fetch_stock_fundamental_info(symbol: str) -> Dict[str, Any]:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", 0.0),
                "forward_pe": info.get("forwardPE", 0.0),
                "pb_ratio": info.get("priceToBook", 0.0),
                "peg_ratio": info.get("pegRatio", 0.0),
                "debt_to_equity": info.get("debtToEquity", 0.0),
                "profit_margins": info.get("profitMargins", 0.0),
                "operating_margins": info.get("operatingMargins", 0.0),
                "roe": info.get("returnOnEquity", 0.0),
                "dividend_yield": info.get("dividendYield", 0.0),
                "institutional_ownership": info.get("heldPercentInstitutions", 0.0),
                "description": info.get("longBusinessSummary", "Business overview unavailable.")
            }
        except Exception:
            return {
                "sector": "N/A", "industry": "N/A", "market_cap": 0, "pe_ratio": 0.0,
                "forward_pe": 0.0, "pb_ratio": 0.0, "peg_ratio": 0.0, "debt_to_equity": 0.0,
                "profit_margins": 0.0, "operating_margins": 0.0, "roe": 0.0, "dividend_yield": 0.0,
                "institutional_ownership": 0.0, "description": "Business overview unavailable."
            }
