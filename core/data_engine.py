"""
Multi-Market Data Ingestion Engine
Handles ticker resolution, financial statement extraction, and technical indicators.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional


class DataEngine:
    @staticmethod
    def format_ticker(symbol: str, market: str) -> str:
        """Formats ticker according to market convention."""
        symbol = symbol.strip().upper()
        if market == "India" and not (symbol.endswith(".NS") or symbol.endswith(".BO")):
            return f"{symbol}.NS"
        return symbol

    @staticmethod
    def fetch_historical_data(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """Fetches historical price series with cleanup and validation."""
        try:
            df = yf.Ticker(ticker).history(period=period, interval=interval)
            if df.empty:
                return pd.DataFrame()
            df.reset_index(inplace=True)
            return df
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def fetch_financial_metrics(ticker: str) -> Dict[str, Any]:
        """Extracts fundamental metrics with explicit fallbacks."""
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
            
            # Fundamentals extraction
            pe = info.get('trailingPE') or info.get('forwardPE') or np.nan
            pb = info.get('priceToBook') or np.nan
            ev_ebitda = info.get('enterpriseToEbitda') or np.nan
            roe = info.get('returnOnEquity') or np.nan
            roce = info.get('returnOnAssets') or np.nan
            rev_growth = info.get('revenueGrowth') or np.nan
            eps_growth = info.get('earningsGrowth') or np.nan
            debt_to_equity = info.get('debtToEquity') or np.nan
            free_cashflow = info.get('freeCashflow') or np.nan
            operating_margins = info.get('operatingMargins') or np.nan
            net_margins = info.get('profitMargins') or np.nan
            dividend_yield = info.get('dividendYield') or 0.0
            beta = info.get('beta') or 1.0
            market_cap = info.get('marketCap') or 0.0
            
            return {
                "pe": pe,
                "pb": pb,
                "ev_ebitda": ev_ebitda,
                "roe": roe,
                "roce": roce,
                "rev_growth": rev_growth,
                "eps_growth": eps_growth,
                "debt_to_equity": debt_to_equity,
                "free_cashflow": free_cashflow,
                "operating_margins": operating_margins,
                "net_margins": net_margins,
                "dividend_yield": dividend_yield,
                "beta": beta,
                "market_cap": market_cap,
                "sector": info.get('sector', 'N/A'),
                "industry": info.get('industry', 'N/A'),
                "business_summary": info.get('longBusinessSummary', 'No summary available.'),
                "inst_ownership": info.get('heldPercentInstitutions', 0.0)
            }
        except Exception:
            return {}
