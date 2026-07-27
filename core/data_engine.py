"""
Multi-Market & Multi-Asset Data Ingestion Engine
Handles universe repositories across Stocks, ETFs, Mutual Funds for US and Indian Markets.
Includes automatic failover to last closed market session data during off-hours or closures.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import ta
from typing import Dict, Any, List, Optional


class DataEngine:
    # Asset Repositories
    US_STOCKS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "JNJ", "JPM", "V", "PG", "UNH", "HD", "MA", "BAC", "XOM", "PFE", "DIS", "CSCO"]
    US_ETFS = ["SPY", "QQQ", "VOO", "VTI", "IWM", "GLD", "AGG", "EEM", "IVV", "XLK", "XLF", "XLV", "XLE", "SCHD", "VUG", "VYM", "ARKK", "SMH", "VNQ", "TLT"]
    US_MUTUAL_FUNDS = ["VFIAX", "VTSAX", "FXAIX", "VWENX", "SWPPX", "VIGAX", "VGSLX", "VBTLX", "PRGFX", "FDGRX"]

    INDIA_STOCKS = [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS", "SBIN.NS", "LT.NS", "ITC.NS", "HINDUNILVR.NS",
        "AXISBANK.NS", "KOTAKBANK.NS", "BAJFINANCE.NS", "MARUTI.NS", "SUNPHARMA.NS", "TATAMOTORS.NS", "NTPC.NS", "POWERGRID.NS", "TITAN.NS", "ULTRACEMCO.NS"
    ]
    INDIA_ETFS = [
        "NIFTYBEES.NS", "BANKBEES.NS", "GOLDBEES.NS", "ITBEES.NS", "JUNIORBEES.NS", "MON100.NS", "LIQUIDBEES.NS", "PHARMABEES.NS", "AUTOBEES.NS", "CPSEETF.NS"
    ]
    INDIA_MUTUAL_FUNDS = [
        "0P0000XW01.BO", "0P0000XVUY.BO", "0P0000XVUX.BO", "0P00013X12.BO", "0P0000XW0K.BO",
        "NIFTYBEES.NS", "BANKBEES.NS", "GOLDBEES.NS", "ITBEES.NS", "JUNIORBEES.NS"
    ]

    @staticmethod
    def format_ticker(symbol: str, market: str) -> str:
        """Formats and sanitizes ticker symbol based on geographic market."""
        symbol = symbol.strip().upper()
        if market == "India":
            if not (symbol.endswith(".NS") or symbol.endswith(".BO")):
                return f"{symbol}.NS"
        return symbol

    @classmethod
    def get_universe(cls, market: str, vehicle: str) -> List[str]:
        """Returns target asset universe based on selected market and investment vehicle."""
        if market == "United States":
            if vehicle == "ETF":
                return cls.US_ETFS
            elif vehicle in ["Mutual Funds", "Mutual Fund"]:
                return cls.US_MUTUAL_FUNDS
            else:
                return cls.US_STOCKS
        else: # India
            if vehicle == "ETF":
                return cls.INDIA_ETFS
            elif vehicle in ["Mutual Funds", "Mutual Fund"]:
                return cls.INDIA_MUTUAL_FUNDS
            else:
                return cls.INDIA_STOCKS

    @staticmethod
    def fetch_historical_data(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetches OHLCV historical price series.
        Automatically handles market closures and off-hours by retrieving and falling back to the last available closed session data.
        """
        try:
            t = yf.Ticker(ticker)
            df = t.history(period=period, interval=interval)
            
            # If intraday fetch returns empty during off-hours, fall back to broader intraday/daily history
            if df is None or df.empty:
                if interval != "1d":
                    df = t.history(period="30d", interval="15m")
                if df is None or df.empty:
                    df = t.history(period="1y", interval="1d")
                    
            if df is None or df.empty:
                return pd.DataFrame()
                
            # Drop NaN rows resulting from market closures / non-trading periods
            df = df.dropna(subset=['Close'])
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
                "sector": info.get('sector', 'Financial Services / Index'),
                "industry": info.get('industry', 'Investment Vehicles'),
                "business_summary": info.get('longBusinessSummary', 'No detailed overview available.'),
                "inst_ownership": info.get('heldPercentInstitutions', 0.0)
            }
        except Exception:
            return {
                "pe": np.nan, "pb": np.nan, "ev_ebitda": np.nan, "roe": np.nan, "roce": np.nan,
                "rev_growth": np.nan, "eps_growth": np.nan, "debt_to_equity": np.nan,
                "free_cashflow": np.nan, "operating_margins": np.nan, "net_margins": np.nan,
                "dividend_yield": 0.0, "beta": 1.0, "market_cap": 0.0,
                "sector": "Diversified", "industry": "Investment Asset",
                "business_summary": "Asset analysis generated dynamically via quantitative historical series.",
                "inst_ownership": 0.0
            }

    @staticmethod
    def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Enriches dataframe with technical indicators."""
        if df.empty or len(df) < 10:
            return df
            
        df['SMA20'] = df['Close'].rolling(window=min(20, len(df))).mean()
        df['SMA50'] = df['Close'].rolling(window=min(50, len(df))).mean()
        df['SMA100'] = df['Close'].rolling(window=min(100, len(df))).mean()
        df['SMA200'] = df['Close'].rolling(window=min(200, len(df))).mean()
        
        df['RSI'] = ta.momentum.rsi(df['Close'], window=min(14, len(df)-1))
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()
        
        df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=min(14, len(df)-1))
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=min(14, len(df)-1))
        
        bollinger = ta.volatility.BollingerBands(df['Close'], window=min(20, len(df)))
        df['BB_high'] = bollinger.bollinger_hband()
        df['BB_low'] = bollinger.bollinger_lband()
        
        # VWAP
        tp = (df['High'] + df['Low'] + df['Close']) / 3.0
        df['VWAP'] = (tp * df['Volume']).cumsum() / df['Volume'].cumsum().replace(0, 1)
        
        return df
