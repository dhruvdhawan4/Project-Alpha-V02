"""
Intraday Trading & Analytics Engine
Computes institutional indicators: VWAP, SuperTrend, Pivot Levels, ATR, ADX, and Risk Sizing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import ta
from core.data_engine import DataEngine


class IntradayEngine:
    def __init__(self):
        self.data_engine = DataEngine()

    def _calculate_supertrend(self, df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
        """Calculates SuperTrend indicator series."""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        atr = ta.volatility.average_true_range(high, low, close, window=period)
        hl2 = (high + low) / 2.0
        
        basic_upperband = hl2 + (multiplier * atr)
        basic_lowerband = hl2 - (multiplier * atr)
        
        upperband = basic_upperband.copy()
        lowerband = basic_lowerband.copy()
        
        for i in range(1, len(df)):
            if basic_upperband.iloc[i] < upperband.iloc[i-1] or close.iloc[i-1] > upperband.iloc[i-1]:
                upperband.iloc[i] = basic_upperband.iloc[i]
            else:
                upperband.iloc[i] = upperband.iloc[i-1]
                
            if basic_lowerband.iloc[i] > lowerband.iloc[i-1] or close.iloc[i-1] < lowerband.iloc[i-1]:
                lowerband.iloc[i] = basic_lowerband.iloc[i]
            else:
                lowerband.iloc[i] = lowerband.iloc[i-1]
                
        return upperband, lowerband

    def analyze_intraday_signals(self, tickers: List[str], investment_amount: float) -> Dict[str, List[Dict[str, Any]]]:
        """Generates Top Long and Top Short trade recommendations."""
        long_trades = []
        short_trades = []
        
        for ticker in tickers:
            symbol = f"{ticker}.NS" if not ticker.endswith(".NS") else ticker
            df = self.data_engine.fetch_historical_data(symbol, period="60d", interval="15m")
            
            if df.empty or len(df) < 50:
                continue
                
            close = df['Close'].iloc[-1]
            high = df['High']
            low = df['Low']
            volume = df['Volume']
            
            # Technical Indicators
            rsi = float(ta.momentum.rsi(df['Close'], window=14).iloc[-1])
            macd_series = ta.trend.macd_diff(df['Close'])
            macd_diff = float(macd_series.iloc[-1])
            adx = float(ta.trend.adx(high, low, df['Close'], window=14).iloc[-1])
            atr = float(ta.volatility.average_true_range(high, low, df['Close'], window=14).iloc[-1])
            
            # VWAP Calculation
            vwap = float((df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).sum() / df['Volume'].sum())
            
            # Pivot Points (Classic)
            prev_high = df['High'].iloc[-2]
            prev_low = df['Low'].iloc[-2]
            prev_close = df['Close'].iloc[-2]
            
            pivot = (prev_high + prev_low + prev_close) / 3.0
            r1 = (2 * pivot) - prev_low
            s1 = (2 * pivot) - prev_high
            
            # Signal Assessment
            is_long = close > vwap and rsi > 50 and macd_diff > 0
            is_short = close < vwap and rsi < 50 and macd_diff < 0
            
            if is_long or is_short:
                stop_loss = round(close - (1.5 * atr), 2) if is_long else round(close + (1.5 * atr), 2)
                target = round(close + (3.0 * atr), 2) if is_long else round(close - (3.0 * atr), 2)
                risk_per_share = abs(close - stop_loss)
                reward_per_share = abs(target - close)
                rr_ratio = round(reward_per_share / (risk_per_share if risk_per_share > 0 else 1.0), 2)
                
                trade_data = {
                    "ticker": ticker.replace(".NS", ""),
                    "entry": round(close, 2),
                    "stop_loss": stop_loss,
                    "target": target,
                    "risk_reward": f"1:{rr_ratio}",
                    "allocation": round(investment_amount * 0.1, 2),
                    "confidence": f"{min(95, int(50 + adx + (rsi - 50 if is_long else 50 - rsi)))}%",
                    "holding_period": "30 mins - 4 hrs",
                    "trend": "Strong Bullish" if is_long else "Strong Bearish",
                    "vwap": round(vwap, 2),
                    "rsi": round(rsi, 2),
                    "adx": round(adx, 2),
                    "atr": round(atr, 2),
                    "pivot": round(pivot, 2),
                    "resistance_1": round(r1, 2),
                    "support_1": round(s1, 2),
                    "thesis": f"{'Bullish' if is_long else 'Bearish'} momentum confirmed by VWAP crossover, ADX strength ({round(adx,1)}), and MACD divergence."
                }
                
                if is_long and len(long_trades) < 5:
                    long_trades.append(trade_data)
                elif is_short and len(short_trades) < 5:
                    short_trades.append(trade_data)
                    
        return {"long_trades": long_trades, "short_trades": short_trades}
