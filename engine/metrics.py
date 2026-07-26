"""
Vectorized Technical Indicators Engine.
Provides deterministic calculation of technical analysis metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any

class TechnicalMetricsEngine:
    
    @staticmethod
    def calculate_sma(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window=window).mean()

    @staticmethod
    def calculate_ema(series: pd.Series, window: int) -> pd.Series:
        return series.ewm(span=window, adjust=False).mean()

    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = TechnicalMetricsEngine.calculate_ema(series, fast)
        ema_slow = TechnicalMetricsEngine.calculate_ema(series, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalMetricsEngine.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def calculate_bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        sma = TechnicalMetricsEngine.calculate_sma(series, window)
        rolling_std = series.rolling(window=window).std()
        upper_band = sma + (rolling_std * num_std)
        lower_band = sma - (rolling_std * num_std)
        return upper_band, sma, lower_band

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift(1))
        low_close = np.abs(df['Low'] - df['Close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def calculate_vwap(df: pd.DataFrame) -> pd.Series:
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3.0
        tp_v = typical_price * df['Volume']
        return tp_v.cumsum() / (df['Volume'].cumsum() + 1e-10)

    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        up_move = df['High'] - df['High'].shift(1)
        down_move = df['Low'].shift(1) - df['Low']
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        atr = TechnicalMetricsEngine.calculate_atr(df, period)
        plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(period).mean() / (atr + 1e-10))
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(period).mean() / (atr + 1e-10))
        
        dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))
        return dx.rolling(period).mean()

    @staticmethod
    def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
        atr = TechnicalMetricsEngine.calculate_atr(df, period)
        hl2 = (df['High'] + df['Low']) / 2.0
        
        basic_upperband = hl2 + (multiplier * atr)
        basic_lowerband = hl2 - (multiplier * atr)
        
        final_upperband = basic_upperband.copy()
        final_lowerband = basic_lowerband.copy()
        supertrend = pd.Series(0.0, index=df.index)
        direction = pd.Series(1, index=df.index)
        
        for i in range(1, len(df)):
            if basic_upperband.iloc[i] < final_upperband.iloc[i-1] or df['Close'].iloc[i-1] > final_upperband.iloc[i-1]:
                final_upperband.iloc[i] = basic_upperband.iloc[i]
            else:
                final_upperband.iloc[i] = final_upperband.iloc[i-1]
                
            if basic_lowerband.iloc[i] > final_lowerband.iloc[i-1] or df['Close'].iloc[i-1] < final_lowerband.iloc[i-1]:
                final_lowerband.iloc[i] = basic_lowerband.iloc[i]
            else:
                final_lowerband.iloc[i] = final_lowerband.iloc[i-1]
                
            if df['Close'].iloc[i] > final_upperband.iloc[i-1]:
                direction.iloc[i] = 1
            elif df['Close'].iloc[i] < final_lowerband.iloc[i-1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i-1]
                
            supertrend.iloc[i] = final_lowerband.iloc[i] if direction.iloc[i] == 1 else final_upperband.iloc[i]
            
        return supertrend, direction

    @staticmethod
    def calculate_pivot_points(df: pd.DataFrame) -> Dict[str, float]:
        last_row = df.iloc[-1]
        p = (last_row['High'] + last_row['Low'] + last_row['Close']) / 3.0
        r1 = (2 * p) - last_row['Low']
        s1 = (2 * p) - last_row['High']
        r2 = p + (last_row['High'] - last_row['Low'])
        s2 = p - (last_row['High'] - last_row['Low'])
        return {"P": round(p, 2), "R1": round(r1, 2), "S1": round(s1, 2), "R2": round(r2, 2), "S2": round(s2, 2)}
