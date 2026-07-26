"""
Long-Term Quantitative Investment Engine
Includes multi-factor dynamic scoring, ML feature importance weighting, and Monte Carlo projections.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, List, Any
from core.data_engine import DataEngine


class LongTermEngine:
    def __init__(self):
        self.data_engine = DataEngine()

    def _compute_factor_weights(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculates factor weights statistically using Random Forest Permutation Importance."""
        if X.empty or len(X) < 5:
            cols = X.columns if not X.empty else ['valuation', 'growth', 'profitability', 'momentum', 'quality']
            return {col: 1.0 / len(cols) for col in cols}
        
        X_filled = X.fillna(X.median())
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_filled, y)
        
        importances = rf.feature_importances_
        total = sum(importances) if sum(importances) > 0 else 1.0
        
        weights = {col: float(imp / total) for col, imp in zip(X.columns, importances)}
        return weights

    def analyze_universe(self, tickers: List[str], market: str, vehicle: str) -> pd.DataFrame:
        """Executes multi-factor evaluation across the asset universe."""
        records = []
        
        for raw_symbol in tickers:
            symbol = self.data_engine.format_ticker(raw_symbol, market)
            df_hist = self.data_engine.fetch_historical_data(symbol, period="2y")
            
            if df_hist.empty or len(df_hist) < 100:
                continue
                
            metrics = self.data_engine.fetch_financial_metrics(symbol)
            if not metrics:
                continue
                
            close = df_hist['Close'].values
            returns_1y = (close[-1] - close[-252]) / close[-252] if len(close) >= 252 else 0.0
            volatility = np.std(np.diff(np.log(close))) * np.sqrt(252)
            sharpe = (returns_1y - 0.05) / volatility if volatility > 0 else 0.0
            
            valuation_score = 1.0 / (metrics['pe'] + 1e-5) if not np.isnan(metrics['pe']) and metrics['pe'] > 0 else 0.01
            growth_score = (metrics['rev_growth'] if not np.isnan(metrics['rev_growth']) else 0.0) + \
                           (metrics['eps_growth'] if not np.isnan(metrics['eps_growth']) else 0.0)
            profit_score = (metrics['roe'] if not np.isnan(metrics['roe']) else 0.0) + \
                           (metrics['roce'] if not np.isnan(metrics['roce']) else 0.0)
            momentum_score = returns_1y
            quality_score = (1.0 / (metrics['debt_to_equity'] + 1.0)) if not np.isnan(metrics['debt_to_equity']) else 0.5
            
            records.append({
                "ticker": raw_symbol,
                "formatted_ticker": symbol,
                "valuation": valuation_score,
                "growth": growth_score,
                "profitability": profit_score,
                "momentum": momentum_score,
                "quality": quality_score,
                "volatility": volatility,
                "sharpe": sharpe,
                "1y_return": returns_1y,
                "metrics": metrics,
                "current_price": close[-1]
            })
            
        if not records:
            return pd.DataFrame()
            
        df_factors = pd.DataFrame(records)
        
        factor_cols = ['valuation', 'growth', 'profitability', 'momentum', 'quality']
        for col in factor_cols:
            std = df_factors[col].std()
            df_factors[f"{col}_z"] = (df_factors[col] - df_factors[col].mean()) / (std if std > 0 else 1.0)
            
        X = df_factors[[f"{c}_z" for c in factor_cols]]
        y = df_factors['sharpe']
        weights = self._compute_factor_weights(X, y)
        
        df_factors['raw_score'] = 0.0
        for col in factor_cols:
            df_factors['raw_score'] += df_factors[f"{col}_z"] * weights.get(f"{col}_z", 0.2)
            
        min_s, max_s = df_factors['raw_score'].min(), df_factors['raw_score'].max()
        range_s = max_s - min_s if max_s != min_s else 1.0
        df_factors['recommendation_score'] = ((df_factors['raw_score'] - min_s) / range_s * 100).round(2)
        
        return df_factors.sort_values(by="recommendation_score", ascending=False)

    def run_monte_carlo_simulation(
        self, initial_investment: float, horizon_years: int, expected_return: float, volatility: float, num_simulations: int = 50000
    ) -> Dict[str, Any]:
        """Runs 50,000 geometric Brownian motion Monte Carlo simulations."""
        dt = 1 / 252
        num_steps = int(horizon_years * 252)
        
        drift = (expected_return - 0.5 * volatility**2) * dt
        vol = volatility * np.sqrt(dt)
        
        daily_returns = np.exp(drift + vol * np.random.normal(0, 1, (num_simulations, num_steps)))
        portfolio_paths = np.zeros((num_simulations, num_steps + 1))
        portfolio_paths[:, 0] = initial_investment
        
        for t in range(1, num_steps + 1):
            portfolio_paths[:, t] = portfolio_paths[:, t - 1] * daily_returns[:, t - 1]
            
        final_values = portfolio_paths[:, -1]
        
        return {
            "worst_case_10th": float(np.percentile(final_values, 10)),
            "expected_case_50th": float(np.percentile(final_values, 50)),
            "best_case_90th": float(np.percentile(final_values, 90)),
            "mean_value": float(np.mean(final_values)),
            "std_dev": float(np.std(final_values)),
            "simulation_paths": portfolio_paths[:100, :]
        }
