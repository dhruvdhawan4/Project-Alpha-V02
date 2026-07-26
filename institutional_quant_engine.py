"""
Institutional Quantitative Portfolio Research & Optimization Engine
===================================================================
Features:
1. Dynamic Expected Return Engine (EWMA Drift + Single-Index CAPM)
2. GARCH(1,1) Volatility Forecasting with RiskMetrics EWMA Fallback
3. Dynamic Covariance Matrix (GARCH Volatilities x Ledoit-Wolf Shrinkage Correlation)
4. Black-Litterman Asset Allocation Model (Reverse Optimization + Views Blending)
5. Multi-Constrained Institutional Optimizer (Transaction Costs, Turnover Caps, Sector Caps)
6. Equal Risk Contribution (Risk Parity) Solver
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False


# =====================================================================
# 1. QUANTITATIVE RETURN & VOLATILITY ENGINE
# =====================================================================
class QuantitativeEngine:
    """
    Quantitative module to replace static constants with dynamic,
    data-driven expected returns and GARCH(1,1) volatility forecasts.
    """

    def __init__(self, risk_free_rate: float = 0.045, trading_days: int = 252):
        self.rf = risk_free_rate
        self.trading_days = trading_days

    def calculate_log_returns(self, price_series: pd.Series) -> pd.Series:
        """Calculates log returns from a price series, dropping NaNs."""
        return np.log(price_series / price_series.shift(1)).dropna()

    def estimate_expected_return(
        self,
        asset_prices: pd.Series,
        benchmark_prices: Optional[pd.Series] = None,
        decay_factor: float = 0.94
    ) -> float:
        """
        Calculates dynamic expected annualized return using a blend of
        EWMA historical drift and CAPM factor loading.
        """
        returns = self.calculate_log_returns(asset_prices)
        if len(returns) < 30:
            raise ValueError("Insufficient price data. Require at least 30 observations.")

        # EWMA Drift Estimation (Weights recent market dynamics higher)
        n = len(returns)
        weights = (1 - decay_factor) * (decay_factor ** np.arange(n - 1, -1, -1))
        weights /= weights.sum()
        ewma_daily_return = np.sum(weights * returns.values)
        ewma_annualized = float(ewma_daily_return * self.trading_days)

        # CAPM Baseline Estimation (if benchmark is provided)
        if benchmark_prices is not None:
            bench_returns = self.calculate_log_returns(benchmark_prices)
            aligned = pd.concat([returns, bench_returns], axis=1, join="inner").dropna()

            if len(aligned) >= 30:
                asset_ret, bench_ret = aligned.iloc[:, 0], aligned.iloc[:, 1]
                cov_matrix = np.cov(asset_ret, bench_ret)
                beta = cov_matrix[0, 1] / cov_matrix[1, 1]

                bench_ewma_daily = np.sum(weights[:len(bench_ret)] * bench_ret.values)
                bench_expected_ann = bench_ewma_daily * self.trading_days

                capm_expected_ann = self.rf + beta * (bench_expected_ann - self.rf)

                # Blend CAPM (60%) with Asset EWMA Drift (40%)
                expected_return = 0.60 * capm_expected_ann + 0.40 * ewma_annualized
                return round(float(expected_return), 4)

        return round(float(ewma_annualized), 4)

    def estimate_garch_volatility(
        self,
        asset_prices: pd.Series,
        horizon_days: int = 22
    ) -> Dict[str, float]:
        """
        Estimates GARCH(1,1) volatility parameters and forecasts annualized volatility.
        Formula: sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
        """
        returns = self.calculate_log_returns(asset_prices) * 100  # Scale for solver stability

        if len(returns) < 100:
            return self._ewma_volatility_fallback(returns / 100)

        # Primary Path: Use `arch` library if available
        if ARCH_AVAILABLE:
            try:
                model = arch_model(returns, vol='Garch', p=1, q=1, dist='normal', rescale=False)
                res = model.fit(disp='off', show_warning=False)

                omega = res.params['omega']
                alpha = res.params['alpha[1]']
                beta = res.params['beta[1]']

                forecast = res.forecast(horizon=horizon_days)
                daily_var_forecast = forecast.variance.iloc[-1].mean()

                annualized_vol = np.sqrt(daily_var_forecast * self.trading_days) / 100
                persistence = alpha + beta
                long_run_var = omega / (1 - persistence) if persistence < 0.999 else daily_var_forecast
                long_run_vol = np.sqrt(long_run_var * self.trading_days) / 100

                return {
                    "method": "GARCH(1,1)",
                    "current_annualized_vol": round(float(annualized_vol), 4),
                    "long_run_annualized_vol": round(float(long_run_vol), 4),
                    "omega": float(omega),
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "persistence": round(float(persistence), 4)
                }
            except Exception:
                return self._ewma_volatility_fallback(returns / 100)

        return self._fit_garch_mle_scipy(returns / 100)

    def _ewma_volatility_fallback(self, returns: pd.Series, lambda_param: float = 0.94) -> Dict[str, float]:
        """RiskMetrics EWMA Volatility fallback."""
        returns_sq = returns ** 2
        var_ewma = returns_sq.ewm(alpha=(1 - lambda_param)).mean().iloc[-1]
        ann_vol = np.sqrt(var_ewma * self.trading_days)

        return {
            "method": "EWMA Fallback (RiskMetrics)",
            "current_annualized_vol": round(float(ann_vol), 4),
            "long_run_annualized_vol": round(float(ann_vol), 4),
            "omega": 0.0,
            "alpha": 1 - lambda_param,
            "beta": lambda_param,
            "persistence": 1.0
        }

    def _fit_garch_mle_scipy(self, returns: pd.Series) -> Dict[str, float]:
        """Custom Scipy MLE optimizer for GARCH(1,1) when arch library is absent."""
        r = returns.values
        T = len(r)

        def garch_log_likelihood(params):
            omega, alpha, beta = params
            variance = np.zeros(T)
            variance[0] = np.var(r)

            for t in range(1, T):
                variance[t] = omega + alpha * (r[t-1]**2) + beta * variance[t-1]

            log_lik = -0.5 * np.sum(np.log(2 * np.pi) + np.log(variance) + (r**2) / variance)
            return -log_lik

        initial_params = [1e-6, 0.05, 0.90]
        bounds = [(1e-8, None), (0.001, 0.3), (0.6, 0.98)]
        constraints = ({'type': 'ineq', 'fun': lambda p: 0.999 - (p[1] + p[2])})

        res = minimize(garch_log_likelihood, initial_params, bounds=bounds, constraints=constraints)

        if res.success:
            omega, alpha, beta = res.x
            var_t = np.var(r)
            for t in range(1, T):
                var_t = omega + alpha * (r[t-1]**2) + beta * var_t

            ann_vol = np.sqrt(var_t * self.trading_days)
            long_run_vol = np.sqrt((omega / (1 - alpha - beta)) * self.trading_days)

            return {
                "method": "GARCH(1,1) via Scipy MLE",
                "current_annualized_vol": round(float(ann_vol), 4),
                "long_run_annualized_vol": round(float(long_run_vol), 4),
                "omega": float(omega),
                "alpha": float(alpha),
                "beta": float(beta),
                "persistence": round(float(alpha + beta), 4)
            }

        return self._ewma_volatility_fallback(returns)


# =====================================================================
# 2. DYNAMIC COVARIANCE ENGINE
# =====================================================================
class DynamicCovarianceEngine:
    """
    Constructs dynamic N x N covariance matrices combining GARCH(1,1) forecasted
    volatilities with Ledoit-Wolf shrinkage correlation matrices:
        Sigma = D_GARCH * R_shrinkage * D_GARCH
    """

    def __init__(self, risk_free_rate: float = 0.045, trading_days: int = 252):
        self.rf = risk_free_rate
        self.trading_days = trading_days
        self.quant_engine = QuantitativeEngine(risk_free_rate=risk_free_rate, trading_days=trading_days)

    def calculate_garch_covariance(
        self,
        prices_df: pd.DataFrame,
        horizon_days: int = 22
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        log_returns = np.log(prices_df / prices_df.shift(1)).dropna()
        assets = prices_df.columns

        # 1. Estimate univariate GARCH(1,1) Volatility for each asset
        garch_vols = []
        vol_metadata = {}
        for col in assets:
            vol_info = self.quant_engine.estimate_garch_volatility(prices_df[col], horizon_days=horizon_days)
            garch_vols.append(vol_info["current_annualized_vol"])
            vol_metadata[col] = vol_info["current_annualized_vol"]

        D_garch = np.diag(garch_vols)

        # 2. Estimate Robust Shrinkage Correlation Matrix (Ledoit-Wolf)
        lw = LedoitWolf()
        sample_cov = lw.fit(log_returns.values).covariance_ * self.trading_days

        stds = np.sqrt(np.diag(sample_cov))
        stds_inv = np.diag(1.0 / stds)
        R_shrink = stds_inv @ sample_cov @ stds_inv
        np.fill_diagonal(R_shrink, 1.0)

        # 3. Reconstruct Dynamic Covariance Matrix
        cov_matrix = D_garch @ R_shrink @ D_garch

        return cov_matrix, R_shrink, vol_metadata


# =====================================================================
# 3. BLACK-LITTERMAN ENGINE
# =====================================================================
class BlackLittermanEngine:
    """
    Implements Black-Litterman Asset Allocation Model.
    Blends Market Equilibrium Returns (derived via Reverse Optimization) with Tactical Views.
    """

    def __init__(self, cov_matrix: np.ndarray, risk_aversion: float = 2.5, tau: float = 0.05):
        self.cov = cov_matrix
        self.delta = risk_aversion
        self.tau = tau
        self.N = cov_matrix.shape[0]

    def calculate_implied_equilibrium_returns(self, market_weights: np.ndarray) -> np.ndarray:
        """Reverse Optimization: Pi = delta * Sigma * w_market"""
        w_mkt = np.array(market_weights).reshape(-1, 1)
        pi = self.delta * (self.cov @ w_mkt)
        return pi.flatten()

    def combine_views(
        self,
        market_weights: np.ndarray,
        P_views: Optional[np.ndarray] = None,
        Q_views: Optional[np.ndarray] = None,
        omega_matrix: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Master Formula:
        E[R] = [ (tau*Sigma)^-1 + P^T * Omega^-1 * P ]^-1 * [ (tau*Sigma)^-1 * Pi + P^T * Omega^-1 * Q ]
        """
        pi = self.calculate_implied_equilibrium_returns(market_weights)

        if P_views is None or Q_views is None or len(P_views) == 0:
            return pi, self.cov

        P = np.array(P_views)
        Q = np.array(Q_views).reshape(-1, 1)

        tau_sigma = self.tau * self.cov
        tau_sigma_inv = np.linalg.pinv(tau_sigma)

        if omega_matrix is None:
            omega_matrix = np.diag(np.diag(P @ tau_sigma @ P.T))

        omega_inv = np.linalg.pinv(omega_matrix)

        M_inv = np.linalg.pinv(tau_sigma_inv + P.T @ omega_inv @ P)
        bl_returns = M_inv @ (tau_sigma_inv @ pi.reshape(-1, 1) + P.T @ omega_inv @ Q)
        bl_cov = self.cov + M_inv

        return bl_returns.flatten(), bl_cov


# =====================================================================
# 4. INSTITUTIONAL PORTFOLIO OPTIMIZER
# =====================================================================
class InstitutionalPortfolioOptimizer:
    """
    Production-grade portfolio optimizer with friction penalty,
    turnover caps, sector constraints, multi-start solver, and Risk Parity.
    """

    def __init__(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_free_rate: float = 0.045,
        risk_aversion: float = 2.5
    ):
        self.mu = np.array(expected_returns)
        self.cov = np.array(cov_matrix)
        self.rf = risk_free_rate
        self.gamma = risk_aversion
        self.N = len(self.mu)

    def portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float, float]:
        ret = np.sum(self.mu * weights)
        vol = np.sqrt(weights.T @ self.cov @ weights)
        sharpe = (ret - self.rf) / vol if vol > 0 else 0.0
        return ret, vol, sharpe

    def optimize_institutional_portfolio(
        self,
        previous_weights: Optional[np.ndarray] = None,
        max_stock_weight: float = 0.15,
        min_stock_weight: float = 0.0,
        max_turnover: float = 0.20,
        transaction_cost_pct: float = 0.0015,
        sector_mapping: Optional[Dict[str, List[int]]] = None,
        sector_caps: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Executes robust optimization under operational frictions.
        Objective: Maximize U(w) = w^T * mu - (gamma / 2) * w^T * Sigma * w - Friction
        """
        if previous_weights is None:
            previous_weights = np.ones(self.N) / self.N

        def objective_function(w):
            port_return = np.sum(w * self.mu)
            port_variance = w.T @ self.cov @ w
            turnover = np.sum(np.abs(w - previous_weights))
            tx_costs = turnover * transaction_cost_pct
            
            utility = port_return - (0.5 * self.gamma * port_variance) - tx_costs
            return -utility

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

        # Turnover Hard Constraint
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: max_turnover - np.sum(np.abs(w - previous_weights))
        })

        # Sector Cap Constraints
        if sector_mapping and sector_caps:
            for sector_name, asset_indices in sector_mapping.items():
                cap = sector_caps.get(sector_name, 1.0)
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda w, idx=asset_indices, c=cap: c - np.sum(w[idx])
                })

        bounds = [(min_stock_weight, max_stock_weight) for _ in range(self.N)]

        # Multi-Start Perturbation Engine
        best_result = None
        min_fun_val = np.inf
        initial_guesses = [
            previous_weights,
            np.ones(self.N) / self.N,
            self._get_risk_parity_weights()
        ]

        for x0 in initial_guesses:
            try:
                res = minimize(
                    objective_function,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 500, 'ftol': 1e-7}
                )
                if res.success and res.fun < min_fun_val:
                    min_fun_val = res.fun
                    best_result = res
            except Exception:
                continue

        # Fallback Engine
        if best_result is None or not best_result.success:
            best_result = minimize(
                objective_function,
                previous_weights,
                method='L-BFGS-B',
                bounds=bounds
            )
            best_result.x = best_result.x / np.sum(best_result.x)

        final_weights = np.round(best_result.x, 4)
        ret, vol, sharpe = self.portfolio_performance(final_weights)
        turnover_executed = float(np.sum(np.abs(final_weights - previous_weights)))
        tx_costs_est = turnover_executed * transaction_cost_pct

        return {
            "objective": "Multi-Constrained Friction-Aware Utility",
            "weights": final_weights,
            "expected_return": round(float(ret), 4),
            "net_expected_return": round(float(ret - tx_costs_est), 4),
            "volatility": round(float(vol), 4),
            "sharpe_ratio": round(float(sharpe), 4),
            "turnover": round(turnover_executed, 4),
            "transaction_cost_impact": round(tx_costs_est, 6),
            "solver_status": "SLSQP Converged" if best_result.success else "Fallback Solver Executed"
        }

    def risk_parity(self) -> Dict[str, Any]:
        """Equal Risk Contribution (ERC) Optimization."""
        def risk_parity_objective(w):
            portfolio_vol = np.sqrt(w.T @ self.cov @ w)
            marginal_risk_contrib = (self.cov @ w) / portfolio_vol
            risk_contrib = w * marginal_risk_contrib
            target_risk = portfolio_vol / self.N
            return np.sum((risk_contrib - target_risk) ** 2)

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = [(0.0, 1.0) for _ in range(self.N)]
        initial_weights = np.ones(self.N) / self.N

        res = minimize(risk_parity_objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

        weights = np.round(res.x, 4)
        ret, vol, sharpe = self.portfolio_performance(weights)
        marginal_risk = (self.cov @ weights) / vol
        risk_contrib_pct = (weights * marginal_risk) / vol

        return {
            "objective": "Equal Risk Contribution (Risk Parity)",
            "weights": weights,
            "risk_contributions_pct": np.round(risk_contrib_pct, 4),
            "expected_return": round(float(ret), 4),
            "volatility": round(float(vol), 4),
            "sharpe_ratio": round(float(sharpe), 4)
        }

    def _get_risk_parity_weights(self) -> np.ndarray:
        vols = np.sqrt(np.diag(self.cov))
        inv_vols = 1.0 / vols
        return inv_vols / np.sum(inv_vols)


# =====================================================================
# 5. END-TO-END PIPELINE WRAPPER
# =====================================================================
def run_institutional_pipeline(
    prices_df: pd.DataFrame,
    market_cap_weights: np.ndarray,
    previous_weights: Optional[np.ndarray] = None,
    tactical_views_P: Optional[np.ndarray] = None,
    tactical_views_Q: Optional[np.ndarray] = None,
    sector_mapping: Optional[Dict[str, List[int]]] = None,
    sector_caps: Optional[Dict[str, float]] = None,
    risk_free_rate: float = 0.045
) -> Dict[str, Any]:
    """
    Executes the complete pipeline:
    Raw Prices -> GARCH + Ledoit-Wolf Covariance -> Black-Litterman -> Friction-Aware Multi-Constraint Optimization.
    """
    assets = prices_df.columns.tolist()

    # 1. Calculate Dynamic Covariance Matrix
    cov_engine = DynamicCovarianceEngine(risk_free_rate=risk_free_rate)
    cov_matrix, corr_matrix, vol_metadata = cov_engine.calculate_garch_covariance(prices_df)

    # 2. Black-Litterman Return Estimation
    bl_engine = BlackLittermanEngine(cov_matrix=cov_matrix)
    bl_returns, bl_cov = bl_engine.combine_views(
        market_weights=market_cap_weights,
        P_views=tactical_views_P,
        Q_views=tactical_views_Q
    )

    # 3. Institutional Optimization
    optimizer = InstitutionalPortfolioOptimizer(expected_returns=bl_returns, cov_matrix=bl_cov, risk_free_rate=risk_free_rate)
    
    mvo_portfolio = optimizer.optimize_institutional_portfolio(
        previous_weights=previous_weights,
        max_stock_weight=0.25,
        max_turnover=0.20,
        sector_mapping=sector_mapping,
        sector_caps=sector_caps
    )
    
    risk_parity_portfolio = optimizer.risk_parity()

    return {
        "assets": assets,
        "garch_volatilities": vol_metadata,
        "black_litterman_returns": dict(zip(assets, np.round(bl_returns, 4))),
        "institutional_mvo_portfolio": mvo_portfolio,
        "risk_parity_portfolio": risk_parity_portfolio
    }


# =====================================================================
# DEMO EXECUTION WITH SYNTHETIC MARKET DATA
# =====================================================================
if __name__ == "__main__":
    print("Initializing Quantitative Portfolio Engine Test Run...\n")
    np.random.seed(42)

    # Generate Synthetic Daily Stock Prices (252 Trading Days x 5 Assets)
    assets = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
    dates = pd.date_range(start="2025-01-01", periods=252, freq="B")
    
    returns_matrix = np.random.multivariate_normal(
        mean=[0.0005, 0.0004, 0.0003, 0.00045, 0.00035],
        cov=[
            [0.0004, 0.0001, 0.00015, 0.0001, 0.00012],
            [0.0001, 0.0003, 0.00008, 0.0002, 0.00009],
            [0.00015, 0.00008, 0.00025, 0.00009, 0.00018],
            [0.0001, 0.0002, 0.00009, 0.00035, 0.0001],
            [0.00012, 0.00009, 0.00018, 0.0001, 0.00028]
        ],
        size=252
    )

    prices_df = pd.DataFrame(100 * np.exp(np.cumsum(returns_matrix, axis=0)), index=dates, columns=assets)

    # Benchmark Market Cap Weights (Nifty 50 proxies)
    market_cap_weights = np.array([0.30, 0.20, 0.25, 0.15, 0.10])
    previous_weights = np.array([0.20, 0.20, 0.20, 0.20, 0.20])

    # Tactical Views:
    # View 1: RELIANCE will return 14% absolute annualized return
    # View 2: INFY will outperform TCS by 3% annualized
    P_views = np.array([
        [1, 0, 0, 0, 0],
        [0, -1, 0, 1, 0]
    ])
    Q_views = np.array([0.14, 0.03])

    # Sector Constraints (0: Energy, 1: Tech, 2: Financials, 3: Tech, 4: Financials)
    sector_mapping = {
        "Technology": [1, 3],
        "Financials": [2, 4]
    }
    sector_caps = {
        "Technology": 0.35,  # Max 35% in IT
        "Financials": 0.40   # Max 40% in Banks
    }

    # Execute Pipeline
    results = run_institutional_pipeline(
        prices_df=prices_df,
        market_cap_weights=market_cap_weights,
        previous_weights=previous_weights,
        tactical_views_P=P_views,
        tactical_views_Q=Q_views,
        sector_mapping=sector_mapping,
        sector_caps=sector_caps
    )

    print("=====================================================")
    print("              GARCH(1,1) VOLATILITIES                ")
    print("=====================================================")
    for k, v in results["garch_volatilities"].items():
        print(f"{k:12s}: {v*100:.2f}%")

    print("\n=====================================================")
    print("          BLACK-LITTERMAN EXPECTED RETURNS           ")
    print("=====================================================")
    for k, v in results["black_litterman_returns"].items():
        print(f"{k:12s}: {v*100:.2f}%")

    print("\n=====================================================")
    print("        INSTITUTIONAL MVO PORTFOLIO (FRICTION-AWARE)  ")
    print("=====================================================")
    mvo = results["institutional_mvo_portfolio"]
    print(f"Allocated Weights : {mvo['weights']}")
    print(f"Expected Return   : {mvo['expected_return']*100:.2f}%")
    print(f"Net Return (Tx)   : {mvo['net_expected_return']*100:.2f}%")
    print(f"Volatility        : {mvo['volatility']*100:.2f}%")
    print(f"Sharpe Ratio      : {mvo['sharpe_ratio']}")
    print(f"Executed Turnover : {mvo['turnover']*100:.2f}%")
    print(f"Solver Status     : {mvo['solver_status']}")

    print("\n=====================================================")
    print("            RISK PARITY (ERC) PORTFOLIO              ")
    print("=====================================================")
    rp = results["risk_parity_portfolio"]
    print(f"Allocated Weights : {rp['weights']}")
    print(f"Risk Contribs %   : {rp['risk_contributions_pct']}")
    print(f"Volatility        : {rp['volatility']*100:.2f}%")
    print(f"Sharpe Ratio      : {rp['sharpe_ratio']}")
