"""
Portfolio optimization module — Black-Litterman, Mean-CVaR, HRP, ERC,
Maximum Diversification, Transaction-Cost-Aware Optimization.
Implements Phases 7, 8, and §10B of claude.md.
"""
import logging

import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list, ward
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

from src.config import (
    RISK_AVERSION, BL_TAU,
    MAX_WEIGHT_FACTOR, MAX_WEIGHT_SINGLE, MAX_WEIGHT_SECTOR,
    TURNOVER_LIMIT, TC_ONE_WAY, SECTOR_MAP
)
from src.garch_utils import ensure_psd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# Black-Litterman
# ──────────────────────────────────────────────────────────

def black_litterman(Sigma, w_mkt, P, Q, Omega, tau=None, delta=None):
    """
    Black-Litterman posterior expected returns and covariance.

    Parameters
    ----------
    Sigma : np.ndarray — (N×N) covariance matrix
    w_mkt : np.ndarray — (N,) market-cap weights
    P : np.ndarray — (K×N) view matrix
    Q : np.ndarray — (K,) view vector (expected returns)
    Omega : np.ndarray — (K×K) view uncertainty
    tau : float — scaling parameter
    delta : float — risk aversion coefficient

    Returns
    -------
    mu_bl : np.ndarray — posterior expected returns
    Sigma_bl : np.ndarray — posterior covariance
    """
    if tau is None:
        tau = BL_TAU
    if delta is None:
        delta = RISK_AVERSION

    # Equilibrium expected returns
    pi = delta * Sigma @ w_mkt

    # BL posterior
    tauSigma = tau * Sigma
    M = np.linalg.inv(np.linalg.inv(tauSigma) + P.T @ np.linalg.inv(Omega) @ P)
    mu_bl = M @ (np.linalg.inv(tauSigma) @ pi + P.T @ np.linalg.inv(Omega) @ Q)
    Sigma_bl = Sigma + M

    return mu_bl, Sigma_bl


def compute_bl_omega_from_rmse(model_rmse, n_views=None):
    """
    Compute BL view uncertainty matrix from model RMSE.

    CRITICAL: Ω = diag(RMSE²), NOT diag(1/RMSE).
    Higher RMSE → higher uncertainty → less weight on view.
    """
    if n_views is None:
        n_views = len(model_rmse)
    omega = np.diag(model_rmse ** 2)
    omega = np.maximum(omega, 1e-4 * np.eye(n_views))
    return omega


def bl_optimal_weights(mu_bl, Sigma_bl, max_weight=None, long_only=True,
                       prev_weights=None, turnover_limit=None,
                       tc=None, sector_map=None, max_sector=None):
    """
    Compute optimal BL weights via mean-variance optimisation with
    institutional constraints: turnover, transaction costs, sector caps.

    Parameters
    ----------
    mu_bl : np.ndarray — posterior expected returns
    Sigma_bl : np.ndarray — posterior covariance
    max_weight : float — per-asset weight cap
    long_only : bool
    prev_weights : np.ndarray or None — previous period weights (for turnover/TC)
    turnover_limit : float or None — max total turnover (one-way sum)
    tc : float or None — one-way transaction cost (deducted from objective)
    sector_map : dict or None — {sector_name: [asset_indices]}
    max_sector : float or None — max weight per sector
    """
    if max_weight is None:
        max_weight = MAX_WEIGHT_FACTOR
    if turnover_limit is None and prev_weights is not None:
        turnover_limit = TURNOVER_LIMIT
    if tc is None:
        tc = TC_ONE_WAY

    n = len(mu_bl)

    # Ensure Sigma_bl is PSD before cvxpy quad_form
    Sigma_bl = ensure_psd(Sigma_bl)
    Sigma_bl = (Sigma_bl + Sigma_bl.T) / 2

    w = cp.Variable(n)
    ret = mu_bl @ w
    risk = cp.quad_form(w, cp.psd_wrap(Sigma_bl))

    # Objective: max return - risk - transaction costs
    objective = ret - 0.5 * RISK_AVERSION * risk
    if prev_weights is not None and tc > 0:
        # TC penalty: proportional to absolute turnover
        objective -= tc * cp.norm1(w - prev_weights)

    constraints = [cp.sum(w) == 1, w <= max_weight]
    if long_only:
        constraints.append(w >= 0)

    # Turnover constraint
    if prev_weights is not None and turnover_limit is not None:
        constraints.append(cp.norm1(w - prev_weights) <= 2 * turnover_limit)

    # Sector constraints
    if sector_map is not None and max_sector is not None:
        for sector, indices in sector_map.items():
            constraints.append(cp.sum(w[indices]) <= max_sector)

    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    if prob.status not in ('optimal', 'optimal_inaccurate'):
        logger.warning(f"BL optimization status: {prob.status}")

    weights = w.value
    if weights is None:
        weights = np.ones(n) / n
        logger.warning("BL optimization failed — returning equal weight")
    else:
        # Numerical cleanup: clip tiny negatives, re-normalize
        weights = np.maximum(weights, 0)
        weights /= weights.sum()

    return weights


# ──────────────────────────────────────────────────────────
# Mean-CVaR
# ──────────────────────────────────────────────────────────

def mean_cvar_optimize(scenario_matrix, alpha=0.05,
                       max_weight=None, target_return=None,
                       prev_weights=None, turnover_limit=None,
                       tc=None, return_weight=0.0):
    """
    Mean-CVaR portfolio optimization (Rockafellar-Uryasev LP formulation)
    with optional return maximization, turnover, and transaction costs.

    Parameters
    ----------
    scenario_matrix : np.ndarray — (S×N) scenario returns
    alpha : float — tail probability
    max_weight : float — per-asset weight cap
    target_return : float or None — minimum return constraint
    prev_weights : np.ndarray or None — previous weights for turnover/TC
    turnover_limit : float or None — max one-way turnover
    tc : float or None — one-way transaction cost
    return_weight : float — weight on expected return in objective (0 = pure min-CVaR)

    Returns
    -------
    np.ndarray — optimal weights
    """
    if max_weight is None:
        max_weight = MAX_WEIGHT_FACTOR
    if tc is None:
        tc = TC_ONE_WAY

    S, N = scenario_matrix.shape
    w = cp.Variable(N)
    z = cp.Variable(S)
    gamma = cp.Variable()

    portfolio_returns = scenario_matrix @ w

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= max_weight,
        z >= 0,
        z >= -portfolio_returns - gamma,
    ]

    if target_return is not None:
        constraints.append(cp.sum(portfolio_returns) / S >= target_return)

    if prev_weights is not None and turnover_limit is not None:
        constraints.append(cp.norm1(w - prev_weights) <= 2 * turnover_limit)

    # Objective: minimize CVaR - return_weight * E[r] + TC penalty
    cvar = gamma + (1 / (alpha * S)) * cp.sum(z)
    objective = cvar
    if return_weight > 0:
        objective -= return_weight * cp.sum(portfolio_returns) / S
    if prev_weights is not None and tc > 0:
        objective += tc * cp.norm1(w - prev_weights)

    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    if prob.status not in ('optimal', 'optimal_inaccurate'):
        logger.warning(f"CVaR optimization status: {prob.status}")

    weights = w.value
    if weights is None:
        weights = np.ones(N) / N
        logger.warning("CVaR optimization failed — returning equal weight")
    else:
        weights = np.maximum(weights, 0)
        weights /= weights.sum()

    return weights


# ──────────────────────────────────────────────────────────
# Hierarchical Risk Parity (HRP)
# ──────────────────────────────────────────────────────────

def hrp_optimize(returns, linkage_method='ward', max_weight=None):
    """
    Hierarchical Risk Parity (López de Prado 2016).
    Does NOT require expected returns — uses only covariance structure.

    Parameters
    ----------
    returns : pd.DataFrame — asset return series
    linkage_method : str — 'ward' (recommended) or 'single' (original paper)
    max_weight : float or None — per-asset weight cap (applied post-hoc)
    """
    cov = returns.cov().values
    corr = returns.corr().values
    n = cov.shape[0]

    # Distance matrix: d(i,j) = sqrt((1 - rho_{ij}) / 2)
    dist = np.sqrt((1 - corr) / 2)
    np.fill_diagonal(dist, 0)
    dist_condensed = squareform(dist, checks=False)

    # Hierarchical clustering — Ward minimizes within-cluster variance
    link = linkage(dist_condensed, method=linkage_method)
    sort_idx = leaves_list(link).tolist()

    # Recursive bisection
    weights = np.zeros(n)
    _recursive_bisect(weights, sort_idx, cov)

    # Weight cap enforcement with proportional redistribution
    if max_weight is not None:
        weights = _cap_weights(weights, max_weight)

    return weights


def _cap_weights(weights, max_weight, max_iter=50):
    """Cap weights at max_weight and proportionally redistribute excess."""
    w = weights.copy()
    for _ in range(max_iter):
        excess_mask = w > max_weight
        if not excess_mask.any():
            break
        excess = (w[excess_mask] - max_weight).sum()
        w[excess_mask] = max_weight
        free_mask = ~excess_mask & (w > 0)
        if free_mask.sum() == 0:
            break
        w[free_mask] += excess * w[free_mask] / w[free_mask].sum()
    w /= w.sum()
    return w


def _recursive_bisect(weights, items, cov):
    """Recursive bisection step for HRP."""
    if len(items) == 1:
        weights[items[0]] = 1.0
        return

    mid = len(items) // 2
    left, right = items[:mid], items[mid:]

    var_left = _cluster_variance(left, cov)
    var_right = _cluster_variance(right, cov)
    alpha = 1 - var_left / (var_left + var_right)

    left_weights = np.zeros(len(weights))
    right_weights = np.zeros(len(weights))
    _recursive_bisect(left_weights, left, cov)
    _recursive_bisect(right_weights, right, cov)

    for i in range(len(weights)):
        weights[i] += alpha * left_weights[i] + (1 - alpha) * right_weights[i]


def _cluster_variance(items, cov):
    """Compute inverse-variance weighted cluster variance."""
    sub_cov = cov[np.ix_(items, items)]
    inv_diag = 1 / np.diag(sub_cov)
    w = inv_diag / inv_diag.sum()
    return w @ sub_cov @ w


# ──────────────────────────────────────────────────────────
# Equal Risk Contribution (ERC)
# ──────────────────────────────────────────────────────────

def risk_budgeting_optimize(cov, budget=None):
    """
    Equal Risk Contribution portfolio.
    Each asset contributes equally to total portfolio risk.
    """
    n = cov.shape[0]
    if budget is None:
        budget = np.ones(n) / n

    def objective(w):
        port_risk = np.sqrt(w @ cov @ w)
        marginal = cov @ w / port_risk
        risk_contrib = w * marginal
        target_contrib = budget * port_risk
        return np.sum((risk_contrib - target_contrib) ** 2)

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0.001, None)] * n
    x0 = np.ones(n) / n

    result = minimize(objective, x0, method='SLSQP',
                      bounds=bounds, constraints=constraints)
    return result.x


# ──────────────────────────────────────────────────────────
# Covariance Estimation
# ──────────────────────────────────────────────────────────

def covariance_ledoit_wolf(returns):
    """Ledoit-Wolf shrinkage covariance estimator with PSD check."""
    lw = LedoitWolf().fit(returns.values)
    cov = lw.covariance_
    cov = ensure_psd(cov)
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


# ──────────────────────────────────────────────────────────
# Portfolio Metrics
# ──────────────────────────────────────────────────────────

def check_sector_constraints(weights_dict, max_sector_weight=None):
    """Verify no sector exceeds weight limit."""
    if max_sector_weight is None:
        max_sector_weight = MAX_WEIGHT_SECTOR

    violations = {}
    for sector, tickers in SECTOR_MAP.items():
        sector_weight = sum(weights_dict.get(t, 0) for t in tickers)
        if sector_weight > max_sector_weight:
            violations[sector] = sector_weight

    return violations


def risk_decomposition(weights, cov):
    """Compute marginal and percentage risk contributions."""
    port_vol = np.sqrt(weights @ cov @ weights)
    marginal = cov @ weights / port_vol
    risk_contrib = weights * marginal
    pct_contrib = risk_contrib / port_vol
    return {
        'marginal_risk': marginal,
        'risk_contrib': risk_contrib,
        'pct_risk_contrib': pct_contrib,
        'portfolio_vol': port_vol,
    }


def concentration_metrics(weights, cov=None):
    """Portfolio concentration analysis."""
    hhi = np.sum(weights ** 2)
    effective_n = 1 / hhi if hhi > 0 else len(weights)

    metrics = {
        'hhi': hhi,
        'effective_n': effective_n,
        'max_weight': np.max(weights),
    }

    if cov is not None:
        vols = np.sqrt(np.diag(cov))
        port_vol = np.sqrt(weights @ cov @ weights)
        metrics['diversification_ratio'] = np.sum(weights * vols) / port_vol

    return metrics


# ──────────────────────────────────────────────────────────
# Maximum Diversification Portfolio
# ──────────────────────────────────────────────────────────

def max_diversification_optimize(cov, max_weight=None, long_only=True):
    """
    Maximum Diversification Portfolio (Choueifaty & Coignard, 2008).
    Maximizes the Diversification Ratio: DR = w'σ / sqrt(w'Σw)
    where σ = vector of asset volatilities.

    Equivalent to minimizing portfolio variance using correlation matrix.
    """
    if max_weight is None:
        max_weight = MAX_WEIGHT_FACTOR

    n = cov.shape[0]
    vols = np.sqrt(np.diag(cov))
    corr = cov / np.outer(vols, vols)
    corr = ensure_psd(corr)
    corr = (corr + corr.T) / 2

    w = cp.Variable(n)
    risk = cp.quad_form(w, cp.psd_wrap(corr))

    constraints = [cp.sum(w) == 1, w <= max_weight]
    if long_only:
        constraints.append(w >= 0)

    prob = cp.Problem(cp.Minimize(risk), constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    weights = w.value
    if weights is None:
        weights = np.ones(n) / n
        logger.warning("Max diversification optimization failed — returning equal weight")
    else:
        weights = np.maximum(weights, 0)
        weights /= weights.sum()

    return weights


# ──────────────────────────────────────────────────────────
# Minimum Variance Portfolio
# ──────────────────────────────────────────────────────────

def min_variance_optimize(cov, max_weight=None, long_only=True):
    """
    Global Minimum Variance Portfolio.
    The only efficient frontier portfolio that doesn't require expected returns.
    """
    if max_weight is None:
        max_weight = MAX_WEIGHT_FACTOR

    n = cov.shape[0]
    cov = ensure_psd(cov)
    cov = (cov + cov.T) / 2

    w = cp.Variable(n)
    risk = cp.quad_form(w, cp.psd_wrap(cov))

    constraints = [cp.sum(w) == 1, w <= max_weight]
    if long_only:
        constraints.append(w >= 0)

    prob = cp.Problem(cp.Minimize(risk), constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    weights = w.value
    if weights is None:
        weights = np.ones(n) / n
        logger.warning("Min variance optimization failed — returning equal weight")
    else:
        weights = np.maximum(weights, 0)
        weights /= weights.sum()

    return weights


# ──────────────────────────────────────────────────────────
# Regime-Conditional Black-Litterman
# ──────────────────────────────────────────────────────────

def regime_conditional_bl(Sigma, w_mkt, P, Q_by_regime, Omega_by_regime,
                          regime_probs, tau=None, delta=None):
    """
    Regime-conditional Black-Litterman: blend posterior returns across regimes
    weighted by filtered regime probabilities.

    mu_blend = sum_k [ P(regime=k) * mu_BL(Q_k, Omega_k) ]

    Parameters
    ----------
    Sigma : np.ndarray — (N×N) covariance matrix
    w_mkt : np.ndarray — market-cap weights
    P : np.ndarray — (K×N) view matrix (same across regimes)
    Q_by_regime : dict — {regime_label: Q_vector}
    Omega_by_regime : dict — {regime_label: Omega_matrix}
    regime_probs : dict — {regime_label: probability}

    Returns
    -------
    mu_blend : np.ndarray — probability-weighted posterior returns
    Sigma_blend : np.ndarray — probability-weighted posterior covariance
    """
    if tau is None:
        tau = BL_TAU
    if delta is None:
        delta = RISK_AVERSION

    n = Sigma.shape[0]
    mu_blend = np.zeros(n)
    Sigma_blend = np.zeros((n, n))

    for regime, prob in regime_probs.items():
        if prob < 0.01:
            continue
        Q = Q_by_regime[regime]
        Omega = Omega_by_regime[regime]
        mu_k, Sigma_k = black_litterman(Sigma, w_mkt, P, Q, Omega, tau, delta)
        mu_blend += prob * mu_k
        # Covariance blend: E[Σ] + Var[μ] (law of total variance)
        Sigma_blend += prob * (Sigma_k + np.outer(mu_k, mu_k))

    # Subtract E[mu]^2 for law of total variance
    Sigma_blend -= np.outer(mu_blend, mu_blend)

    return mu_blend, Sigma_blend


# ──────────────────────────────────────────────────────────
# Robust Black-Litterman (uncertainty in Sigma)
# ──────────────────────────────────────────────────────────

def robust_bl_weights(mu_bl, Sigma_bl, epsilon=0.10,
                      max_weight=None, long_only=True):
    """
    Robust mean-variance optimization with ellipsoidal uncertainty set
    around expected returns. Accounts for estimation error in mu.

    max_w  mu'w - delta/2 * w'Σw - epsilon * ||Σ^{1/2} w||_2

    The penalty term epsilon * ||Σ^{1/2} w||_2 is the worst-case return
    reduction over the uncertainty set {mu : ||mu - mu_hat|| <= epsilon}.

    Parameters
    ----------
    epsilon : float — radius of uncertainty set (higher = more conservative)
    """
    if max_weight is None:
        max_weight = MAX_WEIGHT_FACTOR

    n = len(mu_bl)
    Sigma_bl = ensure_psd(Sigma_bl)
    Sigma_bl = (Sigma_bl + Sigma_bl.T) / 2

    # Cholesky for SOC constraint
    L = np.linalg.cholesky(Sigma_bl)

    w = cp.Variable(n)
    ret = mu_bl @ w
    risk = cp.quad_form(w, cp.psd_wrap(Sigma_bl))
    uncertainty_penalty = epsilon * cp.norm(L.T @ w, 2)

    objective = ret - 0.5 * RISK_AVERSION * risk - uncertainty_penalty

    constraints = [cp.sum(w) == 1, w <= max_weight]
    if long_only:
        constraints.append(w >= 0)

    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    weights = w.value
    if weights is None:
        weights = np.ones(n) / n
        logger.warning("Robust BL optimization failed — returning equal weight")
    else:
        weights = np.maximum(weights, 0)
        weights /= weights.sum()

    return weights


# ──────────────────────────────────────────────────────────
# Portfolio Comparison
# ──────────────────────────────────────────────────────────

def compare_allocations(returns_df, weights_dict, cov=None):
    """
    Compare multiple allocation strategies side by side.

    Parameters
    ----------
    returns_df : pd.DataFrame — asset returns (T × N)
    weights_dict : dict — {strategy_name: weight_array}
    cov : np.ndarray or None — covariance for risk metrics

    Returns
    -------
    pd.DataFrame — comparison table
    """
    results = []
    for name, w in weights_dict.items():
        port_ret = (returns_df * w).sum(axis=1)
        ann_ret = port_ret.mean() * 252
        ann_vol = port_ret.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        cum = (1 + port_ret).cumprod()
        max_dd = ((cum / cum.cummax()) - 1).min()

        row = {
            'strategy': name,
            'ann_return': ann_ret,
            'ann_volatility': ann_vol,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'max_weight': np.max(w),
            'effective_n': 1 / np.sum(w ** 2),
        }

        if cov is not None:
            row['diversification_ratio'] = np.sum(w * np.sqrt(np.diag(cov))) / np.sqrt(w @ cov @ w)

        results.append(row)

    return pd.DataFrame(results).set_index('strategy')
