"""
Risk metrics module — VaR, CVaR, EVT (GPD), Cornish-Fisher, Copula functions.
Implements §9 of claude.md (Copula & Tail Risk Modeling).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm, genpareto, chi2
from scipy.optimize import minimize_scalar
import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# VaR Methods
# ──────────────────────────────────────────────────────────

def var_historical(returns, alpha=0.05):
    """Historical VaR — empirical quantile."""
    return -np.percentile(returns.dropna(), alpha * 100)


def var_gaussian(returns, alpha=0.05):
    """Gaussian VaR — assumes normal distribution."""
    return -(returns.mean() + norm.ppf(alpha) * returns.std())


def var_cornish_fisher(returns, alpha=0.05):
    """
    Cornish-Fisher VaR with monotonicity guard.
    CRITICAL: Uses EXCESS kurtosis (pandas default).
    """
    z = norm.ppf(alpha)
    s = returns.skew()
    k = returns.kurtosis()  # excess kurtosis

    cf_z = (z
            + (z**2 - 1) / 6 * s
            + (z**3 - 3*z) / 24 * k
            - (2*z**3 - 5*z) / 36 * s**2)

    # Monotonicity guard
    if cf_z > z:
        logger.warning("Cornish-Fisher expansion reversed — falling back to Gaussian")
        cf_z = z

    return -(returns.mean() + cf_z * returns.std())


def var_student_t(returns, alpha=0.05):
    """Student's t VaR — MLE fit."""
    from scipy.stats import t as t_dist
    params = t_dist.fit(returns.dropna())
    return -t_dist.ppf(alpha, *params)


def var_evt_gpd(returns, alpha=0.01, threshold_pct=90):
    """
    EVT VaR using Generalised Pareto Distribution (Peaks Over Threshold).

    Parameters
    ----------
    returns : array-like — return series
    alpha : float — tail probability (0.01 = 99% VaR)
    threshold_pct : float — percentile for threshold selection
    """
    losses = -returns[returns < 0].values
    threshold = np.percentile(losses, threshold_pct)
    exceedances = losses[losses > threshold] - threshold

    if len(exceedances) < 15:
        logger.warning(f"Only {len(exceedances)} exceedances (<15). "
                       "EVT results may be unreliable.")

    xi, _, sigma = genpareto.fit(exceedances, floc=0)
    n = len(returns)
    n_u = len(exceedances)

    # EVT VaR
    evt_var = threshold + (sigma / xi) * ((n / n_u * alpha)**(-xi) - 1)

    # EVT ES (Expected Shortfall) — requires xi < 1 for finite ES
    if xi < 1.0:
        evt_es = evt_var / (1 - xi) + (sigma - xi * threshold) / (1 - xi)
    else:
        logger.warning(f"GPD shape xi={xi:.3f} >= 1: ES is infinite. Returning 2x VaR as proxy.")
        evt_es = 2.0 * evt_var

    return {
        'var': evt_var,
        'es': evt_es,
        'xi': xi,
        'sigma': sigma,
        'threshold': threshold,
        'n_exceedances': n_u,
    }


# ──────────────────────────────────────────────────────────
# CVaR / Expected Shortfall
# ──────────────────────────────────────────────────────────

def cvar_historical(returns, alpha=0.05):
    """Historical CVaR (Expected Shortfall)."""
    var = var_historical(returns, alpha)
    tail = returns[returns <= -var]
    return -tail.mean() if len(tail) > 0 else var


def portfolio_cvar(weights, scenario_matrix, alpha=0.05):
    """
    Correct portfolio CVaR from joint return scenarios.
    CRITICAL: Portfolio CVaR ≠ weighted sum of individual CVaRs.
    """
    portfolio_returns = scenario_matrix @ weights
    var_threshold = np.percentile(portfolio_returns, alpha * 100)
    tail = portfolio_returns[portfolio_returns <= var_threshold]
    return -tail.mean() if len(tail) > 0 else -var_threshold


# ──────────────────────────────────────────────────────────
# VaR Backtesting
# ──────────────────────────────────────────────────────────

def kupiec_pof_test(violations, alpha):
    """
    Kupiec Proportion of Failures test for VaR backtesting.
    H0: violation rate = alpha.
    """
    T = len(violations)
    n_viol = int(violations.sum())
    rate = n_viol / T

    if n_viol == 0 or n_viol == T:
        return {'violation_rate': rate, 'lr_stat': np.nan, 'p_value': 0.0,
                'zone': 'red' if rate > 2 * alpha else 'green'}

    lr = -2 * (
        (T - n_viol) * np.log(1 - alpha) + n_viol * np.log(alpha)
        - (T - n_viol) * np.log(1 - rate) - n_viol * np.log(rate)
    )
    p_value = 1 - chi2.cdf(lr, 1)

    # Traffic light zones
    if rate < 1.5 * alpha:
        zone = 'green'
    elif rate < 2 * alpha:
        zone = 'yellow'
    else:
        zone = 'red'

    return {
        'violation_rate': rate,
        'n_violations': n_viol,
        'lr_stat': lr,
        'p_value': p_value,
        'zone': zone,
    }


# ──────────────────────────────────────────────────────────
# Christoffersen Independence Test
# ──────────────────────────────────────────────────────────

def christoffersen_independence_test(violations):
    """
    Christoffersen (1998) test for independence of VaR violations.
    H0: violations are independent Bernoulli draws (no clustering).

    Parameters
    ----------
    violations : array-like of 0/1 — 1 = VaR violation

    Returns
    -------
    dict with lr_stat, p_value, independent (bool)
    """
    violations = np.asarray(violations, dtype=int)
    T = len(violations)

    # Transition counts
    n00 = n01 = n10 = n11 = 0
    for t in range(1, T):
        if violations[t - 1] == 0 and violations[t] == 0:
            n00 += 1
        elif violations[t - 1] == 0 and violations[t] == 1:
            n01 += 1
        elif violations[t - 1] == 1 and violations[t] == 0:
            n10 += 1
        else:
            n11 += 1

    # Transition probabilities
    p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    p = (n01 + n11) / (T - 1) if T > 1 else 0

    # Degenerate cases
    if p01 in (0, 1) or p11 in (0, 1) or p in (0, 1):
        return {'lr_stat': 0.0, 'p_value': 1.0, 'independent': True}

    # Log-likelihood ratio for independence
    lr_ind = -2 * (
        n00 * np.log(1 - p) + n01 * np.log(p)
        + n10 * np.log(1 - p) + n11 * np.log(p)
        - n00 * np.log(1 - p01) - n01 * np.log(p01)
        - n10 * np.log(1 - p11) - n11 * np.log(p11)
    )

    p_value = 1 - chi2.cdf(lr_ind, 1)

    return {
        'lr_stat': lr_ind,
        'p_value': p_value,
        'independent': p_value > 0.05,
        'p01': p01,
        'p11': p11,
    }


def conditional_coverage_test(violations, alpha):
    """
    Christoffersen conditional coverage test = Kupiec POF + Independence.
    Joint test that both the violation rate and independence hold.
    LR_cc = LR_pof + LR_ind ~ chi2(2)
    """
    pof = kupiec_pof_test(violations, alpha)
    ind = christoffersen_independence_test(violations)

    lr_pof = pof.get('lr_stat', 0)
    lr_ind = ind.get('lr_stat', 0)

    if np.isnan(lr_pof):
        lr_pof = 0
    if np.isnan(lr_ind):
        lr_ind = 0

    lr_cc = lr_pof + lr_ind
    p_value = 1 - chi2.cdf(lr_cc, 2)

    return {
        'lr_cc': lr_cc,
        'p_value': p_value,
        'pof_p_value': pof.get('p_value', np.nan),
        'ind_p_value': ind.get('p_value', np.nan),
        'pass': p_value > 0.05,
        'zone': pof.get('zone', 'unknown'),
    }


# ──────────────────────────────────────────────────────────
# Clayton Copula
# ──────────────────────────────────────────────────────────

def fit_clayton_copula(u, v):
    """
    Fit Clayton copula via maximum pseudo-likelihood.
    u, v must be pseudo-observations: ranks / (n+1).
    """
    def neg_loglik(theta):
        if theta <= 0:
            return 1e10
        n = len(u)
        ll = n * np.log(1 + theta)
        ll += -(1 + theta) * np.sum(np.log(u) + np.log(v))
        ll += -(2 + 1/theta) * np.sum(np.log(u**(-theta) + v**(-theta) - 1))
        return -ll

    result = minimize_scalar(neg_loglik, bounds=(0.01, 20), method='bounded')
    theta = result.x
    lambda_L = 2 ** (-1 / theta)  # Lower tail dependence coefficient

    return {
        'theta': theta,
        'lambda_lower': lambda_L,
        'converged': result.success if hasattr(result, 'success') else True,
    }


def to_pseudo_observations(x, y):
    """Convert raw data to pseudo-observations for copula fitting."""
    n = len(x)
    u = pd.Series(x).rank() / (n + 1)
    v = pd.Series(y).rank() / (n + 1)
    return u.values, v.values


# ──────────────────────────────────────────────────────────
# Performance Metrics
# ──────────────────────────────────────────────────────────

def compute_all_metrics(returns, rf=0.0, name='Strategy'):
    """Compute comprehensive portfolio performance metrics."""
    r = returns.dropna()
    if len(r) == 0:
        return {'name': name}

    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    excess_ret = ann_ret - rf
    sharpe = excess_ret / ann_vol if ann_vol > 0 else 0

    cum = (1 + r).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min()

    # Downside deviation: std of returns below rf (daily)
    rf_daily = rf / 252
    downside_returns = r[r < rf_daily] - rf_daily
    downside = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else ann_vol
    sortino = excess_ret / downside if downside > 0 else 0

    var95 = -np.percentile(r, 5)
    cvar95 = -r[r <= -var95].mean() if (r <= -var95).any() else var95

    calmar = excess_ret / abs(max_dd) if max_dd != 0 else 0

    # Maximum drawdown duration (trading days)
    dd_duration = 0
    max_dd_duration = 0
    for i in range(len(drawdown)):
        if drawdown.iloc[i] < 0:
            dd_duration += 1
            max_dd_duration = max(max_dd_duration, dd_duration)
        else:
            dd_duration = 0

    right_tail = np.percentile(r, 95)
    left_tail = abs(np.percentile(r, 5))
    tail_ratio = right_tail / left_tail if left_tail > 0 else 0

    # Omega ratio (threshold = rf daily)
    gains = r[r > rf_daily] - rf_daily
    losses = rf_daily - r[r <= rf_daily]
    omega = gains.sum() / losses.sum() if losses.sum() > 0 else np.inf

    # Ulcer Index (RMS of drawdown)
    ulcer_index = np.sqrt(np.mean(drawdown ** 2))
    ulcer_perf = excess_ret / ulcer_index if ulcer_index > 0 else 0

    # Gain-to-pain ratio
    gain_to_pain = r.sum() / abs(r[r < 0].sum()) if (r < 0).any() else np.inf

    return {
        'name': name,
        'ann_return': ann_ret,
        'ann_volatility': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_dd,
        'max_dd_duration': max_dd_duration,
        'calmar': calmar,
        'omega': omega,
        'ulcer_index': ulcer_index,
        'ulcer_perf_index': ulcer_perf,
        'gain_to_pain': gain_to_pain,
        'var_95': var95,
        'cvar_95': cvar95,
        'skewness': r.skew(),
        'excess_kurtosis': r.kurtosis(),
        'hit_rate': (r > 0).mean(),
        'tail_ratio': tail_ratio,
        'n_observations': len(r),
    }


# ──────────────────────────────────────────────────────────
# Parametric CVaR
# ──────────────────────────────────────────────────────────

def cvar_gaussian(returns, alpha=0.05):
    """
    Gaussian CVaR (Expected Shortfall) — closed-form.
    ES_alpha = -mu + sigma * phi(z_alpha) / alpha
    where phi = standard normal PDF, z_alpha = Phi^{-1}(alpha).
    """
    mu = returns.mean()
    sigma = returns.std()
    z = norm.ppf(alpha)
    return -mu + sigma * norm.pdf(z) / alpha


def cvar_cornish_fisher(returns, alpha=0.05):
    """
    Cornish-Fisher CVaR using modified quantile expansion.
    Accounts for skewness and kurtosis in the tail expectation.
    """
    z = norm.ppf(alpha)
    s = returns.skew()
    k = returns.kurtosis()

    cf_z = (z
            + (z**2 - 1) / 6 * s
            + (z**3 - 3*z) / 24 * k
            - (2*z**3 - 5*z) / 36 * s**2)

    # Monotonicity guard
    if cf_z > z:
        cf_z = z

    # Approximate ES via numerical integration of CF expansion
    # Use the Gaussian ES formula with the CF-adjusted quantile
    mu = returns.mean()
    sigma = returns.std()
    return -mu + sigma * norm.pdf(cf_z) / alpha


# ──────────────────────────────────────────────────────────
# Component & Marginal VaR
# ──────────────────────────────────────────────────────────

def component_var(weights, cov, alpha=0.05):
    """
    Component VaR decomposition — Euler allocation of portfolio VaR.
    Sum of component VaRs = total portfolio VaR (linear homogeneity).

    Parameters
    ----------
    weights : np.ndarray — (N,) portfolio weights
    cov : np.ndarray — (N×N) covariance matrix
    alpha : float — VaR confidence level

    Returns
    -------
    dict with total_var, marginal_var, component_var, pct_contribution
    """
    port_vol = np.sqrt(weights @ cov @ weights)
    z = norm.ppf(alpha)
    total_var = -z * port_vol  # Parametric VaR (assuming zero mean)

    # Marginal VaR: dVaR/dw_i = -z * (Sigma @ w) / sigma_p
    marginal = -z * (cov @ weights) / port_vol

    # Component VaR: CVaR_i = w_i * MVaR_i
    comp_var = weights * marginal

    # Percentage contribution
    pct = comp_var / total_var if total_var > 0 else np.zeros_like(comp_var)

    return {
        'total_var': total_var,
        'marginal_var': marginal,
        'component_var': comp_var,
        'pct_contribution': pct,
        'portfolio_vol': port_vol,
    }


def incremental_var(weights, cov, new_weight_idx, delta_w, alpha=0.05):
    """
    Incremental VaR — change in portfolio VaR from adjusting position i.
    IVaR_i ≈ MVaR_i * delta_w_i (first-order approximation).
    """
    cv = component_var(weights, cov, alpha)
    return cv['marginal_var'][new_weight_idx] * delta_w


# ──────────────────────────────────────────────────────────
# Rolling Risk Metrics
# ──────────────────────────────────────────────────────────

def rolling_var_cvar(returns, window=252, alpha=0.05, method='historical'):
    """
    Time-varying VaR and CVaR using rolling window.

    Parameters
    ----------
    returns : pd.Series
    window : int — rolling window size
    alpha : float — tail probability
    method : str — 'historical', 'gaussian', or 'cornish_fisher'

    Returns
    -------
    pd.DataFrame with columns: VaR, CVaR
    """
    var_series = pd.Series(index=returns.index, dtype=float, name='VaR')
    cvar_series = pd.Series(index=returns.index, dtype=float, name='CVaR')

    for i in range(window, len(returns)):
        r_window = returns.iloc[i - window:i]

        if method == 'historical':
            v = var_historical(r_window, alpha)
            cv = cvar_historical(r_window, alpha)
        elif method == 'gaussian':
            v = var_gaussian(r_window, alpha)
            cv = cvar_gaussian(r_window, alpha)
        elif method == 'cornish_fisher':
            v = var_cornish_fisher(r_window, alpha)
            cv = cvar_cornish_fisher(r_window, alpha)
        else:
            raise ValueError(f"Unknown method: {method}")

        var_series.iloc[i] = v
        cvar_series.iloc[i] = cv

    return pd.DataFrame({'VaR': var_series, 'CVaR': cvar_series})


# ──────────────────────────────────────────────────────────
# EVT Diagnostics
# ──────────────────────────────────────────────────────────

def mean_excess_function(losses, n_thresholds=50):
    """
    Mean Excess Function e(u) = E[X - u | X > u].
    Linearity in u supports GPD tail assumption.
    Used for threshold selection in EVT.

    Returns
    -------
    pd.DataFrame with columns: threshold, mean_excess, n_exceedances
    """
    sorted_losses = np.sort(losses)
    thresholds = np.linspace(
        np.percentile(sorted_losses, 70),
        np.percentile(sorted_losses, 99),
        n_thresholds
    )

    results = []
    for u in thresholds:
        exceedances = sorted_losses[sorted_losses > u]
        if len(exceedances) < 5:
            break
        results.append({
            'threshold': u,
            'mean_excess': (exceedances - u).mean(),
            'n_exceedances': len(exceedances),
        })

    return pd.DataFrame(results)


def hill_estimator(losses, k_range=None):
    """
    Hill (1975) tail index estimator for heavy-tailed distributions.
    alpha_Hill = [1/k * sum_{i=1}^{k} ln(X_{(n-i+1)}) - ln(X_{(n-k)})]^{-1}

    Parameters
    ----------
    losses : array-like — positive losses (exceedances)
    k_range : tuple — (k_min, k_max) for Hill plot

    Returns
    -------
    pd.DataFrame with columns: k, alpha_hill (tail index)
    """
    sorted_losses = np.sort(losses)[::-1]  # Descending
    n = len(sorted_losses)

    if k_range is None:
        k_range = (10, min(n // 2, 500))

    results = []
    for k in range(k_range[0], k_range[1]):
        log_excess = np.log(sorted_losses[:k]) - np.log(sorted_losses[k])
        gamma = log_excess.mean()
        if gamma > 0:
            results.append({
                'k': k,
                'alpha_hill': 1 / gamma,
                'gamma_hill': gamma,
            })

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────
# Improved Copula
# ──────────────────────────────────────────────────────────

def fit_clayton_mle(u, v, init_from_tau=True):
    """
    Fit Clayton copula with Kendall's tau-based initialization.
    Tau = theta / (theta + 2) → theta_init = 2*tau / (1 - tau).
    Better convergence than blind grid search.

    Parameters
    ----------
    u, v : array-like — pseudo-observations in (0,1)
    init_from_tau : bool — use Kendall's tau for initialization
    """
    from scipy.stats import kendalltau

    # Kendall's tau initialization
    tau_val = kendalltau(u, v).statistic
    theta_init = max(0.1, 2 * tau_val / (1 - tau_val)) if tau_val > 0 else 1.0

    def neg_loglik(theta):
        if theta <= 0:
            return 1e10
        n = len(u)
        ll = n * np.log(1 + theta)
        ll += -(1 + theta) * np.sum(np.log(u) + np.log(v))
        ll += -(2 + 1/theta) * np.sum(np.log(u**(-theta) + v**(-theta) - 1))
        return -ll

    result = minimize_scalar(neg_loglik, bounds=(0.01, 30), method='bounded')
    theta = result.x
    lambda_L = 2 ** (-1 / theta)

    # Goodness-of-fit: compare empirical vs model Kendall's tau
    model_tau = theta / (theta + 2)
    tau_error = abs(model_tau - tau_val)

    return {
        'theta': theta,
        'lambda_lower': lambda_L,
        'kendall_tau_empirical': tau_val,
        'kendall_tau_model': model_tau,
        'tau_fit_error': tau_error,
        'converged': result.success if hasattr(result, 'success') else True,
    }


def benchmark_relative_metrics(returns, benchmark_returns, rf=0.0, name='Strategy'):
    """
    Compute benchmark-relative performance metrics.
    Standard institutional reporting (Information Ratio, Tracking Error, Beta, Alpha).

    Parameters
    ----------
    returns : pd.Series — strategy returns
    benchmark_returns : pd.Series — benchmark returns (same frequency)
    rf : float — annualized risk-free rate
    """
    common = returns.index.intersection(benchmark_returns.index)
    r = returns.loc[common].dropna()
    b = benchmark_returns.loc[common].dropna()
    common2 = r.index.intersection(b.index)
    r, b = r.loc[common2], b.loc[common2]

    if len(r) < 30:
        return {'name': name, 'error': 'insufficient_data'}

    excess = r - b
    te = excess.std() * np.sqrt(252)
    ir = excess.mean() * 252 / te if te > 0 else 0

    # CAPM beta and alpha
    cov_rb = np.cov(r, b)[0, 1]
    var_b = np.var(b, ddof=1)
    beta = cov_rb / var_b if var_b > 0 else 1.0
    alpha = (r.mean() - rf/252 - beta * (b.mean() - rf/252)) * 252

    # Up/Down capture ratios
    up_mask = b > 0
    down_mask = b < 0
    up_capture = r[up_mask].mean() / b[up_mask].mean() if up_mask.sum() > 10 else np.nan
    down_capture = r[down_mask].mean() / b[down_mask].mean() if down_mask.sum() > 10 else np.nan

    # Active share approximation (correlation-based)
    active_corr = np.corrcoef(r, b)[0, 1]

    return {
        'name': name,
        'information_ratio': ir,
        'tracking_error': te,
        'beta': beta,
        'alpha': alpha,
        'up_capture': up_capture,
        'down_capture': down_capture,
        'capture_ratio': up_capture / down_capture if down_capture and down_capture != 0 else np.nan,
        'correlation': active_corr,
        'n_observations': len(r),
    }
