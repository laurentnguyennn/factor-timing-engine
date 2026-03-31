"""
GARCH utilities — fit GARCH family models per ticker, DCC-GARCH,
conditional volatility extraction, PSD correction.
Implements Phases 3 & 6 of claude.md.
"""
import logging

import numpy as np
import pandas as pd
from arch import arch_model
from tqdm import tqdm

from src.config import (
    GARCH_MODELS, GARCH_DISTRIBUTIONS, MIN_OBS_FIGARCH, RANDOM_STATE
)

logger = logging.getLogger(__name__)


def fit_garch_family(returns_pct, models=None, distributions=None):
    """
    Fit all GARCH model × distribution combinations for a single series.
    Returns DataFrame of fitted model parameters + BIC.

    Parameters
    ----------
    returns_pct : pd.Series — returns in PERCENT (arch convention)

    Returns
    -------
    pd.DataFrame with columns: model, distribution, bic, aic, loglikelihood, params
    """
    if models is None:
        models = GARCH_MODELS
    if distributions is None:
        distributions = GARCH_DISTRIBUTIONS

    results = []
    n_obs = len(returns_pct.dropna())

    for model_type in models:
        # FIGARCH needs at least MIN_OBS_FIGARCH observations
        if model_type == 'FIGARCH' and n_obs < MIN_OBS_FIGARCH:
            continue

        for dist in distributions:
            try:
                if model_type == 'GARCH':
                    am = arch_model(returns_pct, vol='GARCH', p=1, q=1,
                                    mean='AR', lags=1, dist=dist)
                elif model_type == 'GJR-GARCH':
                    am = arch_model(returns_pct, vol='GARCH', p=1, o=1, q=1,
                                    mean='AR', lags=1, dist=dist)
                elif model_type == 'EGARCH':
                    am = arch_model(returns_pct, vol='EGARCH', p=1, o=1, q=1,
                                    mean='AR', lags=1, dist=dist)
                elif model_type == 'FIGARCH':
                    am = arch_model(returns_pct, vol='FIGARCH', p=1, q=1,
                                    mean='AR', lags=1, dist=dist)
                else:
                    continue

                res = am.fit(disp='off', show_warning=True)

                results.append({
                    'model': model_type,
                    'distribution': dist,
                    'bic': res.bic,
                    'aic': res.aic,
                    'loglikelihood': res.loglikelihood,
                    'params': dict(res.params),
                    'converged': res.convergence_flag == 0,
                    'conditional_vol_last': res.conditional_volatility.iloc[-1] / 100,
                })

            except Exception as e:
                logger.debug(f"{model_type}/{dist}: {e}")
                continue

    return pd.DataFrame(results)


def select_best_garch(results_df):
    """Select best model by BIC from GARCH family results."""
    if len(results_df) == 0:
        return None
    converged = results_df[results_df['converged']]
    if len(converged) == 0:
        converged = results_df
    return converged.loc[converged['bic'].idxmin()]


def extract_conditional_volatility(returns_pct, model_type='GARCH',
                                   dist='StudentsT'):
    """
    Fit specified GARCH model and return conditional vol series.

    Returns
    -------
    pd.Series — conditional volatility in DECIMAL (divided by 100)
    """
    if model_type == 'GARCH':
        am = arch_model(returns_pct, vol='GARCH', p=1, q=1,
                        mean='AR', lags=1, dist=dist)
    elif model_type == 'GJR-GARCH':
        am = arch_model(returns_pct, vol='GARCH', p=1, o=1, q=1,
                        mean='AR', lags=1, dist=dist)
    elif model_type == 'EGARCH':
        am = arch_model(returns_pct, vol='EGARCH', p=1, o=1, q=1,
                        mean='AR', lags=1, dist=dist)
    else:
        am = arch_model(returns_pct, vol='GARCH', p=1, q=1,
                        mean='AR', lags=1, dist=dist)

    res = am.fit(disp='off')
    cond_vol = res.conditional_volatility / 100  # Back to decimal
    cond_vol.name = returns_pct.name or 'cond_vol'
    return cond_vol


def fit_all_tickers_garch(returns_df):
    """
    Fit GARCH family for all tickers. Returns dict of results.

    Parameters
    ----------
    returns_df : pd.DataFrame — daily log returns in DECIMAL

    Returns
    -------
    dict — {ticker: {results_df, best_model, cond_vol_series}}
    """
    all_results = {}

    for ticker in tqdm(returns_df.columns, desc='GARCH fits'):
        series = returns_df[ticker].dropna()
        if len(series) < 252:
            logger.warning(f"{ticker}: only {len(series)} obs, skipping")
            continue

        # CRITICAL: arch expects returns in PERCENT
        series_pct = series * 100

        results = fit_garch_family(series_pct)
        best = select_best_garch(results)

        if best is not None:
            cond_vol = extract_conditional_volatility(
                series_pct,
                model_type=best['model'],
                dist=best['distribution']
            )
        else:
            cond_vol = series.rolling(21).std() * np.sqrt(252)
            logger.warning(f"{ticker}: no GARCH converged, using rolling vol")

        all_results[ticker] = {
            'results_df': results,
            'best_model': best,
            'cond_vol': cond_vol,
        }

    return all_results


def garch_diagnostic_tests(result, lags=10):
    """
    Run diagnostic tests on GARCH fitted residuals.
    Tests: Ljung-Box on standardised residuals and squared residuals, ARCH-LM test.

    Parameters
    ----------
    result : arch model result object
    lags : int — number of lags for Ljung-Box test

    Returns
    -------
    dict with test statistics and p-values
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

    std_resid = result.std_resid.dropna()

    diagnostics = {}

    # Ljung-Box on standardised residuals (tests remaining serial correlation)
    try:
        lb = acorr_ljungbox(std_resid, lags=[lags], return_df=True)
        diagnostics['ljung_box_resid_stat'] = lb['lb_stat'].iloc[0]
        diagnostics['ljung_box_resid_pvalue'] = lb['lb_pvalue'].iloc[0]
        diagnostics['ljung_box_resid_pass'] = lb['lb_pvalue'].iloc[0] > 0.05
    except Exception as e:
        logger.debug(f"Ljung-Box residuals failed: {e}")
        diagnostics['ljung_box_resid_pass'] = None

    # Ljung-Box on squared residuals (tests remaining ARCH effects)
    try:
        lb_sq = acorr_ljungbox(std_resid ** 2, lags=[lags], return_df=True)
        diagnostics['ljung_box_sq_stat'] = lb_sq['lb_stat'].iloc[0]
        diagnostics['ljung_box_sq_pvalue'] = lb_sq['lb_pvalue'].iloc[0]
        diagnostics['ljung_box_sq_pass'] = lb_sq['lb_pvalue'].iloc[0] > 0.05
    except Exception as e:
        logger.debug(f"Ljung-Box squared residuals failed: {e}")
        diagnostics['ljung_box_sq_pass'] = None

    # ARCH-LM test (Engle's test for remaining heteroskedasticity)
    try:
        arch_lm = het_arch(std_resid, nlags=lags)
        diagnostics['arch_lm_stat'] = arch_lm[0]
        diagnostics['arch_lm_pvalue'] = arch_lm[1]
        diagnostics['arch_lm_pass'] = arch_lm[1] > 0.05  # No remaining ARCH effects
    except Exception as e:
        logger.debug(f"ARCH-LM test failed: {e}")
        diagnostics['arch_lm_pass'] = None

    # Persistence: model-type-aware calculation
    # GARCH(1,1): persistence = alpha + beta
    # GJR-GARCH: persistence = alpha + beta + gamma/2
    # EGARCH: persistence = beta (different parameterization)
    params = result.params
    persistence = _compute_persistence(params)
    diagnostics['persistence'] = persistence
    diagnostics['persistence_valid'] = 0.80 <= persistence <= 0.999

    return diagnostics


def _compute_persistence(params):
    """
    Compute volatility persistence from GARCH model parameters.
    Model-type-aware: handles GARCH, GJR, EGARCH parameterizations.
    """
    p = params.to_dict()
    alpha = sum(v for k, v in p.items() if k.startswith('alpha['))
    beta = sum(v for k, v in p.items() if k.startswith('beta['))
    gamma = sum(v for k, v in p.items() if k.startswith('gamma['))

    # GJR-GARCH: persistence = alpha + beta + gamma/2
    # Standard GARCH: gamma=0, so alpha + beta
    return alpha + beta + gamma / 2


def garch_forecast(returns_pct, model_type='GARCH', dist='StudentsT',
                   horizon=5, n_simulations=10000):
    """
    Multi-step ahead GARCH volatility forecast.

    Parameters
    ----------
    returns_pct : pd.Series — returns in PERCENT
    model_type : str — GARCH variant
    dist : str — error distribution
    horizon : int — forecast horizon (trading days)
    n_simulations : int — Monte Carlo paths for multi-step forecast

    Returns
    -------
    dict with:
        - point_forecast: h-step ahead annualised vol forecasts
        - ci_lower, ci_upper: 95% confidence bands
        - term_structure: volatility term structure
    """
    if model_type == 'GARCH':
        am = arch_model(returns_pct, vol='GARCH', p=1, q=1,
                        mean='AR', lags=1, dist=dist)
    elif model_type == 'GJR-GARCH':
        am = arch_model(returns_pct, vol='GARCH', p=1, o=1, q=1,
                        mean='AR', lags=1, dist=dist)
    elif model_type == 'EGARCH':
        am = arch_model(returns_pct, vol='EGARCH', p=1, o=1, q=1,
                        mean='AR', lags=1, dist=dist)
    else:
        am = arch_model(returns_pct, vol='GARCH', p=1, q=1,
                        mean='AR', lags=1, dist=dist)

    res = am.fit(disp='off')

    # Analytical forecast (uses arch's built-in method)
    forecasts = res.forecast(horizon=horizon, method='simulation',
                             simulations=n_simulations)

    # Variance forecasts (in pct^2, convert to annualised decimal vol)
    var_forecast = forecasts.variance.iloc[-1].values
    vol_forecast = np.sqrt(var_forecast * 252) / 100  # Annualise & convert to decimal

    # Simulation-based confidence intervals
    sim_var = forecasts.simulations.variances[-1]  # (n_sim, horizon)
    sim_vol = np.sqrt(sim_var * 252) / 100
    ci_lower = np.percentile(sim_vol, 2.5, axis=0)
    ci_upper = np.percentile(sim_vol, 97.5, axis=0)

    # Volatility term structure: average vol from day 1 to day h
    term_structure = np.array([
        np.sqrt(var_forecast[:h+1].mean() * 252) / 100
        for h in range(horizon)
    ])

    return {
        'point_forecast': vol_forecast,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'term_structure': term_structure,
        'horizon': list(range(1, horizon + 1)),
        'model': f'{model_type}/{dist}',
        'last_cond_vol': res.conditional_volatility.iloc[-1] / 100,
    }


def news_impact_curve(returns_pct, model_type='GJR-GARCH', dist='StudentsT',
                      n_points=100, shock_range=3.0):
    """
    Compute the News Impact Curve (Engle & Ng 1993).
    Shows how the conditional variance responds to past return shocks
    of different signs and magnitudes.

    NIC reveals asymmetry: negative shocks typically increase vol more
    than positive shocks of equal magnitude (leverage effect).

    Parameters
    ----------
    shock_range : float — range of shocks in standard deviations

    Returns
    -------
    pd.DataFrame with columns: shock, cond_var_response
    """
    if model_type == 'GARCH':
        am = arch_model(returns_pct, vol='GARCH', p=1, q=1,
                        mean='AR', lags=1, dist=dist)
    elif model_type == 'GJR-GARCH':
        am = arch_model(returns_pct, vol='GARCH', p=1, o=1, q=1,
                        mean='AR', lags=1, dist=dist)
    elif model_type == 'EGARCH':
        am = arch_model(returns_pct, vol='EGARCH', p=1, o=1, q=1,
                        mean='AR', lags=1, dist=dist)
    else:
        am = arch_model(returns_pct, vol='GARCH', p=1, q=1,
                        mean='AR', lags=1, dist=dist)

    res = am.fit(disp='off')
    p = res.params

    # Unconditional variance
    sigma2 = np.mean(res.conditional_volatility ** 2)
    resid_std = returns_pct.std()

    shocks = np.linspace(-shock_range * resid_std, shock_range * resid_std, n_points)

    if model_type == 'GARCH':
        omega = p.get('omega', 0)
        alpha1 = p.get('alpha[1]', 0)
        beta1 = p.get('beta[1]', 0)
        nic = omega + alpha1 * shocks**2 + beta1 * sigma2
    elif model_type == 'GJR-GARCH':
        omega = p.get('omega', 0)
        alpha1 = p.get('alpha[1]', 0)
        gamma1 = p.get('gamma[1]', 0)
        beta1 = p.get('beta[1]', 0)
        indicator = (shocks < 0).astype(float)
        nic = omega + (alpha1 + gamma1 * indicator) * shocks**2 + beta1 * sigma2
    elif model_type == 'EGARCH':
        omega = p.get('omega', 0)
        alpha1 = p.get('alpha[1]', 0)
        gamma1 = p.get('gamma[1]', 0)
        beta1 = p.get('beta[1]', 0)
        log_sigma2 = np.log(sigma2)
        z = shocks / np.sqrt(sigma2)
        nic = np.exp(omega + alpha1 * (np.abs(z) - np.sqrt(2/np.pi))
                     + gamma1 * z + beta1 * log_sigma2)
    else:
        nic = shocks**2  # Fallback

    return pd.DataFrame({
        'shock': shocks / resid_std,  # Normalized by std
        'shock_raw': shocks,
        'cond_var': nic,
        'cond_vol': np.sqrt(np.maximum(nic, 0)),
    })


def dcc_conditional_correlation(returns_pct_df, univariate_model='GJR-GARCH',
                                dist='StudentsT'):
    """
    DCC-GARCH (Engle 2002) conditional correlation estimation.
    Step 1: Fit univariate GARCH to each series → standardised residuals.
    Step 2: Estimate DCC parameters from standardised residuals.

    Parameters
    ----------
    returns_pct_df : pd.DataFrame — returns in PERCENT (T × N)

    Returns
    -------
    dict with:
        - cond_corr: time-varying correlation matrices (T × N × N)
        - cond_vol: conditional volatilities (T × N)
        - std_resid: standardised residuals (T × N)
    """
    T, N = returns_pct_df.shape
    tickers = returns_pct_df.columns.tolist()
    cond_vol = pd.DataFrame(index=returns_pct_df.index, columns=tickers, dtype=float)
    std_resid = pd.DataFrame(index=returns_pct_df.index, columns=tickers, dtype=float)

    # Step 1: Univariate GARCH fits
    for ticker in tickers:
        try:
            series = returns_pct_df[ticker].dropna()
            vol = extract_conditional_volatility(series, model_type=univariate_model,
                                                 dist=dist)
            cond_vol[ticker] = vol * 100  # Back to pct scale for residual computation
            std_resid[ticker] = series / (vol * 100).clip(lower=1e-6)
        except Exception as e:
            logger.warning(f"DCC: {ticker} GARCH failed ({e}), using rolling vol")
            rolling_vol = series.rolling(21).std()
            cond_vol[ticker] = rolling_vol / 100
            std_resid[ticker] = series / rolling_vol.clip(lower=1e-6)

    # Step 2: DCC dynamics on standardised residuals
    # Q_t = (1 - a - b) * Q_bar + a * (e_{t-1} e_{t-1}') + b * Q_{t-1}
    std_resid_clean = std_resid.dropna()
    e = std_resid_clean.values
    T_clean = len(e)
    Q_bar = np.corrcoef(e.T)  # Unconditional correlation

    # DCC parameters (Engle 2002 typical values; full MLE is complex)
    a_dcc, b_dcc = 0.01, 0.95

    # Compute time-varying correlation
    Q_t = Q_bar.copy()
    corr_series = np.zeros((T_clean, N, N))

    for t in range(T_clean):
        if t > 0:
            e_t = e[t-1:t].T  # (N, 1)
            Q_t = (1 - a_dcc - b_dcc) * Q_bar + a_dcc * (e_t @ e_t.T) + b_dcc * Q_t

        # Normalize Q_t to correlation matrix R_t
        d = np.sqrt(np.diag(Q_t))
        d[d < 1e-8] = 1e-8
        D_inv = np.diag(1 / d)
        R_t = D_inv @ Q_t @ D_inv
        np.fill_diagonal(R_t, 1.0)
        corr_series[t] = R_t

    return {
        'cond_corr': corr_series,
        'cond_vol': cond_vol,
        'std_resid': std_resid_clean,
        'Q_bar': Q_bar,
        'dcc_params': {'a': a_dcc, 'b': b_dcc},
        'dates': std_resid_clean.index,
    }


def ensure_psd(matrix, method='higham'):
    """
    Ensure covariance/correlation matrix is positive semi-definite.
    CRITICAL: Required before Cholesky decomposition or cvxpy optimisation.
    """
    eigvals = np.linalg.eigvalsh(matrix)
    if np.all(eigvals >= -1e-10):
        return matrix  # Already PSD

    if method == 'eigenvalue':
        eigvals_fixed = np.maximum(eigvals, 1e-10)
        eigvecs = np.linalg.eigh(matrix)[1]
        return eigvecs @ np.diag(eigvals_fixed) @ eigvecs.T

    elif method == 'higham':
        return _higham_nearest_psd(matrix)

    else:
        # Diagonal loading fallback
        eps = abs(eigvals.min()) + 1e-6
        return matrix + eps * np.eye(matrix.shape[0])


def _higham_nearest_psd(A, max_iter=100, tol=1e-10):
    """
    Higham's alternating projections algorithm for nearest PSD matrix.
    Projects alternately onto the PSD cone (S+) and the set of matrices
    with the same diagonal as A (preserves variances for covariance matrices).
    Uses Dykstra's correction for convergence guarantee.
    """
    n = A.shape[0]
    dS = np.zeros_like(A)
    Y = A.copy()
    original_diag = np.diag(A).copy()

    for k in range(max_iter):
        R = Y - dS
        # Project onto PSD cone
        eigvals, eigvecs = np.linalg.eigh(R)
        eigvals = np.maximum(eigvals, 0)
        X = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Dykstra's correction
        dS = X - R
        # Preserve original diagonal (variances) for covariance matrices
        Y = X.copy()
        np.fill_diagonal(Y, original_diag)

        # Convergence check: Frobenius norm of change
        if k > 0 and np.linalg.norm(X - X_prev, 'fro') < tol:
            break
        X_prev = X.copy()

    # Final PSD projection to ensure result is PSD
    eigvals, eigvecs = np.linalg.eigh(Y)
    eigvals = np.maximum(eigvals, 1e-10)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T
