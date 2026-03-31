"""
Regime model — HMM wrapper with expanding-window estimation,
filtered probabilities (no look-ahead), and label consistency.
Implements Phase 4 of claude.md.
"""
import os
import hashlib
import logging

import numpy as np
import pandas as pd
import joblib
from hmmlearn import hmm
from sklearn.decomposition import PCA

from src.config import (
    INTERIM_DIR, HMM_N_STATES, HMM_N_RESTARTS, HMM_MIN_WINDOW, RANDOM_STATE,
)

logger = logging.getLogger(__name__)


def expanding_standardise(df, min_window=24):
    """
    Standardise each column using expanding window (no look-ahead).
    z_{i,t} = (x_{i,t} - mean(x_{i,1:t})) / std(x_{i,1:t})
    """
    z = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for col in df.columns:
        expanding_mean = df[col].expanding(min_periods=min_window).mean()
        expanding_std = df[col].expanding(min_periods=min_window).std()
        z[col] = (df[col] - expanding_mean) / expanding_std.clip(lower=1e-8)
    return z


def expanding_pca_composite(z_standardised, min_window=24):
    """
    Compute composite index via expanding-window PCA (first principal component).
    No look-ahead: PCA at time t uses only data up to t.
    """
    T = len(z_standardised)
    composite = np.full(T, np.nan)

    for t in range(min_window, T):
        data_up_to_t = z_standardised.iloc[:t + 1].dropna()
        if len(data_up_to_t) < min_window:
            continue
        pca = PCA(n_components=1)
        pca.fit(data_up_to_t.values)
        composite[t] = pca.transform(
            z_standardised.iloc[t:t + 1].values.reshape(1, -1)
        )[0, 0]

    # Sign convention: positive = good macro
    # Enforce: if 1st PC loads negatively on majority of indicators, flip sign.
    # Most macro indicators (IP, CLI, etc.) are "good when positive", so the
    # composite should correlate positively with the mean of all standardised indicators.
    result = pd.Series(composite, index=z_standardised.index, name='composite_z')
    valid_mask = ~np.isnan(composite)
    if valid_mask.sum() > 0:
        mean_z = z_standardised.mean(axis=1)
        corr = np.corrcoef(result[valid_mask], mean_z[valid_mask])[0, 1]
        if corr < 0:
            result = -result
    return result


def fit_hmm_with_restarts(data, n_states=None, n_restarts=None):
    """
    Fit Gaussian HMM with multiple random restarts to escape local optima.

    Parameters
    ----------
    data : np.ndarray — (T, 1) observations
    n_states : int — number of hidden states
    n_restarts : int — number of random initialisations

    Returns
    -------
    best_model : GaussianHMM — model with highest log-likelihood
    """
    if n_states is None:
        n_states = HMM_N_STATES
    if n_restarts is None:
        n_restarts = HMM_N_RESTARTS

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    best_model = None
    best_score = -np.inf
    scores = []

    for seed in range(n_restarts):
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=500,
            tol=1e-6,
            random_state=seed,
            init_params="stmc",
        )
        try:
            model.fit(data)
            score = model.score(data)
            scores.append(score)
            if score > best_score:
                best_score = score
                best_model = model
        except (ValueError, np.linalg.LinAlgError):
            continue

    if best_model is None:
        raise RuntimeError(f"HMM failed to converge after {n_restarts} restarts")

    if not best_model.monitor_.converged:
        logger.warning("Best HMM did not converge — increase n_iter")

    # Log consistency of top restarts
    scores_sorted = sorted(scores, reverse=True)
    if len(scores_sorted) >= 3:
        spread = scores_sorted[0] - scores_sorted[2]
        logger.info(f"HMM: best LL={best_score:.2f}, "
                    f"top-3 spread={spread:.2f}, "
                    f"{len(scores)}/{n_restarts} restarts converged")

    return best_model


def sort_states_by_mean(model):
    """
    Re-order HMM states so that:
    State 0 = highest mean (Expansion)
    State K-1 = lowest mean (Crisis)

    Returns mapping: old_state → new_state
    """
    means = model.means_.flatten()
    order = np.argsort(means)[::-1]  # Descending by mean
    mapping = {old: new for new, old in enumerate(order)}
    return mapping, order


def label_states(mapping, n_states=3):
    """Map state indices to human-readable labels."""
    labels = {0: 'Expansion', 1: 'Slowdown', 2: 'Crisis'}
    if n_states == 2:
        labels = {0: 'Expansion', 1: 'Crisis'}
    return labels


def filtered_probabilities(model, observations):
    """
    Compute P(S_t | Z_1, ..., Z_t) using forward algorithm only.
    This is the ONLY correct method for live/backtest regime detection.

    NEVER use model.predict_proba() — that uses smoothed (backward) probabilities
    which condition on future data = look-ahead bias.
    """
    if observations.ndim == 1:
        observations = observations.reshape(-1, 1)

    framelogprob = model._compute_log_likelihood(observations)
    n_samples = len(observations)
    n_states = model.n_components

    log_startprob = np.log(model.startprob_ + 1e-300)
    log_transmat = np.log(model.transmat_ + 1e-300)

    fwd = np.zeros((n_samples, n_states))
    fwd[0] = log_startprob + framelogprob[0]

    for t in range(1, n_samples):
        for j in range(n_states):
            fwd[t, j] = (np.logaddexp.reduce(fwd[t - 1] + log_transmat[:, j])
                         + framelogprob[t, j])

    # Normalise to get probabilities
    log_norm = np.logaddexp.reduce(fwd, axis=1, keepdims=True)
    filtered = np.exp(fwd - log_norm)

    return filtered


def bic_model_selection(data, k_range=(2, 3, 4), n_restarts=None):
    """
    Fit HMMs with different numbers of states. Compare BIC.
    BIC = -2 * ln(L) + p * ln(T)
    p = K^2 + 2K - 1 for 1D Gaussian HMM
    """
    if n_restarts is None:
        n_restarts = HMM_N_RESTARTS

    results = []
    for k in k_range:
        try:
            model = fit_hmm_with_restarts(data, n_states=k, n_restarts=n_restarts)
            T = len(data)
            ll = model.score(data.reshape(-1, 1) if data.ndim == 1 else data)
            p = k**2 + 2*k - 1  # K(K-1) transmat + (K-1) startprob + K means + K variances
            bic = -2 * ll + p * np.log(T)
            results.append({'k': k, 'log_likelihood': ll, 'n_params': p, 'bic': bic})
            logger.info(f"K={k}: LL={ll:.2f}, BIC={bic:.2f}")
        except RuntimeError:
            logger.warning(f"K={k}: failed to converge")

    return pd.DataFrame(results)


def get_cached_hmm(data, cache_dir=None, n_states=None, n_restarts=None):
    """Fit or load cached HMM model for expanding window."""
    if cache_dir is None:
        cache_dir = INTERIM_DIR / 'hmm_cache'

    data_bytes = data.tobytes() if isinstance(data, np.ndarray) else data.values.tobytes()
    data_hash = hashlib.md5(data_bytes).hexdigest()[:12]
    cache_path = cache_dir / f'hmm_{data_hash}.pkl'

    if cache_path.exists():
        return joblib.load(cache_path)

    model = fit_hmm_with_restarts(data, n_states=n_states, n_restarts=n_restarts)
    joblib.dump(model, cache_path)
    return model


def label_macro_regime(date):
    """Manual regime labelling for tech portfolio (2016-2026)."""
    date = pd.Timestamp(date)
    if date < pd.Timestamp('2018-01-01'):
        return 'QE Era'
    elif date < pd.Timestamp('2019-01-01'):
        return 'Rate Hike'
    elif date < pd.Timestamp('2020-02-19'):
        return 'Late Cycle'
    elif date < pd.Timestamp('2020-04-01'):
        return 'COVID Crash'
    elif date < pd.Timestamp('2021-12-01'):
        return 'Zero-Rate Recovery'
    elif date < pd.Timestamp('2023-10-01'):
        return 'Inflation/Rate Shock'
    elif date < pd.Timestamp('2025-01-01'):
        return 'AI Boom'
    else:
        return 'Tariff/Geopolitical'


# ──────────────────────────────────────────────────────────
# Regime Transition Analysis
# ──────────────────────────────────────────────────────────

def regime_transition_analysis(model, state_mapping=None):
    """
    Analyze HMM transition dynamics for investment interpretation.
    Reports transition probabilities, expected durations, and
    ergodic (stationary) distribution.

    Parameters
    ----------
    model : GaussianHMM — fitted HMM
    state_mapping : dict or None — {old_state: new_state} from sort_states_by_mean

    Returns
    -------
    dict with transition_matrix, expected_durations, stationary_dist
    """
    A = model.transmat_.copy()
    n_states = model.n_components
    means = model.means_.flatten()
    variances = np.array([model.covars_[i][0, 0] for i in range(n_states)])

    # Re-order if mapping provided
    if state_mapping is not None:
        order = [k for k, v in sorted(state_mapping.items(), key=lambda x: x[1])]
        A = A[np.ix_(order, order)]
        means = means[order]
        variances = variances[order]

    # Expected duration in each state: E[duration] = 1 / (1 - p_ii)
    expected_duration = 1 / (1 - np.diag(A))

    # Ergodic (stationary) distribution: pi * A = pi, sum(pi) = 1
    # Solve as left eigenvector corresponding to eigenvalue 1
    eigenvalues, eigenvectors = np.linalg.eig(A.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    stationary = np.real(eigenvectors[:, idx])
    stationary = stationary / stationary.sum()

    # State labels
    labels = {0: 'Expansion', 1: 'Slowdown', 2: 'Crisis'}
    if n_states == 2:
        labels = {0: 'Expansion', 1: 'Crisis'}

    return {
        'transition_matrix': pd.DataFrame(
            A, index=[labels.get(i, f'S{i}') for i in range(n_states)],
            columns=[labels.get(i, f'S{i}') for i in range(n_states)]
        ),
        'expected_duration_months': pd.Series(
            expected_duration, index=[labels.get(i, f'S{i}') for i in range(n_states)]
        ),
        'stationary_distribution': pd.Series(
            stationary, index=[labels.get(i, f'S{i}') for i in range(n_states)]
        ),
        'state_means': pd.Series(
            means, index=[labels.get(i, f'S{i}') for i in range(n_states)]
        ),
        'state_volatilities': pd.Series(
            np.sqrt(variances), index=[labels.get(i, f'S{i}') for i in range(n_states)]
        ),
    }


def regime_conditional_stats(returns_df, regime_labels):
    """
    Compute factor return statistics conditional on regime.
    Core of the factor-timing thesis: performance varies by regime.

    Parameters
    ----------
    returns_df : pd.DataFrame — factor or asset returns
    regime_labels : pd.Series — regime label for each date

    Returns
    -------
    pd.DataFrame — multi-index (regime, factor) × {ann_return, ann_vol, sharpe, ...}
    """
    common = returns_df.index.intersection(regime_labels.index)
    returns_df = returns_df.loc[common]
    regime_labels = regime_labels.loc[common]

    results = []
    for regime in regime_labels.unique():
        mask = regime_labels == regime
        r = returns_df[mask]
        n_months = mask.sum()

        for col in returns_df.columns:
            s = r[col].dropna()
            if len(s) < 3:
                continue

            ann_ret = s.mean() * 12  # Monthly → annual
            ann_vol = s.std() * np.sqrt(12)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            skew = s.skew()
            kurt = s.kurtosis()

            # Win rate
            win_rate = (s > 0).mean()

            # Worst month
            worst = s.min()

            results.append({
                'regime': regime,
                'factor': col,
                'ann_return': ann_ret,
                'ann_volatility': ann_vol,
                'sharpe': sharpe,
                'skewness': skew,
                'excess_kurtosis': kurt,
                'win_rate': win_rate,
                'worst_month': worst,
                'best_month': s.max(),
                'n_months': n_months,
            })

    return pd.DataFrame(results)


def regime_persistence_metrics(regime_series):
    """
    Analyze regime persistence: how long each regime lasts,
    transition counts, and regime change frequency.

    Parameters
    ----------
    regime_series : pd.Series — regime labels over time (e.g., monthly)

    Returns
    -------
    dict with duration_stats, transition_counts, change_frequency
    """
    regimes = regime_series.dropna()
    if len(regimes) < 2:
        return {}

    # Compute regime durations (consecutive run lengths)
    durations = []
    current_regime = regimes.iloc[0]
    current_duration = 1
    for i in range(1, len(regimes)):
        if regimes.iloc[i] == current_regime:
            current_duration += 1
        else:
            durations.append({'regime': current_regime, 'duration': current_duration,
                              'start': regimes.index[i - current_duration],
                              'end': regimes.index[i - 1]})
            current_regime = regimes.iloc[i]
            current_duration = 1
    durations.append({'regime': current_regime, 'duration': current_duration,
                      'start': regimes.index[len(regimes) - current_duration],
                      'end': regimes.index[len(regimes) - 1]})

    dur_df = pd.DataFrame(durations)

    # Duration statistics by regime
    duration_stats = dur_df.groupby('regime')['duration'].agg(
        ['mean', 'median', 'min', 'max', 'count']
    )
    duration_stats.columns = ['avg_duration', 'median_duration',
                               'min_duration', 'max_duration', 'n_episodes']

    # Transition counts
    transitions = pd.crosstab(regimes.shift(1), regimes, margins=False)

    # Regime change frequency
    changes = (regimes != regimes.shift(1)).sum() - 1  # First obs isn't a change
    change_freq = changes / (len(regimes) - 1)

    return {
        'duration_stats': duration_stats,
        'episodes': dur_df,
        'transition_counts': transitions,
        'change_frequency': change_freq,
        'total_periods': len(regimes),
    }


def smoothed_vs_filtered_diagnostic(model, observations):
    """
    Compare filtered (causal) vs smoothed (non-causal) probabilities.
    Large differences indicate high information content in future data,
    which is exactly what we must NOT use in live trading.

    Returns
    -------
    pd.DataFrame with columns: filtered_*, smoothed_*, divergence
    """
    if observations.ndim == 1:
        observations = observations.reshape(-1, 1)

    # Filtered (forward-only) — what we use
    filt = filtered_probabilities(model, observations)

    # Smoothed (forward + backward) — what we must NOT use
    smoothed = model.predict_proba(observations)

    n_states = model.n_components

    result = pd.DataFrame()
    for k in range(n_states):
        result[f'filtered_{k}'] = filt[:, k]
        result[f'smoothed_{k}'] = smoothed[:, k]

    # KL divergence at each time step (smoothed || filtered)
    kl_div = np.zeros(len(observations))
    for t in range(len(observations)):
        p = smoothed[t] + 1e-10
        q = filt[t] + 1e-10
        kl_div[t] = np.sum(p * np.log(p / q))
    result['kl_divergence'] = kl_div

    # Summary
    mean_kl = kl_div.mean()
    max_kl = kl_div.max()
    # Classification agreement
    agreement = (np.argmax(filt, axis=1) == np.argmax(smoothed, axis=1)).mean()

    logger.info(f"Filtered vs Smoothed: mean KL={mean_kl:.4f}, "
                f"max KL={max_kl:.4f}, agreement={agreement:.1%}")

    return result, {
        'mean_kl_divergence': mean_kl,
        'max_kl_divergence': max_kl,
        'classification_agreement': agreement,
    }
