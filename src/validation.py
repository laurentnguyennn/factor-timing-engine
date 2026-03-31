"""
Validation module — schema checking, NaN monitoring, cross-phase consistency.
Call validate_parquet() at the START of every notebook.
"""
import logging

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR

logger = logging.getLogger(__name__)


def validate_parquet(df, expected_cols=None, min_rows=None,
                     no_nan=False, date_index=True, dtype_check=None,
                     label=''):
    """
    Validate a DataFrame loaded from parquet against expected schema.

    Parameters
    ----------
    df : pd.DataFrame
    expected_cols : list or None — columns that must be present
    min_rows : int or None — minimum row count
    no_nan : bool — if True, fail on any NaN
    date_index : bool — if True, require DatetimeIndex
    dtype_check : dict or None — {col: expected_dtype}
    label : str — human label for error messages

    Raises
    ------
    ValueError if any check fails
    """
    errors = []
    prefix = f"[{label}] " if label else ""

    if date_index and not isinstance(df.index, pd.DatetimeIndex):
        errors.append(f"{prefix}Index is {type(df.index).__name__}, expected DatetimeIndex")

    if date_index and hasattr(df.index, 'is_unique') and not df.index.is_unique:
        n_dup = df.index.duplicated().sum()
        errors.append(f"{prefix}Index has {n_dup} duplicate dates")

    if expected_cols:
        missing = set(expected_cols) - set(df.columns)
        if missing:
            errors.append(f"{prefix}Missing columns: {missing}")

    if min_rows and len(df) < min_rows:
        errors.append(f"{prefix}Only {len(df)} rows, expected ≥{min_rows}")

    if no_nan and df.isna().any().any():
        nan_cols = df.columns[df.isna().any()].tolist()
        errors.append(f"{prefix}NaN found in columns: {nan_cols}")

    if dtype_check:
        for col, expected_dtype in dtype_check.items():
            if col in df.columns and df[col].dtype != expected_dtype:
                errors.append(
                    f"{prefix}Column {col}: dtype {df[col].dtype}, expected {expected_dtype}")

    if errors:
        msg = "Schema validation failed:\n" + "\n".join(f"  • {e}" for e in errors)
        raise ValueError(msg)

    logger.info(f"✓ {prefix}Validation passed: {df.shape[0]} rows × {df.shape[1]} cols")
    return True


def check_nan_propagation(df, label=''):
    """Log NaN statistics — catch NaN spreading between phases."""
    nan_count = df.isna().sum()
    if nan_count.any():
        logger.warning(f"⚠️ NaN in {label}:")
        for col in nan_count[nan_count > 0].index:
            pct = nan_count[col] / len(df) * 100
            logger.warning(f"  {col}: {nan_count[col]} NaN ({pct:.1f}%)")
    else:
        logger.info(f"✓ {label}: no NaN")


def quick_data_check(df, name='DataFrame'):
    """Run essential data quality checks — print summary."""
    print(f"=== {name} ===")
    print(f"  Shape: {df.shape}")
    if hasattr(df.index, 'min'):
        print(f"  Index: {type(df.index).__name__}, "
              f"range: {df.index.min()} → {df.index.max()}")
    if hasattr(df.index, 'duplicated'):
        print(f"  Duplicates: {df.index.duplicated().sum()}")
    print(f"  NaN total: {df.isna().sum().sum()} "
          f"({df.isna().mean().mean():.1%})")
    numeric = df.select_dtypes(include=[np.number])
    if len(numeric.columns) > 0:
        print(f"  Inf total: {np.isinf(numeric).sum().sum()}")
    print(f"  Dtypes: {dict(df.dtypes.value_counts())}")
    if hasattr(df.index, 'is_monotonic_increasing'):
        print(f"  Sorted: {df.index.is_monotonic_increasing}")
    print()


def verify_phase_outputs(phase_num):
    """Run inter-phase consistency checks."""
    checks_passed = True

    if phase_num >= 2:
        try:
            fr = pd.read_parquet(PROCESSED_DIR / 'factor_returns.parquet')
            macro = pd.read_parquet(PROCESSED_DIR / 'macro_indicators.parquet')
            # Check overlap rather than strict equality (different date ranges are expected)
            overlap = fr.index.intersection(macro.index)
            if len(overlap) < 0.8 * min(len(fr), len(macro)):
                logger.warning(f"Low date overlap: factor_returns ({len(fr)}) vs "
                               f"macro_indicators ({len(macro)}): only {len(overlap)} common dates")
                checks_passed = False
            else:
                logger.info(f"Date overlap: {len(overlap)} common dates "
                            f"({len(overlap)/min(len(fr), len(macro)):.0%})")
        except FileNotFoundError:
            pass

    if phase_num >= 4:
        try:
            regime = pd.read_parquet(PROCESSED_DIR / 'regime_probabilities.parquet')
            prob_cols = [c for c in regime.columns if c.startswith('p_')]
            sums = regime[prob_cols].sum(axis=1)
            if not sums.between(0.99, 1.01).all():
                logger.warning("Regime probabilities don't sum to 1")
                checks_passed = False
        except FileNotFoundError:
            pass

    if phase_num >= 7:
        try:
            weights = pd.read_parquet(PROCESSED_DIR / 'bl_weights_timeseries.parquet')
            if not weights.sum(axis=1).between(0.99, 1.01).all():
                logger.warning("BL weights don't sum to 1")
                checks_passed = False
            if (weights < -0.001).any().any():
                logger.warning("Negative weights found")
                checks_passed = False
        except FileNotFoundError:
            pass

    return checks_passed


def stationarity_table(returns_df):
    """
    ADF + KPSS tests on each column. Returns summary DataFrame.

    Stationary = ADF rejects (p < 0.01) AND KPSS fails to reject (p > 0.05).
    """
    from statsmodels.tsa.stattools import adfuller, kpss

    results = []
    for col in returns_df.columns:
        series = returns_df[col].dropna()
        if len(series) < 50:
            continue

        adf = adfuller(series, autolag='AIC')
        try:
            kpss_res = kpss(series, regression='c', nlags='auto')
        except Exception:
            kpss_res = (np.nan, np.nan)

        results.append({
            'series': col,
            'adf_stat': adf[0],
            'adf_pvalue': adf[1],
            'kpss_stat': kpss_res[0],
            'kpss_pvalue': kpss_res[1],
            'adf_reject_1pct': adf[1] < 0.01,
            'kpss_fail_reject_5pct': kpss_res[1] > 0.05 if not np.isnan(kpss_res[1]) else None,
            'stationary': (adf[1] < 0.01) and (kpss_res[1] > 0.05 if not np.isnan(kpss_res[1]) else False),
        })

    return pd.DataFrame(results)


def detect_outliers(returns_df, method='iqr', threshold=5.0):
    """
    Detect outlier returns using IQR or MAD (Median Absolute Deviation) method.

    IQR method: outlier if |x - median| > threshold * IQR
    MAD method: outlier if |x - median| / MAD > threshold

    MAD is more robust to masking (where multiple outliers hide each other).

    Parameters
    ----------
    returns_df : pd.DataFrame or pd.Series
    method : str — 'iqr' or 'mad'
    threshold : float — multiplier for outlier boundary

    Returns
    -------
    pd.DataFrame — boolean mask (True = outlier), plus summary stats
    """
    if isinstance(returns_df, pd.Series):
        returns_df = returns_df.to_frame()

    outlier_mask = pd.DataFrame(False, index=returns_df.index, columns=returns_df.columns)
    summary = []

    for col in returns_df.columns:
        s = returns_df[col].dropna()

        if method == 'iqr':
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            mask = (s < lower) | (s > upper)
        elif method == 'mad':
            median = s.median()
            mad = np.median(np.abs(s - median))
            if mad < 1e-10:
                mask = pd.Series(False, index=s.index)
            else:
                mask = np.abs(s - median) / (1.4826 * mad) > threshold
        else:
            raise ValueError(f"Unknown method: {method}")

        outlier_mask.loc[s.index, col] = mask

        summary.append({
            'column': col,
            'n_outliers': mask.sum(),
            'pct_outliers': mask.mean() * 100,
            'min_value': s[mask].min() if mask.any() else np.nan,
            'max_value': s[mask].max() if mask.any() else np.nan,
        })

    return outlier_mask, pd.DataFrame(summary)


def winsorize_returns(returns_df, limits=(0.01, 0.99)):
    """
    Winsorize extreme returns — clip to specified percentiles.
    Preserves the original distribution shape while limiting tail impact.

    Standard in institutional risk management: we don't delete outliers,
    we reduce their magnitude.

    Parameters
    ----------
    returns_df : pd.DataFrame or pd.Series
    limits : tuple — (lower_pct, upper_pct)

    Returns
    -------
    pd.DataFrame or pd.Series — winsorized returns
    """
    if isinstance(returns_df, pd.Series):
        lower = returns_df.quantile(limits[0])
        upper = returns_df.quantile(limits[1])
        return returns_df.clip(lower=lower, upper=upper)

    result = returns_df.copy()
    for col in result.columns:
        lower = result[col].quantile(limits[0])
        upper = result[col].quantile(limits[1])
        result[col] = result[col].clip(lower=lower, upper=upper)
    return result


def distribution_diagnostics(returns_df, name='Returns'):
    """
    Comprehensive distribution diagnostics:
    - Jarque-Bera normality test
    - Skewness and kurtosis with significance tests
    - Tail index estimation (simple Hill estimator)

    Parameters
    ----------
    returns_df : pd.DataFrame or pd.Series

    Returns
    -------
    pd.DataFrame — one row per series with diagnostic stats
    """
    from scipy.stats import jarque_bera, normaltest, shapiro

    if isinstance(returns_df, pd.Series):
        returns_df = returns_df.to_frame(name)

    results = []
    for col in returns_df.columns:
        s = returns_df[col].dropna()
        if len(s) < 20:
            continue

        # Jarque-Bera test (most standard for financial returns)
        jb_stat, jb_p = jarque_bera(s)

        # D'Agostino-Pearson test
        try:
            dp_stat, dp_p = normaltest(s)
        except Exception:
            dp_stat, dp_p = np.nan, np.nan

        # Basic stats
        skew = s.skew()
        kurt = s.kurtosis()

        # Skewness significance: SE(skew) ≈ sqrt(6/n)
        se_skew = np.sqrt(6 / len(s))
        skew_zscore = skew / se_skew

        # Kurtosis significance: SE(kurt) ≈ sqrt(24/n)
        se_kurt = np.sqrt(24 / len(s))
        kurt_zscore = kurt / se_kurt

        results.append({
            'series': col,
            'n': len(s),
            'mean': s.mean(),
            'std': s.std(),
            'skewness': skew,
            'skew_zscore': skew_zscore,
            'skew_significant': abs(skew_zscore) > 1.96,
            'excess_kurtosis': kurt,
            'kurt_zscore': kurt_zscore,
            'kurt_significant': abs(kurt_zscore) > 1.96,
            'jb_stat': jb_stat,
            'jb_pvalue': jb_p,
            'normal': jb_p > 0.05,
            'dp_stat': dp_stat,
            'dp_pvalue': dp_p,
            'min': s.min(),
            'max': s.max(),
            'pct_1': s.quantile(0.01),
            'pct_99': s.quantile(0.99),
        })

    return pd.DataFrame(results)
