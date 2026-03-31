"""
Feature engineering module — technical indicators, volume features,
and feature matrix assembly for ML pipeline.
Implements Appendix H of claude.md.
"""
import numpy as np
import pandas as pd


def compute_technical_indicators(ohlcv_df):
    """
    Compute comprehensive technical indicators from OHLCV data.

    Parameters
    ----------
    ohlcv_df : pd.DataFrame — columns: Open, High, Low, Close, Volume

    Returns
    -------
    pd.DataFrame — features indexed by date
    """
    close = ohlcv_df['Close']
    high = ohlcv_df['High']
    low = ohlcv_df['Low']
    volume = ohlcv_df['Volume']
    log_ret = np.log(close / close.shift(1))

    feats = pd.DataFrame(index=ohlcv_df.index)

    # === Momentum ===
    feats['log_return_1d'] = log_ret
    feats['momentum_5d'] = close.pct_change(5)
    feats['momentum_21d'] = close.pct_change(21)
    feats['momentum_63d'] = close.pct_change(63)

    # RSI (14-day) — Wilder's exponential smoothing (alpha=1/14), NOT simple moving average
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    feats['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD — min_periods prevents spurious early values
    ema12 = close.ewm(span=12, min_periods=12).mean()
    ema26 = close.ewm(span=26, min_periods=26).mean()
    feats['macd'] = ema12 - ema26
    feats['macd_signal'] = feats['macd'].ewm(span=9).mean()
    feats['macd_histogram'] = feats['macd'] - feats['macd_signal']

    # === Volatility ===
    feats['vol_5d'] = log_ret.rolling(5).std() * np.sqrt(252)
    feats['vol_21d'] = log_ret.rolling(21).std() * np.sqrt(252)
    feats['vol_63d'] = log_ret.rolling(63).std() * np.sqrt(252)

    # Garman-Klass volatility
    log_hl = np.log(high / low)
    log_co = np.log(close / ohlcv_df['Open'])
    feats['gk_vol'] = np.sqrt(
        (0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2).rolling(21).mean() * 252
    )

    # Parkinson volatility
    feats['park_vol'] = np.sqrt(
        (log_hl ** 2 / (4 * np.log(2))).rolling(21).mean() * 252
    )

    # ATR (14-day)
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': abs(high - close.shift(1)),
        'lc': abs(low - close.shift(1))
    }).max(axis=1)
    feats['atr_14'] = tr.rolling(14).mean()

    # Bollinger %B
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    feats['bb_pctb'] = (close - (bb_mid - 2 * bb_std)) / (4 * bb_std)

    # === Volume ===
    feats['volume_zscore'] = (
        (volume - volume.rolling(20).mean()) / volume.rolling(20).std()
    )
    feats['volume_momentum'] = volume.rolling(5).mean() / volume.rolling(20).mean()

    # === Range ===
    feats['high_low_ratio'] = np.log(high / low)
    feats['close_range_pct'] = (close - low) / (high - low).replace(0, np.nan)

    return feats


def yang_zhang_volatility(ohlcv_df, window=60):
    """
    Yang-Zhang (2000) range-based volatility estimator using OHLC data.
    More efficient than close-to-close estimator (uses intra-day information).

    Parameters
    ----------
    ohlcv_df : pd.DataFrame with columns: Open, High, Low, Close
    window : int — lookback window in trading days

    Returns
    -------
    pd.Series — annualised Yang-Zhang volatility
    """
    log_ho = np.log(ohlcv_df['High'] / ohlcv_df['Open'])
    log_lo = np.log(ohlcv_df['Low'] / ohlcv_df['Open'])
    log_co = np.log(ohlcv_df['Close'] / ohlcv_df['Open'])
    log_oc = np.log(ohlcv_df['Open'] / ohlcv_df['Close'].shift(1))

    # Rogers-Satchell component
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    # Bias correction factor
    n = window
    k = 0.34 / (1.34 + (n + 1) / (n - 1))

    # Yang-Zhang variance = overnight var + k * close-to-close var + (1-k) * RS var
    yz_var = (log_oc.rolling(n).var()
              + k * log_co.rolling(n).var()
              + (1 - k) * rs.rolling(n).mean())

    return np.sqrt(yz_var.clip(lower=0) * 252)


def hurst_exponent(series, max_lag=100):
    """
    Hurst exponent via rescaled range (R/S) analysis.
    H < 0.5: mean-reverting
    H = 0.5: random walk (geometric Brownian motion)
    H > 0.5: trending / momentum

    Critical for factor timing: momentum factors work when H > 0.5,
    mean-reversion strategies work when H < 0.5.

    Parameters
    ----------
    series : pd.Series — price or return series
    max_lag : int — maximum lag for R/S computation

    Returns
    -------
    float — estimated Hurst exponent
    """
    series = series.dropna().values
    if len(series) < max_lag:
        max_lag = len(series) // 2

    lags = range(10, max_lag)
    rs_values = []

    for lag in lags:
        # Split into non-overlapping sub-series
        n_subseries = len(series) // lag
        if n_subseries < 1:
            continue

        rs_list = []
        for i in range(n_subseries):
            sub = series[i * lag:(i + 1) * lag]
            mean_sub = sub.mean()
            deviate = np.cumsum(sub - mean_sub)
            R = deviate.max() - deviate.min()
            S = sub.std(ddof=1)
            if S > 1e-10:
                rs_list.append(R / S)

        if rs_list:
            rs_values.append((lag, np.mean(rs_list)))

    if len(rs_values) < 5:
        return 0.5  # Insufficient data

    lags_arr = np.array([x[0] for x in rs_values])
    rs_arr = np.array([x[1] for x in rs_values])

    # Linear regression in log-log space: log(R/S) = H * log(n) + c
    log_lags = np.log(lags_arr)
    log_rs = np.log(rs_arr)
    coeffs = np.polyfit(log_lags, log_rs, 1)

    return coeffs[0]  # Hurst exponent


def rolling_hurst(series, window=252, max_lag=60):
    """
    Time-varying Hurst exponent via rolling window.
    Tracks regime shifts between trending and mean-reverting markets.
    """
    result = pd.Series(index=series.index, dtype=float)
    for i in range(window, len(series)):
        result.iloc[i] = hurst_exponent(series.iloc[i - window:i], max_lag=max_lag)
    return result


def amihud_illiquidity(close, volume, window=21):
    """
    Amihud (2002) illiquidity measure: ILLIQ = |r_t| / Volume_t
    Higher values = less liquid = higher expected returns (illiquidity premium).

    Rolling average for stability. Log-transformed for cross-sectional comparability.

    Parameters
    ----------
    close : pd.Series — closing prices
    volume : pd.Series — daily volume (shares or dollars)
    window : int — averaging window
    """
    abs_ret = np.abs(np.log(close / close.shift(1)))
    # Use dollar volume if available, else raw volume
    illiq_daily = abs_ret / volume.replace(0, np.nan)
    illiq = illiq_daily.rolling(window).mean()
    # Log transform for cross-sectional use
    log_illiq = np.log(illiq.clip(lower=1e-20))
    return log_illiq


def realized_skewness(log_returns, window=63):
    """
    Realized skewness from high-frequency/daily returns.
    Negative skewness = crash risk premium.
    Neuberger (2012) shows this predicts cross-sectional returns.
    """
    roll_mean = log_returns.rolling(window).mean()
    roll_std = log_returns.rolling(window).std()
    z = (log_returns - roll_mean) / roll_std.clip(lower=1e-10)
    return z.rolling(window).apply(lambda x: np.mean(x**3), raw=True)


def idiosyncratic_volatility(asset_returns, factor_returns, window=63):
    """
    Idiosyncratic volatility: residual vol after removing systematic risk.
    Ang et al. (2006): low idio-vol stocks outperform (IVOL puzzle).

    Fits rolling CAPM, returns the standard deviation of residuals.
    """
    if isinstance(factor_returns, pd.DataFrame):
        mkt = factor_returns.iloc[:, 0]
    else:
        mkt = factor_returns

    common = asset_returns.index.intersection(mkt.index)
    asset_returns = asset_returns.loc[common]
    mkt = mkt.loc[common]

    ivol = pd.Series(index=common, dtype=float)
    for i in range(window, len(common)):
        y = asset_returns.iloc[i - window:i].values
        x = mkt.iloc[i - window:i].values
        # Simple OLS: beta = cov(r, m) / var(m)
        beta = np.cov(y, x)[0, 1] / (np.var(x, ddof=1) + 1e-10)
        residuals = y - beta * x
        ivol.iloc[i] = residuals.std() * np.sqrt(252)

    return ivol


def compute_cross_sectional_features(returns_df, window=21):
    """
    Cross-sectional features: relative momentum, vol, and mean-reversion signals.
    These capture where an asset stands relative to its peers.

    Parameters
    ----------
    returns_df : pd.DataFrame — asset returns (T × N)
    window : int — lookback window

    Returns
    -------
    dict of pd.DataFrames — {feature_name: T × N DataFrame}
    """
    # Cross-sectional momentum rank (percentile rank within universe)
    cum_ret = returns_df.rolling(window).sum()
    cs_momentum_rank = cum_ret.rank(axis=1, pct=True)

    # Cross-sectional volatility rank
    vol = returns_df.rolling(window).std()
    cs_vol_rank = vol.rank(axis=1, pct=True)

    # Z-score within cross-section (demeaned by cross-sectional mean)
    cs_mean = cum_ret.mean(axis=1)
    cs_std = cum_ret.std(axis=1)
    cs_zscore = cum_ret.sub(cs_mean, axis=0).div(cs_std.clip(lower=1e-8), axis=0)

    return {
        'cs_momentum_rank': cs_momentum_rank,
        'cs_vol_rank': cs_vol_rank,
        'cs_zscore': cs_zscore,
    }


def synthetic_sentiment(prices, window=20):
    """
    Market-based sentiment proxy using momentum and volatility.
    NOT a substitute for actual NLP — document this limitation.
    """
    ret_5d = prices.pct_change(5)
    vol_20d = prices.pct_change().rolling(window).std()

    sentiment = pd.DataFrame(index=prices.index)
    sentiment['sentiment_mean'] = np.tanh(ret_5d / vol_20d.clip(lower=0.001))
    sentiment['sentiment_std'] = sentiment['sentiment_mean'].rolling(5).std()
    sentiment['sentiment_momentum'] = sentiment['sentiment_mean'].diff(5)
    return sentiment


def build_feature_matrix(ohlcv_df, garch_vol=None, regime_probs=None,
                         sentiment=None):
    """
    Assemble full feature matrix for ML models.
    All features are computed from data available at time t.

    CRITICAL: sentiment features must be ALREADY lagged (t-1).
    """
    feats = compute_technical_indicators(ohlcv_df)

    if garch_vol is not None:
        feats['garch_vol'] = garch_vol.reindex(feats.index)

    if regime_probs is not None:
        for col in regime_probs.columns:
            feats[f'regime_{col}'] = regime_probs[col].reindex(feats.index)

    if sentiment is not None:
        for col in sentiment.columns:
            feats[f'sent_{col}'] = sentiment[col].reindex(feats.index)

    return feats


def create_forward_target(log_returns, horizon=5, target_type='vol'):
    """
    Create forward-looking target variable.
    Target at time t uses data from t+1 onwards.

    Parameters
    ----------
    horizon : int — number of days forward
    target_type : 'vol' or 'direction'
    """
    if target_type == 'vol':
        target = log_returns.rolling(horizon).std().shift(-horizon) * np.sqrt(252)
        target.name = f'fwd_vol_{horizon}d'
    elif target_type == 'direction':
        fwd_ret = log_returns.rolling(horizon).sum().shift(-horizon)
        target = (fwd_ret > 0).astype(int)
        target.name = f'fwd_dir_{horizon}d'
    else:
        raise ValueError(f"Unknown target_type: {target_type}")

    return target
