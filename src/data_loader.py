"""
Data loader module — downloads and caches data from Yahoo Finance, FRED, and
Kenneth French Data Library.
"""
import os
import json
import hashlib
import time
import logging

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from tqdm import tqdm

from src.config import (
    RAW_DIR, PROCESSED_DIR, FRED_SERIES,
    TICKERS, BENCHMARKS, SHORT_HISTORY_TICKERS,
    FACTOR_START, TECH_START, TECH_END,
)

load_dotenv()
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# 1. Kenneth French Data Library
# ──────────────────────────────────────────────────────────

def fetch_french_factors(start='2004'):
    """
    Fetch Fama-French 5 Factors + Momentum from Kenneth French Data Library.

    Returns
    -------
    pd.DataFrame
        Monthly factor returns in DECIMAL (not percent).
        Columns: mkt_rf, smb, hml, rmw, cma, rf, umd
    """
    import pandas_datareader.data as pdr

    logger.info("Fetching Fama-French 5 Factors...")
    ff5 = pdr.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start=start)
    ff5_monthly = ff5[0] / 100  # CRITICAL: convert % to decimal

    logger.info("Fetching Momentum Factor...")
    mom = pdr.DataReader('F-F_Momentum_Factor', 'famafrench', start=start)
    mom_monthly = mom[0] / 100

    # Standardise column names
    ff5_monthly.columns = ['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'rf']
    mom_monthly.columns = ['umd']

    # Merge on date index
    factors = ff5_monthly.join(mom_monthly, how='inner')

    # Convert PeriodIndex to DatetimeIndex (month-end)
    if isinstance(factors.index, pd.PeriodIndex):
        factors.index = factors.index.to_timestamp('M')

    # Normalise to month-end
    factors.index = factors.index + pd.offsets.MonthEnd(0)
    factors.index.name = 'date'

    # Sanity checks
    assert factors['mkt_rf'].max() < 0.30, \
        f"Factor returns appear to be in %, not decimal: max={factors['mkt_rf'].max():.4f}"
    assert not factors.isna().any().any(), "NaN in factor returns"

    logger.info(f"French factors: {factors.shape[0]} months, {factors.columns.tolist()}")
    return factors


# ──────────────────────────────────────────────────────────
# 2. FRED Macro Data
# ──────────────────────────────────────────────────────────

def fetch_fred_series(series_dict=None, start='2003-01-01'):
    """
    Fetch FRED macro time series, resample to monthly, apply transforms.

    Parameters
    ----------
    series_dict : dict or None
        {series_code: description}. Defaults to FRED_SERIES from config.

    Returns
    -------
    pd.DataFrame
        Monthly macro indicators, transformed per §3.2 of claude.md.
    """
    from fredapi import Fred

    api_key = os.environ.get('FRED_API_KEY')
    if not api_key:
        raise EnvironmentError(
            "FRED_API_KEY not found. Set it in .env file.\n"
            "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    fred = Fred(api_key=api_key)
    if series_dict is None:
        series_dict = FRED_SERIES

    raw = {}
    for code, desc in tqdm(series_dict.items(), desc='Fetching FRED'):
        try:
            s = fred.get_series(code, observation_start=start)
            s = pd.to_numeric(s, errors='coerce')  # Handle '.' placeholders
            raw[code] = s
            logger.info(f"✓ {code} ({desc}): {len(s)} obs")
        except Exception as e:
            logger.error(f"✗ {code} ({desc}): {e}")
            raise

    # Resample daily/weekly series to monthly means
    monthly_raw = {}
    for code, s in raw.items():
        monthly_raw[code] = s.resample('ME').mean()

    # Apply transforms (per §3.2 of claude.md)
    indicators = pd.DataFrame(index=monthly_raw['T10Y2Y'].index)
    indicators.index.name = 'date'

    # Direct levels
    indicators['t10y2y'] = monthly_raw['T10Y2Y']
    indicators['baa10y'] = monthly_raw['BAA10Y']
    indicators['vix'] = monthly_raw['VIXCLS']
    indicators['oecd_cli'] = monthly_raw['USALOLITONOSTSAM']
    indicators['unrate'] = monthly_raw['UNRATE']

    # Rate of change transforms
    indicators['claims_roc'] = monthly_raw['ICSA'].pct_change(3)
    indicators['oil_roc'] = monthly_raw['DCOILWTICO'].pct_change(3)
    indicators['indpro_yoy'] = monthly_raw['INDPRO'].pct_change(12)
    indicators['unrate_chg'] = monthly_raw['UNRATE'].diff(12)

    # Real M2 (deflated by CPI, then 12-month % change)
    real_m2 = monthly_raw['M2SL'] / monthly_raw['CPIAUCSL'] * 100
    indicators['real_m2_yoy'] = real_m2.pct_change(12)

    # Forward-fill small gaps (up to 3 months), then drop any remaining NaN
    indicators = indicators.ffill(limit=3)

    logger.info(f"Macro indicators: {indicators.shape}, "
                f"NaN remaining: {indicators.isna().sum().sum()}")
    return indicators


# ──────────────────────────────────────────────────────────
# 3. Yahoo Finance — S&P 500 Prices
# ──────────────────────────────────────────────────────────

def fetch_sp500_tickers():
    """Fetch current S&P 500 constituent list from Wikipedia."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    sp500 = tables[0]
    tickers = sp500['Symbol'].str.replace('.', '-', regex=False).tolist()
    logger.info(f"S&P 500 constituents: {len(tickers)} tickers")
    return tickers


def fetch_prices_batch(tickers, start, end=None, batch_size=50, pause=1.0):
    """
    Download adjusted close prices from Yahoo Finance in batches.

    Parameters
    ----------
    tickers : list
    start : str — start date
    end : str or None — end date
    batch_size : int — download N tickers at a time
    pause : float — seconds between batches (rate limit avoidance)

    Returns
    -------
    pd.DataFrame — DatetimeIndex × tickers, adjusted close prices
    """
    all_prices = {}
    failed = []

    for i in tqdm(range(0, len(tickers), batch_size), desc='Downloading prices'):
        batch = tickers[i:i + batch_size]
        try:
            data = yf.download(batch, start=start, end=end,
                               auto_adjust=True, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                prices = data['Close']
            else:
                prices = data[['Close']]
                prices.columns = batch

            for t in prices.columns:
                if prices[t].notna().sum() > 0:
                    all_prices[t] = prices[t]
                else:
                    failed.append(t)
        except Exception as e:
            logger.warning(f"Batch {i}-{i+batch_size} failed: {e}")
            failed.extend(batch)

        if i + batch_size < len(tickers):
            time.sleep(pause)

    if failed:
        logger.warning(f"Failed tickers ({len(failed)}): {failed[:10]}...")

    result = pd.DataFrame(all_prices)
    result.index = pd.to_datetime(result.index)
    result.index.name = 'date'

    logger.info(f"Downloaded: {result.shape[1]} tickers, "
                f"{result.shape[0]} days, {len(failed)} failed")
    return result


# ──────────────────────────────────────────────────────────
# 4. Tech Portfolio Data
# ──────────────────────────────────────────────────────────

def fetch_tech_portfolio(tickers=None, benchmarks=None,
                         start=None, end=None):
    """
    Download OHLCV data for the 20 tech tickers + benchmarks.
    Handles SQ → XYZ merger.

    Returns
    -------
    pd.DataFrame — DatetimeIndex × tickers, adjusted close prices
    """
    if tickers is None:
        tickers = TICKERS
    if benchmarks is None:
        benchmarks = BENCHMARKS
    if start is None:
        start = TECH_START
    if end is None:
        end = TECH_END

    all_tickers = [t for t in tickers if t != 'XYZ'] + benchmarks
    prices = fetch_prices_batch(all_tickers, start=start, end=end, batch_size=30)

    # Handle SQ → XYZ merger (ticker changed 2025-01-21)
    if 'XYZ' in tickers:
        prices = _merge_sq_xyz(prices, start, end)

    logger.info(f"Tech portfolio: {prices.shape[1]} tickers, {prices.shape[0]} days")
    return prices


def _merge_sq_xyz(prices_df, start, end):
    """Handle Block Inc. ticker change SQ → XYZ (2025-01-21)."""
    transition_date = '2025-01-21'

    try:
        sq = yf.download('SQ', start=start, end=transition_date,
                         auto_adjust=True, progress=False)['Close']
        xyz = yf.download('XYZ', start=transition_date, end=end,
                          auto_adjust=True, progress=False)['Close']

        merged = pd.concat([sq, xyz]).sort_index()
        merged = merged[~merged.index.duplicated(keep='last')]
        prices_df['XYZ'] = merged.reindex(prices_df.index)
        logger.info(f"SQ→XYZ merge: {len(sq)} + {len(xyz)} days")
    except Exception as e:
        logger.warning(f"SQ→XYZ merge failed: {e}. Trying SQ only.")
        try:
            sq = yf.download('SQ', start=start, end=end,
                             auto_adjust=True, progress=False)['Close']
            prices_df['XYZ'] = sq.reindex(prices_df.index)
        except Exception as e2:
            logger.error(f"SQ download also failed: {e2}")

    return prices_df


# ──────────────────────────────────────────────────────────
# 5. Checksums
# ──────────────────────────────────────────────────────────

def compute_checksum(filepath, algorithm='sha256'):
    """Compute SHA-256 checksum for reproducibility verification."""
    h = hashlib.new(algorithm)
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def save_checksums(file_dict, output_path=None):
    """Save checksums for all raw data files."""
    if output_path is None:
        output_path = RAW_DIR / 'checksums.json'

    checksums = {}
    for name, path in file_dict.items():
        if os.path.exists(path):
            checksums[name] = compute_checksum(path)

    with open(output_path, 'w') as f:
        json.dump(checksums, f, indent=2)

    logger.info(f"Saved checksums for {len(checksums)} files → {output_path}")
