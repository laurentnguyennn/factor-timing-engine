# Cross-Sector Factor Timing & Dynamic Allocation Engine

## factor_engine.md — Master Blueprint for Claude Code Execution

**Last Updated:** 2026-03-26  
**Status:** Active Development  
**Target Audience:** Claude Code AI agent executing this project end-to-end based on the given instruction.

---

## TABLE OF CONTENTS

1. [Project Overview](#1-project-overview)
2. [Environment & Dependencies](#2-environment--dependencies)
3. [Data Architecture](#3-data-architecture)
4. [Phase Specifications (Phases 1–12)](#4-phase-specifications)
5. [Dependency Graph](#5-dependency-graph)
6. [ML/DL Model Governance & Anti-Leakage Protocol](#6-mldl-model-governance--anti-leakage-protocol)
7. [Deep Learning Architecture Specification](#7-deep-learning-architecture-specification)
8. [Hybrid Model Ensemble Protocol](#8-hybrid-model-ensemble-protocol)
9. [Copula & Tail Risk Modeling](#9-copula--tail-risk-modeling)
10. [Monte Carlo Simulation Framework](#10-monte-carlo-simulation-framework)
11. [Configuration Management](#11-configuration-management)
12. [Quality Gates & Automated Validation](#12-quality-gates--automated-validation)
13. [Common Pitfalls — DO NOT MAKE THESE ERRORS](#13-common-pitfalls--do-not-make-these-errors)
14. [Execution Notes for Claude Code](#14-execution-notes-for-claude-code)
15. [Troubleshooting Guide](#15-troubleshooting-guide)
16. [Success Criteria](#16-success-criteria)
17. [Appendices](#17-appendices)

---

## 1. PROJECT OVERVIEW

### 1.1 Investment Thesis

**Primary thesis:** Equity factor premia (value, momentum, quality, low-volatility) exhibit
regime-dependent behaviour — their expected returns, volatilities, and cross-correlations shift
materially across macroeconomic regimes. A systematic strategy that (a) detects regime transitions
in real time using observable macro data and (b) dynamically re-weights factor exposures conditional
on the detected regime should deliver superior risk-adjusted returns versus static factor allocation.

**Secondary thesis:** Machine learning and deep learning models can enhance regime detection
and return forecasting beyond classical econometric methods, provided strict anti-leakage
protocols are enforced throughout the pipeline.

**Tertiary thesis:** Tail risk modeling via Extreme Value Theory (EVT) and copula-based
joint crash probability estimation provides actionable risk budgets that static VaR cannot.

### 1.2 Academic Grounding

This project draws on established academic literature across multiple domains:

**Factor Investing & Asset Pricing:**
- Ang & Bekaert (2002) — regime-switching models for asset allocation
- Ilmanen (2011) — factor premia vary with business cycle
- Novy-Marx (2013) — quality factor persistence across regimes
- Baker, Bradley & Wurgler (2011) — low-volatility anomaly
- Fama & French (2015) — five-factor model
- Carhart (1997) — momentum factor

**Portfolio Construction:**
- Black & Litterman (1992) — Bayesian portfolio construction
- Rockafellar & Uryasev (2000) — CVaR optimisation (linear programming formulation)
- He & Litterman (1999) — posterior covariance formulation
- Maillard, Roncalli & Teïletche (2010) — Equal Risk Contribution (ERC)
- López de Prado (2016) — Hierarchical Risk Parity (HRP)
- Ledoit & Wolf (2004) — shrinkage covariance estimation

**Volatility & Risk Modeling:**
- Engle (2002) — Dynamic Conditional Correlation (DCC)
- Bollerslev (1986) — GARCH(1,1)
- Glosten, Jagannathan & Runkle (1993) — GJR-GARCH (leverage effect)
- Nelson (1991) — EGARCH
- Baillie, Bollerslev & Mikkelsen (1996) — FIGARCH (long memory)
- Yang & Zhang (2000) — range-based volatility estimator

**Extreme Value Theory & Copulas:**
- Pickands (1975) — Generalised Pareto Distribution (GPD)
- McNeil & Frey (2000) — EVT for financial risk
- Clayton (1978) — Clayton copula for lower tail dependence
- Joe (1997) — multivariate copula families

**Machine Learning in Finance:**
- Gu, Kelly & Xiu (2020) — ML methods in empirical asset pricing
- Diebold & Mariano (1995) — predictive accuracy comparison
- Mincer & Zarnowitz (1969) — forecast calibration test

**Regime Detection:**
- Hamilton (1989) — Markov switching models
- Rabiner (1989) — HMM tutorial (forward-backward algorithm)
- Baum et al. (1970) — Baum-Welch EM algorithm

**NLP in Finance:**
- Araci (2019) — FinBERT: Financial Sentiment Analysis with BERT
- Loughran & McDonald (2011) — financial text sentiment dictionaries

### 1.3 Project Scope

- **Universe:** S&P 500 (factor construction) + 20 tech-sector individual stocks (portfolio construction)
- **Rebalancing:** Monthly for factor allocation; quarterly retraining for ML/DL models
- **Backtest window:** 2005-01 to 2025-12 (factor timing); 2016-01 to 2026-02 (tech portfolio)
- **Walk-forward:** Expanding window with minimum 60-month estimation period
- **Risk-free rate:** FRED series `DGS3MO` (3-month Treasury) or French Library `RF`

### 1.4 Deliverables Per Phase

Each notebook produces:
1. A validated `.parquet` output consumed by the next phase
2. Standalone analysis charts saved to `outputs/figures/`
3. Summary statistics tables saved to `outputs/tables/`
4. A timestamped log file in `logs/`

**Final outputs:**
- Walk-forward backtest report with statistical significance tests
- Excel summary dashboard (10+ tabs)
- Presentation-ready PowerPoint deck (20–25 slides)
- PDF report (15–20 pages)
- All model artifacts serialised via `joblib` for reproducibility

### 1.5 Working Directory & Project Structure

**Working directory:** `~/factor-timing-engine/`

```
factor-timing-engine/
├── factor_engine.md                        # This file — master blueprint
├── .env                             # API keys (NEVER commit — in .gitignore)
├── .gitignore
├── README.md
├── requirements.txt
├── setup.py                         # Package installation for src/
│
├── data/
│   ├── raw/                         # Downloaded, never modified
│   │   ├── checksums.json           # SHA-256 hashes of raw downloads
│   │   ├── sp500_prices_YYYYMMDD.parquet
│   │   ├── fred_macro.csv
│   │   └── french_factors.csv
│   ├── interim/                     # Cleaned, intermediate
│   │   ├── hmm_cache/               # Cached HMM models (hash-based)
│   │   └── garch_cache/             # Cached GARCH results
│   └── processed/                   # Phase outputs (.parquet) — production data
│       ├── master_data.parquet
│       ├── factor_returns.parquet
│       ├── factor_returns_full.parquet
│       ├── macro_indicators.parquet
│       ├── macro_regimes.parquet
│       ├── regime_labels.parquet
│       ├── regime_probabilities.parquet
│       ├── sp500_daily_prices.parquet
│       ├── sp500_monthly_returns.parquet
│       ├── lowvol_factor_returns.parquet
│       ├── garch_conditional_vol.parquet
│       ├── dcc_conditional_corr.parquet
│       ├── conditional_covariance.parquet
│       ├── sentiment_features.parquet
│       ├── return_scenarios.parquet
│       ├── bl_weights_timeseries.parquet
│       ├── cvar_weights_timeseries.parquet
│       ├── portfolio_weights_timeseries.parquet
│       ├── backtest_returns.parquet
│       ├── backtest_nav.parquet
│       └── vol_forecast_predictions.parquet
│
├── notebooks/
│   ├── 01_data_pipeline.ipynb
│   ├── 02_factor_construction.ipynb
│   ├── 03_factor_validation.ipynb
│   ├── 04_macro_regime_hmm.ipynb
│   ├── 05_regime_conditional_analysis.ipynb
│   ├── 06_dcc_garch.ipynb
│   ├── 07_allocation_blacklitterman.ipynb
│   ├── 08_allocation_mean_cvar.ipynb
│   ├── 09_backtest_engine.ipynb
│   ├── 10_stress_testing.ipynb
│   ├── 11_performance_report.ipynb
│   └── 12_presentation_export.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py                    # Centralised configuration (paths, constants, tickers)
│   ├── data_loader.py               # FRED + yfinance + French library fetch
│   ├── factor_builder.py            # Factor portfolio construction logic
│   ├── feature_engineering.py       # Technical indicators, vol estimators, transforms
│   ├── regime_model.py              # HMM wrapper with expanding-window
│   ├── regime_utils.py              # Regime detection utilities (state ordering, transitions)
│   ├── garch_models.py              # GARCH + DCC estimation
│   ├── garch_utils.py               # GARCH pipeline (fit, select, diagnose)
│   ├── allocation.py                # BL + Mean-CVaR + HRP + ERC optimizers
│   ├── portfolio_optimizer.py       # Portfolio optimization wrappers
│   ├── backtest.py                  # Walk-forward engine
│   ├── backtest_engine.py           # Backtest metrics and comparison
│   ├── risk_metrics.py              # VaR, CVaR, EVT, copula functions
│   ├── ml_pipeline.py               # ML walk-forward (Ridge, RF, XGBoost, LightGBM)
│   ├── dl_models.py                 # Deep learning (LSTM, GRU, TFT)
│   ├── validation.py                # Shared validation helpers (NaN, schema, date alignment)
│   ├── visualization.py             # Plotting utilities (regime overlay, heatmaps, etc.)
│   └── utils.py                     # Shared helpers (IO, logging, checksums)
│
├── tests/
│   ├── test_data_loader.py          # Smoke tests for data fetch + transform
│   ├── test_factor_builder.py       # Edge cases in quintile construction
│   ├── test_regime_model.py         # Filtered vs smoothed probability divergence
│   ├── test_allocation.py           # Weight constraints, PSD checks
│   ├── test_backtest.py             # TC calculation, NAV compounding
│   ├── test_risk_metrics.py         # VaR/CVaR calculation correctness
│   ├── test_ml_pipeline.py          # Walk-forward data isolation
│   └── test_validation.py           # Schema and date checks
│
├── outputs/
│   ├── figures/                     # All plots (300 DPI, PNG)
│   ├── tables/                      # All CSV summary tables
│   ├── reports/                     # PDF report + Excel dashboard
│   └── models/                      # Serialised model artifacts (.pkl, .pt)
│
├── logs/                            # Timestamped run logs per notebook
│
└── docs/
    ├── methodology.md               # Extended methodology writeup
    └── data_dictionary.md           # Detailed schema documentation
```

### 1.6 Ticker Universe (Tech Portfolio Component)

The tech portfolio component uses 20 carefully selected tickers spanning semiconductors,
software, cloud, cybersecurity, and AI:

| # | Ticker | Company | Sector | IPO/DPO | Notes |
|---|--------|---------|--------|---------|-------|
| 1 | NVDA | NVIDIA | Semiconductors | 1999-01 | AI training chips leader |
| 2 | AMD | Advanced Micro Devices | Semiconductors | 1972-09 | CPU/GPU competitor |
| 3 | TSM | Taiwan Semiconductor | Semiconductors | ADR 1997 | Foundry monopoly risk |
| 4 | AVGO | Broadcom | Semiconductors | 2009-08 | Networking + infrastructure |
| 5 | QCOM | Qualcomm | Semiconductors | 1991-12 | Mobile + IoT chips |
| 6 | MU | Micron Technology | Semiconductors | 1984-06 | Memory (DRAM/NAND) |
| 7 | AAPL | Apple | Consumer Tech | 1980-12 | Hardware + services |
| 8 | MSFT | Microsoft | Software/Cloud | 1986-03 | Azure + enterprise |
| 9 | GOOG | Alphabet | Software/Cloud | 2004-08 | Search + cloud + AI |
| 10 | META | Meta Platforms | Software/Social | 2012-05 | Social + metaverse |
| 11 | NFLX | Netflix | Software/Media | 2002-05 | Streaming |
| 12 | CRM | Salesforce | Software/CRM | 2004-06 | Enterprise cloud CRM |
| 13 | ADBE | Adobe | Software/Creative | 1986-08 | Creative + document cloud |
| 14 | NOW | ServiceNow | Software/ITSM | 2012-06 | IT workflow automation |
| 15 | PANW | Palo Alto Networks | Cybersecurity | 2012-07 | Next-gen firewall |
| 16 | CRWD | CrowdStrike | Cybersecurity | 2019-06 | **Short history** |
| 17 | DDOG | Datadog | Observability | 2019-09 | **Short history** |
| 18 | PLTR | Palantir | AI/Analytics | DPO 2020-09 | **Short history** |
| 19 | ANET | Arista Networks | Networking | 2014-06 | Data center switches |
| 20 | XYZ | Block Inc. (fmr. SQ) | Fintech | 2015-11 | **Ticker changed 2025-01-21** |

**Short-history tickers** (CRWD, DDOG, PLTR): These require special handling — FIGARCH
models are excluded (<1500 observations), and walk-forward minimum training windows are
adjusted. Document survivorship bias implications.

**Corporate action:** SQ → XYZ (Block Inc. ticker change effective 2025-01-21). The
`merge_sq_xyz()` function in `src/data_loader.py` handles the seamless concatenation of
historical SQ data with new XYZ data.

**Benchmarks:**

| Benchmark | Ticker/Source | Use |
|-----------|---------------|-----|
| S&P 500 | `^GSPC` | Broad market |
| Nasdaq 100 | `^NDX` | Tech-heavy benchmark |
| Tech ETF | `XLK` | Sector benchmark |
| Semiconductor ETF | `SOXX` | Sub-sector benchmark |
| VIX | `^VIX` | Implied volatility |
| VVIX | `^VVIX` | Vol-of-vol |
| US Dollar Index | `DX-Y.NYB` | Currency risk |

---

## 2. ENVIRONMENT & DEPENDENCIES

### 2.1 Core Requirements

```
# requirements.txt
# === Core ===
python>=3.11,<3.13
pandas>=2.1,<3.0
numpy>=1.24,<2.0
scipy>=1.11
pyarrow>=14                    # Parquet I/O

# === Econometrics & Statistics ===
statsmodels>=0.14              # HAC standard errors, Granger causality, ADF/KPSS
arch>=6.2                      # GARCH family (GARCH, GJR-GARCH, EGARCH, FIGARCH)
hmmlearn>=0.3                  # Gaussian HMM for regime detection

# === Machine Learning ===
scikit-learn>=1.3              # PCA, preprocessing, Ridge, RF, KNN
xgboost>=2.0                   # Gradient boosting (volatility + return forecasting)
lightgbm>=4.1                  # Light gradient boosting
imbalanced-learn>=0.11         # SMOTE for classification tasks ONLY

# === Deep Learning ===
torch>=2.1                     # LSTM, GRU, TFT
# torchvision is NOT required (no image data)
# transformers>=4.35           # FinBERT (optional — large download)

# === Optimisation ===
cvxpy>=1.4                     # Convex optimization (Mean-CVaR, BL, ERC)

# === Visualisation ===
matplotlib>=3.8
seaborn>=0.13

# === Data Access ===
yfinance>=0.2.31               # Yahoo Finance price data
pandas-datareader>=0.10        # FRED access (Fama-French reader)
fredapi>=0.5                   # Alternative FRED API access

# === Utilities ===
openpyxl>=3.1                  # Excel export
python-dotenv>=1.0             # .env file loading
joblib>=1.3                    # Model serialization + caching
tqdm>=4.66                     # Progress bars
shap>=0.43                     # SHAP for model interpretability (optional)

# === NLP (Optional — install separately if using FinBERT) ===
# transformers>=4.35
# sentencepiece>=0.1.99
```

### 2.2 Installation

```bash
# Create conda environment
conda create -n factor-timing python=3.11 -y
conda activate factor-timing

# Install core dependencies
pip install -r requirements.txt

# Install project package in editable mode (enables `from src.xxx import yyy`)
pip install -e .

# Optional: FinBERT (requires ~1.5GB download)
pip install transformers sentencepiece

# Optional: PyTorch with CUDA (if GPU available)
# pip install torch --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "
import pandas, numpy, scipy, statsmodels, arch, hmmlearn
import sklearn, xgboost, lightgbm, cvxpy, torch
print('All core imports successful')
print(f'PyTorch CUDA available: {torch.cuda.is_available()}')
print(f'NumPy version: {numpy.__version__}')
print(f'Pandas version: {pandas.__version__}')
"
```

### 2.3 API Keys

**Use environment variables, never hardcode.**

Create `~/factor-timing-engine/.env`:
```
FRED_API_KEY=your_key_here
```

Load in every notebook:
```python
import os
from dotenv import load_dotenv
load_dotenv()
FRED_API_KEY = os.environ["FRED_API_KEY"]
```

Get a free FRED key at: https://fred.stlouisfed.org/docs/api/api_key.html

### 2.4 .gitignore

```
# Secrets
.env

# Data (reproducible from code)
data/raw/
data/interim/
logs/
__pycache__/

# Model artifacts (large binaries)
*.pkl
*.pt
*.pth
outputs/models/

# OS files
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.pyc
```

### 2.5 Data Sources — No Paid Subscriptions Required

| Source | Data | Access Method | Cost |
|--------|------|---------------|------|
| Yahoo Finance | Price data (OHLCV) | `yfinance` Python package | Free |
| FRED | Macro time series | `fredapi` with API key | Free (API key required) |
| Kenneth French Data Library | Factor returns | `pandas_datareader.famafrench` or direct CSV | Free |
| Wikipedia | S&P 500 constituent list | `pd.read_html` | Free |

### 2.6 Hardware Requirements

| Resource | Minimum | Recommended | Notes |
|----------|---------|-------------|-------|
| RAM | 8 GB | 16 GB | Price panel (~250MB peak) + DL training |
| Disk | 2 GB | 5 GB | Data + model artifacts + outputs |
| CPU | 4 cores | 8 cores | HMM restarts + walk-forward parallelism |
| GPU | Not required | NVIDIA (CUDA) | DL training 5-10x faster with GPU |

**Memory budget by phase:**

| Phase | Peak RAM | Bottleneck |
|-------|----------|------------|
| Phase 1 (Data Pipeline) | ~500 MB | S&P 500 daily prices (500 tickers × 5000 days) |
| Phase 4 (HMM) | ~50 MB | 25 HMM restarts (sequential, not parallel) |
| Phase 6 (DCC-GARCH) | ~100 MB | DCC recursion over 252 months |
| Phase 9 (Backtest) | ~200 MB | Expanding-window re-estimation cache |
| NB07-09 (ML/DL) | ~1 GB | LSTM training + feature matrices |
| NB12 (Monte Carlo) | ~300 MB | 10,000 simulated paths |

---

## 3. DATA ARCHITECTURE

### 3.1 Factor Return Data — Kenneth French Data Library

**Source:** https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

| Dataset | Frequency | Use |
|---------|-----------|-----|
| Fama/French 5 Factors (2×3) | Monthly | SMB, HML, RMW, CMA, Mkt-RF, RF |
| Momentum Factor (Mom) | Monthly | UMD (up minus down) |

These are **pre-constructed** long-short factor portfolios from the full CRSP universe.
They serve as the **primary factor return series** for the factor-timing component.

**Why use French Library instead of self-constructed factors:**
- Survivorship-bias free (uses CRSP dead/delisted stocks)
- Institutional standard — every quant paper benchmarks against these
- Consistent methodology across 60+ years
- Frees project effort for regime detection and allocation (the original contribution)

**Critical parsing note:** French Library CSV files contain **two sections** — monthly data
followed by annual data, separated by a blank line or an "Annual" header. You must detect
the section break and only parse the monthly rows. The `pandas_datareader.famafrench`
module handles this automatically via `pdr.famafrench.FamaFrenchReader`, but if parsing
manually, stop reading at the first blank line after the data begins.

**Factor mapping for this project:**

| Project Factor | French Library Series | Definition | Expected Monthly Mean | Expected Ann. Vol |
|----------------|----------------------|------------|----------------------|-------------------|
| Value | HML (High Minus Low) | Long high B/M, short low B/M | +0.25% | ~10% |
| Momentum | UMD (Up Minus Down) | Long 12-1 month winners, short losers | +0.65% | ~15% |
| Quality | RMW (Robust Minus Weak) | Long high operating profitability, short low | +0.25% | ~8% |
| Low-Volatility | — (construct separately) | Not in French library; see Phase 2 | +0.15% | ~8% |
| Size | SMB (Small Minus Big) | Available but not primary; used for decomposition | — | — |
| Investment | CMA (Conservative Minus Aggressive) | Available; used for decomposition | — | — |

**Low-Volatility factor construction (the one factor we must build):**
Since Kenneth French does not provide a low-volatility factor, we construct it from
S&P 500 constituents using trailing 60-day realized volatility, forming quintile
long-short portfolios. See Phase 2 detailed instructions.

### 3.2 Macro Data — FRED

All series confirmed available on FRED as of 2025.

| Variable | FRED Code | Frequency | Transform | Economic Rationale |
|----------|-----------|-----------|-----------|-------------------|
| Yield curve slope | `T10Y2Y` | Daily → Monthly avg | Level (already a spread) | Inverts before recessions; classic leading indicator |
| Credit spread | `BAA10Y` | Daily → Monthly avg | Level (Baa minus 10Y) | Widens in risk-off; proxy for credit risk appetite |
| VIX | `VIXCLS` | Daily → Monthly avg | Level | Forward-looking equity vol; spikes in crises |
| Initial jobless claims | `ICSA` | Weekly → Monthly avg | 3-month rate of change | Labour market deterioration; leading indicator |
| M2 money supply | `M2SL` | Monthly | Real M2 = M2SL / CPIAUCSL × 100; 12-month % change | Liquidity conditions; monetary policy transmission |
| CPI (for real M2) | `CPIAUCSL` | Monthly | Used to deflate M2 | Not a standalone feature — used only for M2 deflation |
| OECD CLI | `USALOLITONOSTSAM` | Monthly | Level (already standardised around 100) | Composite leading indicator; turning-point signal |
| WTI crude oil | `DCOILWTICO` | Daily → Monthly avg | 3-month % change | Supply/demand shock proxy; cost-push inflation signal |
| Industrial production | `INDPRO` | Monthly | 12-month % change | Real economic activity; coincident indicator |
| Unemployment rate | `UNRATE` | Monthly | Level; also compute 12-month change | Labour market slack; lagging but confirms regime |

**Total macro features: 10 series → 10 transformed indicators**
(CPI is consumed to produce Real M2 but is not a standalone feature in the HMM.)

**Transform details — be precise:**

| Transform | Formula | Notes |
|-----------|---------|-------|
| 3-month rate of change | $(x_t - x_{t-3}) / x_{t-3}$ | Use 3-month lag, not 3-period (ensure monthly frequency first) |
| 12-month % change | $(x_t - x_{t-12}) / x_{t-12}$ | Requires 12 months of warm-up before first valid observation |
| Real M2 | $(M2SL_t / CPIAUCSL_t) \times 100$ | Deflate first, then compute 12-month % change of the deflated series |
| Monthly average | $\bar{x}_t = \frac{1}{N_d} \sum_{d \in \text{month } t} x_d$ | For daily series; $N_d$ = number of trading/observation days in month |

### 3.3 Price Data — yfinance

Used for low-volatility factor construction, tech portfolio, and optional factor replication.

**Factor Universe:**
- **Universe:** S&P 500 current constituents (acknowledge survivorship bias caveat in report)
- **Fields:** Adjusted close prices, daily
- **Period:** 2004-01-01 to 2025-12-31 (extra year for trailing window warm-up)

**Tech Portfolio Universe:**
- **Universe:** 20 tickers listed in §1.6 + 7 benchmark tickers
- **Fields:** OHLCV (Open, High, Low, Close, Volume) — required for Yang-Zhang volatility
- **Period:** 2016-01-01 to 2026-02-28 (or latest available)

**yfinance reliability notes:**
- Yahoo Finance rate-limits aggressive requests. Download in batches of ≤50 tickers with 1-second pauses.
- Use `yfinance.download(tickers, group_by='ticker', auto_adjust=True)` for adjusted prices.
- Retry failed downloads up to 3 times with exponential backoff (2s, 4s, 8s).
- After download, verify: no ticker should have >5% of days as exact zeros (likely data error).
- Save raw download with timestamp to `data/raw/` so re-runs are reproducible without re-fetching.
- **Rate limit workaround:** If download stalls, implement a 15-second cooldown between batches.
- **Delisted tickers:** `yfinance` may return empty DataFrames for delisted stocks. Catch per-ticker
  exceptions, log failures, and proceed with available tickers. Never let one ticker failure
  crash the entire pipeline.

### 3.4 Volume & OHLCV Data Quality Checks

Before proceeding with any analysis, apply these data quality filters:

```python
def validate_ohlcv(df, ticker):
    """Validate OHLCV data quality for a single ticker."""
    checks = {}

    # 1. No negative prices
    checks['negative_prices'] = (df[['Open', 'High', 'Low', 'Close']] < 0).any().any()

    # 2. High >= Low always
    checks['high_lt_low'] = (df['High'] < df['Low']).sum()

    # 3. Close within [Low, High] range
    checks['close_out_of_range'] = (
        (df['Close'] > df['High'] * 1.001) | (df['Close'] < df['Low'] * 0.999)
    ).sum()

    # 4. No exact zeros in Close (likely data error, not a real $0 stock)
    checks['zero_close_pct'] = (df['Close'] == 0).mean()

    # 5. Maximum gap: no gap > 10 consecutive trading days (halts excepted)
    valid_days = df['Close'].dropna()
    max_gap = valid_days.index.to_series().diff().max().days
    checks['max_gap_days'] = max_gap

    # 6. Return spikes: flag |daily return| > 25%
    returns = df['Close'].pct_change()
    checks['spike_count'] = (returns.abs() > 0.25).sum()

    return checks
```

### 3.5 Data Dictionary — Complete Schema Reference

Every `.parquet` output must conform to these schemas. Validation code in `src/validation.py`
checks schemas on load.

**Factor Timing Component:**

| File | Index | Columns | Units | Dtype |
|------|-------|---------|-------|-------|
| `factor_returns.parquet` | `date` (datetime, month-end) | `mkt_rf, smb, hml, rmw, cma, rf, umd` | Decimal returns (0.01 = 1%) | float64 |
| `macro_indicators.parquet` | `date` (datetime, month-end) | `t10y2y, baa10y, vix, claims_roc, real_m2_yoy, oecd_cli, oil_roc, indpro_yoy, unrate, unrate_chg` | Mixed (see transforms) | float64 |
| `sp500_daily_prices.parquet` | `date` (datetime, trading day) | `ticker, adj_close` | USD | float64 |
| `sp500_monthly_returns.parquet` | `date` (datetime, month-end) | `ticker, monthly_return` | Decimal returns | float64 |
| `lowvol_factor_returns.parquet` | `date` (datetime, month-end) | `lowvol` | Decimal returns | float64 |
| `factor_returns_full.parquet` | `date` (datetime, month-end) | `hml, umd, rmw, lowvol` + controls | Decimal returns | float64 |
| `macro_composite_index.parquet` | `date` (datetime, month-end) | `composite_z` | Standardised (z-score) | float64 |
| `regime_probabilities.parquet` | `date` (datetime, month-end) | `p_expansion, p_slowdown, p_crisis, regime_label` | Probabilities [0,1]; label is string | float64 / str |
| `garch_conditional_vol.parquet` | `date` (datetime, month-end) | `vol_hml, vol_umd, vol_rmw, vol_lowvol` | Annualised decimal vol | float64 |
| `dcc_conditional_corr.parquet` | `date` (datetime, month-end) | `corr_hml_umd, corr_hml_rmw, ...` (6 pairs) | Correlation [-1, 1] | float64 |
| `conditional_covariance.parquet` | `date` (datetime, month-end) | 16 entries of flattened 4×4 matrix | Monthly variance/covariance (decimal²) | float64 |
| `bl_weights_timeseries.parquet` | `date` (datetime, month-end) | `w_hml, w_umd, w_rmw, w_lowvol` | Portfolio weight [0, 0.4] | float64 |
| `cvar_weights_timeseries.parquet` | `date` (datetime, month-end) | `w_hml, w_umd, w_rmw, w_lowvol` | Portfolio weight [0, 0.4] | float64 |
| `backtest_returns.parquet` | `date` (datetime, month-end) | Strategy and benchmark columns | Decimal returns (net of TC) | float64 |
| `backtest_nav.parquet` | `date` (datetime, month-end) | Strategy and benchmark columns | Index (base = 100) | float64 |

**Tech Portfolio Component:**

| File | Index | Columns | Units | Dtype |
|------|-------|---------|-------|-------|
| `master_data.parquet` | `date` (datetime, trading day) | 20 tickers + benchmarks (adj close) | USD | float64 |
| `macro_regimes.parquet` | `date` (datetime, trading day) | `macro_regime` | Categorical string | str |
| `garch_parameters.csv` | row index | `ticker, model, distribution, omega, alpha, beta, gamma, aic, bic` | Model parameters | float64 |
| `conditional_vol_series.parquet` | `date` (datetime, trading day) | `vol_{ticker}` per ticker | Annualised decimal vol | float64 |
| `var_cvar_table.csv` | row index | `ticker, alpha, var_historical, var_gaussian, var_cornish_fisher, cvar_*` | Decimal loss | float64 |
| `evt_parameters.csv` | `ticker` | `xi, beta, threshold, n_exceed, evt_var_99, evt_cvar_99, evt_var_999` | GPD parameters | float64 |
| `return_scenarios.parquet` | `date` (datetime, trading day) | 20 ticker columns | Simple returns | float64 |
| `sentiment_features.parquet` | `date` (datetime, trading day) | `sentiment_mean, sentiment_std, sentiment_volume, sentiment_momentum, ticker` | Various | float64 |
| `regime_labels.parquet` | `date` (datetime, trading day) | `regime_state, regime_prob_0, regime_prob_1, ...` | State labels + probabilities | int/float64 |
| `vol_forecast_predictions.parquet` | `date` (datetime, trading day) | `y_true, y_pred, model, ticker` | Annualised vol | float64 |
| `portfolio_weights_timeseries.parquet` | `date` (datetime, trading day) | 20 ticker weight columns | Weight [0, 0.1] | float64 |
| `backtest_performance.csv` | strategy name | Performance metrics columns | Various | float64 |
| `stress_test_results.csv` | row index | `scenario, cumulative_return, max_daily_loss, n_days` | Returns | float64 |

### 3.6 Date Alignment Protocol

All datasets must be aligned before any cross-dataset operation:

```python
def align_dates(dfs, method='inner', freq='M'):
    """
    Align multiple DataFrames to a common date index.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        DataFrames with DatetimeIndex
    method : str
        'inner' (intersection) or 'outer' (union with NaN)
    freq : str
        'M' for month-end, 'B' for business day, 'D' for calendar day

    Returns
    -------
    list of pd.DataFrame
        Aligned DataFrames
    """
    if freq == 'M':
        # Normalise all to month-end
        for i in range(len(dfs)):
            dfs[i].index = dfs[i].index + pd.offsets.MonthEnd(0)

    if method == 'inner':
        common_idx = dfs[0].index
        for df in dfs[1:]:
            common_idx = common_idx.intersection(df.index)
        return [df.loc[common_idx] for df in dfs]
    else:
        all_idx = dfs[0].index
        for df in dfs[1:]:
            all_idx = all_idx.union(df.index)
        return [df.reindex(all_idx) for df in dfs]
```

**Critical rules:**
1. Factor returns from French Library use end-of-month dates
2. FRED monthly series may use first-of-month or end-of-month depending on series
3. Normalise ALL to end-of-month before joining: `df.index = df.index + pd.offsets.MonthEnd(0)`
4. Inner join on date index across factor returns, macro data, and price data
5. **Verify no duplicate dates** after alignment: `assert df.index.is_unique`
6. Trim to common date range: **2005-01 to 2025-12** (252 monthly observations for factor timing)


## 4. PHASE SPECIFICATIONS

---

### PHASE 1 — Data Pipeline
**Notebook:** `01_data_pipeline.ipynb`  
**Dependency:** None (entry point)  
**Estimated runtime:** 5–15 minutes (dominated by yfinance download)

#### Objective
Download, clean, align, and export all raw data into standardised monthly `.parquet` files.

#### Steps

1. **Download French Library data**
   - Fetch Fama/French 5 Factors (monthly) and Momentum Factor (monthly)
   - Use `pandas_datareader.famafrench` (preferred) or direct CSV zip
   - **Critical:** parse dates from YYYYMM → `datetime` with month-end convention
   - **Critical:** convert returns from percentage to decimal (divide by 100) immediately
   - Align column names: `mkt_rf, smb, hml, rmw, cma, rf, umd`

   ```python
   import pandas_datareader.data as pdr
   ff5 = pdr.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start='2004')
   ff5_monthly = ff5[0] / 100  # Convert % to decimal
   mom = pdr.DataReader('F-F_Momentum_Factor', 'famafrench', start='2004')
   mom_monthly = mom[0] / 100
   ```

2. **Download FRED macro data**
   - Use `fredapi.Fred` with API key from `.env`
   - **Edge case:** FRED sometimes returns `'.'` as missing. Use `errors='coerce'`.

   ```python
   from fredapi import Fred
   fred = Fred(api_key=FRED_API_KEY)
   raw = fred.get_series('T10Y2Y', observation_start='2003-01-01')
   raw = pd.to_numeric(raw, errors='coerce')
   monthly = raw.resample('M').mean()
   ```

3. **Download S&P 500 price data** — batches of 50, 1s pauses, SHA-256 checksums
4. **Download Tech Portfolio data** (20 tickers + benchmarks) — handle SQ→XYZ merger

   ```python
   def merge_sq_xyz():
       sq = yf.download('SQ', start='2016-01-01', end='2025-01-21', auto_adjust=True)
       xyz = yf.download('XYZ', start='2025-01-21', auto_adjust=True)
       merged = pd.concat([sq, xyz]).sort_index()
       return merged[~merged.index.duplicated(keep='last')]
   ```

5. **Align all datasets** — end-of-month convention, inner join, verify `df.index.is_unique`
6. **Compute base features** — log returns, simple returns, summary stats, ADF/KPSS tests

#### Validation Gates
- [ ] Factor returns: Mkt-RF monthly mean ≈ 0.004–0.008; HML, UMD, RMW means positive
- [ ] Factor returns: max |monthly return| < 0.30 (% vs decimal sanity check)
- [ ] Macro data: no NaN; all series have 252 monthly observations
- [ ] All `.parquet` files share identical date index
- [ ] Tech portfolio: all 20 tickers present; short-history tickers have correct IPO dates
- [ ] SQ→XYZ merge: no gap/overlap at 2025-01-21
- [ ] Log returns pass ADF test (reject unit root at 1%)

---

### PHASE 2 — Low-Volatility Factor Construction
**Notebook:** `02_factor_construction.ipynb`  
**Dependency:** Phase 1 outputs

#### Objective
Construct the low-volatility factor from S&P 500 stock-level data.

#### Method

**Trailing realised volatility** for each stock $i$ at month-end $t$:

$$\sigma_{i,t} = \sqrt{\frac{252}{D} \sum_{d=1}^{D} (r_{i,d} - \bar{r}_i)^2}$$

- $D = 60$ trailing days; require $D \geq 40$ valid days
- **Quintile portfolios:** rank by $\sigma$ ascending, equal-weight returns in month $t+1$
- **Long-short:** $\text{LowVol}_{t+1} = R_{Q1,t+1} - R_{Q5,t+1}$
- **No look-ahead:** signal at $t$, return at $t+1$

**Yang-Zhang volatility estimator (enhancement):**

```python
def yang_zhang_vol(df, window=60):
    """Range-based estimator using OHLC — more efficient than close-to-close."""
    log_ho = np.log(df['High'] / df['Open'])
    log_lo = np.log(df['Low'] / df['Open'])
    log_co = np.log(df['Close'] / df['Open'])
    log_oc = np.log(df['Open'] / df['Close'].shift(1))
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    n = window
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    yz_var = log_oc.rolling(n).var() + k * log_co.rolling(n).var() + (1-k) * rs.rolling(n).mean()
    return np.sqrt(yz_var * 252)
```

#### Validation Gates
- [ ] LowVol mean monthly return positive (~0.001–0.004)
- [ ] LowVol corr with Mkt-RF negative or near zero
- [ ] ≥30 stocks per quintile at every rebalance date
- [ ] Q1 has lower realised vol than Q5 out-of-sample

---

### PHASE 3 — Factor Validation & GARCH Modeling
**Notebook:** `03_factor_validation.ipynb`  
**Dependency:** Phase 2 outputs

#### Objective
Confirm 4 primary factors capture distinct risk premia. Begin GARCH volatility modeling.

#### Factor Summary Statistics

For each factor compute: annualised mean, vol, Sharpe, Sortino, max drawdown, skewness,
kurtosis, t-statistic. Use **Newey-West HAC standard errors** (lag ≈ 4).

```python
import statsmodels.api as sm
model = sm.OLS(lowvol, sm.add_constant(ff_factors))
results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 4})
```

#### GARCH Pipeline for Tech Portfolio

**20 tickers × 4 models × 4 distributions = 320 fits:**

| Model | Captures |
|-------|----------|
| GARCH(1,1) | Volatility clustering |
| GJR-GARCH(1,1,1) | Leverage effect (bad-news asymmetry) |
| EGARCH(1,1) | Log-variance, no positivity constraint |
| FIGARCH(1,d,1) | Long memory (**exclude for tickers with <1500 obs**) |

| Distribution | When |
|-------------|------|
| Normal | Baseline |
| Student's t | Moderate fat tails |
| Skewed-t | Asymmetric fat tails |
| GED | Alternative heavy tails |

```python
from arch import arch_model

def fit_garch(returns, model_type='GARCH', dist='normal'):
    """CRITICAL: arch expects percentage returns. Multiply by 100."""
    scaled = returns * 100
    specs = {
        'GARCH': dict(vol='GARCH', p=1, o=0, q=1),
        'GJR-GARCH': dict(vol='GARCH', p=1, o=1, q=1),
        'EGARCH': dict(vol='EGARCH', p=1, o=1, q=1),
        'FIGARCH': dict(vol='FIGARCH', p=1, o=0, q=1),
    }
    model = arch_model(scaled, mean='Constant', dist=dist, **specs[model_type])
    try:
        result = model.fit(disp='off', options={'maxiter': 500})
        return result if result.convergence_flag == 0 else None
    except:
        return None
```

#### Validation Gates
- [ ] All 4 factors have positive mean returns; ≥3 have $t > 2$ (HAC SEs)
- [ ] No factor pair correlation exceeds 0.6
- [ ] LowVol regression R² < 0.5
- [ ] GARCH convergence ≥90% of fits; $\alpha + \beta \in [0.85, 0.99]$
- [ ] FIGARCH excluded for CRWD, DDOG, PLTR (<1500 obs)

---

### PHASE 4 — Macro Regime Detection (HMM)
**Notebook:** `04_macro_regime_hmm.ipynb`  
**Dependency:** Phase 1 macro outputs  
**Estimated runtime:** 2–5 minutes

#### Objective
Fit 3-state Gaussian HMM on composite macro index. Detect expansion/slowdown/crisis.

#### Composite Macro Index (Expanding-Window PCA)

$$z_{i,t} = \frac{x_{i,t} - \bar{x}_{i,1:t}}{\sigma_{x_i,1:t}}$$

Start at $t \geq 24$ (24-month minimum for stable statistics).

```python
from sklearn.decomposition import PCA
composite = np.full(T, np.nan)
for t in range(min_window, T):
    pca = PCA(n_components=1)
    pca.fit(z_standardised[:t+1])
    composite[t] = pca.transform(z_standardised[t:t+1].reshape(1, -1))[0, 0]
```

**Sign convention:** first PC loads positive on good indicators (IP, CLI), negative on bad (VIX).

#### HMM Fitting — Multiple Restarts

```python
from hmmlearn import hmm
best_model, best_score = None, -np.inf
for seed in range(25):
    model = hmm.GaussianHMM(n_components=3, covariance_type="full",
                             n_iter=500, tol=1e-6, random_state=seed)
    try:
        model.fit(Z.reshape(-1, 1))
        score = model.score(Z.reshape(-1, 1))
        if score > best_score:
            best_score, best_model = score, model
    except ValueError:
        continue
```

**Regime labelling:** sort states by $\mu_k$ (highest = Expansion, lowest = Crisis).

#### Filtered Probabilities (NO Look-Ahead)

```python
def filtered_probabilities(model, observations):
    """Forward-only algorithm. NEVER use predict_proba() (smoothed = look-ahead)."""
    framelogprob = model._compute_log_likelihood(observations)
    T, K = len(observations), model.n_components
    log_pi = np.log(model.startprob_ + 1e-300)
    log_A = np.log(model.transmat_ + 1e-300)
    fwd = np.zeros((T, K))
    fwd[0] = log_pi + framelogprob[0]
    for t in range(1, T):
        for j in range(K):
            fwd[t, j] = np.logaddexp.reduce(fwd[t-1] + log_A[:, j]) + framelogprob[t, j]
    log_norm = np.logaddexp.reduce(fwd, axis=1, keepdims=True)
    return np.exp(fwd - log_norm)
```

#### BIC Model Selection

$$\text{BIC} = -2 \ln L + p \ln T, \quad p = K^2 + 2K - 1$$

Test $K \in \{2, 3, 4\}$. Cap at $K = 4$ (need ~50 obs per state).

#### Manual Regime Labels for Tech Portfolio

```python
def label_macro_regime(date):
    if date < pd.Timestamp('2018-01-01'): return 'QE Era'
    elif date < pd.Timestamp('2019-01-01'): return 'Rate Hike'
    elif date < pd.Timestamp('2020-02-19'): return 'Late Cycle'
    elif date < pd.Timestamp('2020-04-01'): return 'COVID Crash'
    elif date < pd.Timestamp('2021-12-01'): return 'Zero-Rate Recovery'
    elif date < pd.Timestamp('2023-10-01'): return 'Inflation/Rate Shock'
    elif date < pd.Timestamp('2025-01-01'): return 'AI Boom'
    else: return 'Tariff/Geopolitical'
```

#### Expanding-Window Re-estimation Cache

```python
import joblib, hashlib
def get_hmm_model(data, cache_dir='data/interim/hmm_cache/'):
    data_hash = hashlib.md5(data.tobytes()).hexdigest()[:12]
    path = f"{cache_dir}/hmm_{data_hash}.pkl"
    if os.path.exists(path): return joblib.load(path)
    model = fit_hmm_with_restarts(data, n_restarts=25)
    joblib.dump(model, path)
    return model
```

#### Validation Gates
- [ ] Crisis aligns with NBER recessions (2007-09, 2020) — ≥80% overlap
- [ ] Transition matrix diagonal > 0.7 (persistent regimes)
- [ ] BIC: 3-state < 2-state (or document if 2-state wins)
- [ ] Crisis < 20% of months; expansion > 40%
- [ ] **Filtered** probabilities used, not smoothed
- [ ] HMM converged; top 3 restarts have consistent log-likelihood
- [ ] PCA 1st component explains > 30% variance

---

### PHASE 5 — Regime-Conditional Factor Analysis
**Notebook:** `05_regime_conditional_analysis.ipynb`  
**Dependency:** Phase 3 + Phase 4

#### Objective
Quantify how factor behaviour changes across regimes. This is the **statistical core**.

#### Regime-Conditional Statistics

For each factor $f$ and regime $k$: mean, vol, Sharpe, max drawdown, skewness, hit rate.

**Tests:** Welch's t-test (pairwise) with **Holm-Bonferroni** correction. Kruskal-Wallis
non-parametric test. Bootstrap 10,000 resamples for Sharpe ratio differences.

```python
from statsmodels.stats.multitest import multipletests
reject, pvals_corrected, _, _ = multipletests(pvals_raw, alpha=0.05, method='holm')
```

#### Key Hypotheses

| # | Hypothesis | Test |
|---|-----------|------|
| H1 | Momentum crashes in crisis-to-recovery transitions | One-sided t-test on UMD during transition months |
| H2 | Value outperforms in early expansion | Subsample comparison + bootstrap CI |
| H3 | Quality/LowVol dominate in crisis | Pairwise bootstrap Sharpe difference |
| H4 | Cross-factor correlations increase in crisis | Fisher z-transform test |

#### Validation Gates
- [ ] ≥2 of 4 factors show significant regime-dependent means ($p < 0.05$, Holm-corrected)
- [ ] UMD has worst crisis performance; RMW or LowVol has best
- [ ] Correlations higher in crisis than expansion
- [ ] Each regime has ≥20 months; bootstrap CIs reported

---

### PHASE 6 — DCC-GARCH Dynamic Correlations
**Notebook:** `06_dcc_garch.ipynb`  
**Dependency:** Phase 2

#### Objective
Time-varying volatilities and correlations for the 4 factors → allocation optimisers.

#### Univariate GARCH(1,1) per Factor

```python
from arch import arch_model
for factor in ['hml', 'umd', 'rmw', 'lowvol']:
    model = arch_model(returns[factor] * 100, mean='Constant', vol='GARCH', p=1, q=1)
    result = model.fit(disp='off')
    assert result.convergence_flag == 0
```

#### DCC(1,1): $Q_t = (1-a-b)\bar{Q} + a\hat{z}_{t-1}\hat{z}_{t-1}' + bQ_{t-1}$

Conditional covariance: $\hat{\Sigma}_t = D_t R_t D_t$

**PSD enforcement at every $t$:**
```python
def ensure_psd(sigma):
    eigvals, eigvecs = np.linalg.eigh(sigma)
    eigvals = np.maximum(eigvals, 1e-10)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T
```

#### Validation Gates
- [ ] $\alpha + \beta \in [0.85, 0.99]$; DCC $a + b < 1$
- [ ] Ljung-Box on residuals and squared residuals: $p > 0.05$
- [ ] All $\hat{\Sigma}_t$ are PSD; correlations increase in crisis

---

### PHASE 7 — Black-Litterman Allocation
**Notebook:** `07_allocation_blacklitterman.ipynb`  
**Dependency:** Phase 4 + Phase 5 + Phase 6

#### Objective
Bayesian BL model where regime detection feeds expected return views.

#### Equilibrium Prior: $\pi = \delta \hat{\Sigma}_t w_{eq}$

$\delta \approx 2.5$ (risk aversion); $w_{eq} = [0.25, 0.25, 0.25, 0.25]$

#### Regime Views with Law of Total Variance

$$Q_{f,t} = \sum_{k=1}^{3} p_{k,t} \cdot \bar{r}_{f,k}$$

$$\Omega_{f,t} = \underbrace{\sum_k p_k \sigma_{f,k}^2}_{\text{E[Var]}} + \underbrace{\sum_k p_k (\bar{r}_{f,k} - Q_{f,t})^2}_{\text{Var[E]}}$$

**View capping:** $|Q_{f,t}| \leq 0.03$ (3% monthly). $\Omega$ floor at 0.0001.

#### BL Posterior

$$\mu_{BL} = [(\tau\hat{\Sigma})^{-1} + P'\Omega^{-1}P]^{-1}[(\tau\hat{\Sigma})^{-1}\pi + P'\Omega^{-1}Q]$$

$\tau = 0.05$; $P = I_{4×4}$ (absolute views).

#### Optimization with Constraints

```python
import cvxpy as cp
def bl_optimize(mu_bl, sigma_bl, delta, w_prev=None, max_w=0.4, max_to=0.15):
    n = len(mu_bl)
    w = cp.Variable(n)
    obj = cp.Maximize(mu_bl @ w - (delta/2) * cp.quad_form(w, sigma_bl))
    cons = [cp.sum(w) == 1, w >= 0, w <= max_w]
    if w_prev is not None:
        cons.append(cp.norm(w - w_prev, 1) <= max_to * n)
    cp.Problem(obj, cons).solve(solver=cp.SCS, warm_start=True)
    return w.value
```

#### Validation Gates
- [ ] Weights sum to 1.0 (±1e-6); no weight > 0.40; turnover ≤ 0.15 per factor
- [ ] Crisis: quality/low-vol weight ↑, momentum ↓; Expansion: momentum ↑
- [ ] $|\mu_{BL}| < 0.05$ monthly; $\Omega > 0$ always

---

### PHASE 8 — Mean-CVaR Risk Parity
**Notebook:** `08_allocation_mean_cvar.ipynb`  
**Dependency:** Phase 4 + Phase 5 + Phase 6

#### Objective
Alternative allocation using CVaR risk measure. Benchmark against BL.

#### Risk Parity Base: $w_{RP,i} = (1/\sigma_i) / \sum_j (1/\sigma_j)$

#### Regime Tilt: $\text{tilt}_{i} = 1 + \lambda(\sum_k p_k \text{SR}_{i,k} - \overline{\text{SR}}_i)$

Clip tilt to $[0.5, 2.0]$. Test $\lambda \in \{0.5, 1.0, 1.5, 2.0\}$.

#### CVaR Optimization (Rockafellar-Uryasev LP)

$$\min_{w, \zeta, u} \quad \zeta + \frac{1}{S(1-\alpha)} \sum_s u_s$$

```python
def cvar_optimize(scenarios, mu_regime, r_target, alpha=0.95, max_w=0.4):
    S, N = scenarios.shape
    w, zeta, u = cp.Variable(N), cp.Variable(), cp.Variable(S)
    obj = cp.Minimize(zeta + (1/(S*(1-alpha))) * cp.sum(u))
    cons = [u >= -scenarios @ w - zeta, u >= 0,
            mu_regime @ w >= r_target, cp.sum(w) == 1, w >= 0, w <= max_w]
    cp.Problem(obj, cons).solve(solver=cp.ECOS)
    return w.value
```

#### Validation Gates
- [ ] CVaR weights differ from BL weights
- [ ] CVaR portfolio has lower tail risk (5th percentile)
- [ ] Tilt multipliers bounded [0.5, 2.0]; weights sum to 1.0

---

### PHASE 9 — Walk-Forward Backtest Engine
**Notebook:** `09_backtest_engine.ipynb`  
**Dependency:** Phase 7 + Phase 8  
**Estimated runtime:** 10–30 minutes

#### Walk-Forward Protocol

```
WARM-UP: Months 1–60 (2005-01 to 2009-12) — estimation only
LIVE: Months 61–252 (2010-01 to 2025-12) — 192 out-of-sample months

For each month t from 60 to T-1:
  1. ESTIMATE: expanding window [1, t] — re-estimate PCA, HMM, GARCH/DCC
  2. SIGNAL: filtered P(S_t = k | Z_{1:t}); compute BL/CVaR weights
  3. EXECUTE: r_net,t+1 = Σ w_i,t × r_i,t+1 - TC × Σ|w_i,t - w_i,t-1|
     TC = 0.0025 (25 bps one-way); NAV: V_{t+1} = V_t × (1 + r_net,t+1)
```

#### Benchmarks

| Benchmark | Description |
|-----------|-------------|
| Equal-Weight Static | $w = [0.25]^4$ rebalanced monthly |
| Inverse-Vol Static | Risk parity without regime tilts |
| Market (Mkt-RF + RF) | S&P 500 total return |
| 60/40 | 60% (Mkt-RF + RF) + 40% RF |

#### Performance Metrics

| Metric | Formula |
|--------|---------|
| Annualised return | $12 \times \bar{r}_{net}$ |
| Annualised vol | $\sqrt{12} \times \text{std}(r_{net})$ |
| Sharpe | $\mu / \sigma$ |
| Sortino | $\mu / (\sqrt{12} \times \sigma_{down})$ |
| Max drawdown | $\max_t (1 - V_t / \max_{s \leq t} V_s)$ |
| Calmar | $\mu / \text{MDD}$ |
| Annual turnover | $\sum |w_t - w_{t-1}| \times 12 / \text{years}$ |
| Information ratio | $(\mu_{strat} - \mu_{bench}) / \text{TE}$ |

#### Sharpe Significance: Jobson-Korkie Test

$$z_{JK} = \frac{SR_1 - SR_2}{\sqrt{\frac{1}{T}(2(1-\hat{\rho}) + \frac{1}{2}(SR_1^2 + SR_2^2 - 2 SR_1 SR_2 \hat{\rho}^2))}}$$

#### Validation Gates
- [ ] BL and CVaR both outperform equal-weight static on Sharpe (≥0.15 improvement)
- [ ] Max drawdown of dynamic ≤ static strategies
- [ ] Turnover < 300% annually; TC < 100 bps annually
- [ ] NAV strictly positive; Sharpe significance $p < 0.10$
- [ ] **Negative results documented honestly if they occur**

---

### PHASE 10 — Stress Testing & Tail Risk
**Notebook:** `10_stress_testing.ipynb`  
**Dependency:** Phase 9

#### Named Stress Periods

| Event | Period | Dominant Risk |
|-------|--------|---------------|
| GFC | 2007-10 to 2009-03 | Credit, liquidity |
| European Debt | 2011-07 to 2011-12 | Sovereign contagion |
| Taper Tantrum | 2013-05 to 2013-09 | Rate shock |
| China/Oil | 2015-08 to 2016-02 | Growth, commodity |
| COVID-19 | 2020-02 to 2020-04 | Liquidity, volatility |
| Rate Shock 2022 | 2022-01 to 2022-10 | Duration, inflation |

#### EVT: Generalised Pareto Distribution

$$F_u(y) = 1 - (1 + \xi y / \sigma_u)^{-1/\xi}$$

```python
from scipy.stats import genpareto
losses = -returns[returns < 0]
threshold = np.percentile(losses, 90)
exceedances = losses[losses > threshold] - threshold
xi, _, sigma = genpareto.fit(exceedances, floc=0)
```

$\xi > 0$: heavy tails (expected). Require $N_u \geq 15$ exceedances.

#### Validation Gates
- [ ] Dynamic strategies smaller drawdowns in ≥4 of 6 stress periods
- [ ] GPD shape $\xi > 0$ (heavy tails confirmed)
- [ ] EVT ES₉₉ > historical ES₉₉
- [ ] $N_u \geq 15$ exceedances

---

### PHASE 11 — Performance Report & Dashboard
**Notebook:** `11_performance_report.ipynb`  
**Dependency:** All previous

#### Excel Dashboard (10 tabs)
1. Executive Summary  2. Performance Summary  3. Annual Returns  4. Regime Timeline
5. Weight History  6. Stress Test Results  7. EVT Tail Metrics
8. Factor Conditional Stats  9. Hypothesis Tests  10. Model Diagnostics

#### PDF Report: thesis → methodology → results → limitations → appendix

---

### PHASE 12 — Presentation Export
**Notebook:** `12_presentation_export.ipynb`  
**Dependency:** Phase 11

#### 20-Slide Structure
1. Title  2. Thesis  3. Problem  4. Methodology  5. Data  6. Factor performance
7. Regime detection  8. Regime-conditional Sharpe  9. Momentum crashes
10. DCC-GARCH  11. BL allocation  12. CVaR allocation  13. Backtest NAV
14. Stress testing  15. Alpha decomposition  16. Tail risk  17. Limitations
18. Conclusion  19. Appendix: stats  20. Appendix: diagnostics

---

## 5. DEPENDENCY GRAPH

```
Phase 1 (Data Pipeline)
├──→ Phase 2 (LowVol Construction)
│    └──→ Phase 3 (Factor Validation + GARCH)
│         └──→ Phase 5 (Regime-Conditional) ←── Phase 4
│              ├──→ Phase 7 (BL Allocation) ←── Phase 6 (DCC) ←── Phase 2
│              └──→ Phase 8 (CVaR Allocation) ←── Phase 6
├──→ Phase 4 (HMM Regime Detection)
│
Phase 7 + 8 ──→ Phase 9 (Backtest) ──→ Phase 10 (Stress) ──→ Phase 11 (Report) ──→ Phase 12 (Deck)
```

**Critical path:** 1 → 2 → 3 → 5 → 7 → 9 → 10 → 11 → 12  
**Parallel A:** 1 → 4 (simultaneous with 2 → 3)  
**Parallel B:** 2 → 6 (simultaneous with 4 → 5)

---

## 6. ML/DL MODEL GOVERNANCE & ANTI-LEAKAGE PROTOCOL

### 6.1 Walk-Forward Expanding Window (Mandatory)

**Never use k-fold cross-validation on time series data.** The temporal structure means
future data leaks into training folds. Use **walk-forward expanding window** exclusively:

```python
def walk_forward_predict(X, y, pipeline_factory, retrain_freq=63,
                         initial_train_ratio=0.7, task='regression'):
    """
    Walk-forward expanding-window prediction.

    Parameters
    ----------
    X : pd.DataFrame — features (DatetimeIndex)
    y : pd.Series — target
    retrain_freq : int — retrain every N observations (63 ≈ quarterly)
    initial_train_ratio : float — fraction of data for initial training

    Returns
    -------
    pd.DataFrame with columns: date, y_true, y_pred, [y_prob for classification]
    """
    T = len(X)
    split = int(T * initial_train_ratio)
    results = []
    pipeline = None

    for t in range(split, T):
        # Retrain periodically
        if pipeline is None or (t - split) % retrain_freq == 0:
            pipeline = pipeline_factory()
            pipeline.fit(X.iloc[:t], y.iloc[:t])  # Expanding window: [0, t)

        # Predict ONE step ahead
        pred = pipeline.predict(X.iloc[t:t+1])
        row = {'date': X.index[t], 'y_true': y.iloc[t], 'y_pred': pred[0]}

        if task == 'classification' and hasattr(pipeline, 'predict_proba'):
            row['y_prob'] = pipeline.predict_proba(X.iloc[t:t+1])[0, 1]

        results.append(row)

    return pd.DataFrame(results)
```

### 6.2 Anti-Leakage Checklist

**Before any model training, verify ALL of the following:**

| # | Check | How | Consequence if Violated |
|---|-------|-----|------------------------|
| 1 | **Target alignment** | Target at time $t$ uses only data at $t+h$ or later | Inflated accuracy; model "predicts" the present |
| 2 | **Feature lag** | All features use data ≤ $t-1$ (especially sentiment) | Contemporaneous correlation ≠ predictive power |
| 3 | **Scaler inside pipeline** | `StandardScaler` must be in pipeline, not pre-fit on full data | Test data statistics leak into training |
| 4 | **SMOTE only for classification** | Never apply SMOTE to regression targets | Synthesises impossible continuous values |
| 5 | **No shuffle** | `shuffle=False` in all splits | Temporal ordering destroyed |
| 6 | **Expanding, not rolling** | Training set grows over time | Discarding early data inefficient |
| 7 | **PCA expanding** | PCA loadings computed on data ≤ $t$ | Future variance structure leaks in |
| 8 | **HMM expanding** | HMM fitted on data ≤ $t$ | Future regime transitions leak |

### 6.3 Sentiment Lag Enforcement (Critical)

All sentiment features MUST be lagged by $t-1$:

```python
# WRONG: same-day sentiment → same-day return (contemporaneous)
features['sentiment'] = sentiment_score

# CORRECT: yesterday's sentiment → today's return
features['sentiment'] = sentiment_score.shift(1)

# VERIFY the lag
assert features.loc['2024-01-15', 'sentiment'] == raw_sentiment.loc['2024-01-14']
```

**Why:** A 2 PM headline affects both the sentiment score AND the close-to-close return
of the same day. Using same-day sentiment creates spurious correlation that disappears
out-of-sample.

### 6.4 SMOTE Protocol

```python
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline

# CORRECT: SMOTE inside walk-forward, applied only to training fold
def make_classification_pipeline(model_name, use_smote=True):
    from imblearn.pipeline import Pipeline as ImbPipeline
    steps = [('scaler', StandardScaler())]
    if use_smote:
        steps.append(('smote', SMOTE(random_state=42)))
    steps.append(('model', get_classifier(model_name)))
    return ImbPipeline(steps)

# WRONG: SMOTE on full dataset before split
# WRONG: SMOTE on regression targets
```

### 6.5 Model Comparison Tests

**Diebold-Mariano test** for equal predictive accuracy:

$$DM = \frac{\bar{d}}{\sqrt{\hat{\sigma}_d^2 / T}}, \quad d_t = L(e_{1,t}) - L(e_{2,t})$$

Where $L$ is the loss function (MSE for regression, 0-1 loss for classification).

**Mincer-Zarnowitz test** for forecast calibration:

$$y_t = \alpha + \beta \hat{y}_t + \epsilon_t$$

Test $H_0: \alpha = 0, \beta = 1$ (joint F-test). Rejection means forecasts are biased.

### 6.6 Feature Importance Consistency

Compare SHAP vs permutation importance. If top-5 features differ substantially,
the model may be unstable:

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap_importance = np.abs(shap_values).mean(axis=0)

# Permutation importance
from sklearn.inspection import permutation_importance
perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
```

---

## 7. DEEP LEARNING ARCHITECTURE SPECIFICATION

### 7.1 LSTM Forecaster

```python
import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0)
        # CRITICAL: PyTorch dropout only applies BETWEEN LSTM layers,
        # not after the last layer. Add explicit dropout.
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # Take last timestep
        dropped = self.dropout(last_hidden)  # Explicit dropout after last layer
        return self.fc(dropped).squeeze(-1)
```

### 7.2 GRU Forecaster

```python
class GRUForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers,
                          batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(self.dropout(out[:, -1, :])).squeeze(-1)
```

### 7.3 Sequence Creation

```python
def create_sequences(features, target, lookback=60):
    """Create sliding window sequences for RNN input.
    features: (T, F) array; target: (T,) array
    Returns: X (N, lookback, F), y (N,)
    """
    X, y = [], []
    for i in range(lookback, len(features)):
        X.append(features[i-lookback:i])
        y.append(target[i])
    return np.array(X), np.array(y)
```

### 7.4 Training Protocol

```python
def train_model(model, X_train, y_train, X_val, y_val,
                epochs=100, batch_size=64, lr=1e-3, patience=15):
    """Train with early stopping. Returns history dict."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss, patience_counter = float('inf'), 0
    history = {'train_loss': [], 'val_loss': [], 'stopped_epoch': epochs}

    for epoch in range(epochs):
        model.train()
        # ... batch training loop ...
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.FloatTensor(X_val))
            val_loss = criterion(val_pred, torch.FloatTensor(y_val)).item()
        history['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'outputs/models/best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                history['stopped_epoch'] = epoch
                break
    return history
```

**Key rules:**
- **Temporal split only** — no shuffle, no random split
- **Validation set:** last 15% of training data (not random subset)
- **Early stopping patience:** 15 epochs (prevents overfitting)
- **Gradient clipping:** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
- **GPU detection:** `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`

---

## 8. HYBRID MODEL ENSEMBLE PROTOCOL

### 8.1 Three-Stage Architecture

```
Stage 1: GARCH(1,1) → baseline conditional volatility σ_garch
Stage 2: XGBoost   → predicts residual (σ_actual - σ_garch) using features
Stage 3: LSTM      → adjustment using recent sequence patterns

Final: σ_hybrid = w1 × σ_garch + w2 × σ_xgb + w3 × σ_lstm
```

### 8.2 Weight Optimization

```python
from scipy.optimize import minimize

def hybrid_loss(weights, preds_dict, y_true):
    """Minimize RMSE of weighted combination."""
    w = weights / weights.sum()  # Normalise
    combined = sum(w[i] * preds_dict[name] for i, name in enumerate(preds_dict))
    return np.sqrt(np.mean((y_true - combined) ** 2))

result = minimize(hybrid_loss, x0=[1/3, 1/3, 1/3],
                  args=(preds, y_val), method='Nelder-Mead',
                  bounds=[(0, 1)] * 3)
```

### 8.3 Model Confidence Set

Use Hansen's Model Confidence Set (MCS) to determine which models belong to the "best" set
at a given confidence level. This avoids cherry-picking the single "best" model.

---

## 9. COPULA & TAIL RISK MODELING

### 9.1 VaR Methods (5 Methods × 20 Tickers)

| Method | Formula | Notes |
|--------|---------|-------|
| Historical | Empirical quantile | Non-parametric; no distribution assumption |
| Gaussian | $\text{VaR} = \mu + z_\alpha \sigma$ | Underestimates tails |
| Cornish-Fisher | $\text{VaR} = \mu + (z + \frac{z^2-1}{6}S + \frac{z^3-3z}{24}K_e - \frac{2z^3-5z}{36}S^2)\sigma$ | **Uses EXCESS kurtosis; monotonicity guard required** |
| Student's t | MLE fit, then quantile | Better tail fit |
| EVT (GPD) | POT method, see Phase 10 | Best for extreme tails |

**Cornish-Fisher monotonicity guard:**

```python
def var_cornish_fisher(returns, alpha=0.05):
    """Cornish-Fisher VaR with monotonicity guard.
    CRITICAL: Uses EXCESS kurtosis (scipy.stats.kurtosis default = excess).
    The CF expansion can lose monotonicity at extreme quantiles — guard against this."""
    z = norm.ppf(alpha)
    s = returns.skew()
    k = returns.kurtosis()  # EXCESS kurtosis
    cf_z = z + (z**2 - 1)/6 * s + (z**3 - 3*z)/24 * k - (2*z**3 - 5*z)/36 * s**2

    # Monotonicity guard: CF quantile should be more negative than Gaussian
    if cf_z > z:  # Expansion reversed — fall back to Gaussian
        cf_z = z

    return -(returns.mean() + cf_z * returns.std())
```

### 9.2 Portfolio CVaR — Scenario Matrix Approach

**CRITICAL: Portfolio CVaR ≠ weighted sum of individual CVaRs.**
CVaR is NOT linearly aggregable. You must compute portfolio-level CVaR from joint
return scenarios:

```python
def portfolio_cvar(weights, scenario_matrix, alpha=0.05):
    """Correct portfolio CVaR from joint scenarios."""
    portfolio_returns = scenario_matrix @ weights
    var_threshold = np.percentile(portfolio_returns, alpha * 100)
    tail_returns = portfolio_returns[portfolio_returns <= var_threshold]
    return -tail_returns.mean()
```

### 9.3 Clayton Copula for Lower Tail Dependence

```python
from scipy.optimize import minimize_scalar

def fit_clayton_copula(u, v):
    """Fit Clayton copula via maximum pseudo-likelihood.
    u, v are pseudo-observations (ranks / (n+1))."""
    def neg_loglik(theta):
        if theta <= 0: return 1e10
        n = len(u)
        ll = n * np.log(1 + theta)
        ll += -(1 + theta) * np.sum(np.log(u) + np.log(v))
        ll += -(2 + 1/theta) * np.sum(np.log(u**(-theta) + v**(-theta) - 1))
        return -ll

    result = minimize_scalar(neg_loglik, bounds=(0.01, 20), method='bounded')
    return result.x

# Lower tail dependence coefficient
lambda_L = 2 ** (-1 / theta)  # > 0 means lower tail dependence exists
```

### 9.4 VaR Backtesting

**Kupiec POF test** (proportion of failures):
```python
def kupiec_pof_test(violations, alpha):
    T = len(violations)
    n_viol = violations.sum()
    rate = n_viol / T
    if n_viol == 0 or n_viol == T: return {'p_value': 0}
    lr = -2 * (T * np.log(1-alpha) + n_viol * np.log(alpha)
               - (T-n_viol) * np.log(1-rate) - n_viol * np.log(rate))
    return {'violation_rate': rate, 'p_value': 1 - chi2.cdf(lr, 1)}
```

**Traffic light zones:** Green (rate < 1.5×α), Yellow (1.5–2×α), Red (>2×α).

---

## 10. MONTE CARLO SIMULATION FRAMEWORK

### 10.1 Correlated Return Simulation

```python
def monte_carlo_simulation(mu, cov, n_paths=10000, horizon=252, seed=42):
    """
    Simulate correlated return paths via Cholesky decomposition.

    CRITICAL: Ensure cov is PSD before Cholesky. If not, apply Higham correction.
    """
    np.random.seed(seed)

    # PSD check
    eigvals = np.linalg.eigvalsh(cov)
    if np.any(eigvals < -1e-10):
        cov = ensure_psd(cov)  # From §6 DCC-GARCH

    L = np.linalg.cholesky(cov)
    n_assets = len(mu)
    final_values = np.zeros(n_paths)

    for i in range(n_paths):
        z = np.random.standard_normal((horizon, n_assets))
        daily_ret = mu + z @ L.T
        # Equal-weight portfolio
        port_path = np.exp(daily_ret.mean(axis=1).cumsum())
        final_values[i] = port_path[-1]

    return final_values
```

### 10.2 Metrics from Simulation

```python
final = monte_carlo_simulation(mu, cov)
print(f"Mean 1Y return: {final.mean() - 1:.2%}")
print(f"5th percentile: {np.percentile(final, 5) - 1:.2%}")
print(f"P(loss > 20%): {(final < 0.80).mean():.2%}")
print(f"Simulated VaR(95): {-(np.percentile(final, 5) - 1):.2%}")
```

### 10.3 Hypothetical Stress Scenarios

```python
hypo = {
    'Taiwan Strait': {'TSM': -0.50, 'NVDA': -0.30, 'AAPL': -0.25, 'AMD': -0.20},
    'AI Bubble Burst': {'NVDA': -0.40, 'PLTR': -0.40, 'CRWD': -0.40, 'DDOG': -0.40},
    'Cyber Breach': {'PANW': 0.15, 'CRWD': 0.15, 'META': -0.10, 'GOOG': -0.10},
}
```
---

## 10A. NLP & ALTERNATIVE DATA INTEGRATION

### 10A.1 FinBERT Sentiment Pipeline

**FinBERT** (Araci 2019) is a BERT model fine-tuned on financial text. It classifies
text into positive/negative/neutral sentiment with domain-specific understanding.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model (requires ~1.5GB disk, ~500MB RAM)
tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
finbert = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')

def run_finbert(texts, batch_size=16):
    """
    Run FinBERT on a list of financial texts.

    Returns
    -------
    np.ndarray (len(texts), 3) — [positive, negative, neutral] probabilities
    """
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                          max_length=512, return_tensors='pt')
        with torch.no_grad():
            outputs = finbert(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        all_probs.append(probs.cpu().numpy())
    return np.vstack(all_probs)

# Example
result = run_finbert(['NVIDIA reports record quarterly revenue driven by AI demand'])
# result[0] = [0.92, 0.03, 0.05]  positive/negative/neutral
```

### 10A.2 Synthetic Sentiment Proxy (Fallback)

When live headline data is unavailable, construct a sentiment proxy from market data:

```python
def synthetic_sentiment(prices, window=20):
    """
    Market-based sentiment proxy using momentum and volatility.
    NOT a substitute for actual NLP — document this limitation clearly.
    """
    ret_5d = prices.pct_change(5)
    vol_20d = prices.pct_change().rolling(window).std()

    sentiment = pd.DataFrame(index=prices.index)
    # Normalised momentum as sentiment: tanh(momentum / vol)
    sentiment['sentiment_mean'] = np.tanh(ret_5d / vol_20d.clip(lower=0.001))
    # Sentiment dispersion: vol of sentiment itself
    sentiment['sentiment_std'] = sentiment['sentiment_mean'].rolling(5).std()
    # Directional momentum of sentiment
    sentiment['sentiment_momentum'] = sentiment['sentiment_mean'].diff(5)
    return sentiment
```

### 10A.3 Lag Enforcement Protocol

**ALL sentiment features require t-1 lag before use in any model:**

```python
# Step 1: Compute raw sentiment (same-day)
raw_sentiment = run_finbert(headlines)  # or synthetic_sentiment(prices)

# Step 2: Apply t-1 lag
lagged_sentiment = raw_sentiment.shift(1)

# Step 3: VERIFY the lag is correct
print(f"Original at 2024-01-15: {raw_sentiment.loc['2024-01-15']:.4f}")
print(f"Lagged at 2024-01-16: {lagged_sentiment.loc['2024-01-16']:.4f}")
assert abs(raw_sentiment.loc['2024-01-15'] - lagged_sentiment.loc['2024-01-16']) < 1e-10, \
    "LEAKAGE: t-1 lag not applied correctly"

# Step 4: Drop the first NaN row from lagging
lagged_sentiment = lagged_sentiment.dropna()
```

**Why t-1 matters:** A financial headline published at 2 PM on day $t$ affects both
the FinBERT sentiment score for day $t$ AND the close-to-close return for day $t$.
Using same-day sentiment creates a spurious correlation of ~0.15–0.25 that completely
disappears when the lag is enforced. This is the single most common source of
information leakage in sentiment-based strategies.

### 10A.4 Sentiment Feature Engineering

| Feature | Formula | Interpretation |
|---------|---------|---------------|
| `sentiment_mean` | $\text{tanh}(r_{5d} / \sigma_{20d})$ | Normalised momentum signal |
| `sentiment_std` | rolling std of sentiment_mean | Disagreement / uncertainty |
| `sentiment_momentum` | 5-day change in sentiment_mean | Sentiment trend |
| `sentiment_volume_interaction` | sentiment × volume z-score | High conviction = sentiment × unusual volume |
| `sentiment_regime` | sentiment × regime dummy | Regime-conditional sentiment effect |

---

## 10B. EXPANDED PORTFOLIO OPTIMIZATION METHODS

### 10B.1 Mean-Variance Optimization (Markowitz)

```python
import cvxpy as cp

def mean_variance_optimize(mu, cov, tickers, objective='max_sharpe',
                           max_weight=0.10, rf=0.0):
    """
    Mean-variance optimization with configurable objective.

    Parameters
    ----------
    mu : np.ndarray — annualised expected returns
    cov : np.ndarray — annualised covariance matrix
    objective : str — 'max_sharpe', 'min_volatility', 'target_return'
    max_weight : float — per-asset weight cap
    """
    n = len(mu)
    w = cp.Variable(n)
    port_ret = mu @ w
    port_var = cp.quad_form(w, cov)

    constraints = [cp.sum(w) == 1, w >= 0, w <= max_weight]

    if objective == 'max_sharpe':
        # Maximize Sharpe: max (μ'w - rf) / sqrt(w'Σw)
        # Reformulated as: min w'Σw s.t. μ'w - rf = 1 (scaling trick)
        k = cp.Variable(1, nonneg=True)
        w_k = cp.Variable(n)
        prob = cp.Problem(
            cp.Minimize(cp.quad_form(w_k, cov)),
            [(mu - rf) @ w_k == 1, cp.sum(w_k) == k,
             w_k >= 0, w_k <= max_weight * k]
        )
        prob.solve(solver=cp.SCS)
        return (w_k.value / k.value).flatten()

    elif objective == 'min_volatility':
        prob = cp.Problem(cp.Minimize(port_var), constraints)
        prob.solve(solver=cp.SCS)
        return w.value

    elif objective == 'target_return':
        target = mu.mean()  # Target the average return
        constraints.append(port_ret >= target)
        prob = cp.Problem(cp.Minimize(port_var), constraints)
        prob.solve(solver=cp.SCS)
        return w.value
```

### 10B.2 Hierarchical Risk Parity (HRP)

López de Prado (2016) — uses hierarchical clustering to build a diversified portfolio
without requiring expected return estimates (which are notoriously noisy):

```python
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

def hrp_optimize(returns, tickers):
    """
    Hierarchical Risk Parity (HRP) allocation.
    Does NOT require expected returns — uses only covariance structure.

    Steps:
    1. Compute correlation-based distance matrix
    2. Hierarchical clustering (single linkage)
    3. Quasi-diagonalisation (reorder by cluster)
    4. Recursive bisection for weight allocation
    """
    cov = returns.cov().values
    corr = returns.corr().values

    # Step 1: Distance matrix
    dist = np.sqrt((1 - corr) / 2)  # Distance from correlation
    dist_condensed = squareform(dist, checks=False)

    # Step 2: Hierarchical clustering
    link = linkage(dist_condensed, method='single')
    sort_idx = leaves_list(link).tolist()

    # Step 3: Quasi-diagonal reordering
    cov_sorted = cov[np.ix_(sort_idx, sort_idx)]

    # Step 4: Recursive bisection
    weights = np.zeros(len(tickers))
    cluster_items = [sort_idx]
    cluster_weights = [1.0]

    while cluster_items:
        items = cluster_items.pop(0)
        w = cluster_weights.pop(0)

        if len(items) == 1:
            weights[items[0]] = w
            continue

        mid = len(items) // 2
        left, right = items[:mid], items[mid:]

        # Inverse-variance allocation between clusters
        var_left = cov[np.ix_(left, left)].sum()
        var_right = cov[np.ix_(right, right)].sum()
        alpha = 1 - var_left / (var_left + var_right)

        cluster_items.extend([left, right])
        cluster_weights.extend([w * alpha, w * (1 - alpha)])

    return weights
```

### 10B.3 Equal Risk Contribution (ERC / Risk Budgeting)

```python
from scipy.optimize import minimize

def risk_budgeting_optimize(cov, budget=None, tickers=None):
    """
    Equal Risk Contribution (ERC) portfolio.
    Each asset contributes equally to total portfolio risk.

    Risk contribution of asset i: RC_i = w_i × (Σw)_i / √(w'Σw)
    Target: RC_i = 1/N for all i (equal contribution)
    """
    n = cov.shape[0]
    if budget is None:
        budget = np.ones(n) / n  # Equal risk budgets

    def objective(w):
        port_risk = np.sqrt(w @ cov @ w)
        marginal_risk = cov @ w / port_risk
        risk_contrib = w * marginal_risk
        target_contrib = budget * port_risk
        return np.sum((risk_contrib - target_contrib) ** 2)

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0.001, None)] * n  # No zeros (log barrier)
    x0 = np.ones(n) / n

    result = minimize(objective, x0, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    return result.x
```

### 10B.4 Black-Litterman with RMSE-Based Uncertainty

**CRITICAL: BL view uncertainty (Ω) must be RMSE², NOT 1/RMSE:**

```python
def compute_bl_omega(model_rmse, n_factors=4):
    """
    Compute BL view uncertainty matrix from model RMSE.

    WRONG: Ω = diag(1/RMSE)  — this makes uncertain models MORE confident
    CORRECT: Ω = diag(RMSE²) — higher RMSE → higher uncertainty → less weight on view
    """
    omega = np.diag(model_rmse ** 2)  # RMSE² on diagonal
    # Floor to prevent near-zero uncertainty (overconfident views)
    omega = np.maximum(omega, 1e-4 * np.eye(n_factors))
    return omega
```

### 10B.5 Ledoit-Wolf Shrinkage Covariance

```python
from sklearn.covariance import LedoitWolf

def covariance_ledoit_wolf(returns):
    """
    Ledoit-Wolf shrinkage covariance estimator.
    Shrinks sample covariance toward a structured target (scaled identity).
    Better conditioned than raw sample covariance, especially when T/N is small.

    IMPORTANT: Can still return non-PSD in edge cases with very small samples.
    Always check eigenvalues after computation.
    """
    lw = LedoitWolf().fit(returns.values)
    cov = lw.covariance_

    # PSD check
    eigvals = np.linalg.eigvalsh(cov)
    if np.any(eigvals < -1e-10):
        print(f"Warning: Ledoit-Wolf covariance has negative eigenvalues, applying correction")
        eigvals_fixed = np.maximum(eigvals, 1e-10)
        eigvecs = np.linalg.eigh(cov)[1]
        cov = eigvecs @ np.diag(eigvals_fixed) @ eigvecs.T

    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
```

### 10B.6 Portfolio Performance Metrics

```python
def compute_all_metrics(returns, rf=0.0, name='Strategy'):
    """Compute comprehensive portfolio performance metrics."""
    r = returns.dropna()
    excess = r - rf

    # Basic
    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    # Drawdown
    cum = (1 + r).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min()

    # Downside
    downside = r[r < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0

    # Tail
    var_95 = -np.percentile(r, 5)
    cvar_95 = -r[r <= -var_95].mean() if (r <= -var_95).any() else var_95
    skew = r.skew()
    kurt = r.kurtosis()
    hit_rate = (r > 0).mean()

    # Calmar
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    # Tail ratio
    right_tail = np.percentile(r, 95)
    left_tail = abs(np.percentile(r, 5))
    tail_ratio = right_tail / left_tail if left_tail > 0 else 0

    return {
        'name': name,
        'ann_return': ann_ret,
        'ann_volatility': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_dd,
        'calmar': calmar,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'skewness': skew,
        'excess_kurtosis': kurt,
        'hit_rate': hit_rate,
        'tail_ratio': tail_ratio,
        'n_observations': len(r),
    }
```

---

## 10C. ML PIPELINE — DETAILED SPECIFICATIONS

### 10C.1 Model Zoo

| Model | Type | Use Case | Key Hyperparameters |
|-------|------|----------|-------------------|
| Ridge Regression | Linear | Baseline vol forecast | `alpha` (regularisation) |
| Lasso Regression | Linear | Feature selection | `alpha` |
| Random Forest | Ensemble | Non-linear vol forecast | `n_estimators`, `max_depth`, `min_samples_leaf` |
| XGBoost | Boosting | Best single model (typically) | `n_estimators`, `learning_rate`, `max_depth` |
| LightGBM | Boosting | Faster alternative to XGBoost | Same as XGBoost; `num_leaves` instead of depth |
| KNN | Instance | Local patterns | `n_neighbors`, `weights` |
| Logistic Regression | Linear | Direction classification | `C` (inverse regularisation) |

### 10C.2 Pipeline Factory

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor

def make_regression_pipeline(model_name):
    """Create sklearn pipeline with scaler inside (prevents leakage)."""
    models = {
        'ridge': Ridge(alpha=1.0),
        'lasso': Lasso(alpha=0.01),
        'random_forest': RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=20,
            random_state=42, n_jobs=-1
        ),
        'xgboost': None,  # Requires xgboost import
        'knn': KNeighborsRegressor(n_neighbors=20, weights='distance'),
    }

    if model_name == 'xgboost':
        import xgboost as xgb
        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
    elif model_name == 'lightgbm':
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=200, num_leaves=31, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
        )
    else:
        model = models[model_name]

    return Pipeline([
        ('scaler', StandardScaler()),  # MUST be inside pipeline
        ('model', model)
    ])
```

### 10C.3 Regression Metrics

```python
def regression_metrics(y_true, y_pred):
    """Compute comprehensive regression evaluation metrics."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    residuals = y_true - y_pred
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs(residuals / np.where(y_true != 0, y_true, 1e-10))) * 100

    # Directional accuracy: did we predict the direction of change correctly?
    if len(y_true) > 1:
        actual_dir = np.diff(y_true) > 0
        pred_dir = np.diff(y_pred) > 0
        da = np.mean(actual_dir == pred_dir) * 100
    else:
        da = 50.0

    # R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return {
        'rmse': rmse, 'mae': mae, 'mape': mape,
        'directional_accuracy': da, 'r2': r2,
    }
```

### 10C.4 Classification Metrics

```python
def classification_metrics(y_true, y_pred, y_prob=None):
    """Compute classification metrics including AUC and Brier score."""
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, roc_auc_score, brier_score_loss)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
        metrics['brier_score'] = brier_score_loss(y_true, y_prob)

    return metrics
```

### 10C.5 Multi-Horizon Performance Decay

Test predictions at multiple horizons (1d, 5d, 21d) and document how accuracy decays:

```python
HORIZONS = [1, 5, 21]

for h in HORIZONS:
    target_h = returns.rolling(h).std().shift(-h) * np.sqrt(252)
    # ... run walk-forward with same features but different target ...
    print(f"Horizon {h}d: RMSE={rmse:.6f}, DA={da:.1f}%")

# Expected pattern:
# 1-day:  RMSE ~0.02,  DA ~55%  (most predictable)
# 5-day:  RMSE ~0.04,  DA ~52%
# 21-day: RMSE ~0.08,  DA ~50%  (approaching random walk)
```

### 10C.6 Transaction Cost in ML Trading Signals

```python
def signal_to_pnl(predictions, actual_returns, tc_per_trade=0.0010):
    """
    Convert directional predictions to P&L with transaction costs.
    tc_per_trade: 10 bps per trade (one-way)
    """
    position = np.sign(predictions)  # +1 or -1
    position_change = np.abs(np.diff(position, prepend=0))
    tc = position_change * tc_per_trade

    gross_pnl = position * actual_returns
    net_pnl = gross_pnl - tc

    # Profit factor
    gains = net_pnl[net_pnl > 0].sum()
    losses = abs(net_pnl[net_pnl < 0].sum())
    profit_factor = gains / losses if losses > 0 else float('inf')

    return {
        'gross_sharpe': gross_pnl.mean() / gross_pnl.std() * np.sqrt(252),
        'net_sharpe': net_pnl.mean() / net_pnl.std() * np.sqrt(252),
        'total_tc': tc.sum(),
        'n_trades': (position_change > 0).sum(),
        'profit_factor': profit_factor,
        'hit_rate': (net_pnl > 0).mean(),
    }
```

---

## 10D. RETURN FORECASTING PROTOCOL

### 10D.1 Target Variable Definitions

| Target | Formula | Horizon | Use |
|--------|---------|---------|-----|
| Forward realised vol (5d) | $\sigma_{t+5} = \text{std}(r_{t+1:t+5}) \times \sqrt{252}$ | 1 week | Short-term risk |
| Forward realised vol (21d) | $\sigma_{t+21} = \text{std}(r_{t+1:t+21}) \times \sqrt{252}$ | 1 month | Medium-term risk |
| Forward return direction | $\mathbb{1}(R_{t+1:t+5} > 0)$ | 5 days | Trading signal |
| Forward return magnitude | $R_{t+1:t+5} = P_{t+5}/P_t - 1$ | 5 days | Expected return |

**CRITICAL alignment:** Target at time $t$ uses data from $t+1$ onwards.
Features at time $t$ use data up to and including $t$. Verify:

```python
# Feature: vol computed from r_{t-59:t} (60 trailing days)
features['vol_60d'] = log_returns.rolling(60).std() * np.sqrt(252)

# Target: vol computed from r_{t+1:t+5} (5 forward days)
target = log_returns.rolling(5).std().shift(-5) * np.sqrt(252)

# Verify no overlap
assert features.index[0] + pd.Timedelta(days=60) <= target.dropna().index[0]
```

### 10D.2 Feature Categories

| Category | Features | Count |
|----------|----------|-------|
| Volatility | Realised vol (5d, 21d, 63d), Yang-Zhang, Parkinson, Garman-Klass | 6 |
| Returns | Log return, 5d/21d momentum, RSI(14) | 4 |
| Technical | MACD histogram, Bollinger %B, ATR(14) | 3 |
| Volume | Volume z-score (20d), volume momentum | 2 |
| GARCH | Conditional volatility from best GARCH model | 1 |
| Regime | HMM regime probabilities, macro regime dummy | 2-4 |
| Sentiment | FinBERT score (t-1 lagged), sentiment momentum | 2 |
| **Total** | | **~20** |

---

## 10E. BACKTEST ENGINE — DETAILED IMPLEMENTATION

### 10E.1 Walk-Forward Backtest for Tech Portfolio

```python
def run_portfolio_backtest(returns, weight_strategies, tc=0.0010):
    """
    Run walk-forward backtest for multiple allocation strategies.

    Parameters
    ----------
    returns : pd.DataFrame — daily simple returns (T × N)
    weight_strategies : dict — {name: function(returns_up_to_t) → weights}
    tc : float — one-way transaction cost

    Returns
    -------
    dict of pd.Series — daily strategy returns (net of TC)
    """
    results = {}

    for name, weight_fn in weight_strategies.items():
        n_assets = returns.shape[1]
        net_returns = []
        prev_weights = np.ones(n_assets) / n_assets  # Start equal-weight

        for t in range(252, len(returns)):  # Minimum 1-year burn-in
            # Current weights from strategy
            lookback = returns.iloc[:t]
            try:
                weights = weight_fn(lookback)
            except Exception:
                weights = prev_weights  # Fallback to previous

            # Transaction cost
            turnover = np.abs(weights - prev_weights).sum()
            cost = turnover * tc

            # Portfolio return
            day_return = returns.iloc[t].values @ weights
            net_return = day_return - cost

            net_returns.append({
                'date': returns.index[t],
                'return': net_return,
                'turnover': turnover,
                'cost': cost,
            })

            # Update weights for next period (drift)
            prev_weights = weights * (1 + returns.iloc[t].values)
            prev_weights = prev_weights / prev_weights.sum()  # Renormalise

        results[name] = pd.DataFrame(net_returns).set_index('date')

    return results
```

### 10E.2 Strategy Comparison Framework

```python
def compare_strategies(strategy_returns, rf=0.0):
    """Compare all strategies using consistent metrics."""
    comparison = []
    for name, rets in strategy_returns.items():
        metrics = compute_all_metrics(rets['return'], rf=rf, name=name)
        metrics['annual_turnover'] = rets['turnover'].sum() / (len(rets) / 252)
        metrics['annual_tc_drag'] = rets['cost'].sum() / (len(rets) / 252) * 100  # in bps
        comparison.append(metrics)

    df = pd.DataFrame(comparison).set_index('name')
    return df.round(4)
```

### 10E.3 Rolling Window Analysis

```python
def rolling_sharpe(returns, window=252):
    """Compute rolling 1-year Sharpe ratio."""
    rolling_mean = returns.rolling(window).mean() * 252
    rolling_std = returns.rolling(window).std() * np.sqrt(252)
    return rolling_mean / rolling_std
```

---

## 10F. ADVANCED STATISTICAL TESTS

### 10F.1 Christoffersen Independence Test for VaR

```python
def christoffersen_test(violations, alpha):
    """
    Test independence of VaR violations (no clustering).
    H0: violations are independent Bernoulli draws.
    """
    T = len(violations)
    # Transition counts
    n00 = n01 = n10 = n11 = 0
    for t in range(1, T):
        if violations[t-1] == 0 and violations[t] == 0: n00 += 1
        elif violations[t-1] == 0 and violations[t] == 1: n01 += 1
        elif violations[t-1] == 1 and violations[t] == 0: n10 += 1
        else: n11 += 1

    # Transition probabilities
    p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    p = (n01 + n11) / T

    # Log-likelihood ratio
    if p01 == 0 or p01 == 1 or p11 == 0 or p11 == 1 or p == 0 or p == 1:
        return {'p_value': 1.0}  # Degenerate case

    lr_ind = -2 * (
        n00 * np.log(1-p) + n01 * np.log(p) + n10 * np.log(1-p) + n11 * np.log(p)
        - n00 * np.log(1-p01) - n01 * np.log(p01) - n10 * np.log(1-p11) - n11 * np.log(p11)
    )

    from scipy.stats import chi2
    return {'lr_stat': lr_ind, 'p_value': 1 - chi2.cdf(lr_ind, 1)}
```

### 10F.2 Granger Causality for Macro → Returns

```python
from statsmodels.tsa.stattools import grangercausalitytests

def test_granger_causality(y, x, maxlag=5):
    """
    Test if x Granger-causes y (x has predictive power for y beyond y's own lags).

    Common tests:
    - Does VIX change Granger-cause tech sector returns?
    - Does credit spread change Granger-cause factor returns?
    """
    data = pd.DataFrame({'y': y, 'x': x}).dropna()
    if len(data) < 100:
        return {'best_lag': None, 'p_value': 1.0}

    results = grangercausalitytests(data[['y', 'x']], maxlag=maxlag, verbose=False)

    # Find most significant lag
    best_lag = min(results.keys(),
                  key=lambda k: results[k][0]['ssr_ftest'][1])
    best_p = results[best_lag][0]['ssr_ftest'][1]

    return {'best_lag': best_lag, 'p_value': best_p}
```

### 10F.3 Structural Break Detection

```python
from statsmodels.stats.diagnostic import breaks_cusumolsresid

def test_structural_break(returns, window=120):
    """
    Test for structural breaks in return process using CUSUM.
    Important for: low-vol anomaly weakening post-2010, momentum crashes.
    """
    cusum_stat, p_value, critical = breaks_cusumolsresid(returns.values)
    return {
        'cusum_stat': cusum_stat,
        'p_value': p_value,
        'reject_5pct': p_value < 0.05,
    }
```

### 10F.4 Overfitting Diagnostics

```python
def overfitting_check(train_metrics, test_metrics, threshold=0.20):
    """
    Check if train/test performance gap indicates overfitting.
    Gap > 20% in key metrics = overfitting concern.
    """
    checks = {}
    for metric in ['rmse', 'r2', 'directional_accuracy']:
        if metric in train_metrics and metric in test_metrics:
            train_val = train_metrics[metric]
            test_val = test_metrics[metric]
            if train_val != 0:
                gap = abs(train_val - test_val) / abs(train_val)
            else:
                gap = 0
            checks[metric] = {
                'train': train_val,
                'test': test_val,
                'gap_pct': gap * 100,
                'overfit_flag': gap > threshold,
            }
    return checks
```

### 10F.5 Feature Ablation Study

```python
def feature_ablation(X, y, pipeline_factory, feature_groups):
    """
    Measure impact of removing each feature group on model performance.
    Helps identify which features truly contribute vs noise.

    Parameters
    ----------
    feature_groups : dict — {'group_name': ['col1', 'col2', ...]}
    """
    # Baseline: all features
    baseline = walk_forward_predict(X, y, pipeline_factory)
    baseline_rmse = regression_metrics(baseline['y_true'], baseline['y_pred'])['rmse']

    ablation_results = {'all_features': baseline_rmse}

    for group_name, cols in feature_groups.items():
        # Remove group
        X_ablated = X.drop(columns=cols, errors='ignore')
        if X_ablated.shape[1] == 0:
            continue
        result = walk_forward_predict(X_ablated, y, pipeline_factory)
        rmse = regression_metrics(result['y_true'], result['y_pred'])['rmse']
        ablation_results[f'without_{group_name}'] = rmse

    return ablation_results
```



## 11. CONFIGURATION MANAGEMENT

### 11.1 Centralised Config (`src/config.py`)

All paths, constants, and hyperparameters must be defined in a single configuration file:

```python
# src/config.py
from pathlib import Path

# === Paths ===
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
INTERIM_DIR = DATA_DIR / 'interim'
PROCESSED_DIR = DATA_DIR / 'processed'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
FIGURES_DIR = OUTPUTS_DIR / 'figures'
TABLES_DIR = OUTPUTS_DIR / 'tables'
REPORTS_DIR = OUTPUTS_DIR / 'reports'
MODELS_DIR = OUTPUTS_DIR / 'models'
LOGS_DIR = PROJECT_ROOT / 'logs'

# Create directories
for d in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR,
          REPORTS_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# === File Paths ===
MASTER_DATA_FILE = PROCESSED_DIR / 'master_data.parquet'
MACRO_REGIMES_FILE = PROCESSED_DIR / 'macro_regimes.parquet'
REGIME_LABELS_FILE = PROCESSED_DIR / 'regime_labels.parquet'
COND_VOL_FILE = PROCESSED_DIR / 'conditional_vol_series.parquet'
SENTIMENT_FILE = PROCESSED_DIR / 'sentiment_features.parquet'
RETURN_SCENARIOS_FILE = PROCESSED_DIR / 'return_scenarios.parquet'
VAR_CVAR_FILE = TABLES_DIR / 'var_cvar_table.csv'
EVT_PARAMS_FILE = TABLES_DIR / 'evt_parameters.csv'
BACKTEST_VAR_FILE = TABLES_DIR / 'backtest_var_results.csv'
TRANSITION_MATRIX_FILE = TABLES_DIR / 'transition_matrices.csv'
MODEL_COMPARISON_FILE = TABLES_DIR / 'model_comparison.csv'
CLASS_REPORT_FILE = TABLES_DIR / 'classification_report.csv'
BACKTEST_PERF_FILE = TABLES_DIR / 'backtest_performance.csv'
STRESS_TEST_FILE = TABLES_DIR / 'stress_test_results.csv'

# === Reproducibility ===
RANDOM_STATE = 42
TRAIN_RATIO = 0.70
RETRAIN_FREQ_DAYS = 63      # ~quarterly

# === Ticker Universe ===
TICKERS = [
    'NVDA', 'AMD', 'TSM', 'AVGO', 'QCOM', 'MU',     # Semiconductors
    'AAPL', 'MSFT', 'GOOG', 'META', 'NFLX',          # Big Tech
    'CRM', 'ADBE', 'NOW',                             # Enterprise Software
    'PANW', 'CRWD',                                     # Cybersecurity
    'DDOG', 'PLTR',                                     # Analytics/Observability
    'ANET', 'XYZ',                                      # Networking/Fintech
]

BENCHMARKS = ['^GSPC', '^NDX', 'XLK', 'SOXX', '^VIX', '^VVIX', 'DX-Y.NYB']

SHORT_HISTORY_TICKERS = {
    'CRWD': '2019-06-12',
    'DDOG': '2019-09-19',
    'PLTR': '2020-09-30',
}

# === Key Events ===
KEY_EVENTS = {
    'COVID Crash': ('2020-02-19', '2020-03-23'),
    'Rate Shock 2022': ('2022-01-03', '2022-10-13'),
    'SVB Contagion': ('2023-03-08', '2023-03-15'),
    'DeepSeek/Tariff': ('2025-01-27', '2025-02-10'),
    'China Tech Crackdown': ('2021-07-01', '2021-08-20'),
}

# === GARCH ===
GARCH_MODELS = ['GARCH', 'GJR-GARCH', 'EGARCH', 'FIGARCH']
GARCH_DISTRIBUTIONS = ['normal', 'StudentsT', 'skewt', 'ged']
MIN_OBS_FIGARCH = 1500

# === Deep Learning ===
LSTM_LOOKBACK = 60
LSTM_HIDDEN_DIM = 64
LSTM_N_LAYERS = 2
LSTM_DROPOUT = 0.3
EARLY_STOP_PATIENCE = 15
DL_BATCH_SIZE = 64
DL_LEARNING_RATE = 1e-3
DL_EPOCHS = 100

# === Monte Carlo ===
MC_PATHS = 10000
MC_HORIZON = 252

# === Portfolio ===
MAX_WEIGHT_SINGLE = 0.10      # Max 10% per stock (tech portfolio)
MAX_WEIGHT_SECTOR = 0.30      # Max 30% per sector
MAX_WEIGHT_FACTOR = 0.40      # Max 40% per factor (factor timing)
TURNOVER_LIMIT = 0.20         # Max 20% total turnover per rebalance
TC_ONE_WAY = 0.0010           # 10 bps one-way transaction cost (tech)
TC_FACTOR = 0.0025            # 25 bps one-way (factor portfolio)

# === Macro ===
FRED_SERIES = {
    'T10Y2Y': 'Yield curve slope',
    'BAA10Y': 'Credit spread',
    'VIXCLS': 'VIX',
    'ICSA': 'Initial jobless claims',
    'M2SL': 'M2 money supply',
    'CPIAUCSL': 'CPI (for deflation)',
    'USALOLITONOSTSAM': 'OECD CLI',
    'DCOILWTICO': 'WTI crude oil',
    'INDPRO': 'Industrial production',
    'UNRATE': 'Unemployment rate',
}
```

### 11.2 Hyperparameter Registry

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `RANDOM_STATE` | 42 | Standard seed for reproducibility |
| `TRAIN_RATIO` | 0.70 | 70/30 initial train/test split |
| `RETRAIN_FREQ_DAYS` | 63 | Quarterly retraining (~63 trading days) |
| `LSTM_LOOKBACK` | 60 | ~3 months of daily data context |
| `EARLY_STOP_PATIENCE` | 15 | Generous patience to avoid premature stopping |
| `MC_PATHS` | 10,000 | Sufficient for 95th percentile stability |
| `TC_ONE_WAY` | 10 bps | Conservative for liquid large-cap tech |
| `TC_FACTOR` | 25 bps | Conservative for factor portfolio rebalancing |
| `MAX_WEIGHT_SINGLE` | 10% | Diversification constraint |
| `HMM_N_RESTARTS` | 25 | Sufficient to escape local optima |

---

## 12. QUALITY GATES & AUTOMATED VALIDATION

### 12.1 Schema Validation

```python
# src/validation.py

def validate_parquet(df, expected_cols=None, min_rows=None,
                     no_nan=False, date_index=True, dtype_check=None):
    """
    Validate a DataFrame loaded from parquet against expected schema.
    Call at the START of every notebook to validate inputs.
    """
    errors = []

    if date_index and not isinstance(df.index, pd.DatetimeIndex):
        errors.append(f"Index is {type(df.index)}, expected DatetimeIndex")

    if date_index and not df.index.is_unique:
        errors.append(f"Index has {df.index.duplicated().sum()} duplicate dates")

    if expected_cols:
        missing = set(expected_cols) - set(df.columns)
        if missing:
            errors.append(f"Missing columns: {missing}")

    if min_rows and len(df) < min_rows:
        errors.append(f"Only {len(df)} rows, expected ≥{min_rows}")

    if no_nan and df.isna().any().any():
        nan_cols = df.columns[df.isna().any()].tolist()
        errors.append(f"NaN found in columns: {nan_cols}")

    if dtype_check:
        for col, expected_dtype in dtype_check.items():
            if col in df.columns and df[col].dtype != expected_dtype:
                errors.append(f"Column {col}: dtype {df[col].dtype}, expected {expected_dtype}")

    if errors:
        raise ValueError("Schema validation failed:\n" + "\n".join(f"  • {e}" for e in errors))

    return True
```

### 12.2 Cross-Phase Consistency Checks

```python
def verify_phase_outputs(phase_num):
    """Run inter-phase consistency checks."""
    if phase_num >= 2:
        factor_returns = pd.read_parquet(PROCESSED_DIR / 'factor_returns.parquet')
        macro = pd.read_parquet(PROCESSED_DIR / 'macro_indicators.parquet')
        assert factor_returns.index.equals(macro.index), "Date index mismatch between factors and macro"

    if phase_num >= 4:
        regime = pd.read_parquet(PROCESSED_DIR / 'regime_probabilities.parquet')
        assert regime[['p_expansion', 'p_slowdown', 'p_crisis']].sum(axis=1).between(0.99, 1.01).all(), \
            "Regime probabilities don't sum to 1"

    if phase_num >= 7:
        weights = pd.read_parquet(PROCESSED_DIR / 'bl_weights_timeseries.parquet')
        assert weights.sum(axis=1).between(0.99, 1.01).all(), "BL weights don't sum to 1"
        assert (weights >= -0.001).all().all(), "Negative weights found"
        assert (weights <= 0.401).all().all(), "Weight exceeds 40% cap"
```

### 12.3 Automated Stationarity Testing

```python
def stationarity_table(returns_df):
    """ADF + KPSS tests on each column. Returns summary DataFrame."""
    from statsmodels.tsa.stattools import adfuller, kpss
    results = []
    for col in returns_df.columns:
        series = returns_df[col].dropna()
        if len(series) < 50: continue

        adf = adfuller(series, autolag='AIC')
        kpss_res = kpss(series, regression='c', nlags='auto')

        results.append({
            'series': col,
            'adf_stat': adf[0], 'adf_pvalue': adf[1],
            'kpss_stat': kpss_res[0], 'kpss_pvalue': kpss_res[1],
            'adf_reject_1pct': adf[1] < 0.01,
            'kpss_fail_reject_5pct': kpss_res[1] > 0.05,
            'stationary': (adf[1] < 0.01) and (kpss_res[1] > 0.05),
        })
    return pd.DataFrame(results)
```

### 12.4 NaN Propagation Monitor

```python
def check_nan_propagation(df, label=''):
    """Log NaN statistics — catch NaN spreading between phases."""
    nan_count = df.isna().sum()
    if nan_count.any():
        print(f"⚠️ NaN in {label}:")
        for col in nan_count[nan_count > 0].index:
            pct = nan_count[col] / len(df) * 100
            print(f"  {col}: {nan_count[col]} NaN ({pct:.1f}%)")
    else:
        print(f"✓ {label}: no NaN")
```

---

## 13. COMMON PITFALLS — DO NOT MAKE THESE ERRORS

### 13.1 Factor Timing Pitfalls

| # | Error | Phase | Fix |
|---|-------|-------|-----|
| 1 | **Look-ahead in HMM** | 4 | Use `filtered_probabilities()`, NOT `predict_proba()` |
| 2 | **PCA on full sample** | 4 | Expanding-window standardisation AND PCA |
| 3 | **Factor returns in % vs decimal** | 1 | Divide by 100 immediately; verify max |return| < 0.30 |
| 4 | **Survivorship bias** | 2 | Acknowledge; LowVol may be upward-biased ~0.5-1% ann. |
| 5 | **Geometric vs arithmetic mixing** | 9 | NAV: geometric. Sharpe: arithmetic. |
| 6 | **BIC parameter count wrong** | 4 | $p = K^2 + 2K - 1$ for 1D Gaussian HMM |
| 7 | **DCC non-PSD** | 6 | Higham correction at every $t$ |
| 8 | **TC double-counting** | 9 | Cost = TC × Σ|w_t - w_{t-1}|, one-way |
| 9 | **HMM overfitting** | 4 | Cap at K=4; need ~50 obs/state/param |
| 10 | **Unbounded BL views** | 7 | Cap at ±3% monthly; Ω floor 0.0001 |
| 11 | **BL Ω ignores regime uncertainty** | 7 | Use E[Var] + Var[E] (Law of Total Variance) |
| 12 | **HMM label permutation** | 4,9 | Sort states by μ_k after every re-estimation |
| 13 | **OLS standard errors** | 3 | Use Newey-West HAC (lag ≈ 4) |
| 14 | **GARCH input scaling** | 6 | arch expects %; multiply by 100; divide σ² by 10⁴ |
| 15 | **First-period TC** | 9 | Charge TC from equal-weight to first optimal |

### 13.2 ML/DL Pitfalls

| # | Error | Phase | Fix |
|---|-------|-------|-----|
| 16 | **Scaler fit on full data** | 7-9 | Scaler INSIDE pipeline; fit only on train |
| 17 | **SMOTE on regression** | 7 | SMOTE only for classification; never regression |
| 18 | **k-fold on time series** | 7-9 | Walk-forward expanding window only |
| 19 | **Same-day sentiment** | 6 | Lag ALL sentiment by t-1 |
| 20 | **Shuffle in DataLoader** | 9 | `shuffle=False` for temporal data |
| 21 | **PyTorch dropout position** | 9 | Add explicit `nn.Dropout` after last LSTM layer |
| 22 | **Feature importance instability** | 10 | Compare SHAP vs permutation; document gaps |
| 23 | **Gradient explosion in LSTM** | 9 | Use `clip_grad_norm_(model.parameters(), 1.0)` |

### 13.3 Tail Risk Pitfalls

| # | Error | Phase | Fix |
|---|-------|-------|-----|
| 24 | **Cornish-Fisher monotonicity** | 4 | Guard: if CF quantile > Gaussian, fall back |
| 25 | **Portfolio CVaR as weighted sum** | 11 | Use scenario matrix; CVaR is NOT linearly aggregable |
| 26 | **BL Ω = diag(1/RMSE)** | 7 | WRONG. Ω = diag(RMSE²) — higher error = more uncertainty |
| 27 | **Clayton copula on raw returns** | 4 | Use pseudo-observations: ranks/(n+1) |
| 28 | **GPD threshold too extreme** | 10 | Use 10th percentile; require ≥15 exceedances |
| 29 | **Cholesky on non-PSD cov** | 12 | Check eigenvalues; apply Higham if needed |
| 30 | **Volume z-score denominator** | 1 | Use rolling std, NOT rolling mean |

---

## 14. EXECUTION NOTES FOR CLAUDE CODE

### 14.1 Notebook Execution Protocol

- **Run notebooks sequentially by number.** Each reads `.parquet` from `data/processed/`.
- Every notebook must **start** with input validation:

  ```python
  import pandas as pd
  from src.validation import validate_parquet
  factor_returns = pd.read_parquet('data/processed/factor_returns_full.parquet')
  validate_parquet(factor_returns, expected_cols=['hml', 'umd', 'rmw', 'lowvol'],
                   min_rows=252, no_nan=True, date_index=True)
  ```

- Every notebook must **end** with output export and summary:

  ```python
  output_df.to_parquet('data/processed/output_name.parquet')
  print(f"Exported: {output_df.shape[0]} rows × {output_df.shape[1]} cols")
  print(f"Date range: {output_df.index.min()} to {output_df.index.max()}")
  print(f"Columns: {list(output_df.columns)}")
  print(f"NaN count: {output_df.isna().sum().sum()}")
  ```

### 14.2 Figure Standards

- Save at **300 DPI**, white background, `tight` bbox
- Time series: `figsize=(12, 6)`; Heatmaps: `figsize=(8, 8)`
- Consistent colour scheme:

  | Element | Colour | Hex |
  |---------|--------|-----|
  | Expansion | Green | `#2ecc71` |
  | Slowdown | Amber | `#f39c12` |
  | Crisis | Red | `#e74c3c` |
  | HML (Value) | Blue | `#3498db` |
  | UMD (Momentum) | Red | `#e74c3c` |
  | RMW (Quality) | Green | `#2ecc71` |
  | LowVol | Purple | `#9b59b6` |

- All plots must have: **title, axis labels, legend, grid**

```python
def save_fig(fig, name, dpi=300):
    """Standardised figure saving."""
    path = FIGURES_DIR / f'{name}.png'
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f'Saved: {path}')
```

### 14.3 Reproducibility

- `random_state=42` everywhere (HMM restarts, bootstraps, ML seeds, torch manual seed)
- **Never suppress** HMM convergence warnings, GARCH flags, or cvxpy solver warnings
- **Memory:** peak RAM should not exceed values in §2.6; monitor with `psutil`

### 14.4 Logging

```python
import logging
from datetime import datetime

logging.basicConfig(
    filename=f'logs/phase_{PHASE_NUM}_{datetime.now():%Y%m%d_%H%M}.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)
logger.info(f"Phase {PHASE_NUM} started")
```

### 14.5 GPU Detection for DL Notebooks

```python
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
# Move model and data to device
model = model.to(device)
X_tensor = torch.FloatTensor(X_train).to(device)
```

---

## 15. TROUBLESHOOTING GUIDE

### 15.1 Data Pipeline Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| French Library returns look 100× too large | Forgot to divide by 100 | Verify: `factor_returns.max() < 0.3` after load |
| yfinance download stalls | Rate limiting | Add 15s cooldown between batches; retry 3× with backoff |
| yfinance returns empty for some tickers | Delisted or renamed | Catch per-ticker; log; proceed with available tickers |
| FRED returns `'.'` values | Missing data placeholder | `pd.to_numeric(raw, errors='coerce')` |
| SQ/XYZ merge has gap | Ticker change date mismatch | Verify transition date 2025-01-21; check both tickers |
| ADF test fails to reject | Series may be non-stationary | Check if log returns (not prices) were used |

### 15.2 Econometric Model Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| HMM `ConvergenceWarning` | Insufficient iterations | Increase `n_iter` to 1000; add more restarts |
| HMM state with <5 obs | Overfitting | Reduce K; more restarts; check for outliers |
| GARCH $\alpha + \beta > 1$ | Non-stationary vol | Try GJR-GARCH/EGARCH; split sample |
| FIGARCH fails to converge | Too few observations | Exclude for tickers with <1500 obs |
| PCA 1st PC explains <20% | Indicators too heterogeneous | Use equal-weight fallback |
| DCC log-likelihood explodes | Non-PD Q_t | Add PSD correction inside recursion |

### 15.3 Optimisation Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `cvxpy` returns `infeasible` | Constraints too tight | Relax turnover; lower return target |
| `cvxpy` returns `unbounded` | Missing constraints or wrong sign | Check Maximize vs Minimize |
| BL expected returns > 10% monthly | Unbounded views | Cap at ±3%; add Ω floor |
| Ledoit-Wolf returns non-PSD | Very small sample | Use simple covariance; or apply Higham |

### 15.4 ML/DL Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| LSTM loss doesn't decrease | Learning rate too high/low | Try 1e-4 to 1e-2 range; use scheduler |
| LSTM gradient explodes (NaN loss) | No gradient clipping | Add `clip_grad_norm_(params, 1.0)` |
| Train accuracy >>  test accuracy | Overfitting | More dropout; reduce model size; more data |
| XGBoost importance inconsistent | Gain vs cover vs weight mismatch | Use SHAP for definitive ranking |
| FinBERT OOM error | Batch size too large | Reduce to batch_size=8; use `torch.no_grad()` |
| Walk-forward too slow | Large retrain freq | Reduce features; increase retrain_freq to 126 |

### 15.5 Backtest Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| NAV goes negative | TC > gross return | Cap TC at 50% of gross return |
| Dynamic strategy worse than static | Regime model doesn't add alpha after TC | Valid result — report honestly |
| Sharpe significance $p > 0.10$ | Only 192 months | Hard to achieve significance; report CI instead |

---

## 16. SUCCESS CRITERIA

The project is complete and presentation-ready when:

### 16.1 Technical Completion

1. All 12 notebooks run end-to-end without errors
2. All validation gates in every phase are checked (passed or documented as findings)
3. All `.parquet` outputs conform to the data dictionary schema (§3.5)
4. No API keys, credentials, or sensitive data in notebooks or committed code

### 16.2 Factor Timing Results

5. BL and/or CVaR strategy Sharpe exceeds equal-weight static by ≥0.15
6. Regime model correctly identifies ≥80% of NBER recession months as crisis
7. At least 2 of 4 hypotheses confirmed with $p < 0.05$ (Holm-corrected)
8. Walk-forward backtest uses zero future information (expanding-window, filtered probs)

### 16.3 ML/DL Results

9. Walk-forward predictions use strict anti-leakage protocol (§6.2 checklist all checked)
10. All ML models compared via Diebold-Mariano test with significance reported
11. All DL models trained with early stopping and temporal validation split
12. Sentiment features lagged t-1 with verification logged

### 16.4 Risk Results

13. EVT GPD parameters estimated with ≥15 exceedances
14. VaR backtests pass Kupiec POF test (green zone)
15. Copula tail dependence estimated for key pairs (semiconductors, cybersecurity)
16. Monte Carlo simulation calibrated from historical parameters

### 16.5 Deliverables

17. Excel dashboard with 10 tabs, conditional formatting
18. PowerPoint deck (20–25 slides) with consistent formatting
19. PDF report (15–20 pages) with methodology, results, limitations
20. All figures use consistent colour scheme and 300 DPI
21. Negative results documented honestly if they occur

### 16.6 Robustness

22. Statistical significance of Sharpe improvement reported (JK test or bootstrap)
23. Sub-period stability documented (at least 3 sub-periods)
24. Sensitivity to TC assumptions documented (test at 10, 25, 50 bps)
25. Model audit: train/test gap, feature ablation, overfitting diagnostics

---

## 17. APPENDICES

### A. Mathematical Notation Reference

| Symbol | Meaning |
|--------|---------|
| $r_{f,t}$ | Return of factor $f$ at time $t$ |
| $\sigma_{f,t}$ | Conditional volatility of factor $f$ at time $t$ |
| $S_t$ | Latent HMM state at time $t$ |
| $p_{k,t}$ | Filtered probability of regime $k$ at time $t$ |
| $\hat{\Sigma}_t$ | Conditional covariance matrix at time $t$ |
| $w_t$ | Portfolio weight vector at time $t$ |
| $\pi$ | Black-Litterman equilibrium expected returns |
| $\mu_{BL}$ | BL posterior expected returns |
| $\Omega$ | View uncertainty matrix |
| $Q_t$ | DCC pseudo-correlation matrix |
| $R_t$ | DCC conditional correlation matrix |
| $\xi$ | GPD shape parameter (tail index) |
| $\lambda_L$ | Lower tail dependence coefficient |

### B. FRED Series Availability Check

Run this code to verify all FRED series are accessible before starting the pipeline:

```python
from fredapi import Fred
import os
from dotenv import load_dotenv

load_dotenv()
fred = Fred(api_key=os.environ['FRED_API_KEY'])

SERIES = ['T10Y2Y', 'BAA10Y', 'VIXCLS', 'ICSA', 'M2SL',
          'CPIAUCSL', 'USALOLITONOSTSAM', 'DCOILWTICO', 'INDPRO', 'UNRATE']

for code in SERIES:
    try:
        data = fred.get_series(code, observation_start='2020-01-01')
        print(f"✓ {code}: {len(data)} obs, latest={data.index[-1].date()}")
    except Exception as e:
        print(f"✗ {code}: {e}")
```

### C. Complete Output File Manifest

```
data/processed/          (17 parquet files)
outputs/figures/         (~25 PNG files at 300 DPI)
outputs/tables/          (~15 CSV files)
outputs/reports/         (1 PDF + 1 XLSX + 1 PPTX)
outputs/models/          (~5 PKL + 2 PT files)
logs/                    (12 log files, one per phase)
```

### D. Version Compatibility Notes

| Package | Tested Version | Breaking Change Risk |
|---------|---------------|---------------------|
| `pandas` | 2.1–2.2 | `resample().mean()` behaviour changed in 2.0 |
| `numpy` | 1.24–1.26 | NumPy 2.0 breaks many dependencies; pin <2.0 |
| `arch` | 6.2–7.0 | API stable for GARCH; distribution names may change |
| `hmmlearn` | 0.3.x | `_compute_log_likelihood` is private API; may change |
| `cvxpy` | 1.4–1.5 | Solver availability may vary; SCS always available |
| `torch` | 2.1–2.4 | LSTM API stable; distributed training API changes |
| `yfinance` | 0.2.31+ | Frequent Yahoo API changes cause breakage |

### E. Sector Mapping for Constraints

```python
SECTOR_MAP = {
    'Semiconductors': ['NVDA', 'AMD', 'TSM', 'AVGO', 'QCOM', 'MU'],
    'Big Tech': ['AAPL', 'MSFT', 'GOOG', 'META', 'NFLX'],
    'Enterprise Software': ['CRM', 'ADBE', 'NOW'],
    'Cybersecurity': ['PANW', 'CRWD'],
    'Analytics': ['DDOG', 'PLTR'],
    'Infrastructure': ['ANET', 'XYZ'],
}

def check_sector_constraints(weights, max_sector_weight=0.30):
    """Verify no sector exceeds 30% total weight."""
    for sector, tickers in SECTOR_MAP.items():
        sector_weight = sum(weights.get(t, 0) for t in tickers)
        if sector_weight > max_sector_weight:
            print(f"⚠️ {sector}: {sector_weight:.1%} > {max_sector_weight:.0%} limit")
```

### F. Estimated Runtimes

| Phase | Notebook | Runtime | Bottleneck |
|-------|----------|---------|------------|
| 1 | Data Pipeline | 5–15 min | yfinance download |
| 2 | Factor Construction | 2–5 min | Rolling volatility on 500 stocks |
| 3 | Factor Validation + GARCH | 5–10 min | 320 GARCH fits |
| 4 | HMM Regime Detection | 2–5 min | 25 HMM restarts × 4 state counts |
| 5 | Regime-Conditional Analysis | 1–3 min | Bootstrap (10K resamples) |
| 6 | DCC-GARCH | 3–8 min | DCC MLE optimisation |
| 7 | Black-Litterman | 2–5 min | Expanding-window BL |
| 8 | Mean-CVaR | 2–5 min | LP at each month |
| 9 | Backtest Engine | 10–30 min | Expanding re-estimation |
| 10 | Stress Testing | 3–5 min | EVT fitting |
| 11 | Report & Dashboard | 2–5 min | Excel/PDF generation |
| 12 | Presentation | 2–5 min | PowerPoint generation |
| **Total** | | **~40–100 min** | |

---

### G. Visualization Recipes

#### G.1 Regime Overlay Plot

```python
def plot_regime_overlay(prices, regime_probs, regime_labels, title=''):
    """
    Plot price series with colour-coded regime background shading.
    This is the signature plot of the project — must be publication-quality.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={'height_ratios': [3, 1]},
                                    sharex=True)

    # Top: price with regime shading
    ax1.plot(prices.index, prices, color='black', linewidth=0.8)

    regime_colors = {
        'Expansion': '#2ecc71',
        'Slowdown': '#f39c12',
        'Crisis': '#e74c3c',
    }

    for i in range(len(prices) - 1):
        label = regime_labels.iloc[i]
        if label in regime_colors:
            ax1.axvspan(prices.index[i], prices.index[i+1],
                       alpha=0.15, color=regime_colors[label], linewidth=0)

    ax1.set_ylabel('Price / Index Level')
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Bottom: regime probabilities stacked
    ax2.stackplot(regime_probs.index,
                  regime_probs['p_expansion'],
                  regime_probs['p_slowdown'],
                  regime_probs['p_crisis'],
                  colors=['#2ecc71', '#f39c12', '#e74c3c'],
                  alpha=0.7,
                  labels=['Expansion', 'Slowdown', 'Crisis'])
    ax2.set_ylabel('Regime Probability')
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
```

#### G.2 Factor Return Heatmap by Regime

```python
def plot_regime_heatmap(regime_stats, metric='ann_return'):
    """
    Heatmap showing factor returns by regime — the visual core of the thesis.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    data = regime_stats.pivot(index='factor', columns='regime', values=metric)

    # Reorder columns
    col_order = ['Expansion', 'Slowdown', 'Crisis']
    data = data[col_order]

    im = ax.imshow(data.values, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.index)

    # Annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = f'{data.values[i, j]:.2%}'
            color = 'white' if abs(data.values[i, j]) > 0.15 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=11)

    ax.set_title(f'Factor {metric.replace("_", " ").title()} by Regime', fontsize=13)
    fig.colorbar(im, ax=ax, label=metric)
    plt.tight_layout()
    return fig
```

#### G.3 Drawdown Comparison Chart

```python
def plot_drawdown_comparison(nav_dict, title='Strategy Drawdowns'):
    """
    Drawdown area chart comparing multiple strategies simultaneously.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = plt.cm.Set1(np.linspace(0, 1, len(nav_dict)))

    for (name, nav), color in zip(nav_dict.items(), colors):
        drawdown = (nav / nav.cummax()) - 1
        ax.fill_between(drawdown.index, drawdown, alpha=0.2, color=color)
        ax.plot(drawdown.index, drawdown, label=name, color=color, linewidth=1)

    ax.set_ylabel('Drawdown')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    plt.tight_layout()
    return fig
```

#### G.4 Weight Evolution Stacked Area

```python
def plot_weight_evolution(weights_df, title='Portfolio Weight Evolution'):
    """
    Stacked area chart showing how portfolio weights change over time.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.stackplot(weights_df.index, *[weights_df[c] for c in weights_df.columns],
                 labels=weights_df.columns, alpha=0.8)

    ax.set_ylabel('Portfolio Weight')
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, ncol=4)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig
```

#### G.5 QQ-Plot for Residual Diagnostics

```python
from scipy import stats

def plot_qq(residuals, title='QQ Plot of Standardised Residuals'):
    """
    QQ plot comparing residuals to normal distribution.
    Deviation at tails indicates heavy tails → use Student's t or skewed-t.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(residuals, dist='norm', plot=ax)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
```

#### G.6 SHAP Summary Plot Wrapper

```python
import shap

def plot_shap_summary(model, X_test, feature_names, title='Feature Importance (SHAP)'):
    """
    SHAP summary plot showing feature impact direction and magnitude.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                     show=False, plot_type='bar')
    plt.title(title, fontsize=13)
    plt.tight_layout()
    return fig
```

#### G.7 Efficient Frontier Plot

```python
def plot_efficient_frontier(returns, cov, n_portfolios=5000,
                            optimal_weights=None, title='Efficient Frontier'):
    """
    Plot the efficient frontier with random portfolios and optimal point.
    """
    n_assets = returns.shape[1]
    results = np.zeros((3, n_portfolios))

    for i in range(n_portfolios):
        w = np.random.dirichlet(np.ones(n_assets))
        port_ret = np.sum(returns.mean().values * w) * 252
        port_vol = np.sqrt(w @ cov.values @ w) * np.sqrt(252)
        results[0, i] = port_vol
        results[1, i] = port_ret
        results[2, i] = port_ret / port_vol  # Sharpe

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(results[0], results[1], c=results[2], cmap='viridis',
                         marker='.', s=5, alpha=0.5)
    fig.colorbar(scatter, ax=ax, label='Sharpe Ratio')

    if optimal_weights is not None:
        opt_ret = np.sum(returns.mean().values * optimal_weights) * 252
        opt_vol = np.sqrt(optimal_weights @ cov.values @ optimal_weights) * np.sqrt(252)
        ax.scatter(opt_vol, opt_ret, marker='*', color='red', s=300,
                  label='Optimal', zorder=5)
        ax.legend()

    ax.set_xlabel('Annual Volatility')
    ax.set_ylabel('Annual Return')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
```

---

### H. Feature Engineering Reference

#### H.1 Technical Indicators

```python
def compute_all_technical_indicators(df, ticker=None):
    """
    Compute comprehensive technical indicators from OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame with columns: Open, High, Low, Close, Volume
    """
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    log_ret = np.log(close / close.shift(1))

    features = pd.DataFrame(index=df.index)

    # === Momentum ===
    features['log_return_1d'] = log_ret
    features['momentum_5d'] = close.pct_change(5)
    features['momentum_21d'] = close.pct_change(21)
    features['momentum_63d'] = close.pct_change(63)

    # RSI (14-day)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    features['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    features['macd'] = ema12 - ema26
    features['macd_signal'] = features['macd'].ewm(span=9).mean()
    features['macd_histogram'] = features['macd'] - features['macd_signal']

    # === Volatility ===
    features['vol_5d'] = log_ret.rolling(5).std() * np.sqrt(252)
    features['vol_21d'] = log_ret.rolling(21).std() * np.sqrt(252)
    features['vol_63d'] = log_ret.rolling(63).std() * np.sqrt(252)

    # Garman-Klass volatility
    log_hl = np.log(high / low)
    log_co = np.log(close / df['Open'])
    features['gk_vol'] = np.sqrt(
        (0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2).rolling(21).mean() * 252
    )

    # Parkinson volatility
    features['park_vol'] = np.sqrt(
        (log_hl**2 / (4 * np.log(2))).rolling(21).mean() * 252
    )

    # ATR (14-day)
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': abs(high - close.shift(1)),
        'lc': abs(low - close.shift(1))
    }).max(axis=1)
    features['atr_14'] = tr.rolling(14).mean()

    # Bollinger Bands
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    features['bb_pctb'] = (close - (bb_mid - 2*bb_std)) / (4*bb_std)

    # === Volume ===
    features['volume_zscore'] = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()
    features['volume_momentum'] = volume.rolling(5).mean() / volume.rolling(20).mean()

    # === Range-based ===
    features['high_low_ratio'] = np.log(high / low)
    features['close_range_pct'] = (close - low) / (high - low).replace(0, np.nan)

    return features
```

#### H.2 Macro Feature Transformations

```python
def compute_macro_features(macro_raw):
    """
    Transform raw FRED macro data into model-ready features.
    All transforms documented in §3.2.
    """
    features = pd.DataFrame(index=macro_raw.index)

    # Direct levels (already spreads/indices)
    features['t10y2y'] = macro_raw['T10Y2Y']
    features['baa10y'] = macro_raw['BAA10Y']
    features['vix'] = macro_raw['VIXCLS']
    features['oecd_cli'] = macro_raw['USALOLITONOSTSAM']
    features['unrate'] = macro_raw['UNRATE']

    # Rate of change transforms
    features['claims_roc3m'] = macro_raw['ICSA'].pct_change(3)
    features['oil_roc3m'] = macro_raw['DCOILWTICO'].pct_change(3)
    features['indpro_yoy'] = macro_raw['INDPRO'].pct_change(12)
    features['unrate_chg12m'] = macro_raw['UNRATE'].diff(12)

    # Real M2
    real_m2 = macro_raw['M2SL'] / macro_raw['CPIAUCSL'] * 100
    features['real_m2_yoy'] = real_m2.pct_change(12)

    return features
```

---

### I. Covariance Matrix Cookbook

#### I.1 Methods Comparison

| Method | Sample Size (T/N) | Properties | When to Use |
|--------|------------------|------------|-------------|
| Sample covariance | ≥3 | Unbiased; high variance | Large T/N ratio |
| Ledoit-Wolf shrinkage | ≥1.5 | Better conditioned; biased | Default choice |
| DCC-GARCH | ≥100 months | Time-varying; PSD issues | Dynamic allocation |
| Exponentially weighted | ≥30 | Recency bias | Fast adaptation |
| Factor model | ≥2 | Low rank; structured | Very high N |

#### I.2 PSD Correction Methods

| Method | Quality | Speed | Implementation |
|--------|---------|-------|----------------|
| Eigenvalue clipping | Good | Fast | `eigvals = max(eigvals, eps)` |
| Higham's nearest PSD | Best | Moderate | `scipy` or custom iteration |
| Diagonal loading | Simple | Fastest | `Σ + εI` |
| Spectral cleaning | Good | Fast | RMT-based eigenvalue shrinkage |

```python
def higham_nearest_psd(A, max_iter=100, tol=1e-10):
    """
    Higham's algorithm for the nearest PSD matrix (in Frobenius norm).
    """
    n = A.shape[0]
    S = np.zeros_like(A)
    Y = A.copy()

    for k in range(max_iter):
        R = Y - S
        # Project onto PSD cone
        eigvals, eigvecs = np.linalg.eigh(R)
        eigvals = np.maximum(eigvals, 0)
        X = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Dykstra correction
        S = X - R
        # Project onto unit diagonal
        Y = X.copy()
        np.fill_diagonal(Y, 1.0)

        # Check convergence
        if np.linalg.norm(Y - X, 'fro') < tol:
            break

    return Y
```

---

### J. Excel Dashboard Specification

#### J.1 Tab Layout (10 Tabs)

| Tab # | Name | Content |
|-------|------|---------|
| 1 | Executive Summary | Key metrics table, headline charts, traffic-light status |
| 2 | Performance Summary | Full metrics comparison across all strategies |
| 3 | Annual Returns | Year-by-year returns, calendar heatmap format |
| 4 | Regime Timeline | Macro regime labels with colour coding |
| 5 | Weight History | Monthly weight allocations for BL and CVaR |
| 6 | Stress Test Results | Named event analysis, drawdown comparison |
| 7 | EVT Tail Metrics | GPD parameters, VaR/ES at 95th, 99th, 99.9th |
| 8 | Factor Conditional Stats | Mean, vol, Sharpe by factor × regime |
| 9 | Hypothesis Tests | Welch's t, Kruskal-Wallis, bootstrap results |
| 10 | Model Diagnostics | GARCH params, HMM convergence, ML metrics |

#### J.2 Conditional Formatting Rules

```python
def style_performance_table(df):
    """Apply conditional formatting to performance comparison table."""
    def color_sharpe(val):
        if val > 1.0: return 'background-color: #2ecc71'
        elif val > 0.5: return 'background-color: #f1c40f'
        else: return 'background-color: #e74c3c'

    def color_drawdown(val):
        if val > -0.10: return 'background-color: #2ecc71'
        elif val > -0.25: return 'background-color: #f1c40f'
        else: return 'background-color: #e74c3c'

    styled = df.style
    styled = styled.applymap(color_sharpe, subset=['sharpe'])
    styled = styled.applymap(color_drawdown, subset=['max_drawdown'])
    styled = styled.format({
        'ann_return': '{:.2%}',
        'ann_volatility': '{:.2%}',
        'sharpe': '{:.2f}',
        'max_drawdown': '{:.2%}',
    })
    return styled
```

#### J.3 Excel Export Function

```python
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font, PatternFill, Alignment, Border
from openpyxl.chart import LineChart, Reference

def export_dashboard(all_data, filepath='outputs/reports/dashboard.xlsx'):
    """
    Export comprehensive Excel dashboard with all tabs.
    """
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # Tab 1: Executive Summary
        exec_summary = all_data['executive_summary']
        exec_summary.to_excel(writer, sheet_name='Executive Summary', index=True)

        # Tab 2: Performance Summary
        perf = all_data['performance']
        perf.to_excel(writer, sheet_name='Performance Summary', index=True)

        # Tab 3: Annual Returns
        annual = all_data['annual_returns']
        annual.to_excel(writer, sheet_name='Annual Returns', index=True)

        # ... tabs 4-10 follow same pattern ...

        # Add conditional formatting
        wb = writer.book
        for ws in wb.worksheets:
            ws.sheet_properties.tabColor = '1F77B4'
            # Auto-fit column widths
            for column_cells in ws.columns:
                max_length = max(len(str(cell.value or '')) for cell in column_cells)
                ws.column_dimensions[column_cells[0].column_letter].width = min(max_length + 2, 25)

    print(f"Dashboard exported to {filepath}")
```

---

### K. PDF Report Template

#### K.1 Report Structure (15–20 Pages)

```
1. Executive Summary (1 page)
   - Thesis, key findings, recommendations
   - Headline metrics: Sharpe, drawdown, information ratio

2. Introduction & Motivation (1 page)
   - Academic context, market anomalies
   - Research question and contribution

3. Data Description (1–2 pages)
   - Sources, cleaning methodology, summary statistics
   - Date range, frequency, universe

4. Methodology (3–4 pages)
   - Factor construction (low-vol)
   - HMM regime detection (expanding-window, filtered probabilities)
   - DCC-GARCH dynamic covariance
   - Black-Litterman & Mean-CVaR allocation
   - ML/DL volatility forecasting (walk-forward)
   - Copula tail risk modeling

5. Results (4–5 pages)
   - Regime-conditional factor behaviour
   - Hypothesis test outcomes
   - Backtest performance comparison
   - ML model comparison (Diebold-Mariano)

6. Risk Analysis (2 pages)
   - VaR/CVaR analysis (5 methods)
   - EVT tail risk assessment
   - Stress test results
   - Copula joint crash probabilities

7. Discussion & Limitations (1 page)
   - Survivorship bias caveat
   - Look-ahead bias prevention documentation
   - Data snooping concerns
   - Market impact considerations

8. Conclusion & Future Work (1 page)

9. Appendices (2–3 pages)
   - Detailed statistical tables
   - All validation gate results
   - Model diagnostic plots
```

---

### L. Presentation Design Guide

#### L.1 Slide Format Standards

```python
# PowerPoint generation with python-pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

FONT_TITLE = Pt(28)
FONT_SUBTITLE = Pt(18)
FONT_BODY = Pt(14)
FONT_FOOTNOTE = Pt(10)

COLOR_DARK = RGBColor(0x1A, 0x1A, 0x2E)   # Dark navy
COLOR_ACCENT = RGBColor(0x00, 0xD2, 0xFF)  # Cyan accent
COLOR_RED = RGBColor(0xE7, 0x4C, 0x3C)     # Alert red
COLOR_GREEN = RGBColor(0x2E, 0xCC, 0x71)   # Success green

def add_title_slide(prs, title, subtitle=''):
    """Add a title slide."""
    layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(layout)
    slide.shapes.title.text = title
    slide.placeholders[1].text = subtitle
    return slide

def add_chart_slide(prs, title, image_path, notes=''):
    """Add a slide with a chart image."""
    layout = prs.slide_layouts[5]  # Blank
    slide = prs.slides.add_slide(layout)

    # Title
    txBox = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.8))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = FONT_TITLE
    p.font.bold = True

    # Chart image
    slide.shapes.add_picture(str(image_path),
                             Inches(0.5), Inches(1.2), Inches(9), Inches(5.5))

    # Notes
    if notes:
        slide.notes_slide.notes_text_frame.text = notes

    return slide
```

#### L.2 Key Chart-to-Slide Mapping

| Slide # | Title | Chart Source | Key Message |
|---------|-------|-------------|-------------|
| 6 | Factor Performance | `factor_cumulative_returns.png` | All 4 factors have positive long-run premia |
| 7 | Regime Detection | `composite_index_with_regimes.png` | HMM correctly identifies crisis periods |
| 8 | Regime-Conditional Sharpe | `regime_conditional_sharpe_barplot.png` | **The key insight**: factors behave differently |
| 9 | Momentum Crashes | `momentum_crash_timeline.png` | UMD crashes in crisis→recovery transitions |
| 10 | DCC-GARCH | `conditional_correlation_timeseries.png` | Correlations spike in crisis |
| 11 | BL Allocation | Weight history stacked area | Dynamic weights respond to regime |
| 13 | Backtest NAV | NAV comparison line chart | Dynamic strategy outperforms static |
| 16 | Tail Risk | EVT QQ plot + GPD fit | Heavy tails confirmed; EVT VaR more accurate |

---

### M. Risk Budgeting Framework

#### M.1 Risk Decomposition

$$\text{RC}_i = w_i \frac{(\Sigma w)_i}{\sqrt{w' \Sigma w}}$$

$$\text{\%RC}_i = \frac{\text{RC}_i}{\sum_j \text{RC}_j} = \frac{w_i (\Sigma w)_i}{w' \Sigma w}$$

Verify: $\sum_i \text{\%RC}_i = 1$

```python
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
```

#### M.2 Concentration Metrics

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| Herfindahl-Hirschman (HHI) | $\sum w_i^2$ | 0 = perfect dispersion; 1 = concentrated |
| Effective N | $1 / \sum w_i^2$ | Equivalent number of equal-weight positions |
| Diversification Ratio | $\frac{\sum w_i \sigma_i}{\sigma_p}$ | >1 means diversification benefit exists |
| Max weight | $\max_i w_i$ | Single-name concentration risk |

```python
def concentration_metrics(weights, cov=None):
    """Portfolio concentration analysis."""
    hhi = np.sum(weights ** 2)
    effective_n = 1 / hhi
    max_weight = np.max(weights)

    metrics = {
        'hhi': hhi,
        'effective_n': effective_n,
        'max_weight': max_weight,
    }

    if cov is not None:
        vols = np.sqrt(np.diag(cov))
        port_vol = np.sqrt(weights @ cov @ weights)
        div_ratio = np.sum(weights * vols) / port_vol
        metrics['diversification_ratio'] = div_ratio

    return metrics
```

---

### N. Model Interpretability Guide

#### N.1 SHAP Analysis Protocol

```python
def shap_analysis(model, X_train, X_test, feature_names):
    """
    Complete SHAP analysis for interpretability.
    Run AFTER walk-forward evaluation to explain final model.
    """
    import shap

    # TreeExplainer for tree-based models (XGBoost, LightGBM, RF)
    if hasattr(model, 'get_booster') or hasattr(model, 'feature_importances_'):
        explainer = shap.TreeExplainer(model)
    else:
        # KernelExplainer for others (slower)
        explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 100))

    shap_values = explainer.shap_values(X_test)

    # Summary report
    importance = np.abs(shap_values).mean(axis=0)
    ranked = sorted(zip(feature_names, importance), key=lambda x: -x[1])

    print("SHAP Feature Importance Ranking:")
    for i, (name, imp) in enumerate(ranked):
        print(f"  {i+1}. {name}: {imp:.6f}")

    return shap_values, ranked
```

#### N.2 Partial Dependence Analysis

```python
from sklearn.inspection import PartialDependenceDisplay

def plot_partial_dependence(model, X, features_to_plot, feature_names):
    """
    Plot partial dependence for top features.
    Shows marginal effect of feature on prediction.
    """
    fig, axes = plt.subplots(1, len(features_to_plot), figsize=(5*len(features_to_plot), 4))
    PartialDependenceDisplay.from_estimator(
        model, X, features_to_plot,
        feature_names=feature_names, ax=axes
    )
    plt.suptitle('Partial Dependence Plots', fontsize=14)
    plt.tight_layout()
    return fig
```

---

### O. Data Quality Automation

#### O.1 Pre-Run Data Integrity Check

```python
def pre_run_data_check():
    """
    Run before ANY notebook to ensure data pipeline is intact.
    This catches corrupted files, missing phases, and stale data.
    """
    from pathlib import Path
    import hashlib

    checks = []

    # 1. Check all required processed files exist
    required_files = [
        'factor_returns.parquet',
        'macro_indicators.parquet',
        'master_data.parquet',
    ]

    for f in required_files:
        path = Path(f'data/processed/{f}')
        exists = path.exists()
        checks.append({
            'file': f,
            'exists': exists,
            'size_mb': path.stat().st_size / 1e6 if exists else 0,
            'modified': pd.Timestamp(path.stat().st_mtime, unit='s') if exists else None,
        })

    # 2. Check for stale data (>30 days old)
    for check in checks:
        if check['modified'] and (pd.Timestamp.now() - check['modified']).days > 30:
            check['stale'] = True
        else:
            check['stale'] = False

    # 3. Print summary
    df = pd.DataFrame(checks)
    print("=== Data Integrity Check ===")
    print(df.to_string(index=False))

    missing = df[~df['exists']]
    if len(missing) > 0:
        print(f"\n⚠️ {len(missing)} required files are MISSING!")
        print("Run Phase 1 (01_data_pipeline.ipynb) first.")
        return False

    return True
```

#### O.2 Checksum Verification

```python
import hashlib

def compute_file_checksum(filepath, algorithm='sha256'):
    """Compute SHA-256 checksum of a file for reproducibility verification."""
    h = hashlib.new(algorithm)
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def verify_checksums(checksum_file='data/raw/checksums.json'):
    """Verify all raw data files match stored checksums."""
    import json
    with open(checksum_file) as f:
        stored = json.load(f)

    for filepath, expected_hash in stored.items():
        if not os.path.exists(filepath):
            print(f"✗ MISSING: {filepath}")
            continue
        actual_hash = compute_file_checksum(filepath)
        if actual_hash == expected_hash:
            print(f"✓ {filepath}")
        else:
            print(f"✗ MISMATCH: {filepath}")
            print(f"  Expected: {expected_hash[:16]}...")
            print(f"  Got:      {actual_hash[:16]}...")
```

---

### P. Final Checklists

#### P.1 Pre-Submission Checklist

- [ ] All 12 notebooks execute end-to-end without errors
- [ ] No hardcoded API keys or credentials in notebooks
- [ ] All `.parquet` outputs match data dictionary schema (§3.5)
- [ ] All figures saved at 300 DPI with consistent colour scheme
- [ ] All tables saved as CSV in `outputs/tables/`
- [ ] Excel dashboard has 10 tabs with conditional formatting
- [ ] PDF report is 15–20 pages with methodology + results + limitations
- [ ] PowerPoint has 20–25 slides in consistent format
- [ ] All ML models use walk-forward (no k-fold) with scaler inside pipeline
- [ ] All sentiment features lagged by t-1 with verification logged
- [ ] No SMOTE applied to regression targets
- [ ] HMM uses filtered probabilities (not smoothed)
- [ ] DCC-GARCH produces PSD covariance matrices at every timestep
- [ ] BL views capped at ±3% monthly; Ω uses RMSE² (not 1/RMSE)
- [ ] Transaction costs applied consistently at 25 bps (factors) / 10 bps (tech)
- [ ] Negative results documented honestly if they occur
- [ ] Sharpe ratio significance tested (JK test or bootstrap)
- [ ] Validation gates checked for all 12 phases
- [ ] `.env` file excluded from git
- [ ] Random state = 42 used consistently throughout
- [ ] Memory usage within limits (§2.6)
- [ ] All external data sources credited
- [ ] Version compatibility notes documented

#### P.2 Code Quality Checklist

- [ ] All functions have docstrings with parameter descriptions
- [ ] No deprecated pandas (`.append()`) or numpy (`.bool`) usage
- [ ] Import statements at top of each notebook
- [ ] No circular imports between `src/` modules
- [ ] All file paths use `pathlib.Path` or `src/config.py` constants
- [ ] Exception handling around data downloads (yfinance, FRED)
- [ ] Progress bars (`tqdm`) for long loops (GARCH fits, walk-forward)
- [ ] Logging configured per notebook

#### P.3 Academic Rigour Checklist

- [ ] All statistical tests use appropriate standard errors (HAC for time series)
- [ ] Multiple testing correction applied (Holm-Bonferroni)
- [ ] Confidence intervals reported alongside point estimates
- [ ] Bootstrap standard errors for non-parametric metrics (Sharpe)
- [ ] Survivorship bias acknowledged and quantified where possible
- [ ] Look-ahead bias prevention documented step by step
- [ ] Robustness checks (sub-period, TC sensitivity, model sensitivity) performed
- [ ] Academic references cited for all methods used

---

### Q. Glossary of Technical Terms

| Term | Definition | Context |
|------|-----------|---------|
| **Alpha** | Excess return above a benchmark or factor model prediction | Phase 3 regression intercept; Phase 9 backtest |
| **ARCH effect** | Autoregressive conditional heteroskedasticity — vol clusters | Phase 3 diagnostics; Phase 6 GARCH |
| **Backtest** | Simulated strategy performance using historical data | Phase 9 engine — critical path |
| **Baum-Welch** | EM algorithm for HMM parameter estimation | Phase 4 HMM fitting |
| **BIC** | Bayesian Information Criterion — model selection penalising complexity | Phase 3 GARCH model selection; Phase 4 HMM K selection |
| **Black-Litterman** | Bayesian portfolio model combining equilibrium with views | Phase 7 allocation |
| **Bootstrap** | Resampling method for confidence intervals | Phase 5 Sharpe CIs |
| **Cholesky** | Matrix decomposition for generating correlated random variables | Phase 10 Monte Carlo; Section 10 |
| **Clayton copula** | Copula with lower tail dependence | Section 9 copula modeling |
| **CVaR / ES** | Conditional Value at Risk = Expected Shortfall = average loss beyond VaR | Phase 8 optimization; Section 9 |
| **DCC** | Dynamic Conditional Correlation — time-varying correlation model | Phase 6 |
| **Diebold-Mariano** | Test for equal predictive accuracy between two models | Section 6.5 model comparison |
| **Expanding window** | Estimation using all data from start to current point | Mandatory protocol — all estimation |
| **EVT** | Extreme Value Theory — statistical framework for tail events | Phase 10; Section 9 |
| **Factor premium** | Expected return earned by holding a systematic risk factor | Core thesis |
| **FIGARCH** | Fractionally Integrated GARCH — captures long memory in volatility | Phase 3 GARCH pipeline |
| **Filtered probability** | $P(S_t | Z_{1:t})$ — forward-only, no future information | Phase 4 — MANDATORY |
| **GJR-GARCH** | GARCH with leverage effect (bad news → more vol than good news) | Phase 3; Phase 6 |
| **GPD** | Generalised Pareto Distribution — models exceedances over threshold | Phase 10 EVT |
| **Granger causality** | Statistical test: does X help predict Y beyond Y's own history? | Section 10F |
| **HAC standard errors** | Heteroskedasticity and Autocorrelation Consistent (Newey-West) | Phase 3 regression |
| **HMM** | Hidden Markov Model — latent state time series model | Phase 4 regime detection |
| **Holm-Bonferroni** | Step-down multiple testing correction (less conservative than Bonferroni) | Phase 5 hypothesis tests |
| **HRP** | Hierarchical Risk Parity — allocation using clustering | Section 10B |
| **Information ratio** | Excess return over benchmark / tracking error | Phase 9 performance |
| **Jobson-Korkie** | Test for difference between two Sharpe ratios | Phase 9 significance |
| **Kupiec POF** | Proportion of failures test for VaR backtesting | Section 9.4 |
| **Ledoit-Wolf** | Shrinkage covariance estimator — better conditioned | Section 10B.5 |
| **Look-ahead bias** | Using future information in historical analysis | Must be prevented — §6.2 |
| **Low-volatility anomaly** | Low-vol stocks earn higher risk-adjusted returns than predicted | Phase 2 |
| **Mincer-Zarnowitz** | Test for forecast calibration (unbiasedness) | Section 6.5 |
| **Monte Carlo** | Simulation method using random sampling | Section 10 |
| **Newey-West** | HAC estimator with automatic bandwidth selection | Phase 3 |
| **PCA** | Principal Component Analysis — dimensionality reduction | Phase 4 composite index |
| **PSD** | Positive Semi-Definite — required for valid covariance matrices | Phase 6; Section 10B |
| **Pseudo-observation** | Rank-transformed data divided by (n+1); for copula fitting | Section 9.3 |
| **Regime** | Macroeconomic state (expansion/slowdown/crisis) | Core thesis — Phase 4 |
| **Risk budgeting** | Allocation targeting specific risk contributions per asset | Section 10B.3 |
| **SHAP** | SHapley Additive exPlanations — model interpretability | Section 6.6; Appendix N |
| **Sharpe ratio** | Risk-adjusted return: (mean excess return) / volatility | Throughout |
| **Smoothed probability** | $P(S_t | Z_{1:T})$ — uses future data (NEVER use in backtest) | Phase 4 — FORBIDDEN |
| **Sortino ratio** | Like Sharpe but using downside deviation instead of full vol | Phase 9 |
| **Survivorship bias** | Bias from only analysing stocks that survived (excludes failures) | Phase 2 caveat |
| **Transaction cost** | Cost of trading; modeled as fixed basis points per unit turnover | Phase 9 — 25 bps factors, 10 bps tech |
| **Turnover** | Sum of absolute weight changes per rebalance period | Phase 9 constraint |
| **VaR** | Value at Risk — loss not exceeded with probability (1-α) | Section 9 |
| **Viterbi** | Most likely state sequence algorithm for HMM (NOT for live use) | Phase 4 — for comparison only |
| **Walk-forward** | Backtesting method that re-estimates model at each point | Phase 9; Section 6 |
| **Yang-Zhang** | Range-based volatility estimator using OHLC data | Phase 2 enhancement |

---

### R. Change Log

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-01-15 | Initial | Initial blueprint draft |
| 1.1 | 2025-02-01 | — | Added HMM filtered probability protocol |
| 1.2 | 2025-03-01 | — | Added tech portfolio component (20 tickers) |
| 1.3 | 2025-04-01 | — | Added ML/DL sections (NB07-NB10) |
| 1.4 | 2025-06-01 | — | Added FinBERT sentiment pipeline |
| 1.5 | 2025-09-01 | — | Added copula tail risk modeling |
| 2.0 | 2026-03-26 | — | **Major rewrite:** added 6 new sections (ML governance, DL architecture, hybrid ensemble, copula modeling, Monte Carlo, NLP/FinBERT), expanded pitfalls (15→30), expanded troubleshooting (12→25 entries), added 16 appendices (A-P), added configuration management, quality gates, expanded portfolio optimization (HRP, ERC, Ledoit-Wolf), added feature engineering reference, visualization recipes, Excel/PDF/PPT specifications, final checklists. Grew from ~1700 to 4000+ lines. |

---

### S. Debugging Quick Reference

#### S.1 Common Import Errors

```python
# Problem: ModuleNotFoundError: No module named 'src'
# Fix: Install in editable mode
!pip install -e .  # from project root

# Problem: ModuleNotFoundError: No module named 'arch'
# Fix:
!pip install arch

# Problem: ImportError: cannot import name 'FamaFrenchReader'
# Fix: Use pandas_datareader directly
import pandas_datareader.data as pdr
data = pdr.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start='2004')
```

#### S.2 Common Runtime Errors

```python
# Problem: LinAlgError: Matrix is not positive definite (Cholesky)
# Fix: Apply PSD correction before Cholesky
from src.garch_utils import ensure_psd
cov_fixed = ensure_psd(cov_matrix)
L = np.linalg.cholesky(cov_fixed)

# Problem: ValueError: array must not contain infs or NaNs (GARCH)
# Fix: Drop NaN/inf, verify data is clean
returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
assert not returns.isna().any(), "NaN found in returns"

# Problem: ConvergenceWarning: Model did not converge (HMM)
# Fix: Increase iterations and add restarts
model = GaussianHMM(n_components=3, n_iter=1000, tol=1e-8, random_state=42)

# Problem: SolverError: Problem is infeasible (cvxpy)
# Fix: Loosen constraints
# - Increase max_weight from 0.40 to 0.50
# - Remove turnover constraint temporarily
# - Lower return target

# Problem: RuntimeError: CUDA error: device-side assert triggered (PyTorch)
# Fix: Check for NaN in inputs; verify tensor shapes match
assert not torch.isnan(X_tensor).any(), "NaN in input tensor"
print(f"X shape: {X_tensor.shape}, y shape: {y_tensor.shape}")
```

#### S.3 Data Quality Quick Checks

```python
def quick_data_check(df, name='DataFrame'):
    """Run essential data quality checks in one call."""
    print(f"=== {name} ===")
    print(f"  Shape: {df.shape}")
    print(f"  Index: {type(df.index).__name__}, range: {df.index.min()} → {df.index.max()}")
    print(f"  Duplicates: {df.index.duplicated().sum()}")
    print(f"  NaN total: {df.isna().sum().sum()} ({df.isna().mean().mean():.1%})")
    print(f"  Inf total: {np.isinf(df.select_dtypes(include=[np.number])).sum().sum()}")
    print(f"  Dtypes: {dict(df.dtypes.value_counts())}")
    if hasattr(df.index, 'is_monotonic_increasing'):
        print(f"  Sorted: {df.index.is_monotonic_increasing}")
    print()
```

#### S.4 Memory Usage Monitor

```python
import psutil
import os

def log_memory():
    """Log current memory usage — call at each phase boundary."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory: RSS={mem_info.rss / 1e9:.2f} GB, VMS={mem_info.vms / 1e9:.2f} GB")

    # System-level
    vm = psutil.virtual_memory()
    print(f"System: {vm.percent}% used ({vm.used / 1e9:.1f} / {vm.total / 1e9:.1f} GB)")
```

#### S.5 Timing Decorator

```python
import time
from functools import wraps

def timer(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"⏱️ {func.__name__}: {elapsed:.2f}s")
        return result
    return wrapper

# Usage:
@timer
def fit_all_garch_models(returns, tickers):
    # ... fitting code ...
    pass
```

---

*End of claude.md master blueprint — Version 2.0*
*Total: 17 core sections + 19 appendices (A through S)*
*Target audience: Claude Code AI agent for autonomous execution*

