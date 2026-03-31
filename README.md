# Cross-Sector Factor Timing & Dynamic Allocation Engine

A comprehensive quantitative finance project implementing regime-aware factor timing
and dynamic portfolio allocation across tech-sector stocks.

## Project Overview

This project builds a 12-phase pipeline from raw data to presentation-ready deliverables:

| Phase | Notebook | Description |
|-------|----------|-------------|
| 1 | `NB01_data_pipeline` | Data acquisition (French Library, FRED, Yahoo Finance) |
| 2 | `NB02_factor_construction` | Custom low-volatility factor construction |
| 3 | `NB03_factor_validation_garch` | Factor premium validation + GARCH volatility models |
| 4 | `NB04_hmm_regime_detection` | Hidden Markov Model regime detection (expanding window) |
| 5 | `NB05_regime_conditional` | Regime-conditional analysis + hypothesis testing |
| 6 | `NB06_dcc_garch` | DCC-GARCH dynamic covariance + FinBERT sentiment |
| 7 | `NB07_black_litterman` | Black-Litterman allocation with ML-enhanced views |
| 8 | `NB08_mean_cvar` | Mean-CVaR portfolio optimization |
| 9 | `NB09_backtest_engine` | Walk-forward backtesting with transaction costs |
| 10 | `NB10_stress_testing` | EVT tail risk, Monte Carlo simulation, stress tests |
| 11 | `NB11_report_dashboard` | Excel dashboard + PDF report generation |
| 12 | `NB12_presentation` | PowerPoint deck generation |

## Tech Universe (20 Stocks)

**Semiconductors:** NVDA, AMD, TSM, AVGO, QCOM, MU  
**Big Tech:** AAPL, MSFT, GOOG, META, NFLX  
**Enterprise Software:** CRM, ADBE, NOW  
**Cybersecurity:** PANW, CRWD  
**Analytics:** DDOG, PLTR  
**Infrastructure:** ANET, XYZ

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install -e .  # Editable install for src/ imports

# 3. Set up FRED API key
cp .env.template .env
# Edit .env with your key from https://fred.stlouisfed.org/docs/api/api_key.html

# 4. Run notebooks in order
cd notebooks/
jupyter lab
```

## Key Design Decisions

- **No look-ahead bias:** All HMM states use filtered (not smoothed) probabilities
- **Expanding window:** PCA, GARCH, HMM, and ML models use expanding-window estimation
- **Anti-leakage:** Scaler inside sklearn pipelines; sentiment features lagged t-1
- **Walk-forward only:** No k-fold CV for time series (walk-forward expanding window)
- **Reproducibility:** `random_state=42` everywhere; raw data checksums saved

## Project Structure

```
factor-timing-engine/
├── data/
│   ├── raw/                   # Downloaded data (not committed)
│   ├── interim/               # Intermediate files, caches
│   │   ├── hmm_cache/
│   │   └── garch_cache/
│   └── processed/             # Clean parquet files
├── notebooks/
│   └── NB01–NB12              # Sequential analysis notebooks
├── src/
│   ├── config.py              # Centralised configuration
│   ├── data_loader.py         # Data acquisition
│   ├── validation.py          # Schema validation & QC
│   ├── visualization.py       # Plot utilities
│   ├── garch_utils.py         # GARCH family models
│   ├── regime_model.py        # HMM + expanding-window PCA
│   ├── ml_pipeline.py         # Walk-forward ML engine
│   ├── portfolio_optimization.py  # BL, CVaR, HRP, ERC
│   └── feature_engineering.py # Technical indicators
├── outputs/
│   ├── figures/               # 300 DPI PNG charts
│   ├── tables/                # CSV summary tables
│   ├── reports/               # PDF, XLSX, PPTX
│   └── models/                # Serialised models
├── logs/                      # Per-phase log files
├── docs/
│   └── claude.md              # Master blueprint (4000+ lines)
├── requirements.txt
├── setup.py
├── .env.template
└── .gitignore
```

## Blueprint

See `docs/claude.md` for the complete 4000+ line technical specification.
