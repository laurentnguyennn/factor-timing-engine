"""
Centralised configuration for the Factor Timing Engine.
All paths, constants, tickers, and hyperparameters defined here.
Import this in every notebook and module.
"""
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

# Create all directories
for d in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR,
          REPORTS_DIR, MODELS_DIR, LOGS_DIR,
          INTERIM_DIR / 'hmm_cache', INTERIM_DIR / 'garch_cache']:
    d.mkdir(parents=True, exist_ok=True)

# === File Paths ===
FACTOR_RETURNS_FILE = PROCESSED_DIR / 'factor_returns.parquet'
FACTOR_RETURNS_FULL_FILE = PROCESSED_DIR / 'factor_returns_full.parquet'
MACRO_INDICATORS_FILE = PROCESSED_DIR / 'macro_indicators.parquet'
MASTER_DATA_FILE = PROCESSED_DIR / 'master_data.parquet'
SP500_DAILY_FILE = PROCESSED_DIR / 'sp500_daily_prices.parquet'
SP500_MONTHLY_FILE = PROCESSED_DIR / 'sp500_monthly_returns.parquet'
LOWVOL_FACTOR_FILE = PROCESSED_DIR / 'lowvol_factor_returns.parquet'
MACRO_COMPOSITE_FILE = PROCESSED_DIR / 'macro_composite_index.parquet'
REGIME_PROBS_FILE = PROCESSED_DIR / 'regime_probabilities.parquet'
REGIME_LABELS_FILE = PROCESSED_DIR / 'regime_labels.parquet'
MACRO_REGIMES_FILE = PROCESSED_DIR / 'macro_regimes.parquet'
COND_VOL_FILE = PROCESSED_DIR / 'garch_conditional_vol.parquet'
DCC_CORR_FILE = PROCESSED_DIR / 'dcc_conditional_corr.parquet'
COND_COV_FILE = PROCESSED_DIR / 'conditional_covariance.parquet'
BL_WEIGHTS_FILE = PROCESSED_DIR / 'bl_weights_timeseries.parquet'
CVAR_WEIGHTS_FILE = PROCESSED_DIR / 'cvar_weights_timeseries.parquet'
BACKTEST_RETURNS_FILE = PROCESSED_DIR / 'backtest_returns.parquet'
BACKTEST_NAV_FILE = PROCESSED_DIR / 'backtest_nav.parquet'
SENTIMENT_FILE = PROCESSED_DIR / 'sentiment_features.parquet'
VOL_FORECAST_FILE = PROCESSED_DIR / 'vol_forecast_predictions.parquet'
RETURN_SCENARIOS_FILE = PROCESSED_DIR / 'return_scenarios.parquet'
PORTFOLIO_WEIGHTS_FILE = PROCESSED_DIR / 'portfolio_weights_timeseries.parquet'

# Tables
GARCH_PARAMS_FILE = TABLES_DIR / 'garch_parameters.csv'
GARCH_BEST_FILE = TABLES_DIR / 'garch_best_model_per_ticker.csv'
VAR_CVAR_FILE = TABLES_DIR / 'var_cvar_table.csv'
EVT_PARAMS_FILE = TABLES_DIR / 'evt_parameters.csv'
BACKTEST_PERF_FILE = TABLES_DIR / 'backtest_performance.csv'
STRESS_TEST_FILE = TABLES_DIR / 'stress_test_results.csv'
MODEL_COMPARISON_FILE = TABLES_DIR / 'model_comparison.csv'

# === Reproducibility ===
RANDOM_STATE = 42
TRAIN_RATIO = 0.70
RETRAIN_FREQ_DAYS = 63  # ~quarterly

# === Ticker Universe ===
TICKERS = [
    'NVDA', 'AMD', 'TSM', 'AVGO', 'QCOM', 'MU',     # Semiconductors
    'AAPL', 'MSFT', 'GOOG', 'META', 'NFLX',          # Big Tech
    'CRM', 'ADBE', 'NOW',                             # Enterprise Software
    'PANW', 'CRWD',                                   # Cybersecurity
    'DDOG', 'PLTR',                                   # Analytics/Observability
    'ANET', 'XYZ',                                    # Networking/Fintech
]

BENCHMARKS = ['^GSPC', '^NDX', 'XLK', 'SOXX', '^VIX', '^VVIX', 'DX-Y.NYB']

SHORT_HISTORY_TICKERS = {
    'CRWD': '2019-06-12',
    'DDOG': '2019-09-19',
    'PLTR': '2020-09-30',
}

SECTOR_MAP = {
    'Semiconductors': ['NVDA', 'AMD', 'TSM', 'AVGO', 'QCOM', 'MU'],
    'Big Tech': ['AAPL', 'MSFT', 'GOOG', 'META', 'NFLX'],
    'Enterprise Software': ['CRM', 'ADBE', 'NOW'],
    'Cybersecurity': ['PANW', 'CRWD'],
    'Analytics': ['DDOG', 'PLTR'],
    'Infrastructure': ['ANET', 'XYZ'],
}

# === Key Events ===
KEY_EVENTS = {
    'COVID Crash': ('2020-02-19', '2020-03-23'),
    'Rate Shock 2022': ('2022-01-03', '2022-10-13'),
    'SVB Contagion': ('2023-03-08', '2023-03-15'),
    'DeepSeek/Tariff': ('2025-01-27', '2025-02-10'),
    'China Tech Crackdown': ('2021-07-01', '2021-08-20'),
}

# === Factor Timing Dates ===
FACTOR_START = '2004-01-01'
FACTOR_END = '2026-03-31'
FACTOR_LIVE_START = '2010-01-01'  # After 60-month warm-up

# Tech Portfolio Dates
TECH_START = '2016-01-01'
TECH_END = '2026-03-31'

# === FRED Series ===
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

# === GARCH ===
GARCH_MODELS = ['GARCH', 'GJR-GARCH', 'EGARCH', 'FIGARCH']
GARCH_DISTRIBUTIONS = ['normal', 'StudentsT', 'skewt', 'ged']
MIN_OBS_FIGARCH = 1500

# === HMM ===
HMM_N_STATES = 3
HMM_N_RESTARTS = 25
HMM_MIN_WINDOW = 24  # Months for expanding-window warm-up

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

# === Portfolio Constraints ===
MAX_WEIGHT_SINGLE = 0.10      # Max 10% per stock (tech portfolio)
MAX_WEIGHT_SECTOR = 0.30      # Max 30% per sector
MAX_WEIGHT_FACTOR = 0.40      # Max 40% per factor (factor timing)
TURNOVER_LIMIT = 0.20         # Max 20% total turnover per rebalance
TC_ONE_WAY = 0.0010           # 10 bps one-way transaction cost (tech)
TC_FACTOR = 0.0025            # 25 bps one-way (factor portfolio)
RISK_AVERSION = 2.5           # BL delta
BL_TAU = 0.05                 # BL scaling parameter

# === Colours (consistent throughout) ===
COLORS = {
    'expansion': '#2ecc71',
    'slowdown': '#f39c12',
    'crisis': '#e74c3c',
    'hml': '#3498db',
    'umd': '#e74c3c',
    'rmw': '#2ecc71',
    'lowvol': '#9b59b6',
}
