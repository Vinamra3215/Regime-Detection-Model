
from pathlib import Path
import torch

BASE_DIR    = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

PHASE4_RESULTS_DIR  = PROJECT_DIR / "results" / "phase_4"
PHASE4_CHECKPOINT   = PHASE4_RESULTS_DIR / "checkpoints" / "best_model.pt"
PHASE4_PRICE_SCALER = PHASE4_RESULTS_DIR / "price_scaler.pkl"
PHASE4_SENT_SCALER  = PHASE4_RESULTS_DIR / "sent_scaler.pkl"
PHASE5_RESULTS_DIR  = PROJECT_DIR / "results" / "phase_5"
PHASE5_TEMPERATURE  = PHASE5_RESULTS_DIR / "checkpoints" / "temperature.pt"

PHASE1_LABEL_DIR     = PROJECT_DIR / "results" / "phase_1" / "data" / "labelled"
PHASE3_SENTIMENT_DIR = PROJECT_DIR / "results" / "phase_3" / "sentiment_features"

RESULTS_DIR = PROJECT_DIR / "results" / "phase_7"
PLOTS_DIR   = RESULTS_DIR / "plots"
LOG_DIR     = RESULTS_DIR / "logs"

for d in [RESULTS_DIR, PLOTS_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

NIFTY_50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "BAJFINANCE.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "HCLTECH.NS",
    "SUNPHARMA.NS", "TITAN.NS", "WIPRO.NS", "ULTRACEMCO.NS", "NESTLEIND.NS",
    "BAJAJFINSV.NS", "NTPC.NS", "POWERGRID.NS", "TECHM.NS", "ONGC.NS",
    "TATAMOTORS.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "M&M.NS", "HINDALCO.NS",
    "COALINDIA.NS", "DRREDDY.NS", "DIVISLAB.NS", "CIPLA.NS", "APOLLOHOSP.NS",
    "ADANIPORTS.NS", "ADANIENT.NS", "GRASIM.NS", "HDFCLIFE.NS", "SBILIFE.NS",
    "SHRIRAMFIN.NS", "BPCL.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "INDUSINDBK.NS",
    "BRITANNIA.NS", "ITC.NS", "BAJAJ-AUTO.NS", "BEL.NS", "TRENT.NS",
]

TICKER_TO_IDX = {t: i for i, t in enumerate(NIFTY_50_TICKERS)}
NUM_STOCKS    = len(NIFTY_50_TICKERS)
STOCK_EMBED_DIM = 16

WINDOW_SIZE = 60

PRICE_FEATURE_COLUMNS = [
    "log_return_1d", "log_return_5d",
    "rolling_vol_10d", "rolling_vol_20d",
    "atr_pct", "rsi_14", "macd_histogram",
    "bb_width", "bb_pband",
    "adx", "adx_pos", "adx_neg",
    "volume_ratio", "log_volume_change",
    "ma_dist_20", "ma_dist_50",
    "linreg_slope_10", "linreg_slope_20",
]

SENTIMENT_FEATURE_COLUMNS = [
    "vix_normalized", "vix_change_1d", "vix_change_5d",
    "vix_percentile", "market_return_5d", "market_return_20d",
    "market_breadth",
]

REGIME_TO_IDX = {"Bear": 0, "Sideways": 1, "Bull": 2}
IDX_TO_REGIME = {v: k for k, v in REGIME_TO_IDX.items()}

HIGH_CONF_THRESHOLD = 0.70
MED_CONF_THRESHOLD  = 0.55
UNCERTAINTY_THRESHOLD = 0.15

STRONG_SIGNAL_SIZE = 1.0
WEAK_SIGNAL_SIZE   = 0.5

STOP_LOSS_PCT      = 0.05
TRAILING_STOP_PCT  = 0.03
MAX_PORTFOLIO_RISK = 0.20
MAX_SECTOR_EXPOSURE = 0.40

MIN_HOLDING_DAYS = 3

EVAL_START = "2024-01-01"
EVAL_END   = "2024-12-31"
RISK_FREE_RATE = 0.07
TRADING_DAYS_PER_YEAR = 252
INITIAL_CAPITAL = 1_000_000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

