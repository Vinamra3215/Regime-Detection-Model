
from pathlib import Path
import torch

BASE_DIR           = Path(__file__).resolve().parent
PROJECT_DIR        = BASE_DIR.parent
PHASE4_DIR         = PROJECT_DIR / "Phase_4"
PHASE4_RESULTS_DIR = PROJECT_DIR / "results" / "phase_4"
PHASE4_CHECKPOINT  = PHASE4_RESULTS_DIR / "checkpoints" / "best_model.pt"
PHASE4_PRICE_SCALER  = PHASE4_RESULTS_DIR / "price_scaler.pkl"
PHASE4_SENT_SCALER   = PHASE4_RESULTS_DIR / "sent_scaler.pkl"

RESULTS_DIR    = PROJECT_DIR / "results" / "phase_5"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
LOG_DIR        = RESULTS_DIR / "logs"
PLOTS_DIR      = RESULTS_DIR / "plots"

for d in [RESULTS_DIR, CHECKPOINT_DIR, LOG_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PHASE1_LABEL_DIR     = PROJECT_DIR / "results" / "phase_1" / "data" / "labelled"
PHASE3_SENTIMENT_DIR = PROJECT_DIR / "results" / "phase_3" / "sentiment_features"
PHASE3_MARKET_DIR    = PROJECT_DIR / "results" / "phase_3" / "market_data"

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

TRAIN_END = "2022-12-31"
VAL_END   = "2023-12-31"
WINDOW_SIZE = 60
FORECAST_HORIZON = 1
TRANSITION_WINDOW = 5
BATCH_SIZE = 64

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

NEWS_FEATURE_COLUMNS = [
    "news_sentiment_mean", "news_positive_ratio",
    "news_negative_ratio", "sector_sentiment", "composite_sentiment",
]

REGIME_TO_IDX = {"Bear": 0, "Sideways": 1, "Bull": 2}
IDX_TO_REGIME = {v: k for k, v in REGIME_TO_IDX.items()}
NUM_CLASSES   = 3

TICKER_TO_IDX = {t: i for i, t in enumerate(NIFTY_50_TICKERS)}
NUM_STOCKS    = len(NIFTY_50_TICKERS)
STOCK_EMBED_DIM = 16

MC_SAMPLES        = 50
MC_BATCH_SIZE     = 64

HIGH_CONF_THRESHOLD   = 0.70
UNCERTAINTY_THRESHOLD = 0.15
ABSTAIN_THRESHOLD     = 0.25

CALIBRATION_LR     = 0.01
CALIBRATION_ITERS  = 200

TRANSITION_SMOOTH_WINDOW = 3
TRANSITION_THRESHOLD     = 0.50
MIN_REGIME_DURATION      = 3

MIN_TEST_ACCURACY       = 0.65
MIN_HIGH_CONF_ACCURACY  = 0.80
MIN_PER_CLASS_F1        = 0.55
MIN_TRANSITION_RECALL   = 0.60
MIN_COVERAGE_RATE       = 0.30

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

