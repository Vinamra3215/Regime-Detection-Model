
import os
from pathlib import Path
import torch

BASE_DIR          = Path(__file__).resolve().parent
PROJECT_DIR       = BASE_DIR.parent
PHASE1_DIR        = PROJECT_DIR / "Phase_1"
PHASE1_LABEL_DIR  = PROJECT_DIR / "results" / "phase_1" / "data" / "labelled"
PHASE2_RESULTS_DIR = PROJECT_DIR / "results" / "phase_2_results"
PHASE3_RESULTS_DIR = PROJECT_DIR / "results" / "phase_3"
PHASE3_SENTIMENT_DIR = PHASE3_RESULTS_DIR / "sentiment_features"
PHASE3_MARKET_DIR    = PHASE3_RESULTS_DIR / "market_data"

RESULTS_DIR    = PROJECT_DIR / "results" / "phase_4"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
LOG_DIR        = RESULTS_DIR / "logs"
PLOTS_DIR      = RESULTS_DIR / "plots"

for d in [RESULTS_DIR, CHECKPOINT_DIR, LOG_DIR, PLOTS_DIR]:
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

TRAIN_END   = "2022-12-31"
VAL_END     = "2023-12-31"

WINDOW_SIZE         = 60
FORECAST_HORIZON    = 1
TRANSITION_WINDOW   = 5

PRICE_FEATURE_COLUMNS = [
    "log_return_1d", "log_return_5d",
    "rolling_vol_10d", "rolling_vol_20d",
    "atr_pct",
    "rsi_14",
    "macd_histogram",
    "bb_width", "bb_pband",
    "adx", "adx_pos", "adx_neg",
    "volume_ratio", "log_volume_change",
    "ma_dist_20", "ma_dist_50",
    "linreg_slope_10", "linreg_slope_20",
]

SENTIMENT_FEATURE_COLUMNS = [
    "vix_normalized",
    "vix_change_1d",
    "vix_change_5d",
    "vix_percentile",
    "market_return_5d",
    "market_return_20d",
    "market_breadth",
]

NEWS_FEATURE_COLUMNS = [
    "news_sentiment_mean",
    "news_positive_ratio",
    "news_negative_ratio",
    "sector_sentiment",
    "composite_sentiment",
]

INCLUDE_NEWS_FEATURES = True

REGIME_TO_IDX = {"Bear": 0, "Sideways": 1, "Bull": 2}
IDX_TO_REGIME = {v: k for k, v in REGIME_TO_IDX.items()}
NUM_CLASSES   = 3

STOCK_EMBED_DIM = 16
NUM_STOCKS      = len(NIFTY_50_TICKERS)
TICKER_TO_IDX   = {t: i for i, t in enumerate(NIFTY_50_TICKERS)}

D_MODEL       = 64
N_HEAD        = 4
NUM_LAYERS    = 2
FF_DIM        = 128
DROPOUT       = 0.30

SENT_D_MODEL  = 32
SENT_N_HEAD   = 2
SENT_NUM_LAYERS = 1
SENT_FF_DIM   = 64

ENABLE_FUSION     = True
FUSION_N_HEADS    = 4
FUSION_DROPOUT    = 0.1

BATCH_SIZE    = 64
LEARNING_RATE = 5e-5
WEIGHT_DECAY  = 1e-3
EPOCHS        = 120
PATIENCE      = 25
WARMUP_EPOCHS = 8

USE_FOCAL_LOSS           = True
FOCAL_GAMMA              = 2.0
LABEL_SMOOTHING          = 0.1
CLASS_WEIGHT_MODE        = "inverse_freq"
TRANSITION_LOSS_WEIGHT   = 1.0
TRANSITION_POS_WEIGHT    = 6.0

PHASE2_CHECKPOINT = PHASE2_RESULTS_DIR / "checkpoints" / "best_model.pt"
USE_PRETRAINED_PRICE_ENCODER = True
FREEZE_PRICE_ENCODER_EPOCHS  = 5

TENSORBOARD_LOG_DIR = LOG_DIR / "tensorboard"
TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MIN_TEST_ACCURACY       = 0.65
MIN_HIGH_CONF_ACCURACY  = 0.80
MIN_PER_CLASS_F1        = 0.55
MIN_TRANSITION_RECALL   = 0.60
HIGH_CONF_THRESHOLD     = 0.70
PHASE2_TEST_ACCURACY    = 0.621

