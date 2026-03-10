
import os
from pathlib import Path

BASE_DIR          = Path(__file__).resolve().parent
PROJECT_DIR       = BASE_DIR.parent
PHASE1_DIR        = PROJECT_DIR / "Phase_1"
PHASE1_LABEL_DIR  = PROJECT_DIR / "results" / "phase_1" / "data" / "labelled"
PHASE1_MODEL_DIR  = PROJECT_DIR / "results" / "phase_1" / "data" / "models"

RESULTS_DIR       = PROJECT_DIR / "results" / "phase_2_results"
OUTPUT_DIR        = RESULTS_DIR
CHECKPOINT_DIR    = RESULTS_DIR / "checkpoints"
LOG_DIR           = RESULTS_DIR / "logs"
PLOTS_DIR         = RESULTS_DIR / "plots"

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

FEATURE_COLUMNS = [
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

REGIME_TO_IDX = {"Bear": 0, "Sideways": 1, "Bull": 2}
IDX_TO_REGIME = {v: k for k, v in REGIME_TO_IDX.items()}
NUM_CLASSES   = 3

D_MODEL       = 128
N_HEAD        = 4
NUM_LAYERS    = 4
FF_DIM        = 256
DROPOUT       = 0.35

BATCH_SIZE    = 64
LEARNING_RATE = 3e-5
WEIGHT_DECAY  = 1e-4
EPOCHS        = 100
PATIENCE      = 20
WARMUP_EPOCHS = 5

USE_FOCAL_LOSS   = True
FOCAL_GAMMA      = 2.0
LABEL_SMOOTHING  = 0.1
CLASS_WEIGHT_MODE = "inverse_freq"

TRANSITION_LOSS_WEIGHT = 1.0
TRANSITION_POS_WEIGHT  = 6.0

TENSORBOARD_LOG_DIR = LOG_DIR / "tensorboard"
TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MIN_TEST_ACCURACY       = 0.75
MIN_HIGH_CONF_ACCURACY  = 0.80
MIN_PER_CLASS_F1        = 0.65
MIN_TRANSITION_RECALL   = 0.60
HIGH_CONF_THRESHOLD     = 0.70

