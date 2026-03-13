
from pathlib import Path
import torch

BASE_DIR    = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

PHASE4_DIR         = PROJECT_DIR / "Phase_4"
PHASE4_RESULTS     = PROJECT_DIR / "results" / "phase_4"
PHASE4_CHECKPOINT  = PHASE4_RESULTS / "checkpoints" / "best_model.pt"
PHASE4_PRICE_SCALER = PHASE4_RESULTS / "price_scaler.pkl"
PHASE4_SENT_SCALER  = PHASE4_RESULTS / "sent_scaler.pkl"
PHASE5_TEMP        = PROJECT_DIR / "results" / "phase_5" / "checkpoints" / "temperature.pt"
PHASE1_LABEL_DIR   = PROJECT_DIR / "results" / "phase_1" / "data" / "labelled"

RESULTS_DIR = PROJECT_DIR / "results" / "phase_9"
PLOTS_DIR   = RESULTS_DIR / "plots"
CKPT_DIR    = RESULTS_DIR / "checkpoints"
LOG_DIR     = RESULTS_DIR / "logs"

for d in [RESULTS_DIR, PLOTS_DIR, CKPT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

PRICE_FEATURE_COLUMNS = [
    "log_return_1d", "log_return_5d", "rolling_vol_10d", "rolling_vol_20d",
    "atr_pct", "rsi_14", "macd_histogram", "bb_width", "bb_pband",
    "adx", "adx_pos", "adx_neg", "volume_ratio", "log_volume_change",
    "ma_dist_20", "ma_dist_50", "linreg_slope_10", "linreg_slope_20",
]

WINDOW_SIZE = 60

OBS_DIM = 10

ACTION_DIM = 1

PPO_LEARNING_RATE   = 3e-4
PPO_N_STEPS         = 2048
PPO_BATCH_SIZE      = 64
PPO_N_EPOCHS        = 10
PPO_GAMMA           = 0.99
PPO_GAE_LAMBDA      = 0.95
PPO_CLIP_RANGE      = 0.2
PPO_ENT_COEF        = 0.01
PPO_VF_COEF         = 0.5
PPO_MAX_GRAD_NORM   = 0.5
TOTAL_TIMESTEPS     = 500_000

REWARD_SCALE        = 100.0
RISK_PENALTY_COEF   = 0.3
DRAWDOWN_PENALTY    = 1.0
DRAWDOWN_THRESHOLD  = 0.03
TRADE_COST_PENALTY  = 0.001

TRAIN_START = "2019-07-01"
TRAIN_END   = "2023-12-31"
EVAL_START  = "2024-01-01"
EVAL_END    = "2024-12-31"

INITIAL_CAPITAL = 10_00_000
RISK_FREE_RATE  = 0.07
TRADING_DAYS    = 252

ZERODHA_STT_PCT      = 0.001
ZERODHA_EXCHANGE_PCT = 0.0000345
ZERODHA_GST_PCT      = 0.18
ZERODHA_SEBI_PCT     = 0.000001
ZERODHA_STAMP_PCT    = 0.00015
SLIPPAGE_PCT         = 0.0005

