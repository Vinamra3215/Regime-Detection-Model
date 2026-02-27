import os
from pathlib import Path

BASE_DIR        = Path(__file__).resolve().parent
DATA_RAW_DIR    = BASE_DIR / "outputs" / "data" / "raw"
DATA_LABEL_DIR  = BASE_DIR / "outputs" / "data" / "labelled"
PLOTS_DIR       = BASE_DIR / "outputs" / "plots"

for d in [DATA_RAW_DIR, DATA_LABEL_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

START_DATE = "2019-01-01"
END_DATE   = "2025-01-01"

INDEX_TICKER = "^NSEI"

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

RETURN_WINDOW_SHORT   = 1
RETURN_WINDOW_LONG    = 5
VOL_WINDOW_SHORT      = 10
VOL_WINDOW_LONG       = 20
ATR_WINDOW            = 14
RSI_WINDOW            = 14
MACD_FAST             = 12
MACD_SLOW             = 26
MACD_SIGNAL           = 9
BB_WINDOW             = 20
VOLUME_RATIO_WINDOW   = 5

HMM_N_STATES        = 3
HMM_N_ITER          = 200
HMM_RANDOM_STATE    = 42
HMM_COV_TYPE        = "full"
MIN_DATA_POINTS     = 200

REGIME_LABELS  = {0: "Bear", 1: "Sideways", 2: "Bull"}
REGIME_COLORS  = {"Bull": "#00C853", "Sideways": "#FFD600", "Bear": "#DD2C00"}
REGIME_ORDER   = ["Bull", "Sideways", "Bear"]
