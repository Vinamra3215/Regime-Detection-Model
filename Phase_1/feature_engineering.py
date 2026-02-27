import numpy as np
import pandas as pd
import logging
import ta

from config import (
    RETURN_WINDOW_SHORT, RETURN_WINDOW_LONG,
    VOL_WINDOW_SHORT, VOL_WINDOW_LONG,
    ATR_WINDOW, RSI_WINDOW,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_WINDOW, VOLUME_RATIO_WINDOW,
    MIN_DATA_POINTS
)

log = logging.getLogger(__name__)


def compute_features(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame | None:
    df = df.copy()

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        log.warning(f"[{ticker}] Missing columns: {missing}")
        return None

    df["log_return_1d"] = np.log(df["Close"] / df["Close"].shift(1))
    df["log_return_5d"] = np.log(df["Close"] / df["Close"].shift(5))

    df["rolling_vol_10d"] = df["log_return_1d"].rolling(VOL_WINDOW_SHORT).std()
    df["rolling_vol_20d"] = df["log_return_1d"].rolling(VOL_WINDOW_LONG).std()

    atr_indicator = ta.volatility.AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"], window=ATR_WINDOW
    )
    df["atr_14"] = atr_indicator.average_true_range()
    df["atr_pct"] = df["atr_14"] / df["Close"]

    rsi_indicator = ta.momentum.RSIIndicator(close=df["Close"], window=RSI_WINDOW)
    df["rsi_14"] = rsi_indicator.rsi()

    macd_indicator = ta.trend.MACD(
        close=df["Close"],
        window_fast=MACD_FAST,
        window_slow=MACD_SLOW,
        window_sign=MACD_SIGNAL
    )
    df["macd_histogram"] = macd_indicator.macd_diff()

    bb_indicator = ta.volatility.BollingerBands(close=df["Close"], window=BB_WINDOW)
    df["bb_width"]  = bb_indicator.bollinger_wband()
    df["bb_pband"]  = bb_indicator.bollinger_pband()

    adx_indicator = ta.trend.ADXIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], window=14
    )
    df["adx"]     = adx_indicator.adx()
    df["adx_pos"] = adx_indicator.adx_pos()
    df["adx_neg"] = adx_indicator.adx_neg()

    df["volume_ratio"]       = df["Volume"] / df["Volume"].rolling(VOLUME_RATIO_WINDOW).mean()
    df["log_volume_change"]  = np.log(df["Volume"] / df["Volume"].shift(1) + 1e-9)

    df["ma_20"]      = df["Close"].rolling(20).mean()
    df["ma_50"]      = df["Close"].rolling(50).mean()
    df["ma_dist_20"] = (df["Close"] - df["ma_20"]) / df["ma_20"]
    df["ma_dist_50"] = (df["Close"] - df["ma_50"]) / df["ma_50"]

    df["linreg_slope_10"] = _rolling_linreg_slope(df["Close"], window=10)
    df["linreg_slope_20"] = _rolling_linreg_slope(df["Close"], window=20)

    df.drop(columns=["ma_20", "ma_50", "atr_14"], inplace=True, errors="ignore")

    feature_cols = [c for c in df.columns if c not in {"Open", "High", "Low", "Close", "Volume"}]
    df_clean = df.dropna(subset=feature_cols)

    if len(df_clean) < MIN_DATA_POINTS:
        log.warning(f"[{ticker}] Only {len(df_clean)} rows after cleaning — skipping (min={MIN_DATA_POINTS})")
        return None

    log.info(f"[{ticker}] {len(df_clean)} rows | {len(feature_cols)} features computed")
    return df_clean


def _rolling_linreg_slope(series: pd.Series, window: int) -> pd.Series:
    slopes = np.full(len(series), np.nan)
    vals   = series.values
    x      = np.arange(window)
    x_mean = x.mean()
    x_var  = ((x - x_mean) ** 2).sum()

    for i in range(window - 1, len(vals)):
        y = vals[i - window + 1 : i + 1]
        if np.any(np.isnan(y)):
            continue
        y_mean    = y.mean()
        slope     = ((x - x_mean) * (y - y_mean)).sum() / x_var
        slopes[i] = slope / (y_mean + 1e-9)

    return pd.Series(slopes, index=series.index)


def get_hmm_features(df: pd.DataFrame) -> np.ndarray:
    hmm_cols = [
        "log_return_1d",
        "log_return_5d",
        "rolling_vol_10d",
        "rolling_vol_20d",
        "atr_pct",
        "macd_histogram",
        "rsi_14",
        "adx",
        "ma_dist_20",
        "bb_width",
    ]
    available = [c for c in hmm_cols if c in df.columns]
    arr = df[available].values
    return arr, available


if __name__ == "__main__":
    import yfinance as yf
    raw = yf.download("RELIANCE.NS", start="2019-01-01", end="2025-01-01",
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]
    feat_df = compute_features(raw, ticker="RELIANCE.NS")
    print(feat_df.tail())
    print(f"\nFeatures: {[c for c in feat_df.columns if c not in {'Open','High','Low','Close','Volume'}]}")
