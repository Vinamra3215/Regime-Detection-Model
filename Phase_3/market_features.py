
import numpy as np
import pandas as pd
import yfinance as yf
import logging
from pathlib import Path

from config import (
    NIFTY_50_TICKERS, TICKER_TO_SECTOR,
    PHASE1_LABEL_DIR, MARKET_DATA_DIR,
    VIX_TICKER, NIFTY_INDEX, DATA_START, DATA_END,
)

log = logging.getLogger(__name__)


def download_vix(start: str = DATA_START, end: str = DATA_END) -> pd.DataFrame:
    log.info("Downloading India VIX...")
    try:
        vix = yf.download(VIX_TICKER, start=start, end=end, progress=False)
        if vix.empty:
            log.warning("India VIX download returned empty. Using synthetic VIX.")
            return _synthetic_vix(start, end)
        vix = vix[["Close"]].rename(columns={"Close": "vix_close"})
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        vix.index = pd.to_datetime(vix.index)
        log.info(f"  VIX: {len(vix)} trading days")
        return vix
    except Exception as e:
        log.warning(f"VIX download failed: {e}. Using synthetic VIX.")
        return _synthetic_vix(start, end)


def _synthetic_vix(start: str, end: str) -> pd.DataFrame:
    log.info("  Computing synthetic VIX from Nifty realized volatility...")
    nifty = yf.download(NIFTY_INDEX, start=start, end=end, progress=False)
    if nifty.empty:
        dates = pd.bdate_range(start, end)
        return pd.DataFrame({"vix_close": 15.0}, index=dates)

    close = nifty["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    returns = np.log(close / close.shift(1))
    vol = returns.rolling(20).std() * np.sqrt(252) * 100
    vix = pd.DataFrame({"vix_close": vol}, index=close.index)
    vix["vix_close"] = vix["vix_close"].fillna(15.0)
    return vix


def download_nifty_index(start: str = DATA_START, end: str = DATA_END) -> pd.DataFrame:
    log.info("Downloading Nifty 50 index...")
    try:
        nifty = yf.download(NIFTY_INDEX, start=start, end=end, progress=False)
        if nifty.empty:
            log.warning("Nifty download returned empty.")
            return pd.DataFrame()
        close = nifty[["Close"]].rename(columns={"Close": "nifty_close"})
        if isinstance(close.columns, pd.MultiIndex):
            close.columns = close.columns.get_level_values(0)
        close.index = pd.to_datetime(close.index)
        log.info(f"  Nifty: {len(close)} trading days")
        return close
    except Exception as e:
        log.warning(f"Nifty download failed: {e}")
        return pd.DataFrame()


def compute_market_breadth() -> pd.DataFrame:
    log.info("Computing market breadth from Phase 1 data...")
    daily_returns = {}

    for ticker in NIFTY_50_TICKERS:
        path = PHASE1_LABEL_DIR / f"{ticker}_labelled.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        if "log_return_1d" in df.columns:
            daily_returns[ticker] = df["log_return_1d"]

    if not daily_returns:
        log.warning("No labelled data found for breadth calculation.")
        return pd.DataFrame()

    returns_df = pd.DataFrame(daily_returns)
    breadth = (returns_df > 0).sum(axis=1) / returns_df.count(axis=1)
    breadth_df = pd.DataFrame({"market_breadth": breadth})
    breadth_df.index = pd.to_datetime(breadth_df.index)
    log.info(f"  Breadth: {len(breadth_df)} trading days, "
             f"mean={breadth_df['market_breadth'].mean():.3f}")
    return breadth_df


def compute_market_features() -> pd.DataFrame:
    log.info("\n--- Computing Market-Wide Sentiment Features ---")

    vix_df = download_vix()
    nifty_df = download_nifty_index()
    breadth_df = compute_market_breadth()

    market = vix_df.copy()

    if not nifty_df.empty:
        market = market.join(nifty_df, how="outer")

    if not breadth_df.empty:
        market = market.join(breadth_df, how="outer")

    market = market.sort_index().ffill().bfill()

    if "vix_close" in market.columns:
        vix_rolling_mean = market["vix_close"].rolling(252, min_periods=20).mean()
        vix_rolling_std = market["vix_close"].rolling(252, min_periods=20).std()
        market["vix_normalized"] = ((market["vix_close"] - vix_rolling_mean)
                                     / (vix_rolling_std + 1e-8))
        market["vix_change_1d"] = market["vix_close"].pct_change(1)
        market["vix_change_5d"] = market["vix_close"].pct_change(5)
        market["vix_percentile"] = market["vix_close"].rolling(252, min_periods=20).apply(
            lambda x: (x.iloc[-1] <= x).mean() if len(x) > 0 else 0.5, raw=False
        )

    if "nifty_close" in market.columns:
        market["market_return_5d"] = np.log(
            market["nifty_close"] / market["nifty_close"].shift(5)
        )
        market["market_return_20d"] = np.log(
            market["nifty_close"] / market["nifty_close"].shift(20)
        )

    market = market.fillna(0.0)

    out_path = MARKET_DATA_DIR / "market_features.csv"
    market.to_csv(out_path)
    log.info(f"\nSaved market features: {out_path} ({len(market)} rows)")
    log.info(f"Columns: {list(market.columns)}")

    return market


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    market = compute_market_features()
    print(f"\nMarket features: {market.shape}")
    print(market.tail())

