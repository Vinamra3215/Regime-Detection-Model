import yfinance as yf
import pandas as pd
import time
import logging
from pathlib import Path
from config import (
    NIFTY_50_TICKERS, INDEX_TICKER,
    START_DATE, END_DATE, DATA_RAW_DIR
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def download_ticker(ticker: str, start: str, end: str, retries: int = 3) -> pd.DataFrame | None:
    for attempt in range(retries):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if df.empty:
                log.warning(f"  [{ticker}] Empty data returned.")
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            df.index = pd.to_datetime(df.index)
            df.index.name = "Date"
            log.info(f"  [{ticker}] Downloaded {len(df)} rows ({df.index[0].date()} → {df.index[-1].date()})")
            return df
        except Exception as e:
            log.warning(f"  [{ticker}] Attempt {attempt+1} failed: {e}")
            time.sleep(2)
    log.error(f"  [{ticker}] All retries failed. Skipping.")
    return None


def download_all(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    results = {}
    failed  = []

    log.info(f"Starting download of {len(tickers)} tickers + index ({start} → {end})")

    log.info(f"\n--- Nifty 50 Index ---")
    idx_df = download_ticker(INDEX_TICKER, start, end)
    if idx_df is not None:
        save_path = DATA_RAW_DIR / f"{INDEX_TICKER.replace('^', 'INDEX_')}.csv"
        idx_df.to_csv(save_path)
        results[INDEX_TICKER] = idx_df
    else:
        failed.append(INDEX_TICKER)

    log.info(f"\n--- Nifty 50 Stocks ---")
    for i, ticker in enumerate(tickers, 1):
        log.info(f"[{i:02d}/{len(tickers)}] {ticker}")
        df = download_ticker(ticker, start, end)
        if df is not None:
            save_path = DATA_RAW_DIR / f"{ticker}.csv"
            df.to_csv(save_path)
            results[ticker] = df
        else:
            failed.append(ticker)
        time.sleep(0.3)

    log.info(f"\n✅ Downloaded: {len(results)} | ❌ Failed: {len(failed)}")
    if failed:
        log.warning(f"   Failed tickers: {failed}")

    return results


def load_raw_data(tickers: list[str]) -> dict[str, pd.DataFrame]:
    results = {}
    idx_path = DATA_RAW_DIR / f"{INDEX_TICKER.replace('^', 'INDEX_')}.csv"
    if idx_path.exists():
        results[INDEX_TICKER] = pd.read_csv(idx_path, index_col="Date", parse_dates=True)

    for ticker in tickers:
        path = DATA_RAW_DIR / f"{ticker}.csv"
        if path.exists():
            results[ticker] = pd.read_csv(path, index_col="Date", parse_dates=True)
    log.info(f"Loaded {len(results)} tickers from disk.")
    return results


def get_data(tickers: list[str] = NIFTY_50_TICKERS,
             start: str = START_DATE,
             end: str = END_DATE,
             force_download: bool = False) -> dict[str, pd.DataFrame]:
    cached = [t for t in tickers if (DATA_RAW_DIR / f"{t}.csv").exists()]
    if not force_download and len(cached) == len(tickers):
        log.info("All raw CSVs found on disk. Loading from cache...")
        return load_raw_data(tickers)

    log.info("Downloading fresh data...")
    return download_all(tickers, start, end)


if __name__ == "__main__":
    data = get_data(force_download=False)
    print(f"\nLoaded {len(data)} tickers.")
    sample = list(data.keys())[0]
    print(f"Sample — {sample}:\n{data[sample].tail()}")
