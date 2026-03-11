
import numpy as np
import pandas as pd
import logging
from pathlib import Path

from config import (
    NIFTY_50_TICKERS, TICKER_TO_SECTOR,
    PHASE1_LABEL_DIR, SCORED_NEWS_DIR, MARKET_DATA_DIR,
    SENTIMENT_FEATURES_DIR, SENTIMENT_ROLLING_WINDOWS,
    SENTIMENT_FEATURE_COLUMNS, TRAIN_END, VAL_END,
)

log = logging.getLogger(__name__)


def load_scored_news() -> pd.DataFrame:
    path = SCORED_NEWS_DIR / "scored_news.csv"
    if not path.exists():
        log.warning(f"No scored news at {path}. Running with empty news.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    if "published" in df.columns:
        df["date"] = pd.to_datetime(df["published"], errors="coerce").dt.date
    else:
        df["date"] = pd.to_datetime(df["collected_at"], errors="coerce").dt.date

    df = df.dropna(subset=["date"])
    df["date"] = pd.to_datetime(df["date"])
    log.info(f"Loaded {len(df)} scored articles spanning "
             f"{df['date'].min()} to {df['date'].max()}")
    return df


def aggregate_daily_sentiment(scored_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if scored_df.empty:
        return pd.DataFrame()

    mask = (scored_df["ticker"] == ticker) | (scored_df["ticker"] == "MARKET_GENERAL")
    ticker_news = scored_df[mask].copy()

    if ticker_news.empty:
        return pd.DataFrame()

    daily = ticker_news.groupby("date").agg(
        news_sentiment_mean=("compound", "mean"),
        news_sentiment_std=("compound", "std"),
        news_positive_ratio=("positive", "mean"),
        news_negative_ratio=("negative", "mean"),
        news_count=("compound", "count"),
    ).fillna(0.0)

    daily.index = pd.to_datetime(daily.index)
    return daily


def compute_sector_sentiment(scored_df: pd.DataFrame) -> pd.DataFrame:
    if scored_df.empty:
        return pd.DataFrame()

    if "sector" not in scored_df.columns:
        scored_df["sector"] = scored_df["ticker"].map(
            lambda t: TICKER_TO_SECTOR.get(t, "General")
        )

    sector_daily = scored_df.groupby(["date", "sector"]).agg(
        sector_sentiment=("compound", "mean"),
    ).reset_index()

    pivot = sector_daily.pivot_table(
        index="date", columns="sector", values="sector_sentiment", fill_value=0.0
    )
    pivot.index = pd.to_datetime(pivot.index)

    return pivot


def build_sentiment_features_for_ticker(
    ticker: str,
    scored_df: pd.DataFrame,
    market_df: pd.DataFrame,
    sector_pivot: pd.DataFrame,
) -> pd.DataFrame:
    label_path = PHASE1_LABEL_DIR / f"{ticker}_labelled.csv"
    if not label_path.exists():
        log.warning(f"No Phase 1 data for {ticker}. Skipping.")
        return pd.DataFrame()

    base_df = pd.read_csv(label_path, index_col="Date", parse_dates=True)
    base_df.index = pd.to_datetime(base_df.index)
    dates = base_df.index

    features = pd.DataFrame(index=dates)

    daily_news = aggregate_daily_sentiment(scored_df, ticker)
    if not daily_news.empty:
        features = features.join(daily_news, how="left")

    for col in ["news_sentiment_mean", "news_positive_ratio", "news_negative_ratio"]:
        if col not in features.columns:
            features[col] = 0.0
    if "news_sentiment_std" not in features.columns:
        features["news_sentiment_std"] = 0.0
    if "news_count" not in features.columns:
        features["news_count"] = 0

    features = features.fillna(0.0)

    if not market_df.empty:
        market_cols = [c for c in market_df.columns if c in
                       ["vix_normalized", "vix_change_1d", "vix_change_5d",
                        "vix_percentile", "market_return_5d", "market_return_20d",
                        "market_breadth"]]
        if market_cols:
            features = features.join(market_df[market_cols], how="left")

    sector = TICKER_TO_SECTOR.get(ticker, "General")
    if not sector_pivot.empty and sector in sector_pivot.columns:
        sector_sent = sector_pivot[[sector]].rename(columns={sector: "sector_sentiment"})
        features = features.join(sector_sent, how="left")

    if "sector_sentiment" not in features.columns:
        features["sector_sentiment"] = 0.0

    features["composite_sentiment"] = (
        0.5 * features.get("news_sentiment_mean", 0.0) +
        0.2 * features.get("sector_sentiment", 0.0) +
        0.15 * features.get("market_return_5d", 0.0).clip(-0.1, 0.1) * 10 +
        0.15 * (1.0 - features.get("vix_percentile", 0.5).clip(0, 1)) * 2 - 0.15
    )

    for window in SENTIMENT_ROLLING_WINDOWS:
        col = f"sentiment_momentum_{window}d"
        if "composite_sentiment" in features.columns:
            features[col] = features["composite_sentiment"].diff(window)

    features = features.fillna(0.0)

    available_cols = [c for c in SENTIMENT_FEATURE_COLUMNS if c in features.columns]
    features = features[available_cols]

    return features


def build_all_sentiment_features() -> dict[str, pd.DataFrame]:
    log.info("\n" + "=" * 60)
    log.info("PHASE 3 — Sentiment Feature Engineering")
    log.info("=" * 60)

    scored_df = load_scored_news()

    market_path = MARKET_DATA_DIR / "market_features.csv"
    if market_path.exists():
        market_df = pd.read_csv(market_path, index_col=0, parse_dates=True)
        market_df.index = pd.to_datetime(market_df.index)
        log.info(f"Loaded market features: {market_df.shape}")
    else:
        log.warning("No market features found. Run market_features.py first.")
        market_df = pd.DataFrame()

    sector_pivot = compute_sector_sentiment(scored_df)

    all_features = {}
    success = 0
    for ticker in NIFTY_50_TICKERS:
        features = build_sentiment_features_for_ticker(
            ticker, scored_df, market_df, sector_pivot
        )
        if not features.empty:
            out_path = SENTIMENT_FEATURES_DIR / f"{ticker}_sentiment.csv"
            features.to_csv(out_path)
            all_features[ticker] = features
            success += 1

    log.info(f"\nBuilt sentiment features for {success}/{len(NIFTY_50_TICKERS)} tickers")
    log.info(f"Saved to: {SENTIMENT_FEATURES_DIR}")

    if all_features:
        sample_ticker = list(all_features.keys())[0]
        sample_df = all_features[sample_ticker]
        log.info(f"\nSample ({sample_ticker}): {sample_df.shape}")
        log.info(f"Feature columns: {list(sample_df.columns)}")
        log.info(f"Date range: {sample_df.index.min()} → {sample_df.index.max()}")
        log.info(f"\nFeature statistics (across all tickers):")
        combined = pd.concat(all_features.values())
        log.info(combined.describe().to_string())

    return all_features


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    features = build_all_sentiment_features()
    print(f"\nDone. Features built for {len(features)} tickers.")

