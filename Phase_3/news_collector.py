
import feedparser
import pandas as pd
import numpy as np
import logging
import time
import hashlib
import re
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen
from urllib.error import URLError

from config import (
    NIFTY_50_TICKERS, TICKER_TO_COMPANY, TICKER_TO_SECTOR,
    RSS_FEEDS, GOOGLE_NEWS_RSS, RAW_NEWS_DIR,
    NEWS_REQUEST_TIMEOUT, NEWS_DELAY_BETWEEN, MAX_NEWS_PER_TICKER,
)

log = logging.getLogger(__name__)


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def _parse_date(entry) -> datetime | None:
    for field in ["published_parsed", "updated_parsed"]:
        parsed = getattr(entry, field, None)
        if parsed:
            try:
                return datetime(*parsed[:6])
            except Exception:
                continue
    for field in ["published", "updated"]:
        raw = getattr(entry, field, None)
        if raw:
            for fmt in ["%a, %d %b %Y %H:%M:%S %z",
                        "%a, %d %b %Y %H:%M:%S GMT",
                        "%Y-%m-%dT%H:%M:%S%z"]:
                try:
                    return datetime.strptime(raw.strip(), fmt)
                except ValueError:
                    continue
    return None


def _entry_to_dict(entry, source: str) -> dict | None:
    title = _clean_text(getattr(entry, "title", ""))
    summary = _clean_text(getattr(entry, "summary", ""))
    link = getattr(entry, "link", "")
    pub_date = _parse_date(entry)

    if not title:
        return None

    text = f"{title} {summary}".strip()
    doc_id = hashlib.md5(text.encode()).hexdigest()

    return {
        "doc_id": doc_id,
        "source": source,
        "title": title,
        "summary": summary,
        "text": text,
        "link": link,
        "published": pub_date.isoformat() if pub_date else None,
        "collected_at": datetime.now().isoformat(),
    }


def _match_tickers(text: str, ticker_to_company: dict) -> list[str]:
    text_lower = text.lower()
    matched = []
    for ticker, names in ticker_to_company.items():
        for name in names:
            if name.lower() in text_lower:
                matched.append(ticker)
                break
    return matched


def collect_rss_feeds() -> list[dict]:
    all_articles = []
    seen_ids = set()

    for feed_name, feed_url in RSS_FEEDS.items():
        log.info(f"  Fetching RSS: {feed_name}")
        try:
            feed = feedparser.parse(feed_url)
            count = 0
            for entry in feed.entries:
                article = _entry_to_dict(entry, source=feed_name)
                if article and article["doc_id"] not in seen_ids:
                    seen_ids.add(article["doc_id"])
                    all_articles.append(article)
                    count += 1
            log.info(f"    → {count} articles from {feed_name}")
        except Exception as e:
            log.warning(f"    Failed {feed_name}: {e}")

    log.info(f"  Total from RSS feeds: {len(all_articles)}")
    return all_articles


def collect_google_news(tickers: list[str] = None) -> list[dict]:
    if tickers is None:
        tickers = NIFTY_50_TICKERS

    all_articles = []
    seen_ids = set()

    for ticker in tickers:
        names = TICKER_TO_COMPANY.get(ticker, [])
        if not names:
            continue

        query = names[0]
        url = GOOGLE_NEWS_RSS.format(query=quote(query))
        log.info(f"  Google News: {ticker} ({query})")

        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=NEWS_REQUEST_TIMEOUT) as resp:
                raw = resp.read()
            feed = feedparser.parse(raw)

            count = 0
            for entry in feed.entries[:MAX_NEWS_PER_TICKER]:
                article = _entry_to_dict(entry, source="google_news")
                if article and article["doc_id"] not in seen_ids:
                    article["query_ticker"] = ticker
                    seen_ids.add(article["doc_id"])
                    all_articles.append(article)
                    count += 1
            log.info(f"    → {count} articles for {ticker}")
        except (URLError, TimeoutError) as e:
            log.warning(f"    Failed Google News for {ticker}: {e}")
        except Exception as e:
            log.warning(f"    Unexpected error for {ticker}: {e}")

        time.sleep(NEWS_DELAY_BETWEEN)

    log.info(f"  Total from Google News: {len(all_articles)}")
    return all_articles


def match_articles_to_tickers(articles: list[dict]) -> pd.DataFrame:
    rows = []
    for article in articles:
        text = article.get("text", "")
        query_ticker = article.get("query_ticker")

        matched_tickers = _match_tickers(text, TICKER_TO_COMPANY)

        if query_ticker and query_ticker not in matched_tickers:
            matched_tickers.append(query_ticker)

        if not matched_tickers:
            matched_tickers = ["MARKET_GENERAL"]

        for ticker in matched_tickers:
            row = article.copy()
            row["ticker"] = ticker
            row["sector"] = TICKER_TO_SECTOR.get(ticker, "General")
            rows.append(row)

    df = pd.DataFrame(rows)
    log.info(f"Matched articles → {len(df)} ticker-article pairs "
             f"across {df['ticker'].nunique()} tickers")
    return df


def collect_all_news() -> pd.DataFrame:
    log.info("=" * 60)
    log.info("PHASE 3 — News Collection Pipeline")
    log.info("=" * 60)

    log.info("\nStep 1/3: Collecting RSS feeds...")
    rss_articles = collect_rss_feeds()

    log.info("\nStep 2/3: Collecting Google News per ticker...")
    google_articles = collect_google_news()

    all_articles = rss_articles + google_articles
    log.info(f"\nStep 3/3: Matching {len(all_articles)} articles to tickers...")
    df = match_articles_to_tickers(all_articles)

    out_path = RAW_NEWS_DIR / "all_news.csv"
    df.to_csv(out_path, index=False)
    log.info(f"\nSaved raw news: {out_path} ({len(df)} rows)")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    df = collect_all_news()
    print(f"\nCollected {len(df)} ticker-article pairs")
    print(f"Tickers covered: {df['ticker'].nunique()}")
    print(f"\nSample:\n{df[['ticker', 'source', 'title']].head(10)}")

