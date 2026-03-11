
import argparse
import logging
import sys
import time
from datetime import datetime

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Phase 3 — Sentiment Data Pipeline")
    parser.add_argument("--skip-news", action="store_true",
                        help="Skip news collection (use existing data)")
    parser.add_argument("--skip-scoring", action="store_true",
                        help="Skip FinBERT scoring (use existing scores)")
    parser.add_argument("--skip-market", action="store_true",
                        help="Skip market feature download")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation")
    args = parser.parse_args()

    start_time = time.time()

    print("=" * 60)
    print("  REGIME DETECTION MODEL — Phase 3")
    print("  Sentiment Data Pipeline")
    print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()

    if not args.skip_news:
        print("=" * 40)
        print("  STEP 1/5: News Collection")
        print("=" * 40)
        from news_collector import collect_all_news
        news_df = collect_all_news()
        print(f"  → Collected {len(news_df)} ticker-article pairs\n")
    else:
        print("STEP 1/5: News Collection — SKIPPED\n")

    if not args.skip_scoring:
        print("=" * 40)
        print("  STEP 2/5: FinBERT Sentiment Scoring")
        print("=" * 40)
        from sentiment_scorer import score_news_dataframe
        from config import RAW_NEWS_DIR
        import pandas as pd

        news_path = RAW_NEWS_DIR / "all_news.csv"
        if news_path.exists():
            news_df = pd.read_csv(news_path)
            scored_df = score_news_dataframe(news_df)
            print(f"  → Scored {len(scored_df)} articles\n")
        else:
            print("  ERROR: No raw news found. Run Step 1 first.\n")
            if not args.skip_news:
                sys.exit(1)
    else:
        print("STEP 2/5: FinBERT Scoring — SKIPPED\n")

    if not args.skip_market:
        print("=" * 40)
        print("  STEP 3/5: Market-Wide Features (VIX, Breadth)")
        print("=" * 40)
        from market_features import compute_market_features
        market_df = compute_market_features()
        print(f"  → Market features: {market_df.shape}\n")
    else:
        print("STEP 3/5: Market Features — SKIPPED\n")

    print("=" * 40)
    print("  STEP 4/5: Sentiment Feature Engineering")
    print("=" * 40)
    from feature_engineering import build_all_sentiment_features
    features = build_all_sentiment_features()
    print(f"  → Built features for {len(features)} tickers\n")

    if not args.skip_eval:
        print("=" * 40)
        print("  STEP 5/5: Evaluation & Go/No-Go")
        print("=" * 40)
        from evaluate import run_evaluation
        summary = run_evaluation()
        verdict = summary.get("go_no_go", {}).get("verdict", "UNKNOWN")
        print(f"\n  → Verdict: {verdict}\n")
    else:
        print("STEP 5/5: Evaluation — SKIPPED\n")

    elapsed = time.time() - start_time
    print("=" * 60)
    print(f"  Phase 3 complete. Elapsed: {elapsed:.1f}s")
    print(f"  Results: results/phase_3/")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    main()

