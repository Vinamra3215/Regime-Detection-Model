
import logging
import time
from datetime import datetime

from config import (
    PHASE5_PREDICTIONS, PHASE5_DETAILED,
    NIFTY_50_TICKERS, RESULTS_DIR,
)

log = logging.getLogger(__name__)


def main():
    start_time = time.time()

    print("=" * 60)
    print("  REGIME DETECTION MODEL — Phase 6")
    print("  Trading Signal Generator (Long/Short/Flat)")
    print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    print("\n" + "=" * 40)
    print("  STEP 1/4: Load Phase 5 Predictions")
    print("=" * 40)
    import pandas as pd

    if not PHASE5_PREDICTIONS.exists():
        print(f"  ERROR: Phase 5 predictions not found at {PHASE5_PREDICTIONS}")
        print("  Run Phase 5 first.")
        return

    predictions_df = pd.read_csv(PHASE5_PREDICTIONS)
    print(f"  -> Loaded {len(predictions_df)} ticker predictions")
    print(f"  -> Regime distribution:")
    for regime, count in predictions_df["predicted_regime"].value_counts().items():
        print(f"     {regime}: {count}")

    print("\n" + "=" * 40)
    print("  STEP 2/4: Generate Trading Signals")
    print("=" * 40)
    from signal_generator import generate_signals_snapshot

    signals_df = generate_signals_snapshot(PHASE5_PREDICTIONS)

    print(f"\n  Signal Distribution:")
    for signal, count in signals_df["signal"].value_counts().items():
        pct = count / len(signals_df) * 100
        print(f"    {signal:15s}: {count:3d} ({pct:.1f}%)")

    print(f"\n  Top Confident Signals:")
    for _, row in signals_df.sort_values("signal_strength", key=abs, ascending=False).head(10).iterrows():
        direction = "+" if row["signal_strength"] > 0 else "-" if row["signal_strength"] < 0 else " "
        print(f"    {direction} {row['ticker']:20s} -> {row['signal']:15s} "
              f"(strength={row['signal_strength']:+.2f}, "
              f"conf={row['confidence']:.2f}, unc={row['epistemic_uncertainty']:.4f})")

    print("\n" + "=" * 40)
    print("  STEP 3/4: Evaluate Signal Quality")
    print("=" * 40)
    from evaluate import (
        compute_signal_returns, evaluate_signals,
        run_go_no_go, save_results, create_dashboard,
    )

    ticker_results = []
    for ticker in NIFTY_50_TICKERS:
        result = compute_signal_returns(ticker, signals_df)
        ticker_results.append(result)

    valid = [r for r in ticker_results if r is not None]
    print(f"  -> Evaluated {len(valid)} tickers with price data")

    eval_results = evaluate_signals(signals_df, ticker_results)

    print(f"\n  Regime-Signal Alignment: {eval_results['alignment_rate']*100:.1f}%")
    print(f"  Conservative (FLAT override): {eval_results['conservative_rate']*100:.1f}%")
    print(f"\n  Aggregate Portfolio Metrics:")
    strat = eval_results["aggregate_strategy"]
    bh = eval_results["aggregate_buyhold"]
    print(f"    Strategy:")
    print(f"      Total Return:      {strat['total_return']*100:.2f}%")
    print(f"      Annualized Return: {strat['annualized_return']*100:.2f}%")
    print(f"      Sharpe Ratio:      {strat['sharpe_ratio']:.4f}")
    print(f"      Max Drawdown:      {strat['max_drawdown']*100:.2f}%")
    print(f"      Win Rate:          {strat['win_rate']*100:.1f}%")
    print(f"    Buy & Hold:")
    print(f"      Total Return:      {bh['total_return']*100:.2f}%")
    print(f"      Sharpe Ratio:      {bh['sharpe_ratio']:.4f}")
    print(f"    Outperformance Rate: {eval_results['outperformance_rate']*100:.1f}% of tickers")

    if eval_results["per_ticker"]:
        ticker_df = pd.DataFrame(eval_results["per_ticker"])
        best = ticker_df.nlargest(5, "signal_return")
        worst = ticker_df.nsmallest(5, "signal_return")
        print(f"\n  Top 5 Signal Performers:")
        for _, row in best.iterrows():
            print(f"    {row['ticker']:20s}: signal={row['signal_return']*100:+.2f}% "
                  f"vs buyhold={row['buyhold_return']*100:+.2f}%")
        print(f"\n  Bottom 5 Signal Performers:")
        for _, row in worst.iterrows():
            print(f"    {row['ticker']:20s}: signal={row['signal_return']*100:+.2f}% "
                  f"vs buyhold={row['buyhold_return']*100:+.2f}%")

    print("\n" + "=" * 40)
    print("  STEP 4/4: Go/No-Go for Phase 7")
    print("=" * 40)

    go_results = run_go_no_go(eval_results)

    save_results(signals_df, eval_results, go_results)
    create_dashboard(signals_df, eval_results)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  Phase 6 complete. Elapsed: {elapsed:.1f}s")
    print(f"  Results: {RESULTS_DIR}/")
    print(f"  Verdict: {go_results['verdict']} ({go_results['passed']}/{go_results['total']})")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    main()

