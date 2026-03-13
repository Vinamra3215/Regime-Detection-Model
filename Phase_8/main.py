
import logging
import time
from datetime import datetime

from config import (
    PHASE7_SIGNALS, RESULTS_DIR,
    INITIAL_CAPITAL, EVAL_START, EVAL_END,
)

log = logging.getLogger(__name__)


def main():
    start_time = time.time()

    print("=" * 60)
    print("  REGIME DETECTION MODEL — Phase 8 (FINAL)")
    print("  Backtesting + Paper Trading (Zerodha Kite)")
    print(f"  Capital: Rs {INITIAL_CAPITAL:,.0f}")
    print(f"  Period: {EVAL_START} to {EVAL_END}")
    print(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    print("\n" + "=" * 40)
    print("  STEP 1/5: Load Phase 7 Daily Signals")
    print("=" * 40)
    import pandas as pd

    if not PHASE7_SIGNALS.exists():
        print(f"  ERROR: Phase 7 signals not found at {PHASE7_SIGNALS}")
        return

    signals_df = pd.read_csv(PHASE7_SIGNALS)
    signals_df["date"] = pd.to_datetime(signals_df["date"])
    print(f"  -> {len(signals_df)} daily signals loaded")
    print(f"  -> {signals_df['ticker'].nunique()} tickers")
    print(f"  -> Date range: {signals_df['date'].min()} to {signals_df['date'].max()}")

    sig_dist = signals_df["signal"].value_counts()
    print(f"\n  Signal Distribution:")
    for sig, count in sig_dist.items():
        pct = count / len(signals_df) * 100
        print(f"    {sig:15s}: {count:6d} ({pct:.1f}%)")

    print("\n" + "=" * 40)
    print("  STEP 2/5: Walk-Forward Backtest (Zerodha Costs)")
    print("=" * 40)
    from backtester import run_backtest, compute_buy_and_hold

    portfolio_df, trade_df, total_costs = run_backtest(signals_df)

    if len(portfolio_df) > 0:
        final = portfolio_df["portfolio_value"].iloc[-1]
        pnl = final - INITIAL_CAPITAL
        print(f"\n  Portfolio Summary:")
        print(f"    Initial:     Rs {INITIAL_CAPITAL:>12,.0f}")
        print(f"    Final:       Rs {final:>12,.0f}")
        print(f"    P&L:         Rs {pnl:>+12,.0f} ({pnl/INITIAL_CAPITAL*100:+.2f}%)")
        print(f"    Total Costs: Rs {total_costs:>12,.0f}")
    if len(trade_df) > 0:
        print(f"\n  Trades: {len(trade_df)} total")
        winners = (trade_df["net_pnl"] > 0).sum()
        print(f"  Win Rate: {winners}/{len(trade_df)} "
              f"({winners/len(trade_df)*100:.1f}%)")
        print(f"  Gross P&L: Rs {trade_df['gross_pnl'].sum():+,.0f}")
        print(f"  Net P&L:   Rs {trade_df['net_pnl'].sum():+,.0f}")
        print(f"  Costs:     Rs {trade_df['transaction_costs'].sum():,.0f}")

    print("\n" + "=" * 40)
    print("  STEP 3/5: Buy & Hold Benchmark")
    print("=" * 40)

    buyhold_df = compute_buy_and_hold(signals_df)
    if len(buyhold_df) > 0:
        bh_final = buyhold_df["buyhold_value"].iloc[-1]
        bh_ret = (bh_final / INITIAL_CAPITAL - 1) * 100
        print(f"  Buy & Hold Final: Rs {bh_final:,.0f} ({bh_ret:+.2f}%)")

    print("\n" + "=" * 40)
    print("  STEP 4/5: Performance Evaluation")
    print("=" * 40)
    from evaluate import (
        compute_performance, compute_trade_stats,
        run_go_no_go, create_dashboard, save_all_results,
    )

    strategy_perf = compute_performance(portfolio_df, "Regime Strategy")
    buyhold_perf = compute_performance(
        buyhold_df.rename(columns={"buyhold_value": "portfolio_value"}),
        "Buy & Hold"
    ) if len(buyhold_df) > 0 else {}
    trade_stats = compute_trade_stats(trade_df)

    print(f"\n  {'Metric':<25s} {'Strategy':>12s} {'Buy&Hold':>12s}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    print(f"  {'Total Return %':<25s} {strategy_perf.get('total_return_pct',0):>11.2f}% {buyhold_perf.get('total_return_pct',0):>11.2f}%")
    print(f"  {'Sharpe Ratio':<25s} {strategy_perf.get('sharpe_ratio',0):>12.4f} {buyhold_perf.get('sharpe_ratio',0):>12.4f}")
    print(f"  {'Sortino Ratio':<25s} {strategy_perf.get('sortino_ratio',0):>12.4f} {buyhold_perf.get('sortino_ratio',0):>12.4f}")
    print(f"  {'Max Drawdown %':<25s} {strategy_perf.get('max_drawdown_pct',0):>11.2f}% {buyhold_perf.get('max_drawdown_pct',0):>11.2f}%")
    print(f"  {'Profit Factor':<25s} {strategy_perf.get('profit_factor',0):>12.3f} {buyhold_perf.get('profit_factor',0):>12.3f}")
    print(f"  {'Daily Win Rate %':<25s} {strategy_perf.get('daily_win_rate_pct',0):>11.1f}% {buyhold_perf.get('daily_win_rate_pct',0):>11.1f}%")
    print(f"  {'Positive Months':<25s} {strategy_perf.get('positive_months',0):>8d}/{strategy_perf.get('total_months',0)} {buyhold_perf.get('positive_months',0):>8d}/{buyhold_perf.get('total_months',0)}")

    if trade_stats:
        print(f"\n  Trade Statistics:")
        print(f"    Total Trades:      {trade_stats['total_trades']}")
        print(f"    Win Rate:          {trade_stats['win_rate_pct']:.1f}%")
        print(f"    Avg Holding:       {trade_stats['avg_holding_days']:.1f} days")
        print(f"    Gross P&L:         Rs {trade_stats['total_gross_pnl']:+,.0f}")
        print(f"    Transaction Costs: Rs {trade_stats['total_transaction_costs']:,.0f}")
        print(f"    Net P&L:           Rs {trade_stats['total_net_pnl']:+,.0f}")
        print(f"    Costs as % Capital:{trade_stats['cost_as_pct_of_capital']:.2f}%")

    print("\n" + "=" * 40)
    print("  STEP 5/5: Paper Trading & Final Verdict")
    print("=" * 40)

    from paper_trading import PaperTrader
    paper = PaperTrader()

    latest_date = signals_df["date"].max()
    latest_signals = signals_df[signals_df["date"] == latest_date]
    recommendations = paper.generate_paper_trade_report(latest_signals)

    if len(recommendations) > 0:
        print(f"\n  Today's Paper Trade Recommendations:")
        for _, row in recommendations.iterrows():
            print(f"    {row['action']:4s}  {row['ticker']:20s}  ({row['reason']})")
    else:
        print("  No active trade recommendations for today.")

    go_results = run_go_no_go(strategy_perf, buyhold_perf, trade_stats, total_costs)

    save_all_results(portfolio_df, buyhold_df, trade_df,
                     strategy_perf, buyhold_perf, trade_stats,
                     go_results, total_costs)
    create_dashboard(portfolio_df, buyhold_df, trade_df,
                     strategy_perf, buyhold_perf)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  Phase 8 COMPLETE — Final Phase of Regime Detection Model")
    print(f"  Elapsed: {elapsed:.1f}s")
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

