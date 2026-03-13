
import logging
import time
from datetime import datetime

from config import (
    DEVICE, INITIAL_CAPITAL,
    EVAL_START, EVAL_END,
    RESULTS_DIR,
)

log = logging.getLogger(__name__)


def main():
    start_time = time.time()

    print("=" * 60)
    print("  REGIME DETECTION MODEL — Phase 7")
    print("  Position Sizing & Risk Management")
    print(f"  Device: {DEVICE}")
    print(f"  Capital: Rs {INITIAL_CAPITAL:,.0f}")
    print(f"  Eval period: {EVAL_START} to {EVAL_END}")
    print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    print("\n" + "=" * 40)
    print("  STEP 1/4: Load Phase 4 Model + Scalers")
    print("=" * 40)
    from daily_signals import load_model
    model, price_scaler, sent_scaler, num_sent, temperature = load_model()
    print(f"  -> Model loaded, Temperature: {temperature:.4f}")

    print("\n" + "=" * 40)
    print("  STEP 2/4: Generate Daily Rolling Signals")
    print("=" * 40)
    print(f"  Running MC Dropout (20 samples) on each day's window...")
    print(f"  This will take a few minutes...")

    from daily_signals import generate_daily_signals
    daily_signals = generate_daily_signals(
        model, price_scaler, sent_scaler, num_sent, temperature
    )

    total_signals = sum(len(df) for df in daily_signals.values())
    total_active = sum(
        (df["signal"] != "FLAT").sum() for df in daily_signals.values()
    )
    print(f"\n  -> {len(daily_signals)} tickers, {total_signals} total day-signals")
    print(f"  -> {total_active} active (non-FLAT) signals ({total_active/max(total_signals,1)*100:.1f}%)")

    import pandas as pd
    all_sigs = pd.concat(daily_signals.values(), ignore_index=True)
    print(f"\n  Signal Distribution:")
    for sig, count in all_sigs["signal"].value_counts().items():
        pct = count / len(all_sigs) * 100
        print(f"    {sig:15s}: {count:6d} ({pct:.1f}%)")

    regime_match = (all_sigs["predicted_regime"] == all_sigs["true_regime"]).mean()
    print(f"\n  Daily Regime Accuracy: {regime_match*100:.1f}%")

    print("\n" + "=" * 40)
    print("  STEP 3/4: Walk-Forward Portfolio Simulation")
    print("=" * 40)
    from position_sizer import simulate_portfolio
    portfolio_df, trade_df = simulate_portfolio(daily_signals)

    if len(portfolio_df) > 0:
        print(f"  -> {len(portfolio_df)} trading days simulated")
        print(f"  -> Starting capital: Rs {INITIAL_CAPITAL:,.0f}")
        print(f"  -> Final value:      Rs {portfolio_df['portfolio_value'].iloc[-1]:,.0f}")
        pnl = portfolio_df['portfolio_value'].iloc[-1] - INITIAL_CAPITAL
        print(f"  -> Total P&L:        Rs {pnl:+,.0f} ({pnl/INITIAL_CAPITAL*100:+.2f}%)")
    if len(trade_df) > 0:
        print(f"  -> {len(trade_df)} trades executed")
        winners = (trade_df["return"] > 0).sum()
        print(f"  -> Win rate: {winners}/{len(trade_df)} "
              f"({winners/len(trade_df)*100:.1f}%)")
        stops = (trade_df["exit_reason"] == "STOP_LOSS").sum()
        print(f"  -> Stop-loss exits: {stops}")

    print("\n" + "=" * 40)
    print("  STEP 4/4: Evaluation & Go/No-Go")
    print("=" * 40)
    from evaluate import (
        compute_portfolio_metrics, compute_trade_metrics,
        compute_signal_accuracy, run_go_no_go,
        save_results, create_dashboard,
    )

    portfolio_metrics = compute_portfolio_metrics(portfolio_df)
    trade_metrics = compute_trade_metrics(trade_df)
    signal_accuracy = compute_signal_accuracy(daily_signals)

    print(f"\n  Portfolio Metrics:")
    print(f"    Total Return:      {portfolio_metrics.get('total_return_pct', 0):.2f}%")
    print(f"    Annualized Return: {portfolio_metrics.get('annualized_return_pct', 0):.2f}%")
    print(f"    Sharpe Ratio:      {portfolio_metrics.get('sharpe_ratio', 0):.4f}")
    print(f"    Sortino Ratio:     {portfolio_metrics.get('sortino_ratio', 0):.4f}")
    print(f"    Max Drawdown:      {portfolio_metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"    Win Rate (daily):  {portfolio_metrics.get('win_rate_pct', 0):.1f}%")
    print(f"    Profit Factor:     {portfolio_metrics.get('profit_factor', 0):.3f}")

    if trade_metrics:
        print(f"\n  Trade Metrics:")
        print(f"    Total Trades:      {trade_metrics.get('total_trades', 0)}")
        print(f"    Win Rate:          {trade_metrics.get('win_rate_pct', 0):.1f}%")
        print(f"    Avg Holding:       {trade_metrics.get('avg_holding_days', 0):.1f} days")
        print(f"    Long Win Rate:     {trade_metrics.get('long_win_rate', 0):.1f}%")
        print(f"    Short Win Rate:    {trade_metrics.get('short_win_rate', 0):.1f}%")
        print(f"    Stop-Loss Exits:   {trade_metrics.get('stop_loss_exits', 0)}")
        print(f"    Avg Win:           {trade_metrics.get('avg_win_pct', 0):+.2f}%")
        print(f"    Avg Loss:          {trade_metrics.get('avg_loss_pct', 0):+.2f}%")

    print(f"\n  Signal Accuracy:")
    print(f"    Direction Accuracy: {signal_accuracy.get('signal_direction_accuracy', 0)*100:.1f}%")
    print(f"    Regime Accuracy:    {signal_accuracy.get('regime_accuracy', 0)*100:.1f}%")
    print(f"    Active Signals:     {signal_accuracy.get('n_active_signals', 0)}")

    go_results = run_go_no_go(portfolio_metrics, trade_metrics, signal_accuracy)
    save_results(portfolio_df, trade_df, daily_signals,
                 portfolio_metrics, trade_metrics, signal_accuracy, go_results)
    create_dashboard(portfolio_df, trade_df, daily_signals)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  Phase 7 complete. Elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
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

