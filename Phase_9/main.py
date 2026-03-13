
import logging
import time
from datetime import datetime

from config import (
    DEVICE, INITIAL_CAPITAL,
    TRAIN_START, TRAIN_END, EVAL_START, EVAL_END,
    TOTAL_TIMESTEPS, RESULTS_DIR,
)

log = logging.getLogger(__name__)


def main():
    start_time = time.time()

    print("=" * 60)
    print("  REGIME DETECTION MODEL — Phase 9")
    print("  RL Trading Agent (PPO)")
    print(f"  Device: {DEVICE}")
    print(f"  Capital: Rs {INITIAL_CAPITAL:,.0f}")
    print(f"  Train: {TRAIN_START} to {TRAIN_END}")
    print(f"  Eval:  {EVAL_START} to {EVAL_END}")
    print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    print("\n" + "=" * 40)
    print("  STEP 1/4: Pre-compute Transformer Predictions")
    print("=" * 40)
    from trading_env import load_transformer, precompute_predictions

    model, ps, ss, num_sent = load_transformer()
    print(f"  Transformer loaded ({num_sent} sent features)")

    print(f"\n  Pre-computing training predictions ({TRAIN_START} to {TRAIN_END})...")
    train_data = precompute_predictions(model, ps, ss, num_sent, TRAIN_START, TRAIN_END)
    train_total = sum(len(df) for df in train_data.values())
    print(f"  -> {len(train_data)} tickers, {train_total:,} training steps")

    print(f"\n  Pre-computing eval predictions ({EVAL_START} to {EVAL_END})...")
    eval_data = precompute_predictions(model, ps, ss, num_sent, EVAL_START, EVAL_END)
    eval_total = sum(len(df) for df in eval_data.values())
    print(f"  -> {len(eval_data)} tickers, {eval_total:,} eval steps")

    precompute_time = time.time() - start_time
    print(f"\n  Pre-computation took {precompute_time:.0f}s ({precompute_time/60:.1f} min)")

    print("\n" + "=" * 40)
    print("  STEP 2/4: Train PPO Agent")
    print("=" * 40)
    from train_rl import train_ppo

    train_start = time.time()
    ppo_model = train_ppo(train_data, eval_data)
    train_time = time.time() - train_start
    print(f"\n  Training took {train_time:.0f}s ({train_time/60:.1f} min)")

    print("\n" + "=" * 40)
    print("  STEP 3/4: Evaluate RL Agent on 2024")
    print("=" * 40)
    from train_rl import evaluate_agent
    from evaluate import (
        simulate_portfolio, compute_metrics, load_phase8_metrics,
        run_go_no_go, create_dashboard, save_results,
    )

    records = evaluate_agent(ppo_model, eval_data)
    portfolio_df, total_costs = simulate_portfolio(records)

    if len(portfolio_df) > 0:
        final = portfolio_df["portfolio_value"].iloc[-1]
        pnl = final - INITIAL_CAPITAL
        print(f"\n  RL Portfolio Summary:")
        print(f"    Initial: Rs {INITIAL_CAPITAL:>12,.0f}")
        print(f"    Final:   Rs {final:>12,.0f}")
        print(f"    P&L:     Rs {pnl:>+12,.0f} ({pnl/INITIAL_CAPITAL*100:+.2f}%)")
        print(f"    Costs:   Rs {total_costs:>12,.0f}")

    rl_metrics = compute_metrics(portfolio_df, "RL Agent (PPO)")

    print("\n" + "=" * 40)
    print("  STEP 4/4: RL vs Rule-Based Comparison")
    print("=" * 40)
    p8_metrics = load_phase8_metrics()

    print(f"\n  {'Metric':<25s} {'RL Agent':>12s} {'Rule-Based':>12s}")
    print(f"  {'─'*25} {'─'*12} {'─'*12}")
    print(f"  {'Total Return %':<25s} {rl_metrics.get('total_return_pct',0):>11.2f}% {p8_metrics.get('total_return_pct',0):>11.2f}%")

    rl_sharpe = rl_metrics.get('sharpe', 0)
    p8_sharpe = p8_metrics.get('sharpe_ratio', p8_metrics.get('sharpe', 0))
    print(f"  {'Sharpe Ratio':<25s} {rl_sharpe:>12.4f} {p8_sharpe:>12.4f}")

    print(f"  {'Sortino Ratio':<25s} {rl_metrics.get('sortino',0):>12.4f} {'—':>12s}")
    print(f"  {'Max Drawdown %':<25s} {rl_metrics.get('max_drawdown_pct',0):>11.2f}% {p8_metrics.get('max_drawdown_pct',0):>11.2f}%")
    print(f"  {'Profit Factor':<25s} {rl_metrics.get('profit_factor',0):>12.3f} {p8_metrics.get('profit_factor',0):>12.3f}")
    print(f"  {'Daily Win Rate %':<25s} {rl_metrics.get('daily_win_rate_pct',0):>11.1f}% {'—':>12s}")
    print(f"  {'Volatility %':<25s} {rl_metrics.get('volatility_pct',0):>11.2f}% {'—':>12s}")

    import pandas as pd
    rdf = pd.DataFrame(records)
    if len(rdf) > 0:
        print(f"\n  RL Agent Position Statistics:")
        print(f"    Mean Position:  {rdf['position'].mean():.3f}")
        print(f"    Median:         {rdf['position'].median():.3f}")
        print(f"    Std:            {rdf['position'].std():.3f}")
        print(f"    % in Cash:      {(rdf['position'] < 0.05).mean()*100:.1f}%")
        print(f"    % Fully In:     {(rdf['position'] > 0.80).mean()*100:.1f}%")

    go_results = run_go_no_go(rl_metrics, p8_metrics)
    save_results(portfolio_df, rl_metrics, p8_metrics, go_results, records, total_costs)
    create_dashboard(portfolio_df, rl_metrics, p8_metrics)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  Phase 9 complete. Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
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

