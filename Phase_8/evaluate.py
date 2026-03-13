
import numpy as np
import pandas as pd
import json
import logging

from config import (
    RESULTS_DIR, PLOTS_DIR,
    RISK_FREE_RATE, TRADING_DAYS_PER_YEAR,
    INITIAL_CAPITAL,
)

log = logging.getLogger(__name__)


def compute_performance(portfolio_df, name="Strategy"):
    if len(portfolio_df) == 0:
        return {}

    values = portfolio_df["portfolio_value"].values
    n = len(values)
    total_ret = (values[-1] / values[0]) - 1

    daily_returns = np.diff(values) / values[:-1]
    ann_factor = TRADING_DAYS_PER_YEAR / max(n, 1)
    ann_ret = total_ret * ann_factor
    vol = np.std(daily_returns) * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (ann_ret - RISK_FREE_RATE) / vol if vol > 0 else 0

    neg = daily_returns[daily_returns < 0]
    down_vol = np.std(neg) * np.sqrt(TRADING_DAYS_PER_YEAR) if len(neg) > 0 else 0.001
    sortino = (ann_ret - RISK_FREE_RATE) / down_vol

    running_max = np.maximum.accumulate(values)
    drawdowns = (values - running_max) / running_max
    max_dd = float(np.min(drawdowns))
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 0.001 else 0

    nonzero = daily_returns[daily_returns != 0]
    win_rate = np.mean(nonzero > 0) if len(nonzero) > 0 else 0
    gains = np.sum(daily_returns[daily_returns > 0])
    losses = abs(np.sum(daily_returns[daily_returns < 0]))
    profit_factor = gains / losses if losses > 0 else float('inf')

    pdf = portfolio_df.copy()
    pdf["date"] = pd.to_datetime(pdf["date"])
    pdf = pdf.set_index("date")
    monthly = pdf["portfolio_value"].resample("ME").last()
    monthly_rets = monthly.pct_change().dropna()

    return {
        "name": name,
        "initial_capital": float(values[0]),
        "final_value": float(values[-1]),
        "total_return_pct": float(total_ret * 100),
        "annualized_return_pct": float(ann_ret * 100),
        "volatility_pct": float(vol * 100),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown_pct": float(max_dd * 100),
        "calmar_ratio": float(calmar),
        "daily_win_rate_pct": float(win_rate * 100),
        "profit_factor": float(profit_factor),
        "n_trading_days": n,
        "best_day_pct": float(np.max(daily_returns) * 100) if len(daily_returns) > 0 else 0,
        "worst_day_pct": float(np.min(daily_returns) * 100) if len(daily_returns) > 0 else 0,
        "avg_daily_return_pct": float(np.mean(daily_returns) * 100) if len(daily_returns) > 0 else 0,
        "monthly_returns": monthly_rets.values.tolist() if len(monthly_rets) > 0 else [],
        "positive_months": int((monthly_rets > 0).sum()) if len(monthly_rets) > 0 else 0,
        "total_months": len(monthly_rets),
    }


def compute_trade_stats(trade_df):
    if len(trade_df) == 0:
        return {}

    n = len(trade_df)
    winners = trade_df[trade_df["net_pnl"] > 0]
    losers = trade_df[trade_df["net_pnl"] < 0]

    total_costs = trade_df["transaction_costs"].sum()
    total_gross = trade_df["gross_pnl"].sum()
    total_net = trade_df["net_pnl"].sum()

    return {
        "total_trades": n,
        "winning_trades": len(winners),
        "losing_trades": len(losers),
        "win_rate_pct": len(winners) / n * 100,
        "total_gross_pnl": float(total_gross),
        "total_net_pnl": float(total_net),
        "total_transaction_costs": float(total_costs),
        "cost_as_pct_of_capital": float(total_costs / INITIAL_CAPITAL * 100),
        "avg_win_pnl": float(winners["net_pnl"].mean()) if len(winners) > 0 else 0,
        "avg_loss_pnl": float(losers["net_pnl"].mean()) if len(losers) > 0 else 0,
        "best_trade_pnl": float(trade_df["net_pnl"].max()),
        "worst_trade_pnl": float(trade_df["net_pnl"].min()),
        "avg_holding_days": float(trade_df["holding_days"].mean()),
        "max_holding_days": int(trade_df["holding_days"].max()),
        "stop_loss_count": int((trade_df["exit_reason"] == "STOP_LOSS").sum()),
        "signal_exit_count": int((trade_df["exit_reason"] == "SIGNAL_CHANGE").sum()),
        "end_of_period_count": int((trade_df["exit_reason"] == "END_OF_PERIOD").sum()),
    }


def run_go_no_go(strategy_perf, buyhold_perf, trade_stats, total_costs):
    checks = {}

    net_return = strategy_perf.get("total_return_pct", -100)
    checks["Positive Return After Costs"] = {
        "value": net_return, "threshold": 0.0,
        "result": "PASS" if net_return > 0 else "FAIL",
    }

    max_dd = strategy_perf.get("max_drawdown_pct", -100)
    checks["Max Drawdown > -15%"] = {
        "value": max_dd, "threshold": -15.0,
        "result": "PASS" if max_dd > -15.0 else "FAIL",
    }

    win_rate = trade_stats.get("win_rate_pct", 0)
    checks["Trade Win Rate > 40%"] = {
        "value": win_rate, "threshold": 40.0,
        "result": "PASS" if win_rate > 40.0 else "FAIL",
    }

    cost_pct = trade_stats.get("cost_as_pct_of_capital", 100)
    checks["Transaction Costs < 2%"] = {
        "value": cost_pct, "threshold": 2.0,
        "result": "PASS" if cost_pct < 2.0 else "FAIL",
    }

    pf = strategy_perf.get("profit_factor", 0)
    checks["Profit Factor > 0.9"] = {
        "value": pf, "threshold": 0.9,
        "result": "PASS" if pf > 0.9 else "FAIL",
    }

    passed = sum(1 for c in checks.values() if c["result"] == "PASS")
    total = len(checks)
    verdict = "PRODUCTION-READY" if passed >= 4 else "NEEDS-IMPROVEMENT"

    log.info("")
    log.info("=" * 60)
    log.info("FINAL GO / NO-GO — Production Readiness")
    log.info("=" * 60)
    for name, check in checks.items():
        status = "✅ PASS" if check["result"] == "PASS" else "❌ FAIL"
        log.info(f"  {status}  {name}: {check['value']:.2f} (threshold: {check['threshold']})")
    log.info(f"\n  {'🟢' if passed >= 4 else '🟡'} VERDICT: {verdict} ({passed}/{total})")
    log.info("=" * 60)

    return {"checks": checks, "verdict": verdict, "passed": passed, "total": total}


def create_dashboard(portfolio_df, buyhold_df, trade_df, strategy_perf, buyhold_perf):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        log.warning("Plotly not available")
        return

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "Portfolio Value: Strategy vs Buy & Hold",
            "Drawdown Chart",
            "Monthly Returns",
            "Trade P&L Distribution",
            "Cumulative Transaction Costs",
            "Position Count Over Time",
        ],
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "histogram"}],
               [{"type": "scatter"}, {"type": "scatter"}]],
    )

    dates = pd.to_datetime(portfolio_df["date"])

    fig.add_trace(go.Scatter(
        x=dates, y=portfolio_df["portfolio_value"],
        mode="lines", name="Strategy",
        line=dict(color="#00C853", width=2),
    ), row=1, col=1)

    if len(buyhold_df) > 0:
        bh_dates = pd.to_datetime(buyhold_df["date"])
        fig.add_trace(go.Scatter(
            x=bh_dates, y=buyhold_df["buyhold_value"],
            mode="lines", name="Buy & Hold",
            line=dict(color="#FFA726", width=2, dash="dash"),
        ), row=1, col=1)

    fig.add_hline(y=INITIAL_CAPITAL, line_dash="dot",
                  line_color="white", opacity=0.3, row=1, col=1)

    values = portfolio_df["portfolio_value"].values
    running_max = np.maximum.accumulate(values)
    dd = (values - running_max) / running_max * 100
    fig.add_trace(go.Scatter(
        x=dates, y=dd, mode="lines", name="Drawdown %",
        fill="tozeroy", line=dict(color="#D50000"),
    ), row=1, col=2)

    pdf = portfolio_df.copy()
    pdf["date"] = pd.to_datetime(pdf["date"])
    pdf = pdf.set_index("date")
    monthly = pdf["portfolio_value"].resample("ME").last()
    monthly_rets = monthly.pct_change().dropna() * 100
    colors = ["#00C853" if r > 0 else "#D50000" for r in monthly_rets]
    fig.add_trace(go.Bar(
        x=monthly_rets.index, y=monthly_rets.values,
        name="Monthly Return %", marker_color=colors,
    ), row=2, col=1)

    if len(trade_df) > 0:
        fig.add_trace(go.Histogram(
            x=trade_df["net_pnl"], nbinsx=30,
            name="Trade Net P&L", marker_color="#AB47BC",
        ), row=2, col=2)

    fig.add_trace(go.Scatter(
        x=dates, y=portfolio_df["cumulative_costs"],
        mode="lines", name="Cum. Costs",
        line=dict(color="#FF5722"),
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=portfolio_df["n_positions"],
        mode="lines", name="Positions",
        fill="tozeroy", line=dict(color="#42A5F5"),
    ), row=3, col=2)

    fig.update_layout(
        title="Phase 8 — Backtest Results Dashboard (with Zerodha Costs)",
        height=1100, width=1400,
        template="plotly_dark",
        showlegend=True,
    )

    path = PLOTS_DIR / "phase8_backtest_dashboard.html"
    fig.write_html(str(path))
    log.info(f"Dashboard saved: {path}")


def save_all_results(portfolio_df, buyhold_df, trade_df,
                     strategy_perf, buyhold_perf, trade_stats,
                     go_results, total_costs):
    portfolio_df.to_csv(RESULTS_DIR / "backtest_portfolio.csv", index=False)
    if len(buyhold_df) > 0:
        buyhold_df.to_csv(RESULTS_DIR / "buyhold_portfolio.csv", index=False)
    if len(trade_df) > 0:
        trade_df.to_csv(RESULTS_DIR / "backtest_trades.csv", index=False)

    summary = {
        "strategy": strategy_perf,
        "buy_and_hold": buyhold_perf,
        "trades": trade_stats,
        "total_transaction_costs": total_costs,
        "go_no_go": go_results,
    }
    with open(RESULTS_DIR / "phase8_backtest_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    with open(RESULTS_DIR / "phase8_backtest_summary.txt", "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  PHASE 8 — FINAL BACKTEST REPORT\n")
        f.write("  Zerodha Kite Realistic Costs\n")
        f.write("=" * 60 + "\n\n")

        f.write("STRATEGY PERFORMANCE\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Initial Capital:     Rs {strategy_perf['initial_capital']:>12,.0f}\n")
        f.write(f"  Final Value:         Rs {strategy_perf['final_value']:>12,.0f}\n")
        f.write(f"  Total Return:           {strategy_perf['total_return_pct']:>10.2f}%\n")
        f.write(f"  Annualized Return:      {strategy_perf['annualized_return_pct']:>10.2f}%\n")
        f.write(f"  Sharpe Ratio:           {strategy_perf['sharpe_ratio']:>10.4f}\n")
        f.write(f"  Sortino Ratio:          {strategy_perf['sortino_ratio']:>10.4f}\n")
        f.write(f"  Max Drawdown:           {strategy_perf['max_drawdown_pct']:>10.2f}%\n")
        f.write(f"  Daily Win Rate:         {strategy_perf['daily_win_rate_pct']:>10.1f}%\n")
        f.write(f"  Profit Factor:          {strategy_perf['profit_factor']:>10.3f}\n")
        f.write(f"  Positive Months:        {strategy_perf['positive_months']}/{strategy_perf['total_months']}\n\n")

        f.write("BUY & HOLD BENCHMARK\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Final Value:         Rs {buyhold_perf.get('final_value', 0):>12,.0f}\n")
        f.write(f"  Total Return:           {buyhold_perf.get('total_return_pct', 0):>10.2f}%\n")
        f.write(f"  Sharpe Ratio:           {buyhold_perf.get('sharpe_ratio', 0):>10.4f}\n\n")

        if trade_stats:
            f.write("TRADE STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Total Trades:           {trade_stats['total_trades']:>10d}\n")
            f.write(f"  Win Rate:               {trade_stats['win_rate_pct']:>10.1f}%\n")
            f.write(f"  Avg Holding:            {trade_stats['avg_holding_days']:>10.1f} days\n")
            f.write(f"  Total Costs:         Rs {trade_stats['total_transaction_costs']:>12,.0f}\n")
            f.write(f"  Costs % of Capital:     {trade_stats['cost_as_pct_of_capital']:>10.2f}%\n")
            f.write(f"  Gross P&L:           Rs {trade_stats['total_gross_pnl']:>12,.0f}\n")
            f.write(f"  Net P&L:             Rs {trade_stats['total_net_pnl']:>12,.0f}\n\n")

        f.write("GO/NO-GO\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Verdict: {go_results['verdict']} ({go_results['passed']}/{go_results['total']})\n")
        for name, check in go_results["checks"].items():
            status = "PASS" if check["result"] == "PASS" else "FAIL"
            f.write(f"  {status}  {name}: {check['value']:.2f}\n")

    log.info(f"All results saved to {RESULTS_DIR}/")

