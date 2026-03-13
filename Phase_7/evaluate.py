
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


def compute_portfolio_metrics(portfolio_df):
    if len(portfolio_df) == 0:
        return {}

    returns = portfolio_df["daily_return"].values
    n_days = len(returns)

    total_return = (portfolio_df["portfolio_value"].iloc[-1] / INITIAL_CAPITAL) - 1
    ann_factor = TRADING_DAYS_PER_YEAR / max(n_days, 1)
    ann_return = total_return * ann_factor

    vol = np.std(returns) * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (ann_return - RISK_FREE_RATE) / vol if vol > 0 else 0.0

    neg_returns = returns[returns < 0]
    downside_vol = np.std(neg_returns) * np.sqrt(TRADING_DAYS_PER_YEAR) if len(neg_returns) > 0 else 0.001
    sortino = (ann_return - RISK_FREE_RATE) / downside_vol

    cumulative = portfolio_df["portfolio_value"].values
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_dd = float(np.min(drawdowns))

    calmar = ann_return / abs(max_dd) if abs(max_dd) > 0.001 else 0.0

    nonzero = returns[returns != 0]
    win_rate = np.mean(nonzero > 0) if len(nonzero) > 0 else 0.0

    gains = np.sum(returns[returns > 0])
    losses = np.abs(np.sum(returns[returns < 0]))
    profit_factor = gains / losses if losses > 0 else float('inf')

    avg_positions = portfolio_df["active_positions"].mean()

    return {
        "initial_capital": INITIAL_CAPITAL,
        "final_value": float(portfolio_df["portfolio_value"].iloc[-1]),
        "total_return_pct": float(total_return * 100),
        "annualized_return_pct": float(ann_return * 100),
        "volatility_pct": float(vol * 100),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown_pct": float(max_dd * 100),
        "calmar_ratio": float(calmar),
        "win_rate_pct": float(win_rate * 100),
        "profit_factor": float(profit_factor),
        "avg_daily_positions": float(avg_positions),
        "n_trading_days": n_days,
    }


def compute_trade_metrics(trade_df):
    if len(trade_df) == 0:
        return {}

    n_trades = len(trade_df)
    winners = trade_df[trade_df["return"] > 0]
    losers = trade_df[trade_df["return"] < 0]

    avg_win = winners["return"].mean() if len(winners) > 0 else 0
    avg_loss = losers["return"].mean() if len(losers) > 0 else 0
    best_trade = trade_df["return"].max()
    worst_trade = trade_df["return"].min()
    avg_hold = trade_df["holding_days"].mean()

    stop_outs = (trade_df["exit_reason"] == "STOP_LOSS").sum()
    signal_exits = (trade_df["exit_reason"] == "SIGNAL_CHANGE").sum()

    long_trades = trade_df[trade_df["direction"] == "LONG"]
    short_trades = trade_df[trade_df["direction"] == "SHORT"]

    return {
        "total_trades": n_trades,
        "winning_trades": len(winners),
        "losing_trades": len(losers),
        "win_rate_pct": float(len(winners) / n_trades * 100),
        "avg_win_pct": float(avg_win * 100),
        "avg_loss_pct": float(avg_loss * 100),
        "best_trade_pct": float(best_trade * 100),
        "worst_trade_pct": float(worst_trade * 100),
        "avg_holding_days": float(avg_hold),
        "stop_loss_exits": int(stop_outs),
        "signal_change_exits": int(signal_exits),
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
        "long_win_rate": float(
            len(long_trades[long_trades["return"] > 0]) / max(len(long_trades), 1) * 100),
        "short_win_rate": float(
            len(short_trades[short_trades["return"] > 0]) / max(len(short_trades), 1) * 100),
    }


def compute_signal_accuracy(daily_signals_dict):
    correct = 0
    total = 0
    regime_correct = 0

    for ticker, df in daily_signals_dict.items():
        for _, row in df.iterrows():
            signal = row.get("signal", "FLAT")
            actual = row.get("actual_return", 0)
            true_regime = row.get("true_regime", "Unknown")
            pred_regime = row.get("predicted_regime", "Unknown")

            if signal == "FLAT":
                continue

            total += 1
            if signal.endswith("LONG") and actual > 0:
                correct += 1
            elif signal.endswith("SHORT") and actual < 0:
                correct += 1

            if pred_regime == true_regime:
                regime_correct += 1

    return {
        "signal_direction_accuracy": correct / max(total, 1),
        "regime_accuracy": regime_correct / max(total, 1),
        "n_active_signals": total,
    }


def run_go_no_go(portfolio_metrics, trade_metrics, signal_accuracy):
    checks = {}

    sharpe = portfolio_metrics.get("sharpe_ratio", -999)
    checks["Sharpe Ratio > -0.5"] = {
        "value": sharpe, "threshold": -0.5,
        "result": "PASS" if sharpe > -0.5 else "FAIL",
    }

    max_dd = portfolio_metrics.get("max_drawdown_pct", -100)
    checks["Max Drawdown > -30%"] = {
        "value": max_dd, "threshold": -30.0,
        "result": "PASS" if max_dd > -30.0 else "FAIL",
    }

    win_rate = trade_metrics.get("win_rate_pct", 0)
    checks["Trade Win Rate > 40%"] = {
        "value": win_rate, "threshold": 40.0,
        "result": "PASS" if win_rate > 40.0 else "FAIL",
    }

    sig_acc = signal_accuracy.get("signal_direction_accuracy", 0) * 100
    checks["Signal Direction Accuracy > 45%"] = {
        "value": sig_acc, "threshold": 45.0,
        "result": "PASS" if sig_acc > 45.0 else "FAIL",
    }

    if trade_metrics.get("total_trades", 0) > 0:
        stop_pct = trade_metrics["stop_loss_exits"] / trade_metrics["total_trades"] * 100
    else:
        stop_pct = 0
    checks["Stop-Loss Rate < 50%"] = {
        "value": stop_pct, "threshold": 50.0,
        "result": "PASS" if stop_pct < 50.0 else "FAIL",
    }

    passed = sum(1 for c in checks.values() if c["result"] == "PASS")
    total = len(checks)
    verdict = "GO" if passed >= 3 else "NO-GO"

    log.info("")
    log.info("=" * 60)
    log.info("GO / NO-GO — Phase 7 → Phase 8")
    log.info("=" * 60)
    for name, check in checks.items():
        status = "✅ PASS" if check["result"] == "PASS" else "❌ FAIL"
        log.info(f"  {status}  {name}: {check['value']:.2f} (threshold: {check['threshold']})")
    log.info(f"\n  {'🟢' if verdict == 'GO' else '🔴'} VERDICT: {verdict} ({passed}/{total})")
    log.info("=" * 60)

    return {"checks": checks, "verdict": verdict, "passed": passed, "total": total}


def save_results(portfolio_df, trade_df, daily_signals_dict,
                 portfolio_metrics, trade_metrics, signal_accuracy, go_results):
    portfolio_df.to_csv(RESULTS_DIR / "portfolio_timeseries.csv", index=False)

    if len(trade_df) > 0:
        trade_df.to_csv(RESULTS_DIR / "trade_log.csv", index=False)

    all_signals = pd.concat(daily_signals_dict.values(), ignore_index=True)
    all_signals.to_csv(RESULTS_DIR / "daily_signals_all.csv", index=False)

    summary = {
        "portfolio_metrics": portfolio_metrics,
        "trade_metrics": trade_metrics,
        "signal_accuracy": signal_accuracy,
        "go_no_go": go_results,
    }
    with open(RESULTS_DIR / "phase7_evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    with open(RESULTS_DIR / "phase7_evaluation_summary.txt", "w") as f:
        f.write("Phase 7 — Position Sizing & Risk Management Summary\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Initial Capital: Rs {portfolio_metrics['initial_capital']:,.0f}\n")
        f.write(f"Final Value:     Rs {portfolio_metrics['final_value']:,.0f}\n")
        f.write(f"Total Return:    {portfolio_metrics['total_return_pct']:.2f}%\n")
        f.write(f"Sharpe Ratio:    {portfolio_metrics['sharpe_ratio']:.4f}\n")
        f.write(f"Sortino Ratio:   {portfolio_metrics['sortino_ratio']:.4f}\n")
        f.write(f"Max Drawdown:    {portfolio_metrics['max_drawdown_pct']:.2f}%\n\n")
        if trade_metrics:
            f.write(f"Total Trades:    {trade_metrics['total_trades']}\n")
            f.write(f"Win Rate:        {trade_metrics['win_rate_pct']:.1f}%\n")
            f.write(f"Avg Hold:        {trade_metrics['avg_holding_days']:.1f} days\n")
            f.write(f"Stop-Loss Exits: {trade_metrics['stop_loss_exits']}\n\n")
        f.write(f"Go/No-Go: {go_results['verdict']} ({go_results['passed']}/{go_results['total']})\n")

    log.info(f"Results saved to {RESULTS_DIR}/")


def create_dashboard(portfolio_df, trade_df, daily_signals_dict):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        log.warning("Plotly not available, skipping dashboard")
        return

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "Portfolio Value (Walk-Forward)",
            "Daily P&L",
            "Active Positions Over Time",
            "Cumulative P&L",
            "Trade Returns Distribution",
            "Signal Distribution Over Time",
        ],
        specs=[[{"type": "scatter"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "histogram"}, {"type": "bar"}]],
    )

    if len(portfolio_df) > 0:
        dates = pd.to_datetime(portfolio_df["date"])

        fig.add_trace(go.Scatter(
            x=dates, y=portfolio_df["portfolio_value"],
            mode="lines", name="Portfolio Value",
            line=dict(color="#00C853", width=2),
        ), row=1, col=1)
        fig.add_hline(y=INITIAL_CAPITAL, line_dash="dash",
                      line_color="white", opacity=0.5, row=1, col=1)

        colors = ["#00C853" if x >= 0 else "#D50000" for x in portfolio_df["daily_pnl"]]
        fig.add_trace(go.Bar(
            x=dates, y=portfolio_df["daily_pnl"],
            name="Daily P&L", marker_color=colors,
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=dates, y=portfolio_df["active_positions"],
            mode="lines", name="Active Positions",
            line=dict(color="#42A5F5"),
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=dates, y=portfolio_df["cumulative_pnl"],
            mode="lines", name="Cumulative P&L",
            fill="tozeroy",
            line=dict(color="#AB47BC"),
        ), row=2, col=2)

    if len(trade_df) > 0:
        fig.add_trace(go.Histogram(
            x=trade_df["return"] * 100,
            name="Trade Returns (%)",
            nbinsx=30,
            marker_color="#FFA726",
        ), row=3, col=1)

    all_signals = pd.concat(daily_signals_dict.values(), ignore_index=True)
    sig_counts = all_signals.groupby("signal").size().reset_index(name="count")
    colors_map = {
        "STRONG_LONG": "#00C853", "WEAK_LONG": "#69F0AE",
        "FLAT": "#78909C",
        "WEAK_SHORT": "#FF8A80", "STRONG_SHORT": "#D50000",
    }
    fig.add_trace(go.Bar(
        x=sig_counts["signal"],
        y=sig_counts["count"],
        marker_color=[colors_map.get(s, "#999") for s in sig_counts["signal"]],
        name="Signal Counts",
    ), row=3, col=2)

    fig.update_layout(
        title="Phase 7 — Position Sizing & Risk Management Dashboard",
        height=1000, width=1400,
        template="plotly_dark",
        showlegend=False,
    )

    path = PLOTS_DIR / "phase7_portfolio_dashboard.html"
    fig.write_html(str(path))
    log.info(f"Dashboard saved: {path}")

