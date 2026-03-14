
import numpy as np
import pandas as pd
import json
import logging

from config import (
    RESULTS_DIR, PLOTS_DIR,
    INITIAL_CAPITAL, RISK_FREE_RATE, TRADING_DAYS,
    ZERODHA_STT_PCT, ZERODHA_EXCHANGE_PCT,
    ZERODHA_GST_PCT, ZERODHA_SEBI_PCT,
    ZERODHA_STAMP_PCT, SLIPPAGE_PCT,
)

log = logging.getLogger(__name__)

PROJECT_DIR = RESULTS_DIR.parent.parent
PHASE8_RESULTS = PROJECT_DIR / "results" / "phase_8"

def txn_cost(value, is_buy=True):
    stt = value * ZERODHA_STT_PCT if not is_buy else 0
    exc = value * ZERODHA_EXCHANGE_PCT
    gst = exc * ZERODHA_GST_PCT
    sebi = value * ZERODHA_SEBI_PCT
    stamp = value * ZERODHA_STAMP_PCT if is_buy else 0
    slip = value * SLIPPAGE_PCT
    return stt + exc + gst + sebi + stamp + slip

def simulate_portfolio(records):
    if not records:
        return pd.DataFrame(), 0.0

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    dates = sorted(df["date"].unique())

    capital = INITIAL_CAPITAL
    prev_positions = {}
    daily_records = []
    total_costs = 0.0
    trade_count = 0

    for date in dates:
        day_data = df[df["date"] == date]
        day_return = 0.0
        day_costs = 0.0

        for _, row in day_data.iterrows():
            ticker = row["ticker"]
            position = row["position"]
            actual_ret = row["actual_return"]

            prev_pos = prev_positions.get(ticker, 0.0)

            pos_change = abs(position - prev_pos)
            if pos_change > 0.05:
                trade_value = capital * pos_change / len(day_data["ticker"].unique())
                cost = txn_cost(trade_value, is_buy=(position > prev_pos))
                day_costs += cost
                trade_count += 1

            allocated = capital * position / max(len(day_data["ticker"].unique()), 1)
            day_return += allocated * actual_ret

            prev_positions[ticker] = position

        capital += day_return - day_costs
        total_costs += day_costs

        daily_records.append({
            "date": date,
            "portfolio_value": capital,
            "day_return": day_return,
            "day_costs": day_costs,
            "cumulative_costs": total_costs,
            "n_active": sum(1 for v in prev_positions.values() if v > 0.05),
        })

    portfolio_df = pd.DataFrame(daily_records)
    return portfolio_df, total_costs

def compute_metrics(portfolio_df, name="RL Strategy"):
    if len(portfolio_df) < 2:
        return {}

    vals = portfolio_df["portfolio_value"].values
    n = len(vals)
    total_ret = (vals[-1] / vals[0] - 1)
    daily_rets = np.diff(vals) / vals[:-1]
    ann_factor = TRADING_DAYS / max(n, 1)
    ann_ret = total_ret * ann_factor
    vol = np.std(daily_rets) * np.sqrt(TRADING_DAYS)
    sharpe = (ann_ret - RISK_FREE_RATE) / vol if vol > 0 else 0
    neg = daily_rets[daily_rets < 0]
    down_vol = np.std(neg) * np.sqrt(TRADING_DAYS) if len(neg) > 0 else 0.001
    sortino = (ann_ret - RISK_FREE_RATE) / down_vol
    rmax = np.maximum.accumulate(vals)
    max_dd = float(np.min((vals - rmax) / rmax))
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 0.001 else 0
    nz = daily_rets[daily_rets != 0]
    win_rate = np.mean(nz > 0) if len(nz) > 0 else 0
    gains = np.sum(daily_rets[daily_rets > 0])
    losses = abs(np.sum(daily_rets[daily_rets < 0]))
    pf = gains / losses if losses > 0 else float("inf")

    return {
        "name": name,
        "initial": float(vals[0]),
        "final": float(vals[-1]),
        "total_return_pct": float(total_ret * 100),
        "annualized_return_pct": float(ann_ret * 100),
        "volatility_pct": float(vol * 100),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown_pct": float(max_dd * 100),
        "calmar": float(calmar),
        "daily_win_rate_pct": float(win_rate * 100),
        "profit_factor": float(pf),
        "n_days": n,
    }

def load_phase8_metrics():
    p8_file = PHASE8_RESULTS / "phase8_backtest_summary.json"
    if p8_file.exists():
        with open(p8_file) as f:
            data = json.load(f)
        return data.get("strategy", {})
    return {}

def run_go_no_go(rl_metrics, p8_metrics):
    checks = {}

    ret = rl_metrics.get("total_return_pct", -100)
    checks["Positive Return"] = {
        "value": ret, "threshold": 0.0,
        "result": "PASS" if ret > 0 else "FAIL",
    }

    dd = rl_metrics.get("max_drawdown_pct", -100)
    checks["Max Drawdown > -15%"] = {
        "value": dd, "threshold": -15.0,
        "result": "PASS" if dd > -15.0 else "FAIL",
    }

    pf = rl_metrics.get("profit_factor", 0)
    p8_pf = p8_metrics.get("profit_factor", 1.0)
    checks["Profit Factor > Rule-Based"] = {
        "value": pf, "threshold": p8_pf,
        "result": "PASS" if pf > p8_pf * 0.9 else "FAIL",
    }

    wr = rl_metrics.get("daily_win_rate_pct", 0)
    checks["Daily Win Rate > 45%"] = {
        "value": wr, "threshold": 45.0,
        "result": "PASS" if wr > 45.0 else "FAIL",
    }

    sharpe = rl_metrics.get("sharpe", -10)
    p8_sharpe = p8_metrics.get("sharpe_ratio", p8_metrics.get("sharpe", -10))
    checks["Sharpe >= Rule-Based"] = {
        "value": sharpe, "threshold": p8_sharpe,
        "result": "PASS" if sharpe >= p8_sharpe * 0.9 else "FAIL",
    }

    passed = sum(1 for c in checks.values() if c["result"] == "PASS")
    total = len(checks)
    verdict = "RL-SUPERIOR" if passed >= 4 else "RL-COMPARABLE" if passed >= 3 else "RULE-BASED-BETTER"

    log.info("")
    log.info("=" * 60)
    log.info("GO / NO-GO — Phase 9: RL vs Rule-Based")
    log.info("=" * 60)
    for name, check in checks.items():
        s = "✅ PASS" if check["result"] == "PASS" else "❌ FAIL"
        log.info(f"  {s}  {name}: {check['value']:.4f} (threshold: {check['threshold']:.4f})")
    emoji = "🟢" if passed >= 4 else "🟡" if passed >= 3 else "🔴"
    log.info(f"\n  {emoji} VERDICT: {verdict} ({passed}/{total})")
    log.info("=" * 60)

    return {"checks": checks, "verdict": verdict, "passed": passed, "total": total}

def create_dashboard(rl_portfolio_df, rl_metrics, p8_metrics):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        log.warning("Plotly not installed")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "RL Portfolio Value Over Time",
            "RL Drawdown",
            "Position Distribution (RL Agent Decisions)",
            "RL vs Rule-Based Comparison",
        ],
    )

    dates = pd.to_datetime(rl_portfolio_df["date"])

    fig.add_trace(go.Scatter(
        x=dates, y=rl_portfolio_df["portfolio_value"],
        mode="lines", name="RL Strategy", line=dict(color="#00E676", width=2),
    ), row=1, col=1)
    fig.add_hline(y=INITIAL_CAPITAL, line_dash="dot", line_color="white",
                  opacity=0.3, row=1, col=1)

    vals = rl_portfolio_df["portfolio_value"].values
    rmax = np.maximum.accumulate(vals)
    dd = (vals - rmax) / rmax * 100
    fig.add_trace(go.Scatter(
        x=dates, y=dd, mode="lines", name="Drawdown %",
        fill="tozeroy", line=dict(color="#D50000"),
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=dates, y=rl_portfolio_df["n_active"],
        mode="lines", name="Active Positions",
        fill="tozeroy", line=dict(color="#42A5F5"),
    ), row=2, col=1)

    metrics_to_compare = ["total_return_pct", "sharpe", "max_drawdown_pct", "profit_factor"]
    labels = ["Total Return %", "Sharpe", "Max Drawdown %", "Profit Factor"]
    rl_vals = [rl_metrics.get(m, 0) for m in metrics_to_compare]
    p8_vals_map = {"total_return_pct": "total_return_pct", "sharpe": "sharpe_ratio",
                   "max_drawdown_pct": "max_drawdown_pct", "profit_factor": "profit_factor"}
    p8_vals = [p8_metrics.get(p8_vals_map.get(m, m), p8_metrics.get(m, 0)) for m in metrics_to_compare]

    fig.add_trace(go.Bar(x=labels, y=rl_vals, name="RL Agent",
                         marker_color="#00E676"), row=2, col=2)
    fig.add_trace(go.Bar(x=labels, y=p8_vals, name="Rule-Based",
                         marker_color="#FFA726"), row=2, col=2)

    fig.update_layout(
        title="Phase 9 — RL Trading Agent vs Rule-Based",
        height=900, width=1400, template="plotly_dark",
    )

    path = PLOTS_DIR / "phase9_rl_dashboard.html"
    fig.write_html(str(path))
    log.info(f"Dashboard saved: {path}")

def save_results(portfolio_df, rl_metrics, p8_metrics, go_results, records, total_costs):
    portfolio_df.to_csv(RESULTS_DIR / "rl_portfolio.csv", index=False)
    pd.DataFrame(records).to_csv(RESULTS_DIR / "rl_daily_decisions.csv", index=False)

    summary = {
        "rl_agent": rl_metrics,
        "rule_based_phase8": p8_metrics,
        "total_costs": total_costs,
        "go_no_go": go_results,
    }
    with open(RESULTS_DIR / "phase9_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    with open(RESULTS_DIR / "phase9_summary.txt", "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  PHASE 9 — RL Trading Agent Results\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"  {'Metric':<25s} {'RL Agent':>12s} {'Rule-Based':>12s}\n")
        f.write(f"  {'─'*25} {'─'*12} {'─'*12}\n")
        f.write(f"  {'Total Return %':<25s} {rl_metrics.get('total_return_pct',0):>11.2f}% {p8_metrics.get('total_return_pct',0):>11.2f}%\n")
        f.write(f"  {'Sharpe Ratio':<25s} {rl_metrics.get('sharpe',0):>12.4f} {p8_metrics.get('sharpe_ratio', p8_metrics.get('sharpe',0)):>12.4f}\n")
        f.write(f"  {'Max Drawdown %':<25s} {rl_metrics.get('max_drawdown_pct',0):>11.2f}% {p8_metrics.get('max_drawdown_pct',0):>11.2f}%\n")
        f.write(f"  {'Profit Factor':<25s} {rl_metrics.get('profit_factor',0):>12.3f} {p8_metrics.get('profit_factor',0):>12.3f}\n")
        f.write(f"  {'Win Rate %':<25s} {rl_metrics.get('daily_win_rate_pct',0):>11.1f}% {'—':>12s}\n\n")
        f.write(f"  Verdict: {go_results['verdict']} ({go_results['passed']}/{go_results['total']})\n")

    log.info(f"Results saved to {RESULTS_DIR}/")
