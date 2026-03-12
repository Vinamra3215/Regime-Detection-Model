
import numpy as np
import pandas as pd
import logging
import json

from config import (
    RESULTS_DIR, PLOTS_DIR,
    PHASE1_LABEL_DIR, EVAL_START,
    RISK_FREE_RATE, TRADING_DAYS_PER_YEAR,
    NIFTY_50_TICKERS, REGIME_TO_IDX,
)

log = logging.getLogger(__name__)


def compute_signal_returns(ticker, signals_df):
    label_path = PHASE1_LABEL_DIR / f"{ticker}_labelled.csv"
    if not label_path.exists():
        return None

    price_df = pd.read_csv(label_path, index_col="Date", parse_dates=True)
    if "log_return_1d" not in price_df.columns:
        return None

    eval_start = pd.Timestamp(EVAL_START)
    price_df = price_df[price_df.index >= eval_start]
    if len(price_df) < 10:
        return None

    signal_to_pos = {
        "STRONG_LONG": 1.0,
        "WEAK_LONG": 0.5,
        "FLAT": 0.0,
        "WEAK_SHORT": -0.5,
        "STRONG_SHORT": -1.0,
    }

    daily_returns = price_df["log_return_1d"].values
    n = len(price_df)

    positions = np.zeros(n)
    regime_correct = np.zeros(n)

    if signals_df is not None and len(signals_df) > 0:
        row = signals_df[signals_df["ticker"] == ticker]
        if len(row) > 0:
            signal = row.iloc[0].get("signal", "FLAT")
            strength = signal_to_pos.get(signal, 0.0)
            positions[:] = strength

    strategy_returns = positions * daily_returns

    buyhold_returns = daily_returns.copy()

    if "Regime" in price_df.columns:
        true_regimes = price_df["Regime"].values
    else:
        true_regimes = np.full(n, "Unknown")

    return {
        "ticker": ticker,
        "n_days": n,
        "strategy_returns": strategy_returns,
        "buyhold_returns": buyhold_returns,
        "positions": positions,
        "true_regimes": true_regimes,
        "dates": price_df.index,
    }


def compute_portfolio_metrics(returns, name="Strategy"):
    if len(returns) == 0 or np.all(returns == 0):
        return {
            "name": name,
            "total_return": 0.0,
            "annualized_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "n_days": len(returns),
        }

    total = np.sum(returns)
    n_days = len(returns)
    ann_factor = TRADING_DAYS_PER_YEAR / max(n_days, 1)

    ann_return = total * ann_factor
    vol = np.std(returns) * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (ann_return - RISK_FREE_RATE) / vol if vol > 0 else 0.0

    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    max_dd = np.min(drawdowns) if len(drawdowns) > 0 else 0.0

    nonzero = returns[returns != 0]
    if len(nonzero) > 0:
        win_rate = np.mean(nonzero > 0)
        gains = np.sum(nonzero[nonzero > 0])
        losses = np.abs(np.sum(nonzero[nonzero < 0]))
        profit_factor = gains / losses if losses > 0 else float('inf')
    else:
        win_rate = 0.0
        profit_factor = 0.0

    return {
        "name": name,
        "total_return": float(total),
        "annualized_return": float(ann_return),
        "volatility": float(vol),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "n_days": n_days,
    }


def evaluate_signals(signals_df, ticker_results):
    signal_counts = signals_df["signal"].value_counts().to_dict()
    total = len(signals_df)

    alignment = []
    for _, row in signals_df.iterrows():
        regime = row.get("predicted_regime", "Sideways")
        signal = row.get("signal", "FLAT")

        if regime == "Bull" and signal in ("STRONG_LONG", "WEAK_LONG"):
            alignment.append("ALIGNED")
        elif regime == "Bear" and signal in ("STRONG_SHORT", "WEAK_SHORT"):
            alignment.append("ALIGNED")
        elif regime == "Sideways" and signal == "FLAT":
            alignment.append("ALIGNED")
        elif signal == "FLAT":
            alignment.append("CONSERVATIVE")
        else:
            alignment.append("MISALIGNED")

    alignment_counts = pd.Series(alignment).value_counts().to_dict()

    all_strategy_returns = []
    all_buyhold_returns = []
    per_ticker_results = []

    for result in ticker_results:
        if result is None:
            continue
        strat = compute_portfolio_metrics(result["strategy_returns"], f"Strategy_{result['ticker']}")
        bh = compute_portfolio_metrics(result["buyhold_returns"], f"BuyHold_{result['ticker']}")
        per_ticker_results.append({
            "ticker": result["ticker"],
            "signal_return": strat["total_return"],
            "buyhold_return": bh["total_return"],
            "signal_sharpe": strat["sharpe_ratio"],
            "buyhold_sharpe": bh["sharpe_ratio"],
            "outperforms": strat["total_return"] > bh["total_return"],
        })
        all_strategy_returns.extend(result["strategy_returns"])
        all_buyhold_returns.extend(result["buyhold_returns"])

    agg_strategy = compute_portfolio_metrics(np.array(all_strategy_returns), "Aggregate_Strategy")
    agg_buyhold  = compute_portfolio_metrics(np.array(all_buyhold_returns), "Aggregate_BuyHold")

    n_outperform = sum(1 for r in per_ticker_results if r["outperforms"])

    return {
        "signal_distribution": signal_counts,
        "total_signals": total,
        "alignment": alignment_counts,
        "alignment_rate": alignment_counts.get("ALIGNED", 0) / max(total, 1),
        "conservative_rate": alignment_counts.get("CONSERVATIVE", 0) / max(total, 1),
        "aggregate_strategy": agg_strategy,
        "aggregate_buyhold": agg_buyhold,
        "outperformance_rate": n_outperform / max(len(per_ticker_results), 1),
        "n_tickers_evaluated": len(per_ticker_results),
        "per_ticker": per_ticker_results,
    }


def run_go_no_go(eval_results):
    checks = {}

    alignment = eval_results["alignment_rate"]
    checks["Signal-Regime Alignment"] = {
        "value": alignment,
        "threshold": 0.80,
        "result": "PASS" if alignment >= 0.80 else "FAIL",
    }

    strat_sharpe = eval_results["aggregate_strategy"]["sharpe_ratio"]
    bh_sharpe = eval_results["aggregate_buyhold"]["sharpe_ratio"]
    checks["Strategy Sharpe > 0"] = {
        "value": strat_sharpe,
        "threshold": 0.0,
        "result": "PASS" if strat_sharpe > 0 else "FAIL",
    }

    flat_rate = eval_results["signal_distribution"].get("FLAT", 0) / max(eval_results["total_signals"], 1)
    checks["Non-trivial Signals"] = {
        "value": 1.0 - flat_rate,
        "threshold": 0.10,
        "result": "PASS" if (1.0 - flat_rate) >= 0.10 else "FAIL",
    }

    max_dd = eval_results["aggregate_strategy"]["max_drawdown"]
    checks["Max Drawdown"] = {
        "value": max_dd,
        "threshold": -0.50,
        "result": "PASS" if max_dd > -0.50 else "FAIL",
    }

    passed = sum(1 for c in checks.values() if c["result"] == "PASS")
    total  = len(checks)
    verdict = "GO" if passed >= 3 else "NO-GO"

    log.info("")
    log.info("=" * 60)
    log.info("GO / NO-GO — Phase 6 → Phase 7")
    log.info("=" * 60)
    for name, check in checks.items():
        status = "✅ PASS" if check["result"] == "PASS" else "❌ FAIL"
        log.info(f"  {status}  {name}: {check['value']:.4f} (threshold: {check['threshold']})")
    log.info(f"\n  {'🟢' if verdict == 'GO' else '🔴'} VERDICT: {verdict} ({passed}/{total})")
    log.info("=" * 60)

    return {"checks": checks, "verdict": verdict, "passed": passed, "total": total}


def save_results(signals_df, eval_results, go_results):
    signals_df.to_csv(RESULTS_DIR / "trading_signals.csv", index=False)

    if eval_results.get("per_ticker"):
        pd.DataFrame(eval_results["per_ticker"]).to_csv(
            RESULTS_DIR / "per_ticker_performance.csv", index=False
        )

    summary = {
        "signal_distribution": eval_results["signal_distribution"],
        "alignment": eval_results["alignment"],
        "alignment_rate": eval_results["alignment_rate"],
        "aggregate_strategy": eval_results["aggregate_strategy"],
        "aggregate_buyhold": eval_results["aggregate_buyhold"],
        "outperformance_rate": eval_results["outperformance_rate"],
        "go_no_go": go_results,
    }
    with open(RESULTS_DIR / "phase6_evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    with open(RESULTS_DIR / "phase6_evaluation_summary.txt", "w") as f:
        f.write("Phase 6 — Trading Signal Generator Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Signals: {eval_results['total_signals']}\n")
        f.write(f"Distribution: {eval_results['signal_distribution']}\n\n")
        f.write(f"Alignment Rate: {eval_results['alignment_rate']*100:.1f}%\n")
        f.write(f"Conservative Rate: {eval_results['conservative_rate']*100:.1f}%\n\n")
        f.write(f"Aggregate Strategy Sharpe: {eval_results['aggregate_strategy']['sharpe_ratio']:.4f}\n")
        f.write(f"Aggregate BuyHold Sharpe:  {eval_results['aggregate_buyhold']['sharpe_ratio']:.4f}\n")
        f.write(f"Outperformance Rate: {eval_results['outperformance_rate']*100:.1f}%\n\n")
        f.write(f"Go/No-Go: {go_results['verdict']} ({go_results['passed']}/{go_results['total']})\n")
        for name, check in go_results["checks"].items():
            status = "PASS" if check["result"] == "PASS" else "FAIL"
            f.write(f"  {status}  {name}: {check['value']:.4f}\n")

    log.info(f"Results saved to {RESULTS_DIR}/")


def create_dashboard(signals_df, eval_results):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        log.warning("Plotly not available, skipping dashboard")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Signal Distribution",
            "Regime-Signal Alignment",
            "Confidence by Signal Type",
            "Uncertainty by Signal Type",
        ],
        specs=[[{"type": "bar"}, {"type": "pie"}],
               [{"type": "box"}, {"type": "box"}]],
    )

    dist = eval_results["signal_distribution"]
    colors = {
        "STRONG_LONG": "#00C853", "WEAK_LONG": "#69F0AE",
        "FLAT": "#78909C",
        "WEAK_SHORT": "#FF8A80", "STRONG_SHORT": "#D50000",
    }
    for signal_type in ["STRONG_LONG", "WEAK_LONG", "FLAT", "WEAK_SHORT", "STRONG_SHORT"]:
        count = dist.get(signal_type, 0)
        if count > 0:
            fig.add_trace(go.Bar(
                x=[signal_type], y=[count], name=signal_type,
                marker_color=colors.get(signal_type, "#999"),
                showlegend=False,
            ), row=1, col=1)

    align = eval_results["alignment"]
    fig.add_trace(go.Pie(
        labels=list(align.keys()),
        values=list(align.values()),
        marker_colors=["#00C853", "#78909C", "#D50000"],
        showlegend=True,
    ), row=1, col=2)

    for signal_type in signals_df["signal"].unique():
        subset = signals_df[signals_df["signal"] == signal_type]
        fig.add_trace(go.Box(
            y=subset["confidence"], name=signal_type,
            marker_color=colors.get(signal_type, "#999"),
            showlegend=False,
        ), row=2, col=1)
        fig.add_trace(go.Box(
            y=subset["epistemic_uncertainty"], name=signal_type,
            marker_color=colors.get(signal_type, "#999"),
            showlegend=False,
        ), row=2, col=2)

    fig.update_layout(
        title="Phase 6 — Trading Signal Dashboard",
        height=800, width=1200,
        template="plotly_dark",
    )

    dash_path = PLOTS_DIR / "phase6_signal_dashboard.html"
    fig.write_html(str(dash_path))
    log.info(f"Dashboard saved: {dash_path}")

