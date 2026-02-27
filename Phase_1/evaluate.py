import numpy as np
import pandas as pd
import pickle
import logging
import sys
from pathlib import Path
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    DATA_LABEL_DIR, PLOTS_DIR, NIFTY_50_TICKERS,
    REGIME_COLORS, REGIME_ORDER
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PASS  = "✅ PASS"
FAIL  = "❌ FAIL"
SOFT  = "🔶 SOFT"

MIN_AVG_DURATION_DAYS   = 10
MIN_REGIME_SHARPE_WIN   = 0.60
MIN_POSTERIOR_SHARPNESS = 0.60
MIN_CROSS_TICKER_AGREE  = 0.50
PVALUE_THRESHOLD        = 0.05


def load_labelled(tickers: list[str]) -> dict[str, pd.DataFrame]:
    data = {}
    for ticker in tickers:
        path = DATA_LABEL_DIR / f"{ticker}_labelled.csv"
        if path.exists():
            df = pd.read_csv(path, index_col="Date", parse_dates=True)
            if "Regime" in df.columns and len(df) > 100:
                data[ticker] = df
    log.info(f"Loaded {len(data)} labelled tickers.")
    return data


def load_hmm_models(tickers: list[str]) -> dict:
    models = {}
    models_dir = DATA_LABEL_DIR.parent / "models"
    for ticker in tickers:
        path = models_dir / f"{ticker}_hmm.pkl"
        if path.exists():
            with open(path, "rb") as f:
                models[ticker] = pickle.load(f)
    return models


def metric_return_separation(df: pd.DataFrame) -> dict:
    groups = {}
    for regime in REGIME_ORDER:
        mask = df["Regime"] == regime
        if mask.sum() > 10 and "log_return_1d" in df.columns:
            groups[regime] = df.loc[mask, "log_return_1d"].dropna().values

    if len(groups) < 2:
        return {"pass": False, "reason": "insufficient data per regime", "means": {}}

    means   = {r: v.mean() for r, v in groups.items()}
    ordered = (
        means.get("Bull", 0) > means.get("Sideways", -999) and
        means.get("Sideways", -999) > means.get("Bear", -9999)
    )

    pairs   = [("Bull", "Bear"), ("Bull", "Sideways"), ("Sideways", "Bear")]
    p_vals  = []
    for a, b in pairs:
        if a in groups and b in groups:
            _, p = stats.ttest_ind(groups[a], groups[b], equal_var=False)
            p_vals.append(p)

    min_p   = min(p_vals) if p_vals else 1.0
    passed  = ordered and min_p < PVALUE_THRESHOLD

    return {
        "pass":    passed,
        "ordered": ordered,
        "min_pvalue": min_p,
        "means":   means,
    }


def metric_volatility_separation(df: pd.DataFrame) -> dict:
    vols = {}
    for regime in REGIME_ORDER:
        mask = df["Regime"] == regime
        if mask.sum() > 10 and "log_return_1d" in df.columns:
            vols[regime] = df.loc[mask, "log_return_1d"].dropna().std()

    if not vols:
        return {"pass": False, "vols": {}}

    bear_higher = vols.get("Bear", 0) >= vols.get("Bull", 999)
    passed      = bear_higher

    return {"pass": passed, "vols": vols}


def metric_regime_persistence(df: pd.DataFrame) -> dict:
    durations = {r: [] for r in REGIME_ORDER}
    current   = None
    run       = 0

    for r in df["Regime"].values:
        if r == current:
            run += 1
        else:
            if current is not None:
                durations[current].append(run)
            current = r
            run     = 1
    if current:
        durations[current].append(run)

    avg_durs = {r: (np.mean(v) if v else 0) for r, v in durations.items()}
    passed   = all(d >= MIN_AVG_DURATION_DAYS for d in avg_durs.values() if d > 0)

    return {"pass": passed, "avg_durations": avg_durs}


def metric_entropy(df: pd.DataFrame) -> dict:
    counts = df["Regime"].value_counts(normalize=True)
    entropy = -sum(p * np.log(p + 1e-12) for p in counts.values)
    max_e   = np.log(3)
    ratio   = entropy / max_e
    passed  = ratio < 0.95

    return {"pass": passed, "entropy": entropy, "max_entropy": max_e, "ratio": ratio}


def metric_posterior_confidence(df: pd.DataFrame) -> dict:
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if not prob_cols:
        return {"pass": False, "avg_max_prob": None, "reason": "no posterior columns"}

    max_probs   = df[prob_cols].max(axis=1)
    avg_max     = max_probs.mean()
    passed      = avg_max >= MIN_POSTERIOR_SHARPNESS

    return {"pass": passed, "avg_max_prob": avg_max}


def metric_regime_filtered_strategy(df: pd.DataFrame) -> dict:
    if "log_return_1d" not in df.columns or "Regime" not in df.columns:
        return {"pass": False, "reason": "missing columns"}

    df = df.dropna(subset=["log_return_1d", "Regime"]).copy()

    df["bh_cum"]  = df["log_return_1d"].cumsum()

    signal = df["Regime"].map({"Bull": 1.0, "Bear": -0.5, "Sideways": 0.0})
    df["strat_ret"] = df["log_return_1d"] * signal
    df["strat_cum"] = df["strat_ret"].cumsum()

    def sharpe(r):
        if r.std() == 0:
            return 0.0
        return (r.mean() / r.std()) * np.sqrt(252)

    bh_sharpe    = sharpe(df["log_return_1d"])
    strat_sharpe = sharpe(df["strat_ret"])

    total_bh     = df["bh_cum"].iloc[-1]
    total_strat  = df["strat_cum"].iloc[-1]

    bh_dd        = _max_drawdown(df["bh_cum"].values)
    strat_dd     = _max_drawdown(df["strat_cum"].values)

    passed       = strat_sharpe > bh_sharpe

    return {
        "pass":          passed,
        "bh_sharpe":     bh_sharpe,
        "strat_sharpe":  strat_sharpe,
        "bh_return":     total_bh,
        "strat_return":  total_strat,
        "bh_maxdd":      bh_dd,
        "strat_maxdd":   strat_dd,
        "df":            df,
    }


def _max_drawdown(cum_returns: np.ndarray) -> float:
    peak = np.maximum.accumulate(cum_returns)
    dd   = cum_returns - peak
    return float(dd.min())


def metric_forward_return_test(df: pd.DataFrame) -> dict:
    if "log_return_1d" not in df.columns:
        return {"pass": False}

    df = df.copy()
    df["fwd_1d"] = df["log_return_1d"].shift(-1)
    df["fwd_5d"] = df["log_return_1d"].rolling(5).sum().shift(-5)

    results = {}
    for horizon, col in [("1d", "fwd_1d"), ("5d", "fwd_5d")]:
        bull_r = df.loc[df["Regime"] == "Bull", col].dropna()
        bear_r = df.loc[df["Regime"] == "Bear", col].dropna()

        if len(bull_r) > 10 and len(bear_r) > 10:
            _, p_bull = stats.ttest_1samp(bull_r, 0)
            _, p_bear = stats.ttest_1samp(bear_r, 0)
            results[horizon] = {
                "bull_mean": bull_r.mean(),
                "bear_mean": bear_r.mean(),
                "p_bull":    p_bull,
                "p_bear":    p_bear,
                "bull_positive": bull_r.mean() > 0 and p_bull < PVALUE_THRESHOLD,
                "bear_negative": bear_r.mean() < 0 and p_bear < PVALUE_THRESHOLD,
            }

    passed = all(
        r.get("bull_positive", False) and r.get("bear_negative", False)
        for r in results.values()
    )

    return {"pass": passed, "details": results}


def metric_cross_ticker_agreement(all_dfs: dict[str, pd.DataFrame]) -> dict:
    regime_series = {}
    for ticker, df in all_dfs.items():
        if "Regime" in df.columns:
            regime_series[ticker] = df["Regime"].map({"Bull": 1, "Sideways": 0, "Bear": -1})

    if len(regime_series) < 5:
        return {"pass": False, "avg_agreement": None}

    aligned   = pd.DataFrame(regime_series).dropna(how="all")
    mode_vals = aligned.mode(axis=1)[0]

    agreements = []
    for col in aligned.columns:
        agree = (aligned[col] == mode_vals).mean()
        agreements.append(agree)

    avg_agree = np.mean(agreements)
    passed    = avg_agree >= MIN_CROSS_TICKER_AGREE

    return {"pass": passed, "avg_agreement": avg_agree}


def run_all_metrics(all_dfs: dict[str, pd.DataFrame]) -> dict:
    ticker_results = {}

    for ticker, df in all_dfs.items():
        ticker_results[ticker] = {
            "return_sep":    metric_return_separation(df),
            "vol_sep":       metric_volatility_separation(df),
            "persistence":   metric_regime_persistence(df),
            "entropy":       metric_entropy(df),
            "posterior":     metric_posterior_confidence(df),
            "strategy":      metric_regime_filtered_strategy(df),
            "forward_ret":   metric_forward_return_test(df),
        }

    cross_ticker = metric_cross_ticker_agreement(all_dfs)

    return {"per_ticker": ticker_results, "cross_ticker": cross_ticker}


def print_report(results: dict):
    per_ticker   = results["per_ticker"]
    cross_ticker = results["cross_ticker"]
    tickers      = list(per_ticker.keys())
    n            = len(tickers)

    def pct_pass(metric_key):
        passing = sum(1 for t in tickers if per_ticker[t][metric_key].get("pass", False))
        return passing, n, 100 * passing / n if n else 0

    print("\n" + "═" * 70)
    print("  📊  PHASE 1 EVALUATION REPORT — GO / NO-GO FOR PHASE 2")
    print("═" * 70)

    hard_metrics = []
    soft_metrics = []

    rs_pass, rs_n, rs_pct   = pct_pass("return_sep")
    vs_pass, _, vs_pct      = pct_pass("vol_sep")
    pe_pass, _, pe_pct      = pct_pass("persistence")
    en_pass, _, en_pct      = pct_pass("entropy")
    po_pass, _, po_pct      = pct_pass("posterior")
    st_pass, _, st_pct      = pct_pass("strategy")
    fr_pass, _, fr_pct      = pct_pass("forward_ret")
    ct_pass                 = cross_ticker.get("pass", False)
    ct_agree                = cross_ticker.get("avg_agreement", 0)

    print("\n  ── HARD REQUIREMENTS (all must pass for Go) ─────────────────────")
    print(f"  {'Metric':<45} {'Pass Rate':>10}  {'Status':>10}")
    print("  " + "─" * 65)

    hard_rows = [
        ("Return Separation (Bull>Sideways>Bear, p<0.05)", rs_pct, rs_pct >= 70),
        ("Regime Persistence (avg duration ≥10 days)",    pe_pct, pe_pct >= 70),
        ("Regime-Filtered Strategy > Buy&Hold",           st_pct, st_pct >= 60),
        ("Posterior Confidence (avg max prob ≥ 0.60)",    po_pct, po_pct >= 70),
    ]

    all_hard_pass = True
    for label, pct, passed in hard_rows:
        status = PASS if passed else FAIL
        if not passed:
            all_hard_pass = False
        print(f"  {label:<45} {pct:>9.1f}%  {status}")

    print("\n  ── SOFT CHECKS (informational) ──────────────────────────────────")
    soft_rows = [
        ("Volatility Separation (Bear ≥ Bull vol)",       vs_pct),
        ("Low State Entropy (not random noise)",          en_pct),
        ("Forward Return Significance (1d & 5d)",         fr_pct),
        (f"Cross-Ticker Agreement ({ct_agree*100:.1f}% match mode)", 100 * ct_pass),
    ]
    for label, pct in soft_rows:
        icon = SOFT if pct >= 60 else FAIL
        print(f"  {label:<45} {pct:>9.1f}%  {icon}")

    print("\n  ── DETAILED STATS (sample: first 5 tickers) ─────────────────────")
    print(f"  {'Ticker':<20} {'Bull Mean':>10} {'Bear Mean':>10} {'AvgDur(B)':>10} {'Strat Sharpe':>13} {'BH Sharpe':>10}")
    print("  " + "─" * 75)
    for ticker in tickers[:10]:
        r  = per_ticker[ticker]
        bm = r["return_sep"]["means"].get("Bull", float("nan"))
        br = r["return_sep"]["means"].get("Bear", float("nan"))
        pd_ = r["persistence"]["avg_durations"].get("Bull", 0)
        ss = r["strategy"].get("strat_sharpe", float("nan"))
        bhs = r["strategy"].get("bh_sharpe", float("nan"))
        label = ticker.replace(".NS", "")
        print(f"  {label:<20} {bm:>10.5f} {br:>10.5f} {pd_:>10.1f} {ss:>13.3f} {bhs:>10.3f}")

    print("\n" + "═" * 70)
    verdict = "🟢  GO — Phase 1 labels are reliable. Proceed to Phase 2." if all_hard_pass \
              else "🔴  NO-GO — Phase 1 labels need improvement before Phase 2."
    print(f"  VERDICT: {verdict}")
    print("═" * 70 + "\n")

    return all_hard_pass


def plot_evaluation_dashboard(results: dict, all_dfs: dict[str, pd.DataFrame]):
    per_ticker = results["per_ticker"]
    tickers    = list(per_ticker.keys())

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "Mean Return by Regime (per ticker)",
            "Strategy vs Buy-and-Hold Sharpe",
            "Avg Regime Duration (days)",
            "Posterior Confidence (avg max prob)",
            "Regime-Filtered Cumulative Return (RELIANCE.NS)",
            "Cross-Ticker Regime Agreement",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    _plot_returns_by_regime(fig, per_ticker, tickers, row=1, col=1)
    _plot_sharpe_comparison(fig, per_ticker, tickers, row=1, col=2)
    _plot_duration(fig, per_ticker, tickers, row=2, col=1)
    _plot_posterior_conf(fig, per_ticker, tickers, row=2, col=2)
    _plot_cumulative_return(fig, per_ticker, tickers, row=3, col=1)
    _plot_cross_ticker(fig, all_dfs, row=3, col=2)

    fig.update_layout(
        template="plotly_dark",
        height=1400,
        title=dict(text="<b>Phase 1 — HMM Evaluation Dashboard</b>", x=0.5, font=dict(size=20)),
        paper_bgcolor="#0D0D0D",
        plot_bgcolor="#1A1A1A",
        font=dict(family="Inter, Arial", color="#E0E0E0"),
        showlegend=False,
    )
    fig.update_yaxes(gridcolor="#2A2A2A", zerolinecolor="#555")
    fig.update_xaxes(gridcolor="#2A2A2A")

    path = PLOTS_DIR / "evaluation_dashboard.html"
    fig.write_html(str(path))
    log.info(f"Saved evaluation dashboard: {path}")


def _plot_returns_by_regime(fig, per_ticker, tickers, row, col):
    for regime, color in REGIME_COLORS.items():
        means = [per_ticker[t]["return_sep"]["means"].get(regime, 0) for t in tickers]
        labels = [t.replace(".NS", "") for t in tickers]
        fig.add_trace(go.Box(
            y=means, name=regime,
            marker_color=color,
            boxpoints="all", pointpos=0,
            hovertemplate=f"{regime}<br>Mean Return: %{{y:.5f}}<extra></extra>",
        ), row=row, col=col)


def _plot_sharpe_comparison(fig, per_ticker, tickers, row, col):
    labels   = [t.replace(".NS", "") for t in tickers]
    strat_sh = [per_ticker[t]["strategy"].get("strat_sharpe", 0) for t in tickers]
    bh_sh    = [per_ticker[t]["strategy"].get("bh_sharpe", 0) for t in tickers]

    colors  = ["#00C853" if s > b else "#DD2C00" for s, b in zip(strat_sh, bh_sh)]

    fig.add_trace(go.Bar(
        x=labels, y=strat_sh, name="Strategy",
        marker_color=colors, opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Strategy Sharpe: %{y:.3f}<extra></extra>",
    ), row=row, col=col)

    fig.add_trace(go.Scatter(
        x=labels, y=bh_sh, mode="markers",
        marker=dict(color="#FFD600", size=5, symbol="diamond"),
        name="Buy & Hold",
        hovertemplate="<b>%{x}</b><br>B&H Sharpe: %{y:.3f}<extra></extra>",
    ), row=row, col=col)


def _plot_duration(fig, per_ticker, tickers, row, col):
    labels = [t.replace(".NS", "") for t in tickers]
    for regime, color in REGIME_COLORS.items():
        durs = [per_ticker[t]["persistence"]["avg_durations"].get(regime, 0) for t in tickers]
        fig.add_trace(go.Bar(
            x=labels, y=durs, name=regime,
            marker_color=color, opacity=0.8,
        ), row=row, col=col)

    fig.add_hline(y=MIN_AVG_DURATION_DAYS, line_dash="dash",
                  line_color="#FF6D00", annotation_text="Min 10d", row=row, col=col)
    fig.update_layout(**{f"xaxis{row*2-1}_tickangle": -45})


def _plot_posterior_conf(fig, per_ticker, tickers, row, col):
    labels = [t.replace(".NS", "") for t in tickers]
    confs  = [per_ticker[t]["posterior"].get("avg_max_prob", 0) for t in tickers]
    colors = ["#00C853" if c >= MIN_POSTERIOR_SHARPNESS else "#DD2C00" for c in confs]

    fig.add_trace(go.Bar(
        x=labels, y=confs, marker_color=colors,
        hovertemplate="<b>%{x}</b><br>Avg max prob: %{y:.3f}<extra></extra>",
    ), row=row, col=col)
    fig.add_hline(y=MIN_POSTERIOR_SHARPNESS, line_dash="dash",
                  line_color="#FF6D00", annotation_text="Min 0.60", row=row, col=col)


def _plot_cumulative_return(fig, per_ticker, tickers, row, col):
    ref    = "RELIANCE.NS" if "RELIANCE.NS" in tickers else tickers[0]
    strat  = per_ticker[ref]["strategy"]

    if "df" in strat:
        df = strat["df"]
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bh_cum"],
            mode="lines", line=dict(color="#FFD600", width=1.5),
            name="Buy & Hold",
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["strat_cum"],
            mode="lines", line=dict(color="#00C853", width=1.5),
            name="Regime Strategy",
        ), row=row, col=col)


def _plot_cross_ticker(fig, all_dfs: dict, row, col):
    regime_series = {}
    for ticker, df in all_dfs.items():
        if "Regime" in df.columns:
            regime_series[ticker] = df["Regime"].map({"Bull": 1, "Sideways": 0, "Bear": -1})

    aligned  = pd.DataFrame(regime_series).dropna(how="all")
    mode_ser = aligned.mode(axis=1)[0]

    pct_bull = (aligned == 1).mean(axis=1)
    pct_bear = (aligned == -1).mean(axis=1)
    pct_side = (aligned == 0).mean(axis=1)

    fig.add_trace(go.Scatter(
        x=aligned.index, y=pct_bull.rolling(10).mean(),
        mode="lines", line=dict(color="#00C853", width=1.2), name="% Bull",
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=aligned.index, y=pct_bear.rolling(10).mean(),
        mode="lines", line=dict(color="#DD2C00", width=1.2), name="% Bear",
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=aligned.index, y=pct_side.rolling(10).mean(),
        mode="lines", line=dict(color="#FFD600", width=1.2), name="% Sideways",
    ), row=row, col=col)


def main():
    print("\n" + "═" * 70)
    print("  🔬  PHASE 1 — EVALUATION METRICS")
    print("  Checking if HMM labels are reliable enough for Phase 2")
    print("═" * 70 + "\n")

    all_dfs = load_labelled(NIFTY_50_TICKERS)

    if not all_dfs:
        log.error("No labelled data found. Run main.py first.")
        sys.exit(1)

    log.info("Running all evaluation metrics...")
    results = run_all_metrics(all_dfs)

    go_decision = print_report(results)

    log.info("Generating evaluation dashboard...")
    plot_evaluation_dashboard(results, all_dfs)

    print(f"  📊 Evaluation plot saved to: {PLOTS_DIR}/evaluation_dashboard.html\n")

    sys.exit(0 if go_decision else 1)


if __name__ == "__main__":
    main()
