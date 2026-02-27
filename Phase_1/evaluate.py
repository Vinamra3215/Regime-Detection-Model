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

PASS = "✅ PASS"
FAIL = "❌ FAIL"
SOFT = "🔶 SOFT"

MIN_AVG_DURATION_DAYS   = 10
MIN_REGIME_SHARPE_WIN   = 0.60
MIN_POSTERIOR_SHARPNESS = 0.60
MIN_CROSS_TICKER_AGREE  = 0.50
PVALUE_THRESHOLD        = 0.05

MARKET_ERAS = {
    "Pre-COVID":   ("2019-01-01", "2020-01-31"),
    "COVID Crash": ("2020-02-01", "2020-04-30"),
    "Recovery":    ("2020-05-01", "2021-12-31"),
    "2022 Bear":   ("2022-01-01", "2022-12-31"),
    "2023 Rally":  ("2023-01-01", "2024-12-31"),
}


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

    pairs  = [("Bull", "Bear"), ("Bull", "Sideways"), ("Sideways", "Bear")]
    p_vals = []
    for a, b in pairs:
        if a in groups and b in groups:
            _, p = stats.ttest_ind(groups[a], groups[b], equal_var=False)
            p_vals.append(p)

    min_p  = min(p_vals) if p_vals else 1.0
    passed = ordered and min_p < PVALUE_THRESHOLD

    return {"pass": passed, "ordered": ordered, "min_pvalue": min_p, "means": means}


def metric_volatility_separation(df: pd.DataFrame) -> dict:
    vols = {}
    for regime in REGIME_ORDER:
        mask = df["Regime"] == regime
        if mask.sum() > 10 and "log_return_1d" in df.columns:
            vols[regime] = df.loc[mask, "log_return_1d"].dropna().std()

    if not vols:
        return {"pass": False, "vols": {}}

    passed = vols.get("Bear", 0) >= vols.get("Bull", 999)
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
    counts  = df["Regime"].value_counts(normalize=True)
    entropy = -sum(p * np.log(p + 1e-12) for p in counts.values)
    max_e   = np.log(3)
    ratio   = entropy / max_e
    passed  = ratio < 0.95

    return {"pass": passed, "entropy": entropy, "max_entropy": max_e, "ratio": ratio}


def metric_posterior_confidence(df: pd.DataFrame) -> dict:
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if not prob_cols:
        return {"pass": False, "avg_max_prob": None, "reason": "no posterior columns"}

    max_probs = df[prob_cols].max(axis=1)
    avg_max   = max_probs.mean()
    passed    = avg_max >= MIN_POSTERIOR_SHARPNESS

    return {"pass": passed, "avg_max_prob": avg_max}


def metric_regime_filtered_strategy(df: pd.DataFrame) -> dict:
    if "log_return_1d" not in df.columns or "Regime" not in df.columns:
        return {"pass": False, "reason": "missing columns"}

    df = df.dropna(subset=["log_return_1d", "Regime"]).copy()
    df["bh_cum"] = df["log_return_1d"].cumsum()

    signal          = df["Regime"].map({"Bull": 1.0, "Bear": -0.5, "Sideways": 0.0})
    df["strat_ret"] = df["log_return_1d"] * signal
    df["strat_cum"] = df["strat_ret"].cumsum()

    def sharpe(r):
        return (r.mean() / r.std()) * np.sqrt(252) if r.std() != 0 else 0.0

    bh_sharpe    = sharpe(df["log_return_1d"])
    strat_sharpe = sharpe(df["strat_ret"])

    return {
        "pass":         strat_sharpe > bh_sharpe,
        "bh_sharpe":    bh_sharpe,
        "strat_sharpe": strat_sharpe,
        "bh_return":    df["bh_cum"].iloc[-1],
        "strat_return": df["strat_cum"].iloc[-1],
        "bh_maxdd":     _max_drawdown(df["bh_cum"].values),
        "strat_maxdd":  _max_drawdown(df["strat_cum"].values),
        "df":           df,
    }


def _max_drawdown(cum_returns: np.ndarray) -> float:
    peak = np.maximum.accumulate(cum_returns)
    return float((cum_returns - peak).min())


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
                "bull_mean":    bull_r.mean(),
                "bear_mean":    bear_r.mean(),
                "p_bull":       p_bull,
                "p_bear":       p_bear,
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

    aligned  = pd.DataFrame(regime_series).dropna(how="all")
    mode_ser = aligned.mode(axis=1)[0]

    avg_agree = np.mean([(aligned[col] == mode_ser).mean() for col in aligned.columns])
    return {"pass": avg_agree >= MIN_CROSS_TICKER_AGREE, "avg_agreement": avg_agree}


def metric_sma_baseline(df: pd.DataFrame) -> dict:
    if "Close" not in df.columns or "log_return_1d" not in df.columns:
        return {"pass": False, "reason": "missing columns"}

    df = df.copy().dropna(subset=["log_return_1d"])
    df["sma_20"]     = df["Close"].rolling(20).mean()
    df["sma_50"]     = df["Close"].rolling(50).mean()
    df["sma_signal"] = np.where(df["sma_20"] > df["sma_50"], 1.0, -0.5)

    df["sma_ret"] = df["log_return_1d"] * df["sma_signal"].shift(1)
    df["bh_ret"]  = df["log_return_1d"]

    signal_regime = df["Regime"].map({"Bull": 1.0, "Bear": -0.5, "Sideways": 0.0})
    df["hmm_ret"] = df["log_return_1d"] * signal_regime.shift(1)

    df = df.dropna(subset=["sma_ret", "hmm_ret"])

    def sharpe(r):
        return (r.mean() / r.std()) * np.sqrt(252) if r.std() != 0 else 0.0

    sma_sh = sharpe(df["sma_ret"])
    hmm_sh = sharpe(df["hmm_ret"])
    bh_sh  = sharpe(df["bh_ret"])

    passed = hmm_sh > sma_sh

    return {
        "pass":        passed,
        "sma_sharpe":  sma_sh,
        "hmm_sharpe":  hmm_sh,
        "bh_sharpe":   bh_sh,
        "sma_cum":     df["sma_ret"].cumsum().iloc[-1],
        "hmm_cum":     df["hmm_ret"].cumsum().iloc[-1],
        "bh_cum":      df["bh_ret"].cumsum().iloc[-1],
        "df":          df,
    }


def metric_era_breakdown(df: pd.DataFrame) -> dict:
    if "log_return_1d" not in df.columns or "Regime" not in df.columns:
        return {}

    era_results = {}
    for era_name, (start, end) in MARKET_ERAS.items():
        mask   = (df.index >= start) & (df.index <= end)
        era_df = df[mask].dropna(subset=["log_return_1d", "Regime"])

        if len(era_df) < 20:
            era_results[era_name] = {"rows": 0, "available": False}
            continue

        signal         = era_df["Regime"].map({"Bull": 1.0, "Bear": -0.5, "Sideways": 0.0})
        strat_ret      = era_df["log_return_1d"] * signal
        bh_ret         = era_df["log_return_1d"]

        def sharpe(r):
            return (r.mean() / r.std()) * np.sqrt(252) if r.std() != 0 else 0.0

        def max_dd(r):
            c = r.cumsum()
            return float((c - np.maximum.accumulate(c)).min())

        regime_dist = era_df["Regime"].value_counts(normalize=True).to_dict()

        era_results[era_name] = {
            "available":      True,
            "rows":           len(era_df),
            "strat_sharpe":   sharpe(strat_ret),
            "bh_sharpe":      sharpe(bh_ret),
            "strat_maxdd":    max_dd(strat_ret),
            "bh_maxdd":       max_dd(bh_ret),
            "strat_beats_bh": sharpe(strat_ret) > sharpe(bh_ret),
            "regime_dist":    regime_dist,
            "dominant_regime": era_df["Regime"].mode()[0] if len(era_df) > 0 else "N/A",
        }

    return era_results


def run_all_metrics(all_dfs: dict[str, pd.DataFrame]) -> dict:
    ticker_results = {}

    for ticker, df in all_dfs.items():
        ticker_results[ticker] = {
            "return_sep":  metric_return_separation(df),
            "vol_sep":     metric_volatility_separation(df),
            "persistence": metric_regime_persistence(df),
            "entropy":     metric_entropy(df),
            "posterior":   metric_posterior_confidence(df),
            "strategy":    metric_regime_filtered_strategy(df),
            "forward_ret": metric_forward_return_test(df),
            "sma_base":    metric_sma_baseline(df),
            "era":         metric_era_breakdown(df),
        }

    cross_ticker = metric_cross_ticker_agreement(all_dfs)
    return {"per_ticker": ticker_results, "cross_ticker": cross_ticker}


def print_report(results: dict) -> bool:
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

    rs_pass, _, rs_pct  = pct_pass("return_sep")
    pe_pass, _, pe_pct  = pct_pass("persistence")
    po_pass, _, po_pct  = pct_pass("posterior")
    st_pass, _, st_pct  = pct_pass("strategy")
    vs_pass, _, vs_pct  = pct_pass("vol_sep")
    en_pass, _, en_pct  = pct_pass("entropy")
    fr_pass, _, fr_pct  = pct_pass("forward_ret")
    sb_pass, _, sb_pct  = pct_pass("sma_base")
    ct_pass             = cross_ticker.get("pass", False)
    ct_agree            = cross_ticker.get("avg_agreement", 0)

    print("\n  ── HARD REQUIREMENTS ────────────────────────────────────────────")
    print(f"  {'Metric':<45} {'Pass Rate':>10}  {'Status':>10}")
    print("  " + "─" * 65)

    hard_rows = [
        ("Return Separation (Bull>Sideways>Bear, p<0.05)", rs_pct, rs_pct >= 70),
        ("Regime Persistence (avg duration ≥10 days)",    pe_pct, pe_pct >= 70),
        ("Regime-Filtered Strategy > Buy & Hold",         st_pct, st_pct >= 60),
        ("Posterior Confidence (avg max prob ≥ 0.60)",    po_pct, po_pct >= 70),
        ("HMM Strategy > SMA Crossover Baseline",         sb_pct, sb_pct >= 60),
    ]

    all_hard_pass = True
    for label, pct, passed in hard_rows:
        status = PASS if passed else FAIL
        if not passed:
            all_hard_pass = False
        print(f"  {label:<45} {pct:>9.1f}%  {status}")

    print("\n  ── SOFT CHECKS ──────────────────────────────────────────────────")
    soft_rows = [
        ("Volatility Separation (Bear ≥ Bull vol)",      vs_pct),
        ("Low State Entropy (not random noise)",          en_pct),
        ("Forward Return Significance (1d & 5d)",         fr_pct),
        (f"Cross-Ticker Agreement ({ct_agree*100:.1f}%)",  100 * ct_pass),
    ]
    for label, pct in soft_rows:
        icon = SOFT if pct >= 60 else FAIL
        print(f"  {label:<45} {pct:>9.1f}%  {icon}")

    print("\n  ── SMA BASELINE COMPARISON (sample: 10 tickers) ─────────────────")
    print(f"  {'Ticker':<20} {'HMM Sharpe':>12} {'SMA Sharpe':>12} {'B&H Sharpe':>12}  {'Beat SMA?':>10}")
    print("  " + "─" * 70)
    for ticker in tickers[:10]:
        sb = per_ticker[ticker]["sma_base"]
        if not sb.get("sma_sharpe"):
            continue
        label = ticker.replace(".NS", "")
        beat  = "✅" if sb.get("pass") else "❌"
        print(f"  {label:<20} {sb['hmm_sharpe']:>12.3f} {sb['sma_sharpe']:>12.3f} {sb['bh_sharpe']:>12.3f}  {beat:>10}")

    print("\n  ── ERA-BASED BREAKDOWN (avg across all tickers) ─────────────────")
    print(f"  {'Era':<20} {'Rows':>6} {'HMM Sharpe':>12} {'B&H Sharpe':>12} {'Dominant':>12}  {'HMM>B&H?':>9}")
    print("  " + "─" * 75)

    for era_name in MARKET_ERAS:
        era_data = [
            per_ticker[t]["era"].get(era_name, {})
            for t in tickers
            if per_ticker[t]["era"].get(era_name, {}).get("available", False)
        ]
        if not era_data:
            print(f"  {era_name:<20} {'N/A':>6}")
            continue

        avg_strat  = np.mean([e["strat_sharpe"] for e in era_data])
        avg_bh     = np.mean([e["bh_sharpe"]    for e in era_data])
        avg_rows   = int(np.mean([e["rows"]      for e in era_data]))
        dom_regime = max(
            {r: sum(e["regime_dist"].get(r, 0) for e in era_data) for r in REGIME_ORDER},
            key=lambda x: sum(e["regime_dist"].get(x, 0) for e in era_data)
        )
        beat = "✅" if avg_strat > avg_bh else "❌"
        print(f"  {era_name:<20} {avg_rows:>6} {avg_strat:>12.3f} {avg_bh:>12.3f} {dom_regime:>12}  {beat:>9}")

    print("\n  ── DETAILED STATS (sample: 10 tickers) ──────────────────────────")
    print(f"  {'Ticker':<20} {'Bull Mean':>10} {'Bear Mean':>10} {'AvgDur(B)':>10} {'Strat Sharpe':>13} {'BH Sharpe':>10}")
    print("  " + "─" * 75)
    for ticker in tickers[:10]:
        r   = per_ticker[ticker]
        bm  = r["return_sep"]["means"].get("Bull",     float("nan"))
        br  = r["return_sep"]["means"].get("Bear",     float("nan"))
        pd_ = r["persistence"]["avg_durations"].get("Bull", 0)
        ss  = r["strategy"].get("strat_sharpe", float("nan"))
        bhs = r["strategy"].get("bh_sharpe",    float("nan"))
        print(f"  {ticker.replace('.NS',''):<20} {bm:>10.5f} {br:>10.5f} {pd_:>10.1f} {ss:>13.3f} {bhs:>10.3f}")

    print("\n" + "═" * 70)
    v = "🟢  GO — Phase 1 labels are reliable. Proceed to Phase 2." if all_hard_pass \
        else "🔴  NO-GO — Phase 1 labels need improvement before Phase 2."
    print(f"  VERDICT: {v}")
    print("═" * 70 + "\n")

    return all_hard_pass


def plot_evaluation_dashboard(results: dict, all_dfs: dict[str, pd.DataFrame]):
    per_ticker = results["per_ticker"]
    tickers    = list(per_ticker.keys())

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            "Mean Return by Regime (per ticker)",
            "HMM vs SMA vs Buy-and-Hold Sharpe",
            "Avg Regime Duration (days)",
            "Posterior Confidence (avg max prob)",
            "Regime-Filtered Cumulative Return (RELIANCE.NS)",
            "Cross-Ticker Regime Composition Over Time",
            "Era-Based Strategy Sharpe",
            "Era-Based Max Drawdown",
        ],
        vertical_spacing=0.10,
        horizontal_spacing=0.10,
    )

    _plot_returns_by_regime(fig, per_ticker, tickers, row=1, col=1)
    _plot_sma_comparison(fig, per_ticker, tickers, row=1, col=2)
    _plot_duration(fig, per_ticker, tickers, row=2, col=1)
    _plot_posterior_conf(fig, per_ticker, tickers, row=2, col=2)
    _plot_cumulative_return(fig, per_ticker, tickers, row=3, col=1)
    _plot_cross_ticker(fig, all_dfs, row=3, col=2)
    _plot_era_sharpe(fig, per_ticker, tickers, row=4, col=1)
    _plot_era_drawdown(fig, per_ticker, tickers, row=4, col=2)

    fig.update_layout(
        template="plotly_dark",
        height=1800,
        title=dict(text="<b>Phase 1 — Finalized HMM Evaluation Dashboard</b>", x=0.5, font=dict(size=20)),
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
        fig.add_trace(go.Box(
            y=means, name=regime, marker_color=color,
            boxpoints="all", pointpos=0,
        ), row=row, col=col)


def _plot_sma_comparison(fig, per_ticker, tickers, row, col):
    labels  = [t.replace(".NS", "") for t in tickers]
    hmm_sh  = [per_ticker[t]["sma_base"].get("hmm_sharpe", 0) for t in tickers]
    sma_sh  = [per_ticker[t]["sma_base"].get("sma_sharpe", 0) for t in tickers]
    bh_sh   = [per_ticker[t]["sma_base"].get("bh_sharpe",  0) for t in tickers]
    colors  = ["#00C853" if h > s else "#DD2C00" for h, s in zip(hmm_sh, sma_sh)]

    fig.add_trace(go.Bar(x=labels, y=hmm_sh, name="HMM",
                         marker_color=colors, opacity=0.85), row=row, col=col)
    fig.add_trace(go.Scatter(x=labels, y=sma_sh, mode="markers",
                             marker=dict(color="#FFD600", size=5, symbol="diamond"),
                             name="SMA"), row=row, col=col)
    fig.add_trace(go.Scatter(x=labels, y=bh_sh, mode="markers",
                             marker=dict(color="#888", size=4, symbol="circle"),
                             name="B&H"), row=row, col=col)


def _plot_duration(fig, per_ticker, tickers, row, col):
    labels = [t.replace(".NS", "") for t in tickers]
    for regime, color in REGIME_COLORS.items():
        durs = [per_ticker[t]["persistence"]["avg_durations"].get(regime, 0) for t in tickers]
        fig.add_trace(go.Bar(x=labels, y=durs, name=regime,
                             marker_color=color, opacity=0.8), row=row, col=col)
    fig.add_hline(y=MIN_AVG_DURATION_DAYS, line_dash="dash",
                  line_color="#FF6D00", row=row, col=col)


def _plot_posterior_conf(fig, per_ticker, tickers, row, col):
    labels = [t.replace(".NS", "") for t in tickers]
    confs  = [per_ticker[t]["posterior"].get("avg_max_prob", 0) for t in tickers]
    colors = ["#00C853" if c >= MIN_POSTERIOR_SHARPNESS else "#DD2C00" for c in confs]
    fig.add_trace(go.Bar(x=labels, y=confs, marker_color=colors), row=row, col=col)
    fig.add_hline(y=MIN_POSTERIOR_SHARPNESS, line_dash="dash",
                  line_color="#FF6D00", row=row, col=col)


def _plot_cumulative_return(fig, per_ticker, tickers, row, col):
    ref   = "RELIANCE.NS" if "RELIANCE.NS" in tickers else tickers[0]
    strat = per_ticker[ref]["strategy"]
    sma   = per_ticker[ref]["sma_base"]

    if "df" in strat:
        df = strat["df"]
        fig.add_trace(go.Scatter(x=df.index, y=df["bh_cum"],
                                 line=dict(color="#888", width=1.2), name="B&H"), row=row, col=col)
        fig.add_trace(go.Scatter(x=df.index, y=df["strat_cum"],
                                 line=dict(color="#00C853", width=1.5), name="HMM"), row=row, col=col)
    if "df" in sma:
        df2 = sma["df"]
        sma_cum = df2["sma_ret"].cumsum()
        fig.add_trace(go.Scatter(x=df2.index, y=sma_cum,
                                 line=dict(color="#FFD600", width=1.2, dash="dash"), name="SMA"), row=row, col=col)


def _plot_cross_ticker(fig, all_dfs, row, col):
    regime_series = {}
    for ticker, df in all_dfs.items():
        if "Regime" in df.columns:
            regime_series[ticker] = df["Regime"].map({"Bull": 1, "Sideways": 0, "Bear": -1})

    aligned = pd.DataFrame(regime_series).dropna(how="all")
    for regime, color in [("Bull", "#00C853"), ("Bear", "#DD2C00"), ("Sideways", "#FFD600")]:
        val     = {"Bull": 1, "Bear": -1, "Sideways": 0}[regime]
        pct     = (aligned == val).mean(axis=1).rolling(10).mean()
        fig.add_trace(go.Scatter(x=aligned.index, y=pct,
                                 line=dict(color=color, width=1.2), name=f"% {regime}"), row=row, col=col)


def _plot_era_sharpe(fig, per_ticker, tickers, row, col):
    era_names  = list(MARKET_ERAS.keys())
    hmm_sharpe = []
    bh_sharpe  = []

    for era in era_names:
        era_data = [per_ticker[t]["era"].get(era, {}) for t in tickers
                    if per_ticker[t]["era"].get(era, {}).get("available", False)]
        hmm_sharpe.append(np.mean([e["strat_sharpe"] for e in era_data]) if era_data else 0)
        bh_sharpe.append(np.mean([e["bh_sharpe"]     for e in era_data]) if era_data else 0)

    colors = ["#00C853" if h > b else "#DD2C00" for h, b in zip(hmm_sharpe, bh_sharpe)]
    fig.add_trace(go.Bar(x=era_names, y=hmm_sharpe, name="HMM",
                         marker_color=colors, opacity=0.9), row=row, col=col)
    fig.add_trace(go.Scatter(x=era_names, y=bh_sharpe, mode="lines+markers",
                             line=dict(color="#FFD600", dash="dash"), name="B&H"), row=row, col=col)


def _plot_era_drawdown(fig, per_ticker, tickers, row, col):
    era_names   = list(MARKET_ERAS.keys())
    hmm_dd      = []
    bh_dd       = []

    for era in era_names:
        era_data = [per_ticker[t]["era"].get(era, {}) for t in tickers
                    if per_ticker[t]["era"].get(era, {}).get("available", False)]
        hmm_dd.append(np.mean([e["strat_maxdd"] for e in era_data]) if era_data else 0)
        bh_dd.append(np.mean([e["bh_maxdd"]     for e in era_data]) if era_data else 0)

    fig.add_trace(go.Bar(x=era_names, y=hmm_dd, name="HMM Drawdown",
                         marker_color="#DD2C00", opacity=0.7), row=row, col=col)
    fig.add_trace(go.Bar(x=era_names, y=bh_dd, name="B&H Drawdown",
                         marker_color="#888", opacity=0.5), row=row, col=col)
    fig.update_layout(**{f"barmode": "group"})


def main():
    print("\n" + "═" * 70)
    print("  🔬  PHASE 1 — FINALIZED EVALUATION METRICS")
    print("  Checking if HMM labels are reliable enough for Phase 2")
    print("═" * 70 + "\n")

    all_dfs = load_labelled(NIFTY_50_TICKERS)

    if not all_dfs:
        log.error("No labelled data found. Run main.py first.")
        sys.exit(1)

    log.info("Running all evaluation metrics (including SMA baseline + era breakdown)...")
    results     = run_all_metrics(all_dfs)
    go_decision = print_report(results)

    log.info("Generating finalized evaluation dashboard...")
    plot_evaluation_dashboard(results, all_dfs)

    print(f"  📊 Dashboard saved: {PLOTS_DIR}/evaluation_dashboard.html\n")
    sys.exit(0 if go_decision else 1)


if __name__ == "__main__":
    main()
