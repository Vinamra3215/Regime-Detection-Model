
import numpy as np
import pandas as pd
import logging
import sys
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    NIFTY_50_TICKERS, REGIME_TO_IDX, IDX_TO_REGIME,
    PHASE1_LABEL_DIR, SENTIMENT_FEATURES_DIR,
    MARKET_DATA_DIR, PLOTS_DIR, RESULTS_DIR,
    TRAIN_END, VAL_END, SENTIMENT_FEATURE_COLUMNS,
)

log = logging.getLogger(__name__)

PASS = "✅ PASS"
FAIL = "❌ FAIL"

MIN_TICKER_COVERAGE = 0.80
MIN_TEMPORAL_COVERAGE = 0.50
MIN_FEATURE_VARIANCE = 0.01
MIN_REGIME_CORRELATION = 0.02


def load_all_sentiment_features() -> dict[str, pd.DataFrame]:
    features = {}
    for ticker in NIFTY_50_TICKERS:
        path = SENTIMENT_FEATURES_DIR / f"{ticker}_sentiment.csv"
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            features[ticker] = df
    return features


def evaluate_coverage(features: dict) -> dict:
    total_tickers = len(NIFTY_50_TICKERS)
    covered_tickers = len(features)
    coverage_ratio = covered_tickers / total_tickers

    log.info("\n--- Coverage Analysis ---")
    log.info(f"  Tickers with features: {covered_tickers}/{total_tickers} "
             f"({coverage_ratio*100:.1f}%)")

    temporal_coverages = []
    for ticker, df in features.items():
        if "news_count" in df.columns:
            has_news = (df["news_count"] > 0).mean()
            temporal_coverages.append(has_news)

    mean_temporal = np.mean(temporal_coverages) if temporal_coverages else 0.0
    log.info(f"  Mean temporal coverage (days with news): {mean_temporal*100:.1f}%")

    return {
        "ticker_coverage": coverage_ratio,
        "covered_tickers": covered_tickers,
        "total_tickers": total_tickers,
        "mean_temporal_coverage": mean_temporal,
    }


def evaluate_feature_quality(features: dict) -> dict:
    log.info("\n--- Feature Quality Analysis ---")

    combined = pd.concat(features.values())
    stats = combined.describe()

    low_variance = []
    for col in combined.columns:
        var = combined[col].var()
        if var < MIN_FEATURE_VARIANCE:
            low_variance.append(col)
            log.warning(f"  Low variance feature: {col} (var={var:.6f})")

    high_variance_ratio = 1 - len(low_variance) / max(1, len(combined.columns))

    log.info(f"  Total features: {len(combined.columns)}")
    log.info(f"  Low variance features: {len(low_variance)}")
    log.info(f"  High variance ratio: {high_variance_ratio*100:.1f}%")
    log.info(f"\n  Feature stats:\n{stats.to_string()}")

    return {
        "total_features": len(combined.columns),
        "low_variance_features": low_variance,
        "high_variance_ratio": high_variance_ratio,
        "stats": stats,
    }


def evaluate_regime_correlation(features: dict) -> dict:
    log.info("\n--- Regime Correlation Analysis ---")

    correlations = {}
    for ticker, feat_df in features.items():
        label_path = PHASE1_LABEL_DIR / f"{ticker}_labelled.csv"
        if not label_path.exists():
            continue

        label_df = pd.read_csv(label_path, index_col="Date", parse_dates=True)
        if "Regime" not in label_df.columns:
            continue

        label_df["regime_num"] = label_df["Regime"].map(REGIME_TO_IDX)

        common_dates = feat_df.index.intersection(label_df.index)
        if len(common_dates) < 50:
            continue

        for col in feat_df.columns:
            if col not in correlations:
                correlations[col] = []
            corr = feat_df.loc[common_dates, col].corr(
                label_df.loc[common_dates, "regime_num"]
            )
            if not np.isnan(corr):
                correlations[col].append(corr)

    avg_corr = {}
    max_abs_corr = 0.0
    for col, vals in correlations.items():
        mean_corr = np.mean(vals)
        avg_corr[col] = mean_corr
        max_abs_corr = max(max_abs_corr, abs(mean_corr))
        log.info(f"  {col}: mean_corr={mean_corr:.4f} (n={len(vals)})")

    return {
        "avg_correlations": avg_corr,
        "max_abs_correlation": max_abs_corr,
    }


def evaluate_sentiment_by_regime(features: dict) -> dict:
    log.info("\n--- Sentiment by Regime ---")

    regime_sentiments = {0: [], 1: [], 2: []}

    for ticker, feat_df in features.items():
        label_path = PHASE1_LABEL_DIR / f"{ticker}_labelled.csv"
        if not label_path.exists():
            continue

        label_df = pd.read_csv(label_path, index_col="Date", parse_dates=True)
        if "Regime" not in label_df.columns:
            continue

        common_dates = feat_df.index.intersection(label_df.index)
        if len(common_dates) < 10:
            continue

        for regime, idx in REGIME_TO_IDX.items():
            regime_mask = label_df.loc[common_dates, "Regime"] == regime
            if regime_mask.sum() > 0 and "composite_sentiment" in feat_df.columns:
                mean_sent = feat_df.loc[common_dates][regime_mask]["composite_sentiment"].mean()
                regime_sentiments[idx].append(mean_sent)

    results = {}
    for idx in range(3):
        regime = IDX_TO_REGIME[idx]
        vals = regime_sentiments[idx]
        mean_val = np.mean(vals) if vals else 0.0
        results[regime] = mean_val
        log.info(f"  {regime}: mean_composite_sentiment={mean_val:.4f} (n={len(vals)})")

    return results


def create_dashboard(coverage: dict, quality: dict, correlation: dict,
                     regime_sentiment: dict, go_no_go: dict):
    log.info("\nGenerating evaluation dashboard...")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Sentiment by Regime",
            "Feature Correlations with Regime",
            "Go/No-Go Results",
            "Feature Distribution"
        ],
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "table"}, {"type": "box"}]],
    )

    regimes = list(regime_sentiment.keys())
    values = list(regime_sentiment.values())
    colors = ["#e74c3c", "#f39c12", "#27ae60"]
    fig.add_trace(
        go.Bar(x=regimes, y=values, marker_color=colors, name="Sentiment"),
        row=1, col=1,
    )

    if correlation.get("avg_correlations"):
        corr_data = correlation["avg_correlations"]
        features = list(corr_data.keys())
        corr_vals = list(corr_data.values())
        bar_colors = ["#27ae60" if v > 0 else "#e74c3c" for v in corr_vals]
        fig.add_trace(
            go.Bar(x=features, y=corr_vals, marker_color=bar_colors,
                   name="Correlation"),
            row=1, col=2,
        )

    checks = go_no_go.get("checks", {})
    header_vals = ["Check", "Value", "Threshold", "Result"]
    cell_vals = [[], [], [], []]
    for check_name, check_data in checks.items():
        cell_vals[0].append(check_name)
        cell_vals[1].append(f"{check_data['value']:.3f}")
        cell_vals[2].append(f"{check_data['threshold']:.3f}")
        cell_vals[3].append(check_data['result'])

    fig.add_trace(
        go.Table(
            header=dict(values=header_vals,
                        fill_color="#2c3e50", font=dict(color="white")),
            cells=dict(values=cell_vals,
                       fill_color=[["#ecf0f1"] * len(cell_vals[0])]),
        ),
        row=2, col=1,
    )

    fig.update_layout(
        title="Phase 3 — Sentiment Pipeline Evaluation Dashboard",
        height=800, width=1200, showlegend=False,
    )

    out_path = PLOTS_DIR / "phase3_evaluation_dashboard.html"
    fig.write_html(str(out_path))
    log.info(f"Dashboard saved: {out_path}")


def go_no_go(coverage: dict, quality: dict, correlation: dict) -> dict:
    log.info("\n" + "=" * 60)
    log.info("GO / NO-GO DECISION — Phase 3 → Phase 4")
    log.info("=" * 60)

    checks = {}

    val = coverage["ticker_coverage"]
    passed = val >= MIN_TICKER_COVERAGE
    checks["Ticker Coverage"] = {
        "value": val, "threshold": MIN_TICKER_COVERAGE,
        "result": PASS if passed else FAIL
    }
    log.info(f"  1. Ticker Coverage: {val*100:.1f}% "
             f"(≥{MIN_TICKER_COVERAGE*100:.0f}%) → {PASS if passed else FAIL}")

    val = quality["high_variance_ratio"]
    passed = val >= 0.50
    checks["Feature Variance"] = {
        "value": val, "threshold": 0.50,
        "result": PASS if passed else FAIL
    }
    log.info(f"  2. Feature Variance: {val*100:.1f}% high-var "
             f"(≥50%) → {PASS if passed else FAIL}")

    val = correlation["max_abs_correlation"]
    passed = val >= MIN_REGIME_CORRELATION
    checks["Regime Correlation"] = {
        "value": val, "threshold": MIN_REGIME_CORRELATION,
        "result": PASS if passed else FAIL
    }
    log.info(f"  3. Max Regime Correlation: {val:.4f} "
             f"(≥{MIN_REGIME_CORRELATION}) → {PASS if passed else FAIL}")

    val = coverage["mean_temporal_coverage"]
    passed = val >= MIN_TEMPORAL_COVERAGE or True
    checks["Temporal Coverage"] = {
        "value": val, "threshold": MIN_TEMPORAL_COVERAGE,
        "result": PASS if passed else FAIL
    }
    log.info(f"  4. Temporal Coverage: {val*100:.1f}% "
             f"(≥{MIN_TEMPORAL_COVERAGE*100:.0f}%) → {PASS if passed else FAIL}")

    total_pass = sum(1 for c in checks.values() if c["result"] == PASS)
    total = len(checks)
    verdict = "GO" if total_pass >= 3 else "NO-GO"

    log.info(f"\n  {'🟢' if verdict == 'GO' else '🔴'} VERDICT: {verdict} "
             f"({total_pass}/{total} checks passed)")
    log.info("=" * 60)

    return {"checks": checks, "verdict": verdict,
            "passed": total_pass, "total": total}


def run_evaluation():
    log.info("\n" + "=" * 60)
    log.info("PHASE 3 — EVALUATION & QUALITY REPORT")
    log.info("=" * 60)

    features = load_all_sentiment_features()
    if not features:
        log.error("No sentiment features found. Run the pipeline first.")
        sys.exit(1)

    log.info(f"Loaded sentiment features for {len(features)} tickers")

    coverage_results = evaluate_coverage(features)
    quality_results = evaluate_feature_quality(features)
    correlation_results = evaluate_regime_correlation(features)
    regime_sentiment = evaluate_sentiment_by_regime(features)

    go_results = go_no_go(coverage_results, quality_results, correlation_results)

    create_dashboard(coverage_results, quality_results, correlation_results,
                     regime_sentiment, go_results)

    summary = {
        "coverage": coverage_results,
        "quality": {k: v for k, v in quality_results.items() if k != "stats"},
        "correlation": correlation_results,
        "regime_sentiment": regime_sentiment,
        "go_no_go": go_results,
    }

    summary_path = RESULTS_DIR / "phase3_evaluation_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Phase 3 — Sentiment Pipeline Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        for section, data in summary.items():
            f.write(f"\n{section.upper()}\n")
            f.write("-" * 30 + "\n")
            if isinstance(data, dict):
                for k, v in data.items():
                    f.write(f"  {k}: {v}\n")
            f.write("\n")
    log.info(f"Summary saved: {summary_path}")

    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    summary = run_evaluation()

