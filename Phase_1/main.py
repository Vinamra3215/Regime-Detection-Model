import argparse
import logging
import time
import sys
from pathlib import Path
from datetime import datetime

_base_dir = Path(__file__).resolve().parent
_project_dir = _base_dir.parent
_log_dir = _project_dir / "results" / "phase_1" / "logs"
_log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(_log_dir / "pipeline.log"), mode="w"),
    ]
)
log = logging.getLogger(__name__)

from config import NIFTY_50_TICKERS, RESULTS_DIR
from data_download import get_data
from feature_engineering import compute_features
from hmm_labeler import run_labeling, load_labelled_data
from visualize import generate_all_plots

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1 — Regime Labeling Pipeline")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--per-ticker-plots", type=int, default=10)
    return parser.parse_args()

def print_banner():
    print("\n" + "═" * 65)
    print("  🧠  PHASE 1 — HMM REGIME LABELING PIPELINE")
    print(f"       Nifty 50 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 65 + "\n")

def print_summary(labelled: dict, elapsed: float):
    print("\n" + "═" * 65)
    print(f"  📊  REGIME LABELING SUMMARY — {len(labelled)} Stocks")
    print("═" * 65)
    print(f"  {'Ticker':<25} {'Bull%':>7} {'Bear%':>7} {'Sideways%':>10}  {'Days':>6}")
    print("  " + "─" * 58)

    totals = {"Bull": 0, "Bear": 0, "Sideways": 0, "Days": 0}

    for ticker, df in labelled.items():
        if "Regime" not in df.columns:
            continue
        counts = df["Regime"].value_counts()
        total  = len(df)
        bull   = counts.get("Bull", 0)
        bear   = counts.get("Bear", 0)
        side   = counts.get("Sideways", 0)
        label  = ticker.replace(".NS", "").replace("^", "")

        print(f"  {label:<25} {100*bull/total:>6.1f}% {100*bear/total:>6.1f}% {100*side/total:>9.1f}%  {total:>6}")

        totals["Bull"]     += bull
        totals["Bear"]     += bear
        totals["Sideways"] += side
        totals["Days"]     += total

    total_all = totals["Days"]
    if total_all > 0:
        print("  " + "─" * 58)
        print(f"  {'AVERAGE':<25} "
              f"{100*totals['Bull']/total_all:>6.1f}% "
              f"{100*totals['Bear']/total_all:>6.1f}% "
              f"{100*totals['Sideways']/total_all:>9.1f}%  {total_all:>6}")

    print("═" * 65)
    print(f"  ⏱  Total time: {elapsed:.1f}s")
    print("═" * 65 + "\n")

def main():
    args = parse_args()
    print_banner()
    t0 = time.time()

    if args.ticker:
        tickers = [args.ticker if args.ticker.endswith(".NS") else args.ticker + ".NS"]
    else:
        tickers = NIFTY_50_TICKERS

    print("\n📥  STEP 1/4 — Data Download")
    print("─" * 50)
    t1       = time.time()
    raw_data = get_data(tickers=tickers, force_download=args.force_download)
    log.info(f"Download complete in {time.time()-t1:.1f}s — {len(raw_data)} tickers loaded.")

    if not raw_data:
        log.error("No data loaded. Exiting.")
        sys.exit(1)

    print("\n⚙️   STEP 2/4 — Feature Engineering")
    print("─" * 50)
    t2          = time.time()
    feature_dfs = {}
    failed_fe   = []

    for ticker, df in raw_data.items():
        if ticker == "^NSEI":
            continue
        feat_df = compute_features(df, ticker)
        if feat_df is not None:
            feature_dfs[ticker] = feat_df
        else:
            failed_fe.append(ticker)

    log.info(f"Feature engineering complete in {time.time()-t2:.1f}s")
    log.info(f"  ✅ Success: {len(feature_dfs)} | ❌ Skipped: {len(failed_fe)}")

    if not feature_dfs:
        log.error("No features computed. Exiting.")
        sys.exit(1)

    print("\n🧠  STEP 3/4 — HMM Regime Labeling")
    print("─" * 50)
    t3       = time.time()
    labelled = run_labeling(feature_dfs, save_models=True)
    log.info(f"HMM labeling complete in {time.time()-t3:.1f}s")

    if not labelled:
        log.error("No tickers successfully labelled. Exiting.")
        sys.exit(1)

    if not args.skip_plots:
        print("\n📊  STEP 4/4 — Generating Plotly Visualisations")
        print("─" * 50)
        t4 = time.time()
        generate_all_plots(labelled, per_ticker_limit=args.per_ticker_plots)
        log.info(f"Visualisation complete in {time.time()-t4:.1f}s")
    else:
        log.info("Skipping plots (--skip-plots flag set).")

    elapsed = time.time() - t0
    print_summary(labelled, elapsed)

    print(f"\n✅  Phase 1 complete! Output files:")
    print(f"   • Labelled CSVs : {RESULTS_DIR}/data/labelled/")
    print(f"   • HMM models    : {RESULTS_DIR}/data/models/")
    print(f"   • Plotly charts : {RESULTS_DIR}/plots/")
    print(f"   • Pipeline log  : {RESULTS_DIR}/logs/pipeline.log\n")

if __name__ == "__main__":
    main()
