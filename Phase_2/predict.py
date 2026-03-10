
import argparse
import numpy as np
import pandas as pd
import torch
import pickle
import logging
import sys
from pathlib import Path

from config import (
    DEVICE, CHECKPOINT_DIR, OUTPUT_DIR,
    IDX_TO_REGIME, FEATURE_COLUMNS, WINDOW_SIZE,
    PHASE1_LABEL_DIR, TICKER_TO_IDX,
)
from model import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def load_model_and_scaler():
    ckpt_path = CHECKPOINT_DIR / "best_model.pt"
    if not ckpt_path.exists():
        log.error(f"No checkpoint at {ckpt_path}. Run train.py first.")
        sys.exit(1)

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = build_model(checkpoint["num_features"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    scaler_path = OUTPUT_DIR / "feature_scaler.pkl"
    if not scaler_path.exists():
        log.error(f"No scaler at {scaler_path}. Run training to generate it.")
        sys.exit(1)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    log.info(f"Loaded model (epoch {checkpoint['epoch']}) and scaler.")
    return model, scaler, checkpoint["num_features"]


def prepare_input(df: pd.DataFrame, scaler, num_features: int,
                  window_size: int = WINDOW_SIZE) -> torch.Tensor:
    available_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    if len(available_cols) < 5:
        raise ValueError(f"Too few features available: {available_cols}")

    features = df[available_cols].tail(window_size).values.astype(np.float32)

    features_df = pd.DataFrame(features)
    features_df = features_df.ffill().bfill()
    features = features_df.values.astype(np.float32)

    if len(features) < window_size:
        raise ValueError(f"Not enough data: need {window_size} rows, got {len(features)}")

    scaled = scaler.transform(features)

    tensor = torch.FloatTensor(scaled).unsqueeze(0)
    return tensor


@torch.no_grad()
def predict_regime(model, input_tensor: torch.Tensor, ticker: str = None) -> dict:
    model.eval()
    input_tensor = input_tensor.to(DEVICE)

    stock_idx = TICKER_TO_IDX.get(ticker, 0) if ticker else 0
    stock_ids = torch.LongTensor([stock_idx]).to(DEVICE)

    output = model(input_tensor, stock_ids=stock_ids)

    probs     = output["regime_probs"][0].cpu().numpy()
    pred_idx  = probs.argmax()
    regime    = IDX_TO_REGIME[pred_idx]
    confidence = probs[pred_idx]

    trans_prob = output["transition_prob"][0].item()

    return {
        "predicted_regime": regime,
        "confidence":       confidence,
        "probabilities":    {IDX_TO_REGIME[i]: float(probs[i]) for i in range(len(probs))},
        "transition_prob":  trans_prob,
        "regime_stable":    trans_prob < 0.5,
    }


def predict_ticker(ticker: str, model=None, scaler=None):
    if model is None or scaler is None:
        model, scaler, num_features = load_model_and_scaler()
    else:
        num_features = model.num_features

    path = PHASE1_LABEL_DIR / f"{ticker}_labelled.csv"
    if not path.exists():
        log.error(f"No labelled data for {ticker}. Run Phase 1 first.")
        return None

    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    input_tensor = prepare_input(df, scaler, num_features)
    result = predict_regime(model, input_tensor, ticker=ticker)

    result["ticker"] = ticker
    result["latest_date"] = str(df.index[-1].date())
    result["latest_close"] = float(df["Close"].iloc[-1])

    if "Regime" in df.columns:
        result["hmm_regime"] = df["Regime"].iloc[-1]

    return result


def predict_all(tickers: list[str]):
    model, scaler, num_features = load_model_and_scaler()

    results = []
    for ticker in tickers:
        try:
            result = predict_ticker(ticker, model, scaler)
            if result:
                results.append(result)
        except Exception as e:
            log.error(f"[{ticker}] Prediction failed: {e}")

    return results


def print_predictions(results: list[dict]):
    print("\n" + "═" * 80)
    print("  🔮  REGIME PREDICTIONS")
    print("═" * 80)
    print(f"  {'Ticker':<18} {'Regime':<12} {'Conf':>6} {'P(Bull)':>8} {'P(Bear)':>8} "
          f"{'P(Side)':>8} {'Trans':>6} {'HMM':>10}")
    print("  " + "─" * 78)

    for r in results:
        label = r["ticker"].replace(".NS", "")
        regime = r["predicted_regime"]
        conf   = r["confidence"]
        p_bull = r["probabilities"]["Bull"]
        p_bear = r["probabilities"]["Bear"]
        p_side = r["probabilities"]["Sideways"]
        trans  = r["transition_prob"]
        hmm    = r.get("hmm_regime", "N/A")

        icon = {"Bull": "🟢", "Bear": "🔴", "Sideways": "🟡"}.get(regime, "⚪")

        print(f"  {icon} {label:<16} {regime:<12} {conf:>5.1%} {p_bull:>8.3f} {p_bear:>8.3f} "
              f"{p_side:>8.3f} {trans:>5.1%} {hmm:>10}")

    regime_counts = {}
    for r in results:
        regime = r["predicted_regime"]
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

    print("\n  " + "─" * 78)
    print(f"  Summary: {len(results)} stocks | " +
          " | ".join(f"{k}: {v}" for k, v in sorted(regime_counts.items())))

    agree = sum(1 for r in results if r.get("hmm_regime") == r["predicted_regime"])
    print(f"  HMM Agreement: {agree}/{len(results)} ({100*agree/len(results):.1f}%)")
    print("═" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Phase 2 — Regime Prediction")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Single ticker (e.g., RELIANCE.NS)")
    parser.add_argument("--all", action="store_true",
                        help="Predict all Nifty 50 stocks")
    args = parser.parse_args()

    if args.ticker:
        ticker = args.ticker if args.ticker.endswith(".NS") else args.ticker + ".NS"
        result = predict_ticker(ticker)
        if result:
            print_predictions([result])
    elif args.all:
        from config import NIFTY_50_TICKERS
        results = predict_all(NIFTY_50_TICKERS)
        print_predictions(results)
    else:
        from config import NIFTY_50_TICKERS
        results = predict_all(NIFTY_50_TICKERS[:10])
        print_predictions(results)


if __name__ == "__main__":
    main()

