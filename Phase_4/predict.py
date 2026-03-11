
import argparse
import pickle
import logging
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from config import (
    DEVICE, CHECKPOINT_DIR, RESULTS_DIR, NIFTY_50_TICKERS,
    IDX_TO_REGIME, NUM_CLASSES, TICKER_TO_IDX,
    PRICE_FEATURE_COLUMNS, SENTIMENT_FEATURE_COLUMNS,
    NEWS_FEATURE_COLUMNS, INCLUDE_NEWS_FEATURES,
    PHASE1_LABEL_DIR, PHASE3_SENTIMENT_DIR, PHASE3_MARKET_DIR,
    WINDOW_SIZE,
)
from model import build_model

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def load_model_and_scalers():
    ckpt = torch.load(CHECKPOINT_DIR / "best_model.pt",
                      map_location=DEVICE, weights_only=False)

    model = build_model(
        ckpt["num_price_features"], ckpt["num_sent_features"]
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    price_scaler = pickle.load(open(RESULTS_DIR / "price_scaler.pkl", "rb"))
    sent_scaler  = pickle.load(open(RESULTS_DIR / "sent_scaler.pkl", "rb"))

    return model, price_scaler, sent_scaler, ckpt


def prepare_ticker_data(ticker: str, price_scaler, sent_scaler):

    label_path = PHASE1_LABEL_DIR / f"{ticker}_labelled.csv"
    if not label_path.exists():
        return None, None, None

    df_price = pd.read_csv(label_path, index_col=0, parse_dates=True)

    sent_path = PHASE3_SENTIMENT_DIR / f"{ticker}_sentiment.csv"
    if sent_path.exists():
        df_sent = pd.read_csv(sent_path, index_col=0, parse_dates=True)
    else:
        df_sent = pd.read_csv(PHASE3_MARKET_DIR / "market_features.csv",
                              index_col=0, parse_dates=True)

    common_dates = df_price.index.intersection(df_sent.index)
    if len(common_dates) < WINDOW_SIZE:
        log.warning(f"{ticker}: insufficient aligned data "
                    f"({len(common_dates)} < {WINDOW_SIZE})")
        return None, None, None

    df_price = df_price.loc[common_dates]
    df_sent  = df_sent.loc[common_dates]

    sent_cols = list(SENTIMENT_FEATURE_COLUMNS)
    if INCLUDE_NEWS_FEATURES:
        sent_cols += [c for c in NEWS_FEATURE_COLUMNS if c in df_sent.columns]

    price_raw = df_price[PRICE_FEATURE_COLUMNS].values
    sent_raw  = df_sent[sent_cols].values

    price_scaled = price_scaler.transform(
        np.nan_to_num(price_raw, nan=0.0, posinf=0.0, neginf=0.0)
    )
    sent_scaled = sent_scaler.transform(
        np.nan_to_num(sent_raw, nan=0.0, posinf=0.0, neginf=0.0)
    )

    price_window = price_scaled[-WINDOW_SIZE:]
    sent_window  = sent_scaled[-WINDOW_SIZE:]
    last_date    = common_dates[-1]

    return price_window, sent_window, last_date


@torch.no_grad()
def predict_ticker(model, ticker, price_scaler, sent_scaler):
    price_window, sent_window, last_date = prepare_ticker_data(
        ticker, price_scaler, sent_scaler
    )
    if price_window is None:
        return None

    X_price = torch.tensor(price_window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    X_sent  = torch.tensor(sent_window,  dtype=torch.float32).unsqueeze(0).to(DEVICE)

    stock_id = TICKER_TO_IDX.get(ticker, 0)
    stock_ids = torch.tensor([stock_id], dtype=torch.long).to(DEVICE)

    output = model(X_price, X_sent, stock_ids=stock_ids)

    probs = output["regime_probs"].cpu().numpy()[0]
    pred  = probs.argmax()
    conf  = probs[pred]
    trans_prob = output["transition_prob"].cpu().item()

    return {
        "ticker": ticker,
        "date": str(last_date.date()) if hasattr(last_date, 'date') else str(last_date),
        "predicted_regime": IDX_TO_REGIME[pred],
        "confidence": conf,
        "probabilities": {IDX_TO_REGIME[i]: float(probs[i])
                          for i in range(NUM_CLASSES)},
        "transition_probability": trans_prob,
    }


def predict_all(tickers=NIFTY_50_TICKERS):

    model, price_scaler, sent_scaler, ckpt = load_model_and_scalers()

    print("\n" + "=" * 80)
    print(f"  PHASE 4 — SENTIMENT-ENRICHED REGIME PREDICTIONS")
    print(f"  Model from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.4f})")
    print("=" * 80)

    results = []
    for ticker in tickers:
        pred = predict_ticker(model, ticker, price_scaler, sent_scaler)
        if pred is None:
            print(f"  {ticker:<20} -- skipped (no data)")
            continue

        results.append(pred)
        regime = pred["predicted_regime"]
        conf   = pred["confidence"]
        trans  = pred["transition_probability"]

        flag = "!" if trans > 0.5 else " "
        conf_bar = "#" * int(conf * 20)
        print(f"  {ticker:<20} {regime:<10} "
              f"conf={conf:.3f} [{conf_bar:<20}] "
              f"trans={trans:.3f} {flag}")

    if results:
        df = pd.DataFrame(results)
        out_path = RESULTS_DIR / "predictions.csv"
        df.to_csv(out_path, index=False)
        log.info(f"Saved predictions to {out_path}")

        print(f"\n  Regime Distribution:")
        for regime in IDX_TO_REGIME.values():
            count = sum(1 for r in results if r["predicted_regime"] == regime)
            pct = count / len(results) * 100
            print(f"    {regime:<10}: {count:3d} ({pct:.1f}%)")

        trans_alerts = [r for r in results if r["transition_probability"] > 0.5]
        if trans_alerts:
            print(f"\n  Transition Alerts ({len(trans_alerts)}):")
            for r in sorted(trans_alerts,
                            key=lambda x: x["transition_probability"],
                            reverse=True):
                print(f"    {r['ticker']:<20} "
                      f"trans_prob={r['transition_probability']:.3f}")

    print("\n" + "=" * 80 + "\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=NIFTY_50_TICKERS)
    args = parser.parse_args()
    predict_all(args.tickers)

