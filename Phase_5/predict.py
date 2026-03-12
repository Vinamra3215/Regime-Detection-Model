
import numpy as np
import pandas as pd
import torch
import pickle
import logging

from config import (
    DEVICE, MC_SAMPLES, RESULTS_DIR,
    NIFTY_50_TICKERS, TICKER_TO_IDX,
    IDX_TO_REGIME, PHASE4_CHECKPOINT,
    PHASE4_PRICE_SCALER, PHASE4_SENT_SCALER,
    PRICE_FEATURE_COLUMNS, SENTIMENT_FEATURE_COLUMNS,
    NEWS_FEATURE_COLUMNS,
    PHASE1_LABEL_DIR, PHASE3_SENTIMENT_DIR,
    WINDOW_SIZE, CHECKPOINT_DIR,
    HIGH_CONF_THRESHOLD, UNCERTAINTY_THRESHOLD, ABSTAIN_THRESHOLD,
)

from mc_dropout import load_phase4_model, mc_predict_batch, compute_uncertainty_metrics

log = logging.getLogger(__name__)


def predict_ticker(ticker, model, price_scaler, sent_scaler,
                   n_samples=MC_SAMPLES, expected_sent=7):
    label_path = PHASE1_LABEL_DIR / f"{ticker}_labelled.csv"
    sent_path  = PHASE3_SENTIMENT_DIR / f"{ticker}_sentiment.csv"

    if not label_path.exists():
        return None

    df = pd.read_csv(label_path, index_col="Date", parse_dates=True)

    all_sent = list(SENTIMENT_FEATURE_COLUMNS) + list(NEWS_FEATURE_COLUMNS)
    sent_cols = all_sent[:expected_sent]

    if sent_path.exists():
        sent_df = pd.read_csv(sent_path, index_col=0, parse_dates=True)
        available = [c for c in sent_cols if c in sent_df.columns]
        if available:
            df = df.join(sent_df[available], how="left")

    for col in sent_cols:
        if col not in df.columns:
            df[col] = 0.0
    df = df.fillna(0.0)

    price_cols = [c for c in PRICE_FEATURE_COLUMNS if c in df.columns]

    last_window = df.iloc[-WINDOW_SIZE:]
    if len(last_window) < WINDOW_SIZE:
        return None

    price_data = np.nan_to_num(last_window[price_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    sent_data  = np.nan_to_num(last_window[sent_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    price_scaled = price_scaler.transform(price_data.reshape(-1, len(price_cols))).reshape(1, WINDOW_SIZE, len(price_cols))
    sent_scaled  = sent_scaler.transform(sent_data.reshape(-1, len(sent_cols))).reshape(1, WINDOW_SIZE, len(sent_cols))

    x_price = torch.FloatTensor(price_scaled).to(DEVICE)
    x_sent  = torch.FloatTensor(sent_scaled).to(DEVICE)
    stock_id = torch.LongTensor([TICKER_TO_IDX.get(ticker, 0)]).to(DEVICE)

    regime_samples, trans_samples = mc_predict_batch(model, x_price, x_sent, stock_id, n_samples)
    metrics = compute_uncertainty_metrics(regime_samples)

    pred_class = int(metrics["predicted_class"][0])
    confidence = float(metrics["confidence"][0])
    mi = float(metrics["mutual_information"][0])

    if confidence >= HIGH_CONF_THRESHOLD and mi < UNCERTAINTY_THRESHOLD:
        signal = "CONFIDENT"
    elif mi >= ABSTAIN_THRESHOLD:
        signal = "ABSTAIN"
    else:
        signal = "LOW_CONF"

    return {
        "ticker": ticker,
        "date": str(df.index[-1].date()),
        "predicted_regime": IDX_TO_REGIME[pred_class],
        "confidence": confidence,
        "epistemic_uncertainty": mi,
        "transition_probability": float(trans_samples.mean()),
        "signal": signal,
        "probabilities": {IDX_TO_REGIME[i]: float(metrics["mean_probs"][0][i]) for i in range(3)},
    }


def predict_all(n_samples=MC_SAMPLES):
    model, num_price, num_sent = load_phase4_model()

    with open(PHASE4_PRICE_SCALER, "rb") as f:
        price_scaler = pickle.load(f)
    with open(PHASE4_SENT_SCALER, "rb") as f:
        sent_scaler = pickle.load(f)

    results = []
    for ticker in NIFTY_50_TICKERS:
        pred = predict_ticker(ticker, model, price_scaler, sent_scaler,
                              n_samples, expected_sent=num_sent)
        if pred:
            results.append(pred)
            emoji = {"CONFIDENT": "G", "LOW_CONF": "Y", "ABSTAIN": "R"}
            print(f"  [{emoji.get(pred['signal'], '?')}] "
                  f"{ticker:20s} -> {pred['predicted_regime']:>10s} "
                  f"(conf={pred['confidence']:.2f}, unc={pred['epistemic_uncertainty']:.4f}) "
                  f"[{pred['signal']}]")

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "predictions_with_uncertainty.csv", index=False)

    confident = sum(1 for r in results if r["signal"] == "CONFIDENT")
    low_conf  = sum(1 for r in results if r["signal"] == "LOW_CONF")
    abstain   = sum(1 for r in results if r["signal"] == "ABSTAIN")
    print(f"\n  Summary: {confident} confident, {low_conf} low-conf, {abstain} abstain out of {len(results)} stocks")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    predict_all()

