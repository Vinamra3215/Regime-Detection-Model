
import sys
import importlib.util
import numpy as np
import pandas as pd
import torch
import pickle
import logging
from pathlib import Path

from config import (
    DEVICE, WINDOW_SIZE,
    PHASE4_CHECKPOINT, PHASE4_PRICE_SCALER, PHASE4_SENT_SCALER,
    PHASE5_TEMPERATURE,
    PHASE1_LABEL_DIR, PHASE3_SENTIMENT_DIR,
    PRICE_FEATURE_COLUMNS, SENTIMENT_FEATURE_COLUMNS,
    NIFTY_50_TICKERS, TICKER_TO_IDX,
    REGIME_TO_IDX, IDX_TO_REGIME,
    HIGH_CONF_THRESHOLD, MED_CONF_THRESHOLD,
    UNCERTAINTY_THRESHOLD, EVAL_START, EVAL_END,
)

log = logging.getLogger(__name__)

PHASE4_DIR = Path(__file__).resolve().parent.parent / "Phase_4"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_model():
    p5_config = sys.modules.get("config")

    p4_config = _load_module("config", str(PHASE4_DIR / "config.py"))
    p4_model = _load_module("phase4_model", str(PHASE4_DIR / "model.py"))

    if p5_config is not None:
        sys.modules["config"] = p5_config

    ckpt = torch.load(PHASE4_CHECKPOINT, map_location=DEVICE, weights_only=False)
    num_price = ckpt["num_price_features"]
    num_sent = ckpt["num_sent_features"]

    model = p4_model.build_model(num_price, num_sent)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)

    with open(PHASE4_PRICE_SCALER, "rb") as f:
        price_scaler = pickle.load(f)
    with open(PHASE4_SENT_SCALER, "rb") as f:
        sent_scaler = pickle.load(f)

    temperature = 1.0
    if PHASE5_TEMPERATURE.exists():
        temp_data = torch.load(PHASE5_TEMPERATURE, map_location="cpu", weights_only=False)
        temperature = temp_data["temperature"]

    log.info(f"Model loaded: {num_price} price, {num_sent} sent features, T={temperature:.4f}")
    return model, price_scaler, sent_scaler, num_sent, temperature


def mc_predict_single(model, x_price, x_sent, stock_id, n_samples=20):
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

    regime_samples = []
    trans_samples = []

    with torch.no_grad():
        for _ in range(n_samples):
            out = model(x_price, x_sent, stock_ids=stock_id)
            regime_samples.append(out["regime_probs"].cpu().numpy())
            trans_samples.append(out["transition_prob"].squeeze(-1).cpu().numpy())

    regime_arr = np.stack(regime_samples, axis=0)
    mean_probs = regime_arr.mean(axis=0)[0]

    eps = 1e-10
    pred_entropy = -np.sum(mean_probs * np.log(mean_probs + eps))
    indiv_entropy = -np.sum(regime_arr * np.log(regime_arr + eps), axis=2).mean(axis=0)[0]
    mutual_info = max(pred_entropy - indiv_entropy, 0.0)

    pred_class = int(np.argmax(mean_probs))
    confidence = float(mean_probs[pred_class])
    transition_prob = float(np.mean(trans_samples))

    return {
        "predicted_regime": IDX_TO_REGIME[pred_class],
        "confidence": confidence,
        "epistemic_uncertainty": mutual_info,
        "transition_probability": transition_prob,
        "probs": mean_probs,
    }


def generate_signal(pred):
    regime = pred["predicted_regime"]
    conf = pred["confidence"]
    unc = pred["epistemic_uncertainty"]
    trans = pred["transition_probability"]

    if unc >= UNCERTAINTY_THRESHOLD * 1.5:
        return "FLAT", 0.0

    if regime == "Sideways" or trans >= 0.70:
        return "FLAT", 0.0

    if regime == "Bull":
        if conf >= HIGH_CONF_THRESHOLD and unc < UNCERTAINTY_THRESHOLD:
            return "STRONG_LONG", 1.0
        elif conf >= MED_CONF_THRESHOLD:
            return "WEAK_LONG", 0.5
        return "FLAT", 0.0

    if regime == "Bear":
        if conf >= HIGH_CONF_THRESHOLD and unc < UNCERTAINTY_THRESHOLD:
            return "STRONG_SHORT", -1.0
        elif conf >= MED_CONF_THRESHOLD:
            return "WEAK_SHORT", -0.5
        return "FLAT", 0.0

    return "FLAT", 0.0


def generate_daily_signals(model, price_scaler, sent_scaler, num_sent, temperature):
    eval_start = pd.Timestamp(EVAL_START)
    eval_end = pd.Timestamp(EVAL_END)

    all_sent_cols = list(SENTIMENT_FEATURE_COLUMNS)
    sent_cols = all_sent_cols[:num_sent]

    all_results = {}
    total_tickers = len(NIFTY_50_TICKERS)

    for t_idx, ticker in enumerate(NIFTY_50_TICKERS):
        label_path = PHASE1_LABEL_DIR / f"{ticker}_labelled.csv"
        sent_path = PHASE3_SENTIMENT_DIR / f"{ticker}_sentiment.csv"

        if not label_path.exists():
            continue

        df = pd.read_csv(label_path, index_col="Date", parse_dates=True)

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
        if len(price_cols) < 5 or "Regime" not in df.columns:
            continue

        eval_mask = (df.index >= eval_start) & (df.index <= eval_end)
        eval_indices = np.where(eval_mask)[0]

        if len(eval_indices) == 0:
            continue

        daily_records = []

        for day_idx in eval_indices:
            if day_idx < WINDOW_SIZE:
                continue

            window_start = day_idx - WINDOW_SIZE
            window_end = day_idx

            price_data = df.iloc[window_start:window_end][price_cols].values.astype(np.float32)
            sent_data = df.iloc[window_start:window_end][sent_cols].values.astype(np.float32)

            price_data = np.nan_to_num(price_data, nan=0.0, posinf=0.0, neginf=0.0)
            sent_data = np.nan_to_num(sent_data, nan=0.0, posinf=0.0, neginf=0.0)

            price_scaled = price_scaler.transform(price_data).reshape(1, WINDOW_SIZE, len(price_cols))
            sent_scaled = sent_scaler.transform(sent_data).reshape(1, WINDOW_SIZE, len(sent_cols))

            price_scaled = np.nan_to_num(price_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            sent_scaled = np.nan_to_num(sent_scaled, nan=0.0, posinf=0.0, neginf=0.0)

            x_price = torch.FloatTensor(price_scaled).to(DEVICE)
            x_sent = torch.FloatTensor(sent_scaled).to(DEVICE)
            stock_id = torch.LongTensor([TICKER_TO_IDX.get(ticker, 0)]).to(DEVICE)

            pred = mc_predict_single(model, x_price, x_sent, stock_id, n_samples=20)
            signal, position = generate_signal(pred)

            actual_return = df.iloc[day_idx].get("log_return_1d", 0.0)
            true_regime = df.iloc[day_idx].get("Regime", "Unknown")

            daily_records.append({
                "date": df.index[day_idx],
                "ticker": ticker,
                "predicted_regime": pred["predicted_regime"],
                "true_regime": true_regime,
                "confidence": pred["confidence"],
                "epistemic_uncertainty": pred["epistemic_uncertainty"],
                "transition_probability": pred["transition_probability"],
                "signal": signal,
                "position_size": position,
                "actual_return": actual_return,
            })

        if daily_records:
            all_results[ticker] = pd.DataFrame(daily_records)

        if (t_idx + 1) % 10 == 0:
            log.info(f"  Processed {t_idx+1}/{total_tickers} tickers "
                     f"({len(daily_records)} days for {ticker})")

    log.info(f"Generated daily signals for {len(all_results)} tickers")
    return all_results

