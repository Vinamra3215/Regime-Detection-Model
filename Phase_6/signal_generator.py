
import numpy as np
import pandas as pd
import logging

from config import (
    HIGH_CONF_THRESHOLD, MED_CONF_THRESHOLD,
    UNCERTAINTY_THRESHOLD, ABSTAIN_THRESHOLD,
    TRANSITION_PROB_EXIT, TRANSITION_PROB_WARN,
    MIN_HOLDING_DAYS,
)

log = logging.getLogger(__name__)


def generate_raw_signal(regime, confidence, uncertainty, transition_prob=0.0):
    if uncertainty >= ABSTAIN_THRESHOLD:
        return "FLAT", 0.0

    if transition_prob >= TRANSITION_PROB_EXIT:
        return "FLAT", 0.0

    if regime == "Sideways":
        return "FLAT", 0.0

    if regime == "Bull":
        if confidence >= HIGH_CONF_THRESHOLD and uncertainty < UNCERTAINTY_THRESHOLD:
            strength = min(confidence, 1.0)
            if transition_prob >= TRANSITION_PROB_WARN:
                return "WEAK_LONG", strength * 0.5
            return "STRONG_LONG", strength
        elif confidence >= MED_CONF_THRESHOLD:
            return "WEAK_LONG", confidence * 0.5
        else:
            return "FLAT", 0.0

    if regime == "Bear":
        if confidence >= HIGH_CONF_THRESHOLD and uncertainty < UNCERTAINTY_THRESHOLD:
            strength = min(confidence, 1.0)
            if transition_prob >= TRANSITION_PROB_WARN:
                return "WEAK_SHORT", -strength * 0.5
            return "STRONG_SHORT", -strength
        elif confidence >= MED_CONF_THRESHOLD:
            return "WEAK_SHORT", -confidence * 0.5
        else:
            return "FLAT", 0.0

    return "FLAT", 0.0


def apply_holding_filter(signals_df, min_hold=MIN_HOLDING_DAYS):
    filtered = signals_df.copy()
    signals = filtered["signal"].values.copy()
    n = len(signals)

    i = 0
    while i < n:
        current = signals[i]
        j = i + 1
        while j < n and signals[j] == current:
            j += 1

        duration = j - i
        if duration < min_hold and current != "FLAT" and j < n:
            for k in range(i, min(i + min_hold, n)):
                signals[k] = current

        i = j

    filtered["signal"] = signals
    return filtered


def generate_signals_for_ticker(ticker, predictions_df):
    df = predictions_df.copy()
    signals = []
    strengths = []

    for _, row in df.iterrows():
        regime = row.get("predicted_regime", "Sideways")
        conf   = row.get("confidence", 0.0)
        unc    = row.get("epistemic_uncertainty", 1.0)
        trans  = row.get("transition_probability", 0.0)

        signal, strength = generate_raw_signal(regime, conf, unc, trans)
        signals.append(signal)
        strengths.append(strength)

    df["signal"] = signals
    df["signal_strength"] = strengths

    df = apply_holding_filter(df)

    return df


def generate_signals_snapshot(predictions_csv_path):
    df = pd.read_csv(predictions_csv_path)

    signals = []
    for _, row in df.iterrows():
        regime = row.get("predicted_regime", "Sideways")
        conf   = row.get("confidence", 0.0)
        unc    = row.get("epistemic_uncertainty", 1.0)
        trans  = row.get("transition_probability", 0.0)

        signal, strength = generate_raw_signal(regime, conf, unc, trans)
        signals.append({
            "ticker": row["ticker"],
            "date": row.get("date", ""),
            "predicted_regime": regime,
            "confidence": conf,
            "epistemic_uncertainty": unc,
            "transition_probability": trans,
            "signal": signal,
            "signal_strength": strength,
        })

    result = pd.DataFrame(signals)

    counts = result["signal"].value_counts()
    log.info(f"Signal Distribution: {dict(counts)}")

    return result

