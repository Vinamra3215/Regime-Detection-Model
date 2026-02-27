import numpy as np
import pandas as pd
import logging
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

from config import (
    HMM_N_STATES, HMM_N_ITER, HMM_RANDOM_STATE, HMM_COV_TYPE,
    REGIME_LABELS, DATA_LABEL_DIR
)
from feature_engineering import get_hmm_features

log = logging.getLogger(__name__)


def train_hmm(features: np.ndarray, ticker: str = "") -> tuple[GaussianHMM, StandardScaler]:
    scaler = StandardScaler()
    X      = scaler.fit_transform(features)

    model = GaussianHMM(
        n_components    = HMM_N_STATES,
        covariance_type = HMM_COV_TYPE,
        n_iter          = HMM_N_ITER,
        random_state    = HMM_RANDOM_STATE,
        verbose         = False,
    )
    model.fit(X)
    log.info(f"[{ticker}] HMM converged={model.monitor_.converged} | score={model.score(X):.2f}")
    return model, scaler


def map_states_to_regimes(model: GaussianHMM, feature_names: list[str]) -> dict[int, str]:
    try:
        ret_idx = feature_names.index("log_return_1d")
    except ValueError:
        ret_idx = 0

    state_means   = model.means_[:, ret_idx]
    sorted_states = np.argsort(state_means)

    state_map = {
        int(sorted_states[0]): "Bear",
        int(sorted_states[1]): "Sideways",
        int(sorted_states[2]): "Bull",
    }
    for state, regime in state_map.items():
        log.info(f"  State {state} → {regime} (mean_return={state_means[state]:.5f})")
    return state_map


def label_regimes(df: pd.DataFrame, model: GaussianHMM,
                  scaler: StandardScaler, feature_names: list[str],
                  ticker: str = "") -> pd.DataFrame:
    X_raw, _     = get_hmm_features(df)
    X_scaled     = scaler.transform(X_raw)
    hidden_states = model.predict(X_scaled)
    state_probs  = model.predict_proba(X_scaled)

    state_map = map_states_to_regimes(model, feature_names)

    df = df.copy()
    df["HMM_State"] = hidden_states
    df["Regime"]    = df["HMM_State"].map(state_map)

    for state_id, regime_name in state_map.items():
        df[f"prob_{regime_name}"] = state_probs[:, state_id]

    return df


def smooth_regimes(df: pd.DataFrame, min_days: int = 3) -> pd.DataFrame:
    df      = df.copy()
    regimes = df["Regime"].values.tolist()
    n       = len(regimes)

    run_start = 0
    for i in range(1, n + 1):
        if i == n or regimes[i] != regimes[i - 1]:
            run_len = i - run_start
            if run_len < min_days and run_start > 0:
                prev = regimes[run_start - 1]
                regimes[run_start:i] = [prev] * run_len
            run_start = i

    df["Regime"] = regimes
    return df


def run_labeling(feature_dfs: dict[str, pd.DataFrame],
                 save_models: bool = True) -> dict[str, pd.DataFrame]:
    labelled   = {}
    models_dir = DATA_LABEL_DIR.parent / "models"
    models_dir.mkdir(exist_ok=True)

    for ticker, df in feature_dfs.items():
        log.info(f"\n{'─'*50}")
        log.info(f"[{ticker}] Fitting HMM...")

        try:
            X_raw, feat_names = get_hmm_features(df)

            if len(X_raw) < 100:
                log.warning(f"[{ticker}] Too few rows for HMM. Skipping.")
                continue

            model, scaler = train_hmm(X_raw, ticker)
            labelled_df   = label_regimes(df, model, scaler, feat_names, ticker)
            labelled_df   = smooth_regimes(labelled_df, min_days=3)

            counts = labelled_df["Regime"].value_counts()
            total  = len(labelled_df)
            log.info(f"[{ticker}] Regime breakdown:")
            for regime in ["Bull", "Bear", "Sideways"]:
                c = counts.get(regime, 0)
                log.info(f"  {regime:>10}: {c:5d} days ({100*c/total:.1f}%)")

            save_path = DATA_LABEL_DIR / f"{ticker}_labelled.csv"
            labelled_df.to_csv(save_path)
            labelled[ticker] = labelled_df

            if save_models:
                model_path = models_dir / f"{ticker}_hmm.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump({"model": model, "scaler": scaler, "features": feat_names}, f)

        except Exception as e:
            log.error(f"[{ticker}] Error during HMM labeling: {e}", exc_info=True)
            continue

    log.info(f"\n✅ Labelled {len(labelled)} tickers successfully.")
    return labelled


def load_labelled_data(tickers: list[str]) -> dict[str, pd.DataFrame]:
    results = {}
    for ticker in tickers:
        path = DATA_LABEL_DIR / f"{ticker}_labelled.csv"
        if path.exists():
            results[ticker] = pd.read_csv(path, index_col="Date", parse_dates=True)
    log.info(f"Loaded {len(results)} labelled tickers from disk.")
    return results


if __name__ == "__main__":
    import yfinance as yf
    from feature_engineering import compute_features

    raw = yf.download("RELIANCE.NS", start="2019-01-01", end="2025-01-01",
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]

    feat_df  = compute_features(raw, "RELIANCE.NS")
    labelled = run_labeling({"RELIANCE.NS": feat_df})
    print(labelled["RELIANCE.NS"][["Close", "Regime", "prob_Bull", "prob_Bear", "prob_Sideways"]].tail(20))
