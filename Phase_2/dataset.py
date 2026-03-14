
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import logging
import pickle
from pathlib import Path
from typing import Optional

from config import (
    PHASE1_LABEL_DIR, NIFTY_50_TICKERS,
    FEATURE_COLUMNS, REGIME_TO_IDX, NUM_CLASSES,
    WINDOW_SIZE, FORECAST_HORIZON, TRANSITION_WINDOW,
    TRAIN_END, VAL_END, BATCH_SIZE, OUTPUT_DIR,
    CLASS_WEIGHT_MODE, TICKER_TO_IDX,
)

log = logging.getLogger(__name__)

class RegimeDataset(Dataset):

    def __init__(self, windows: np.ndarray, labels: np.ndarray,
                 transitions: np.ndarray, dates: np.ndarray,
                 tickers: np.ndarray):
        self.windows     = torch.FloatTensor(windows)
        self.labels      = torch.LongTensor(labels)
        self.transitions = torch.FloatTensor(transitions)
        self.dates       = dates
        self.tickers     = tickers
        self.stock_ids   = torch.LongTensor(
            [TICKER_TO_IDX.get(t, 0) for t in tickers]
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx], self.transitions[idx], self.stock_ids[idx]

def load_phase1_data(tickers: list[str] = NIFTY_50_TICKERS,
                     label_dir: Path = PHASE1_LABEL_DIR) -> dict[str, pd.DataFrame]:
    data = {}
    for ticker in tickers:
        path = label_dir / f"{ticker}_labelled.csv"
        if path.exists():
            df = pd.read_csv(path, index_col="Date", parse_dates=True)
            if "Regime" in df.columns and len(df) > 100:
                data[ticker] = df
    log.info(f"Loaded {len(data)} labelled tickers from Phase 1.")
    return data

def _compute_transition_labels(regimes: np.ndarray, horizon: int = TRANSITION_WINDOW) -> np.ndarray:
    n = len(regimes)
    transitions = np.zeros(n, dtype=np.float32)
    for i in range(n):
        future_end = min(i + horizon + 1, n)
        future_regimes = regimes[i + 1 : future_end]
        if len(future_regimes) > 0 and np.any(future_regimes != regimes[i]):
            transitions[i] = 1.0
    return transitions

def _validate_features(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    available = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(available)
    if missing:
        log.warning(f"Missing feature columns: {missing}")
    return available

def build_windows(data: dict[str, pd.DataFrame],
                  feature_cols: list[str] = FEATURE_COLUMNS,
                  window_size: int = WINDOW_SIZE,
                  forecast_horizon: int = FORECAST_HORIZON,
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_windows     = []
    all_labels      = []
    all_transitions = []
    all_dates       = []
    all_tickers     = []

    for ticker, df in data.items():
        available_cols = _validate_features(df, feature_cols)
        if len(available_cols) < 5:
            log.warning(f"[{ticker}] Too few features ({len(available_cols)}). Skipping.")
            continue

        features = df[available_cols].values.astype(np.float32)
        regimes  = df["Regime"].map(REGIME_TO_IDX).values.astype(np.int64)
        dates    = df.index.values

        trans = _compute_transition_labels(regimes, TRANSITION_WINDOW)

        features = np.where(np.isinf(features), np.nan, features)
        features_df = pd.DataFrame(features, columns=available_cols)
        features_df = features_df.ffill().bfill().fillna(0.0)
        for col in features_df.columns:
            median = features_df[col].median()
            std = features_df[col].std()
            if std > 0:
                features_df[col] = features_df[col].clip(median - 10 * std, median + 10 * std)
        features = features_df.values.astype(np.float32)

        n = len(features)
        for i in range(window_size, n - forecast_horizon):
            window = features[i - window_size : i]
            label  = regimes[i + forecast_horizon - 1]
            transition = trans[i]
            target_date = dates[i + forecast_horizon - 1]

            if np.isnan(label) or label < 0 or label >= NUM_CLASSES:
                continue

            all_windows.append(window)
            all_labels.append(label)
            all_transitions.append(transition)
            all_dates.append(target_date)
            all_tickers.append(ticker)

    if not all_windows:
        raise ValueError("No valid windows created. Check Phase 1 data.")

    windows     = np.array(all_windows, dtype=np.float32)
    labels      = np.array(all_labels, dtype=np.int64)
    transitions = np.array(all_transitions, dtype=np.float32)
    dates_arr   = np.array(all_dates)
    tickers_arr = np.array(all_tickers)

    log.info(f"Built {len(windows)} windows | shape: {windows.shape} | "
             f"Label dist: Bear={np.sum(labels==0)}, Sideways={np.sum(labels==1)}, Bull={np.sum(labels==2)}")

    return windows, labels, transitions, dates_arr, tickers_arr

def fit_scaler(windows: np.ndarray) -> RobustScaler:
    N, W, F = windows.shape
    flat = windows.reshape(-1, F)
    scaler = RobustScaler()
    scaler.fit(flat)
    return scaler

def apply_scaler(windows: np.ndarray, scaler: RobustScaler) -> np.ndarray:
    N, W, F = windows.shape
    flat = windows.reshape(-1, F)
    scaled = scaler.transform(flat)
    return scaled.reshape(N, W, F).astype(np.float32)

def compute_class_weights(labels: np.ndarray, mode: str = CLASS_WEIGHT_MODE) -> torch.Tensor:
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)

    if mode == "inverse_freq":
        total = counts.sum()
        weights = total / (NUM_CLASSES * counts + 1e-6)
    elif mode == "effective_num":
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / (effective_num + 1e-6)
    else:
        weights = np.ones(NUM_CLASSES)

    weights = weights / weights.sum() * NUM_CLASSES
    log.info(f"Class weights ({mode}): Bear={weights[0]:.3f}, Sideways={weights[1]:.3f}, Bull={weights[2]:.3f}")
    return torch.FloatTensor(weights)

def create_dataloaders(
    tickers: list[str] = NIFTY_50_TICKERS,
    batch_size: int = BATCH_SIZE,
    save_scaler: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    data = load_phase1_data(tickers)
    if not data:
        raise ValueError("No Phase 1 data found. Run Phase 1 first.")

    windows, labels, transitions, dates, tickers_arr = build_windows(data)

    train_mask = dates <= np.datetime64(TRAIN_END)
    val_mask   = (dates > np.datetime64(TRAIN_END)) & (dates <= np.datetime64(VAL_END))
    test_mask  = dates > np.datetime64(VAL_END)

    log.info(f"Split: Train={train_mask.sum()}, Val={val_mask.sum()}, Test={test_mask.sum()}")

    if train_mask.sum() == 0:
        raise ValueError(f"No training samples. Check TRAIN_END={TRAIN_END}")

    scaler = fit_scaler(windows[train_mask])

    if save_scaler:
        scaler_path = OUTPUT_DIR / "feature_scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        log.info(f"Saved scaler to {scaler_path}")

    windows_scaled = apply_scaler(windows, scaler)

    train_ds = RegimeDataset(
        windows_scaled[train_mask], labels[train_mask],
        transitions[train_mask], dates[train_mask], tickers_arr[train_mask]
    )
    val_ds = RegimeDataset(
        windows_scaled[val_mask], labels[val_mask],
        transitions[val_mask], dates[val_mask], tickers_arr[val_mask]
    )
    test_ds = RegimeDataset(
        windows_scaled[test_mask], labels[test_mask],
        transitions[test_mask], dates[test_mask], tickers_arr[test_mask]
    )

    class_weights = compute_class_weights(labels[train_mask])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    log.info(f"DataLoaders created: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    return train_loader, val_loader, test_loader, class_weights

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    train_loader, val_loader, test_loader, cw = create_dataloaders()

    X, y, t, s = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  X (features):   {X.shape}")
    print(f"  y (regime):     {y.shape}")
    print(f"  t (transition): {t.shape}")
    print(f"  s (stock_id):   {s.shape}")
    print(f"\nClass weights: {cw}")
    print(f"Label distribution in batch: {torch.bincount(y, minlength=3)}")
    print(f"Stock ID range in batch: [{s.min().item()}, {s.max().item()}]")
