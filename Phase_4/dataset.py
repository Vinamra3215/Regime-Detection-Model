
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import logging
import pickle
from pathlib import Path

from config import (
    PHASE1_LABEL_DIR, PHASE3_SENTIMENT_DIR,
    NIFTY_50_TICKERS, TICKER_TO_IDX,
    PRICE_FEATURE_COLUMNS, SENTIMENT_FEATURE_COLUMNS,
    NEWS_FEATURE_COLUMNS, INCLUDE_NEWS_FEATURES,
    REGIME_TO_IDX, NUM_CLASSES,
    WINDOW_SIZE, FORECAST_HORIZON, TRANSITION_WINDOW,
    TRAIN_END, VAL_END, BATCH_SIZE, RESULTS_DIR,
    CLASS_WEIGHT_MODE,
)

log = logging.getLogger(__name__)


class SentimentRegimeDataset(Dataset):

    def __init__(self, price_windows, sent_windows, labels, transitions,
                 dates, tickers):
        self.price_windows = torch.FloatTensor(price_windows)
        self.sent_windows  = torch.FloatTensor(sent_windows)
        self.labels        = torch.LongTensor(labels)
        self.transitions   = torch.FloatTensor(transitions)
        self.dates         = dates
        self.tickers       = tickers
        self.stock_ids     = torch.LongTensor(
            [TICKER_TO_IDX.get(t, 0) for t in tickers]
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.price_windows[idx], self.sent_windows[idx],
                self.labels[idx], self.transitions[idx], self.stock_ids[idx])


def load_phase1_data(tickers=NIFTY_50_TICKERS, label_dir=PHASE1_LABEL_DIR):
    data = {}
    for ticker in tickers:
        path = label_dir / f"{ticker}_labelled.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        if "Regime" in df.columns and len(df) > 100:
            data[ticker] = df
    log.info(f"Loaded {len(data)} labelled tickers from Phase 1.")
    return data


def load_sentiment_data(tickers=NIFTY_50_TICKERS,
                        sent_dir=PHASE3_SENTIMENT_DIR):
    data = {}
    for ticker in tickers:
        path = sent_dir / f"{ticker}_sentiment.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        data[ticker] = df
    log.info(f"Loaded sentiment features for {len(data)} tickers.")
    return data


def _get_sentiment_columns():
    cols = list(SENTIMENT_FEATURE_COLUMNS)
    if INCLUDE_NEWS_FEATURES:
        cols.extend(NEWS_FEATURE_COLUMNS)
    return cols


def _compute_transition_labels(regimes, horizon=TRANSITION_WINDOW):
    n = len(regimes)
    transitions = np.zeros(n, dtype=np.float32)
    for i in range(n):
        future_end = min(i + horizon + 1, n)
        future_regimes = regimes[i + 1 : future_end]
        if len(future_regimes) > 0 and np.any(future_regimes != regimes[i]):
            transitions[i] = 1.0
    return transitions


def _clean_features(features, col_names):
    features = np.where(np.isinf(features), np.nan, features)
    df = pd.DataFrame(features, columns=col_names)
    df = df.ffill().bfill().fillna(0.0)
    for col in df.columns:
        median = df[col].median()
        std = df[col].std()
        if std > 0:
            df[col] = df[col].clip(median - 10 * std, median + 10 * std)
    return df.values.astype(np.float32)


def build_windows(price_data, sentiment_data):
    sent_cols = _get_sentiment_columns()

    all_price_windows = []
    all_sent_windows  = []
    all_labels        = []
    all_transitions   = []
    all_dates         = []
    all_tickers       = []

    for ticker, price_df in price_data.items():
        price_cols = [c for c in PRICE_FEATURE_COLUMNS if c in price_df.columns]
        if len(price_cols) < 5:
            log.warning(f"[{ticker}] Too few price features ({len(price_cols)}). Skipping.")
            continue

        sent_df = sentiment_data.get(ticker)
        if sent_df is None:
            log.warning(f"[{ticker}] No sentiment data. Skipping.")
            continue

        available_sent_cols = [c for c in sent_cols if c in sent_df.columns]
        if len(available_sent_cols) < 3:
            log.warning(f"[{ticker}] Too few sentiment features. Skipping.")
            continue

        common_dates = price_df.index.intersection(sent_df.index).sort_values()
        if len(common_dates) < WINDOW_SIZE + FORECAST_HORIZON + 10:
            log.warning(f"[{ticker}] Too few common dates ({len(common_dates)}). Skipping.")
            continue

        price_features = price_df.loc[common_dates, price_cols].values.astype(np.float32)
        sent_features = sent_df.loc[common_dates, available_sent_cols].values.astype(np.float32)
        regimes = price_df.loc[common_dates, "Regime"].map(REGIME_TO_IDX).values.astype(np.int64)
        dates = common_dates.values

        price_features = _clean_features(price_features, price_cols)
        sent_features = _clean_features(sent_features, available_sent_cols)

        if len(available_sent_cols) < len(sent_cols):
            full_sent = np.zeros((len(common_dates), len(sent_cols)), dtype=np.float32)
            for i, col in enumerate(sent_cols):
                if col in available_sent_cols:
                    j = available_sent_cols.index(col)
                    full_sent[:, i] = sent_features[:, j]
            sent_features = full_sent

        trans = _compute_transition_labels(regimes)

        n = len(price_features)
        for i in range(WINDOW_SIZE, n - FORECAST_HORIZON):
            price_window = price_features[i - WINDOW_SIZE : i]
            sent_window  = sent_features[i - WINDOW_SIZE : i]
            label = regimes[i + FORECAST_HORIZON - 1]
            transition = trans[i]
            target_date = dates[i + FORECAST_HORIZON - 1]

            if np.isnan(label) or label < 0 or label >= NUM_CLASSES:
                continue

            all_price_windows.append(price_window)
            all_sent_windows.append(sent_window)
            all_labels.append(label)
            all_transitions.append(transition)
            all_dates.append(target_date)
            all_tickers.append(ticker)

    if not all_price_windows:
        raise ValueError("No valid windows created. Check Phase 1 and Phase 3 data.")

    price_windows = np.array(all_price_windows, dtype=np.float32)
    sent_windows  = np.array(all_sent_windows, dtype=np.float32)
    labels        = np.array(all_labels, dtype=np.int64)
    transitions   = np.array(all_transitions, dtype=np.float32)
    dates_arr     = np.array(all_dates)
    tickers_arr   = np.array(all_tickers)

    log.info(f"Built {len(price_windows)} windows | "
             f"Price: {price_windows.shape} | Sent: {sent_windows.shape} | "
             f"Label dist: Bear={np.sum(labels==0)}, "
             f"Sideways={np.sum(labels==1)}, Bull={np.sum(labels==2)}")

    return price_windows, sent_windows, labels, transitions, dates_arr, tickers_arr


def fit_scalers(price_windows, sent_windows):
    N_p, W_p, F_p = price_windows.shape
    N_s, W_s, F_s = sent_windows.shape

    price_scaler = RobustScaler()
    price_scaler.fit(price_windows.reshape(-1, F_p))

    sent_scaler = RobustScaler()
    sent_scaler.fit(sent_windows.reshape(-1, F_s))

    return price_scaler, sent_scaler


def apply_scaler(windows, scaler):
    N, W, F = windows.shape
    scaled = scaler.transform(windows.reshape(-1, F))
    return scaled.reshape(N, W, F).astype(np.float32)


def compute_class_weights(labels, mode=CLASS_WEIGHT_MODE):
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
    log.info(f"Class weights ({mode}): Bear={weights[0]:.3f}, "
             f"Sideways={weights[1]:.3f}, Bull={weights[2]:.3f}")
    return torch.FloatTensor(weights)


def create_dataloaders(tickers=NIFTY_50_TICKERS, batch_size=BATCH_SIZE,
                       save_scalers=True):
    price_data = load_phase1_data(tickers)
    sentiment_data = load_sentiment_data(tickers)

    if not price_data:
        raise ValueError("No Phase 1 data found.")
    if not sentiment_data:
        raise ValueError("No Phase 3 sentiment data found.")

    price_windows, sent_windows, labels, transitions, dates, tickers_arr = \
        build_windows(price_data, sentiment_data)

    train_mask = dates <= np.datetime64(TRAIN_END)
    val_mask   = (dates > np.datetime64(TRAIN_END)) & (dates <= np.datetime64(VAL_END))
    test_mask  = dates > np.datetime64(VAL_END)

    log.info(f"Split: Train={train_mask.sum()}, Val={val_mask.sum()}, Test={test_mask.sum()}")

    if train_mask.sum() == 0:
        raise ValueError(f"No training samples. Check TRAIN_END={TRAIN_END}")

    price_scaler, sent_scaler = fit_scalers(
        price_windows[train_mask], sent_windows[train_mask]
    )

    if save_scalers:
        scaler_path = RESULTS_DIR / "price_scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(price_scaler, f)
        scaler_path = RESULTS_DIR / "sent_scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(sent_scaler, f)
        log.info(f"Saved scalers to {RESULTS_DIR}")

    price_scaled = apply_scaler(price_windows, price_scaler)
    sent_scaled  = apply_scaler(sent_windows, sent_scaler)

    train_ds = SentimentRegimeDataset(
        price_scaled[train_mask], sent_scaled[train_mask],
        labels[train_mask], transitions[train_mask],
        dates[train_mask], tickers_arr[train_mask]
    )
    val_ds = SentimentRegimeDataset(
        price_scaled[val_mask], sent_scaled[val_mask],
        labels[val_mask], transitions[val_mask],
        dates[val_mask], tickers_arr[val_mask]
    )
    test_ds = SentimentRegimeDataset(
        price_scaled[test_mask], sent_scaled[test_mask],
        labels[test_mask], transitions[test_mask],
        dates[test_mask], tickers_arr[test_mask]
    )

    class_weights = compute_class_weights(labels[train_mask])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    num_price_features = price_scaled.shape[-1]
    num_sent_features  = sent_scaled.shape[-1]

    log.info(f"DataLoaders: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    log.info(f"Price features: {num_price_features}, Sentiment features: {num_sent_features}")

    return (train_loader, val_loader, test_loader, class_weights,
            num_price_features, num_sent_features)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    (train_loader, val_loader, test_loader, cw,
     n_price, n_sent) = create_dataloaders()

    X_p, X_s, y, t, s = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  X_price (features):     {X_p.shape}")
    print(f"  X_sentiment (features): {X_s.shape}")
    print(f"  y (regime):             {y.shape}")
    print(f"  t (transition):         {t.shape}")
    print(f"  s (stock_id):           {s.shape}")
    print(f"\nClass weights: {cw}")

