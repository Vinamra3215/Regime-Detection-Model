
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import logging

from config import (
    NIFTY_50_TICKERS, TICKER_TO_IDX,
    PHASE1_LABEL_DIR, PHASE3_SENTIMENT_DIR,
    PRICE_FEATURE_COLUMNS, SENTIMENT_FEATURE_COLUMNS,
    NEWS_FEATURE_COLUMNS,
    WINDOW_SIZE, FORECAST_HORIZON, TRANSITION_WINDOW,
    TRAIN_END, VAL_END, BATCH_SIZE,
    PHASE4_PRICE_SCALER, PHASE4_SENT_SCALER, PHASE4_CHECKPOINT,
    REGIME_TO_IDX,
)

log = logging.getLogger(__name__)


class Phase5Dataset(Dataset):
    def __init__(self, price_windows, sent_windows, labels, transitions, stock_ids):
        self.price_windows = torch.FloatTensor(price_windows)
        self.sent_windows  = torch.FloatTensor(sent_windows)
        self.labels        = torch.LongTensor(labels)
        self.transitions   = torch.FloatTensor(transitions)
        self.stock_ids     = torch.LongTensor(stock_ids)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.price_windows[idx], self.sent_windows[idx],
                self.labels[idx], self.transitions[idx], self.stock_ids[idx])


def _get_sent_cols(expected_sent):
    all_cols = list(SENTIMENT_FEATURE_COLUMNS) + list(NEWS_FEATURE_COLUMNS)
    if expected_sent <= len(SENTIMENT_FEATURE_COLUMNS):
        return list(SENTIMENT_FEATURE_COLUMNS)[:expected_sent]
    return all_cols[:expected_sent]


def load_and_merge_data(ticker, sent_cols):
    label_path = PHASE1_LABEL_DIR / f"{ticker}_labelled.csv"
    sent_path  = PHASE3_SENTIMENT_DIR / f"{ticker}_sentiment.csv"

    if not label_path.exists():
        return None

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
    return df


def build_windows(sent_cols):
    all_price, all_sent, all_labels = [], [], []
    all_trans, all_stocks, all_dates = [], [], []

    for ticker in NIFTY_50_TICKERS:
        df = load_and_merge_data(ticker, sent_cols)
        if df is None:
            continue

        price_cols = [c for c in PRICE_FEATURE_COLUMNS if c in df.columns]
        if len(price_cols) < 5 or "Regime" not in df.columns:
            continue

        price_features = df[price_cols].values.astype(np.float32)
        sent_features  = df[sent_cols].values.astype(np.float32)
        regimes = df["Regime"].map(REGIME_TO_IDX).values
        stock_idx = TICKER_TO_IDX.get(ticker, 0)

        for i in range(len(df) - WINDOW_SIZE - FORECAST_HORIZON):
            end = i + WINDOW_SIZE
            target_idx = end + FORECAST_HORIZON - 1
            if target_idx >= len(df):
                break

            pw = price_features[i:end]
            sw = sent_features[i:end]
            if np.isnan(pw).any() or np.isnan(sw).any():
                continue

            label = regimes[target_idx]
            if np.isnan(label):
                continue

            trans_end = min(target_idx + TRANSITION_WINDOW, len(regimes))
            future = regimes[target_idx:trans_end]
            transition = 1 if len(set(future[~np.isnan(future)])) > 1 else 0

            all_price.append(pw)
            all_sent.append(sw)
            all_labels.append(int(label))
            all_trans.append(float(transition))
            all_stocks.append(stock_idx)
            all_dates.append(df.index[target_idx])

    log.info(f"Built {len(all_labels)} windows | "
             f"Price: ({len(all_labels)}, {WINDOW_SIZE}, {len(price_cols)}) | "
             f"Sent: ({len(all_labels)}, {WINDOW_SIZE}, {len(sent_cols)})")

    return (np.array(all_price), np.array(all_sent), np.array(all_labels),
            np.array(all_trans), np.array(all_stocks), np.array(all_dates))


def create_dataloaders():
    ckpt = torch.load(PHASE4_CHECKPOINT, map_location="cpu", weights_only=False)
    expected_sent = ckpt["num_sent_features"]
    log.info(f"Phase 4 checkpoint expects {expected_sent} sentiment features")
    del ckpt

    sent_cols = _get_sent_cols(expected_sent)
    log.info(f"Using {len(sent_cols)} sent features: {sent_cols}")

    price_arr, sent_arr, labels, trans, stocks, dates = build_windows(sent_cols)

    train_end = pd.Timestamp(TRAIN_END)
    val_end   = pd.Timestamp(VAL_END)
    train_mask = dates <= train_end
    val_mask   = (dates > train_end) & (dates <= val_end)
    test_mask  = dates > val_end

    with open(PHASE4_PRICE_SCALER, "rb") as f:
        price_scaler = pickle.load(f)
    with open(PHASE4_SENT_SCALER, "rb") as f:
        sent_scaler = pickle.load(f)

    n, w, pf = price_arr.shape
    _, _, sf = sent_arr.shape

    price_flat = np.nan_to_num(price_arr.reshape(-1, pf), nan=0.0, posinf=0.0, neginf=0.0)
    sent_flat  = np.nan_to_num(sent_arr.reshape(-1, sf), nan=0.0, posinf=0.0, neginf=0.0)

    price_scaled = np.nan_to_num(price_scaler.transform(price_flat).reshape(n, w, pf), nan=0.0, posinf=0.0, neginf=0.0)
    sent_scaled  = np.nan_to_num(sent_scaler.transform(sent_flat).reshape(n, w, sf), nan=0.0, posinf=0.0, neginf=0.0)

    def make_loader(mask, shuffle=False):
        ds = Phase5Dataset(price_scaled[mask], sent_scaled[mask],
                           labels[mask], trans[mask], stocks[mask])
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(train_mask)
    val_loader   = make_loader(val_mask)
    test_loader  = make_loader(test_mask)

    log.info(f"DataLoaders: Train={train_mask.sum()}, Val={val_mask.sum()}, Test={test_mask.sum()}")
    return train_loader, val_loader, test_loader

