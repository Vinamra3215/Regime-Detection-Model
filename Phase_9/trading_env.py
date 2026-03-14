
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
import pickle
import sys
import importlib.util
import logging

from config import (
    DEVICE, WINDOW_SIZE, OBS_DIM,
    PRICE_FEATURE_COLUMNS, NIFTY_50_TICKERS, TICKER_TO_IDX,
    PHASE4_DIR, PHASE4_CHECKPOINT, PHASE4_PRICE_SCALER, PHASE4_SENT_SCALER,
    PHASE1_LABEL_DIR,
    REWARD_SCALE, RISK_PENALTY_COEF, DRAWDOWN_PENALTY,
    DRAWDOWN_THRESHOLD, TRADE_COST_PENALTY,
)

log = logging.getLogger(__name__)

def load_transformer():
    p_config = sys.modules.get("config")
    spec = importlib.util.spec_from_file_location("config", str(PHASE4_DIR / "config.py"))
    p4c = importlib.util.module_from_spec(spec)
    sys.modules["config"] = p4c
    spec.loader.exec_module(p4c)

    spec2 = importlib.util.spec_from_file_location("p4m", str(PHASE4_DIR / "model.py"))
    p4m = importlib.util.module_from_spec(spec2)
    sys.modules["p4m"] = p4m
    spec2.loader.exec_module(p4m)

    if p_config:
        sys.modules["config"] = p_config
    else:
        del sys.modules["config"]

    ckpt = torch.load(PHASE4_CHECKPOINT, map_location=DEVICE, weights_only=False)
    model = p4m.build_model(ckpt["num_price_features"], ckpt["num_sent_features"])
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()

    with open(PHASE4_PRICE_SCALER, "rb") as f:
        price_scaler = pickle.load(f)
    with open(PHASE4_SENT_SCALER, "rb") as f:
        sent_scaler = pickle.load(f)

    return model, price_scaler, sent_scaler, ckpt["num_sent_features"]

def mc_dropout_predict(model, x_price, x_sent, stock_id, n_samples=10):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(x_price, x_sent, stock_ids=stock_id)
            samples.append(out["regime_probs"].cpu().numpy())

    arr = np.stack(samples, 0)
    mean_probs = arr.mean(axis=0)[0]
    pred = int(np.argmax(mean_probs))
    conf = float(mean_probs[pred])

    eps = 1e-10
    pred_ent = -np.sum(mean_probs * np.log(mean_probs + eps))
    indiv_ent = -np.sum(arr * np.log(arr + eps), axis=2).mean(axis=0)[0]
    uncertainty = max(pred_ent - indiv_ent, 0.0)

    return mean_probs, conf, uncertainty

def precompute_predictions(model, price_scaler, sent_scaler, num_sent,
                           start_date, end_date):
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    all_data = {}

    for ticker in NIFTY_50_TICKERS:
        path = PHASE1_LABEL_DIR / f"{ticker}_labelled.csv"
        if not path.exists():
            continue

        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        if len(df) < WINDOW_SIZE + 20:
            continue

        price_cols = [c for c in PRICE_FEATURE_COLUMNS if c in df.columns]
        if len(price_cols) < 15:
            continue

        eval_mask = (df.index >= start_ts) & (df.index <= end_ts)
        eval_indices = np.where(eval_mask)[0]

        if len(eval_indices) < 10:
            continue

        records = []
        for day_idx in eval_indices:
            if day_idx < WINDOW_SIZE:
                continue

            p_data = df.iloc[day_idx - WINDOW_SIZE:day_idx][price_cols].values.astype(np.float32)
            p_data = np.nan_to_num(p_data, nan=0.0, posinf=0.0, neginf=0.0)
            p_sc = price_scaler.transform(p_data).reshape(1, WINDOW_SIZE, len(price_cols))
            p_sc = np.nan_to_num(p_sc, nan=0.0, posinf=0.0, neginf=0.0)

            s_data = np.zeros((WINDOW_SIZE, num_sent), dtype=np.float32)
            s_sc = sent_scaler.transform(s_data).reshape(1, WINDOW_SIZE, num_sent)
            s_sc = np.nan_to_num(s_sc, nan=0.0, posinf=0.0, neginf=0.0)

            xp = torch.FloatTensor(p_sc).to(DEVICE)
            xs = torch.FloatTensor(s_sc).to(DEVICE)
            sid = torch.LongTensor([TICKER_TO_IDX.get(ticker, 0)]).to(DEVICE)

            probs, conf, unc = mc_dropout_predict(model, xp, xs, sid)

            actual_ret = df.iloc[day_idx]["log_return_1d"] if "log_return_1d" in df.columns else 0.0
            vol_5d = df.iloc[max(0,day_idx-5):day_idx]["log_return_1d"].std() if "log_return_1d" in df.columns else 0.01
            ret_5d = df.iloc[day_idx]["log_return_5d"] if "log_return_5d" in df.columns else 0.0

            records.append({
                "date": df.index[day_idx],
                "prob_bear": probs[0],
                "prob_side": probs[1],
                "prob_bull": probs[2],
                "confidence": conf,
                "uncertainty": unc,
                "actual_return": actual_ret,
                "vol_5d": vol_5d,
                "ret_5d": ret_5d,
            })

        if records:
            all_data[ticker] = pd.DataFrame(records)

    return all_data

class TradingEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, precomputed_data, mode="train"):
        super().__init__()
        self.mode = mode
        self.data = precomputed_data
        self.tickers = list(precomputed_data.keys())

        self.index = []
        for t_idx, ticker in enumerate(self.tickers):
            n_days = len(precomputed_data[ticker])
            for d_idx in range(n_days):
                self.index.append((t_idx, d_idx))

        if mode == "train":
            np.random.shuffle(self.index)

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([0.0]), high=np.array([1.0]), dtype=np.float32
        )

        self.step_idx = 0
        self.position = 0.0
        self.unrealized_ret = 0.0
        self.days_held = 0
        self.peak_value = 1.0
        self.portfolio_value = 1.0
        self.current_ticker = 0
        self.episode_returns = []

    def _get_obs(self, t_idx, d_idx):
        ticker = self.tickers[t_idx]
        row = self.data[ticker].iloc[d_idx]

        obs = np.array([
            row["prob_bear"],
            row["prob_side"],
            row["prob_bull"],
            row["confidence"],
            row["uncertainty"],
            self.position,
            self.unrealized_ret,
            self.days_held / 30.0,
            row["vol_5d"] * 10.0,
            row["ret_5d"] * 10.0,
        ], dtype=np.float32)

        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.position = 0.0
        self.unrealized_ret = 0.0
        self.days_held = 0
        self.peak_value = 1.0
        self.portfolio_value = 1.0
        self.episode_returns = []

        if self.mode == "train":
            np.random.shuffle(self.index)

        self.step_idx = 0

        if len(self.index) > 0:
            t_idx, d_idx = self.index[0]
            self.current_ticker = t_idx
            return self._get_obs(t_idx, d_idx), {}
        return np.zeros(OBS_DIM, dtype=np.float32), {}

    def step(self, action):
        if self.step_idx >= len(self.index):
            return np.zeros(OBS_DIM, dtype=np.float32), 0.0, True, False, {}

        t_idx, d_idx = self.index[self.step_idx]
        ticker = self.tickers[t_idx]
        row = self.data[ticker].iloc[d_idx]

        new_position = float(np.clip(action[0], 0.0, 1.0))
        actual_return = row["actual_return"]
        uncertainty = row["uncertainty"]

        pnl_reward = new_position * actual_return * REWARD_SCALE

        risk_penalty = -RISK_PENALTY_COEF * new_position * uncertainty

        position_change = abs(new_position - self.position)
        trade_cost = -TRADE_COST_PENALTY * position_change * REWARD_SCALE

        self.portfolio_value *= (1.0 + new_position * actual_return)
        self.peak_value = max(self.peak_value, self.portfolio_value)
        dd = (self.portfolio_value - self.peak_value) / self.peak_value
        dd_penalty = DRAWDOWN_PENALTY * min(dd + DRAWDOWN_THRESHOLD, 0.0) * REWARD_SCALE

        reward = pnl_reward + risk_penalty + trade_cost + dd_penalty

        if new_position > 0.05:
            if self.position > 0.05 and t_idx == self.current_ticker:
                self.days_held += 1
                self.unrealized_ret += actual_return
            else:
                self.days_held = 1
                self.unrealized_ret = actual_return
        else:
            self.days_held = 0
            self.unrealized_ret = 0.0

        self.position = new_position
        self.current_ticker = t_idx
        self.episode_returns.append(float(reward))

        self.step_idx += 1
        terminated = self.step_idx >= len(self.index)
        truncated = False

        if terminated:
            obs = np.zeros(OBS_DIM, dtype=np.float32)
        else:
            next_t, next_d = self.index[self.step_idx]
            if self.mode == "eval" and next_t != t_idx:
                self.position = 0.0
                self.unrealized_ret = 0.0
                self.days_held = 0
            obs = self._get_obs(next_t, next_d)

        info = {
            "pnl_reward": float(pnl_reward),
            "actual_return": float(actual_return),
            "position": float(new_position),
            "portfolio_value": float(self.portfolio_value),
        }

        return obs, float(reward), terminated, truncated, info
