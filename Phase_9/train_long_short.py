
import sys, importlib.util, numpy as np, pandas as pd, torch, pickle
import gymnasium as gym
from gymnasium import spaces
import logging, time, json
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
PHASE4_DIR  = PROJECT_DIR / "Phase_4"
PHASE4_RESULTS = PROJECT_DIR / "results" / "phase_4"
PHASE4_CKPT = PHASE4_RESULTS / "checkpoints" / "best_model.pt"
PHASE4_PS   = PHASE4_RESULTS / "price_scaler.pkl"
PHASE4_SS   = PHASE4_RESULTS / "sent_scaler.pkl"
PHASE1_LABELS = PROJECT_DIR / "results" / "phase_1" / "data" / "labelled"
LONG_ONLY_MODEL = PROJECT_DIR / "results" / "phase_9" / "checkpoints" / "ppo_trading_agent"

RESULTS_DIR = PROJECT_DIR / "results" / "phase_9" / "long_short"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NIFTY_50 = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
    "HINDUNILVR.NS","SBIN.NS","BHARTIARTL.NS","BAJFINANCE.NS","KOTAKBANK.NS",
    "LT.NS","AXISBANK.NS","ASIANPAINT.NS","MARUTI.NS","HCLTECH.NS",
    "SUNPHARMA.NS","TITAN.NS","WIPRO.NS","ULTRACEMCO.NS","NESTLEIND.NS",
    "BAJAJFINSV.NS","NTPC.NS","POWERGRID.NS","TECHM.NS","ONGC.NS",
    "TATAMOTORS.NS","TATASTEEL.NS","JSWSTEEL.NS","M&M.NS","HINDALCO.NS",
    "COALINDIA.NS","DRREDDY.NS","DIVISLAB.NS","CIPLA.NS","APOLLOHOSP.NS",
    "ADANIPORTS.NS","ADANIENT.NS","GRASIM.NS","HDFCLIFE.NS","SBILIFE.NS",
    "SHRIRAMFIN.NS","BPCL.NS","EICHERMOT.NS","HEROMOTOCO.NS","INDUSINDBK.NS",
    "BRITANNIA.NS","ITC.NS","BAJAJ-AUTO.NS","BEL.NS","TRENT.NS",
]
T2I = {t:i for i,t in enumerate(NIFTY_50)}
PCOLS = [
    "log_return_1d","log_return_5d","rolling_vol_10d","rolling_vol_20d",
    "atr_pct","rsi_14","macd_histogram","bb_width","bb_pband",
    "adx","adx_pos","adx_neg","volume_ratio","log_volume_change",
    "ma_dist_20","ma_dist_50","linreg_slope_10","linreg_slope_20",
]
W = 60; OBS_DIM = 10
STT=0.001; EXC=0.0000345; GST=0.18; SEBI=0.000001; STAMP=0.00015; SLIP=0.0005
CAPITAL = 10_00_000

def _slope(s, w):
    out = np.full(len(s), np.nan); v = s.values
    x = np.arange(w); xm = x.mean(); xv = ((x-xm)**2).sum()
    for i in range(w-1, len(v)):
        y = v[i-w+1:i+1]
        if np.any(np.isnan(y)): continue
        ym = y.mean(); out[i] = ((x-xm)*(y-ym)).sum()/xv/(ym+1e-9)
    return pd.Series(out, index=s.index)

def features(df):
    import ta; df = df.copy()
    if not {"Open","High","Low","Close","Volume"}.issubset(df.columns): return None
    df["log_return_1d"]=np.log(df["Close"]/df["Close"].shift(1))
    df["log_return_5d"]=np.log(df["Close"]/df["Close"].shift(5))
    df["rolling_vol_10d"]=df["log_return_1d"].rolling(10).std()
    df["rolling_vol_20d"]=df["log_return_1d"].rolling(20).std()
    a=ta.volatility.AverageTrueRange(df["High"],df["Low"],df["Close"],14)
    df["atr_pct"]=a.average_true_range()/df["Close"]
    df["rsi_14"]=ta.momentum.RSIIndicator(df["Close"],14).rsi()
    df["macd_histogram"]=ta.trend.MACD(df["Close"]).macd_diff()
    b=ta.volatility.BollingerBands(df["Close"],20)
    df["bb_width"]=b.bollinger_wband(); df["bb_pband"]=b.bollinger_pband()
    ax=ta.trend.ADXIndicator(df["High"],df["Low"],df["Close"],14)
    df["adx"]=ax.adx(); df["adx_pos"]=ax.adx_pos(); df["adx_neg"]=ax.adx_neg()
    df["volume_ratio"]=df["Volume"]/df["Volume"].rolling(20).mean()
    df["log_volume_change"]=np.log(df["Volume"]/df["Volume"].shift(1)+1e-9)
    m20=df["Close"].rolling(20).mean(); m50=df["Close"].rolling(50).mean()
    df["ma_dist_20"]=(df["Close"]-m20)/m20; df["ma_dist_50"]=(df["Close"]-m50)/m50
    df["linreg_slope_10"]=_slope(df["Close"],10); df["linreg_slope_20"]=_slope(df["Close"],20)
    return df.dropna(subset=PCOLS)

def load_transformer():
    pc = sys.modules.get("config")
    s = importlib.util.spec_from_file_location("config", str(PHASE4_DIR/"config.py"))
    c = importlib.util.module_from_spec(s); sys.modules["config"]=c; s.loader.exec_module(c)
    s2 = importlib.util.spec_from_file_location("p4m", str(PHASE4_DIR/"model.py"))
    m = importlib.util.module_from_spec(s2); sys.modules["p4m"]=m; s2.loader.exec_module(m)
    if pc: sys.modules["config"]=pc
    else: del sys.modules["config"]
    ckpt = torch.load(PHASE4_CKPT, map_location=DEVICE, weights_only=False)
    mdl = m.build_model(ckpt["num_price_features"], ckpt["num_sent_features"])
    mdl.load_state_dict(ckpt["model_state"]); mdl.to(DEVICE); mdl.eval()
    with open(PHASE4_PS,"rb") as f: ps=pickle.load(f)
    with open(PHASE4_SS,"rb") as f: ss=pickle.load(f)
    return mdl, ps, ss, ckpt["num_sent_features"]

def mc_pred(mdl, xp, xs, sid, n=10):
    for m in mdl.modules():
        if isinstance(m, torch.nn.Dropout): m.train()
    samps = []
    with torch.no_grad():
        for _ in range(n):
            out = mdl(xp, xs, stock_ids=sid)
            p = out["regime_probs"].cpu().numpy()
            if np.any(np.isnan(p)) or np.any(np.isinf(p)):
                p = np.array([[1/3, 1/3, 1/3]])
            samps.append(p)
    arr = np.stack(samps,0); mp = arr.mean(0)[0]
    mp = np.nan_to_num(mp, nan=1/3, posinf=1, neginf=0)
    mp = np.clip(mp, 0, 1)
    s = mp.sum()
    if s < 1e-6: mp = np.array([1/3, 1/3, 1/3])
    else: mp = mp / s
    pred=int(np.argmax(mp)); conf=float(mp[pred])
    eps=1e-10; pe=-np.sum(mp*np.log(mp+eps))
    ie=-np.sum(arr*np.log(arr+eps),axis=2).mean(0)[0]
    unc = max(pe-ie, 0)
    if np.isnan(unc) or np.isinf(unc): unc = 0.5
    return mp, conf, unc

class LongShortTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, data, mode="train"):
        super().__init__()
        self.mode = mode; self.data = data
        self.tickers = list(data.keys())

        self.index = []
        for ti, t in enumerate(self.tickers):
            for di in range(len(data[t])):
                self.index.append((ti, di))
        if mode == "train": np.random.shuffle(self.index)

        self.observation_space = spaces.Box(-10, 10, (OBS_DIM,), np.float32)
        self.action_space = spaces.Box(
            low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32
        )

        self.step_idx = 0; self.position = 0.0
        self.unr_ret = 0.0; self.days_held = 0
        self.peak = 1.0; self.pv = 1.0
        self.cur_ticker = 0

    def _obs(self, ti, di):
        t = self.tickers[ti]; row = self.data[t].iloc[di]
        obs = np.array([
            row["prob_bear"], row["prob_side"], row["prob_bull"],
            row["confidence"], row["uncertainty"],
            self.position,
            self.unr_ret, self.days_held/30.0,
            row["vol_5d"]*10, row["ret_5d"]*10,
        ], dtype=np.float32)
        return np.nan_to_num(obs, nan=0, posinf=1, neginf=-1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.position=0; self.unr_ret=0; self.days_held=0
        self.peak=1; self.pv=1
        if self.mode=="train": np.random.shuffle(self.index)
        self.step_idx = 0
        if self.index:
            ti,di = self.index[0]; self.cur_ticker=ti
            return self._obs(ti,di), {}
        return np.zeros(OBS_DIM, np.float32), {}

    def step(self, action):
        if self.step_idx >= len(self.index):
            return np.zeros(OBS_DIM, np.float32), 0.0, True, False, {}

        ti,di = self.index[self.step_idx]
        t = self.tickers[ti]; row = self.data[t].iloc[di]

        new_pos = float(np.clip(action[0], -1.0, 1.0))
        ret = float(np.clip(row["actual_return"], -0.15, 0.15))
        unc = float(np.clip(row["uncertainty"], 0, 2.0))

        pnl = new_pos * ret * 50.0

        risk = -0.2 * abs(new_pos) * unc

        chg = abs(new_pos - self.position)
        cost = -0.05 * chg

        short_penalty = 0.0
        if new_pos < -0.05:
            short_penalty = -0.03 * abs(new_pos)

        step_ret = new_pos * ret
        step_ret = float(np.clip(step_ret, -0.10, 0.10))
        self.pv *= (1.0 + step_ret)
        self.pv = max(self.pv, 0.01)
        self.peak = max(self.peak, self.pv)
        dd = (self.pv - self.peak) / self.peak
        dd_pen = 50.0 * min(dd + 0.03, 0)

        reward = float(np.clip(pnl + risk + cost + short_penalty + dd_pen, -10.0, 10.0))

        if abs(new_pos) > 0.05:
            if abs(self.position) > 0.05 and ti == self.cur_ticker:
                self.days_held += 1
                self.unr_ret = float(np.clip(self.unr_ret + ret, -1, 1))
            else:
                self.days_held = 1; self.unr_ret = ret
        else:
            self.days_held = 0; self.unr_ret = 0

        self.position = new_pos; self.cur_ticker = ti
        self.step_idx += 1
        done = self.step_idx >= len(self.index)

        if done:
            obs = np.zeros(OBS_DIM, np.float32)
        else:
            nti, ndi = self.index[self.step_idx]
            if self.mode == "eval" and nti != ti:
                self.position=0; self.unr_ret=0; self.days_held=0
            obs = self._obs(nti, ndi)

        info = {"position": float(new_pos), "actual_return": float(ret),
                "portfolio_value": float(self.pv)}
        return obs, float(reward), done, False, info

def precompute_from_labelled(mdl, ps, ss, ns, start, end):
    st = pd.Timestamp(start); en = pd.Timestamp(end)
    data = {}
    for ticker in NIFTY_50:
        path = PHASE1_LABELS / f"{ticker}_labelled.csv"
        if not path.exists(): continue
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        if len(df) < W+20: continue
        pcols = [c for c in PCOLS if c in df.columns]
        if len(pcols) < 15: continue
        emask = (df.index>=st)&(df.index<=en)
        eidx = np.where(emask)[0]
        if len(eidx) < 10: continue
        recs = []
        for di in eidx:
            if di < W: continue
            pd_ = df.iloc[di-W:di][pcols].values.astype(np.float32)
            pd_ = np.nan_to_num(pd_, nan=0, posinf=0, neginf=0)
            psc = ps.transform(pd_).reshape(1,W,len(pcols))
            psc = np.nan_to_num(psc, nan=0, posinf=0, neginf=0)
            psc = np.clip(psc, -5, 5)
            sd_ = np.zeros((W,ns), np.float32)
            ssc = ss.transform(sd_).reshape(1,W,ns)
            ssc = np.nan_to_num(ssc, nan=0, posinf=0, neginf=0)
            ssc = np.clip(ssc, -5, 5)
            xp = torch.FloatTensor(psc).to(DEVICE)
            xs = torch.FloatTensor(ssc).to(DEVICE)
            sid = torch.LongTensor([T2I.get(ticker,0)]).to(DEVICE)
            probs, conf, unc = mc_pred(mdl, xp, xs, sid)
            ret = df.iloc[di]["log_return_1d"] if "log_return_1d" in df.columns else 0
            vol5 = df.iloc[max(0,di-5):di]["log_return_1d"].std() if "log_return_1d" in df.columns else 0.01
            ret5 = df.iloc[di]["log_return_5d"] if "log_return_5d" in df.columns else 0
            ret = float(np.nan_to_num(ret, nan=0, posinf=0, neginf=0))
            vol5 = float(np.nan_to_num(vol5, nan=0.01, posinf=0.01, neginf=0.01))
            ret5 = float(np.nan_to_num(ret5, nan=0, posinf=0, neginf=0))
            if np.any(np.isnan(probs)): continue
            recs.append({
                "date": df.index[di], "prob_bear": float(probs[0]), "prob_side": float(probs[1]),
                "prob_bull": float(probs[2]), "confidence": float(conf), "uncertainty": float(unc),
                "actual_return": ret, "vol_5d": vol5, "ret_5d": ret5,
            })
        if recs:
            tdf = pd.DataFrame(recs)
            tdf = tdf.dropna()
            if len(tdf) > 5: data[ticker] = tdf
    return data

def precompute_from_yfinance(mdl, ps, ss, ns, dl_start, dl_end, eval_start, eval_end):
    import yfinance as yf
    es = pd.Timestamp(eval_start); ee = pd.Timestamp(eval_end)
    data = {}
    for ticker in NIFTY_50:
        try:
            raw = yf.download(ticker, start=dl_start, end=dl_end,
                              auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0] for c in raw.columns]
            if len(raw)<100: continue
            df = features(raw)
            if df is None or len(df)<W+10: continue
            pcols = [c for c in PCOLS if c in df.columns]
            emask = (df.index>=es)&(df.index<=ee)
            eidx = np.where(emask)[0]
            if len(eidx)<10: continue
            recs = []
            for di in eidx:
                if di < W: continue
                pd_ = df.iloc[di-W:di][pcols].values.astype(np.float32)
                pd_ = np.nan_to_num(pd_, nan=0, posinf=0, neginf=0)
                psc = ps.transform(pd_).reshape(1,W,len(pcols))
                psc = np.nan_to_num(psc, nan=0, posinf=0, neginf=0)
                psc = np.clip(psc, -5, 5)
                sd_ = np.zeros((W,ns), np.float32)
                ssc = ss.transform(sd_).reshape(1,W,ns)
                ssc = np.nan_to_num(ssc, nan=0, posinf=0, neginf=0)
                ssc = np.clip(ssc, -5, 5)
                xp = torch.FloatTensor(psc).to(DEVICE)
                xs = torch.FloatTensor(ssc).to(DEVICE)
                sid = torch.LongTensor([T2I.get(ticker,0)]).to(DEVICE)
                probs, conf, unc = mc_pred(mdl, xp, xs, sid)
                ret = df.iloc[di]["log_return_1d"] if "log_return_1d" in df.columns else 0
                vol5 = df.iloc[max(0,di-5):di]["log_return_1d"].std() if "log_return_1d" in df.columns else 0.01
                ret5 = df.iloc[di]["log_return_5d"] if "log_return_5d" in df.columns else 0
                ret = float(np.nan_to_num(ret, nan=0, posinf=0, neginf=0))
                vol5 = float(np.nan_to_num(vol5, nan=0.01, posinf=0.01, neginf=0.01))
                ret5 = float(np.nan_to_num(ret5, nan=0, posinf=0, neginf=0))
                if np.any(np.isnan(probs)): continue
                recs.append({
                    "date": df.index[di], "prob_bear": float(probs[0]), "prob_side": float(probs[1]),
                    "prob_bull": float(probs[2]), "confidence": float(conf), "uncertainty": float(unc),
                    "actual_return": ret, "vol_5d": vol5, "ret_5d": ret5,
                })
            if recs:
                tdf = pd.DataFrame(recs)
                tdf = tdf.dropna()
                if len(tdf) > 5: data[ticker] = tdf
        except: pass
    return data

class LogCB(BaseCallback):
    def __init__(self, freq=10000, verbose=0):
        super().__init__(verbose); self.freq = freq
    def _on_step(self):
        if self.n_calls % self.freq == 0 and self.model.ep_info_buffer:
            mr = np.mean([e["r"] for e in self.model.ep_info_buffer])
            log.info(f"  Step {self.n_calls:>7d}/500000 | Mean Reward: {mr:>8.2f}")
        return True

def txn_cost(v, buy=True):
    return (0 if buy else v*STT)+v*EXC+v*EXC*GST+v*SEBI+(v*STAMP if buy else 0)+v*SLIP

def simulate(agent, data, name):
    env = LongShortTradingEnv(data, mode="eval")
    obs, _ = env.reset()
    recs = []; step = 0
    while True:
        action, _ = agent.predict(obs, deterministic=True)
        obs, rew, done, _, info = env.step(action)
        if step < len(env.index):
            ti, di = env.index[step]
            t = env.tickers[ti]; row = env.data[t].iloc[di]
            recs.append({
                "date": row["date"], "ticker": t,
                "position": info["position"],
                "actual_return": info["actual_return"],
            })
        step += 1
        if done: break

    df = pd.DataFrame(recs)
    df["date"] = pd.to_datetime(df["date"])
    dates = sorted(df["date"].unique())

    capital = CAPITAL; prev = {}; drecs = []; tc = 0; trades = 0
    n_long = 0; n_short = 0; n_flat = 0

    for date in dates:
        dd = df[df["date"]==date]
        dret = 0; dcost = 0; nt = max(len(dd),1)
        for _, row in dd.iterrows():
            t = row["ticker"]; pos = row["position"]; aret = row["actual_return"]
            pv = prev.get(t, 0)
            chg = abs(pos - pv)
            if chg > 0.05:
                tv = capital * chg / nt
                dcost += txn_cost(tv, pos > pv)
                trades += 1
            alloc = capital * pos / nt
            dret += alloc * aret
            prev[t] = pos
            if pos > 0.05: n_long += 1
            elif pos < -0.05: n_short += 1
            else: n_flat += 1

        capital += dret - dcost; tc += dcost
        na = sum(1 for v in prev.values() if abs(v) > 0.05)
        drecs.append({"date": date, "value": capital, "n_active": na})

    pdf = pd.DataFrame(drecs)
    vals = pdf["value"].values
    tret = (vals[-1]/vals[0]-1)*100
    drets = np.diff(vals)/vals[:-1]
    vol = np.std(drets)*np.sqrt(252)*100
    ann = tret * 252/len(vals)
    sharpe = (ann/100-0.07)/(vol/100) if vol>0 else 0
    rmax = np.maximum.accumulate(vals)
    maxdd = float(np.min((vals-rmax)/rmax)*100)
    nz = drets[drets!=0]
    wr = np.mean(nz>0)*100 if len(nz)>0 else 0
    g = np.sum(drets[drets>0]); l = abs(np.sum(drets[drets<0]))
    pf = g/l if l>0 else 0

    total_decisions = n_long + n_short + n_flat
    result = {
        "period": name, "total_ret": tret, "sharpe": sharpe,
        "max_dd": maxdd, "vol": vol, "win_rate": wr, "pf": pf,
        "trades": trades, "costs": tc, "final": float(vals[-1]),
        "long_pct": n_long/total_decisions*100 if total_decisions else 0,
        "short_pct": n_short/total_decisions*100 if total_decisions else 0,
        "flat_pct": n_flat/total_decisions*100 if total_decisions else 0,
    }

    print(f"\n  {'─'*55}")
    print(f"  RESULTS: {name}")
    print(f"  {'─'*55}")
    print(f"  Total Return:   {tret:+.2f}%")
    print(f"  Sharpe:         {sharpe:.4f}")
    print(f"  Max Drawdown:   {maxdd:.2f}%")
    print(f"  Win Rate:       {wr:.1f}%")
    print(f"  Profit Factor:  {pf:.3f}")
    print(f"  Trades:         {trades}")
    print(f"  Txn Costs:      Rs {tc:,.0f}")
    print(f"  Decisions: LONG {n_long} ({n_long/total_decisions*100:.1f}%) | "
          f"SHORT {n_short} ({n_short/total_decisions*100:.1f}%) | "
          f"FLAT {n_flat} ({n_flat/total_decisions*100:.1f}%)")

    pdf.to_csv(RESULTS_DIR / f"{name.replace(' ','_').lower()}_portfolio.csv", index=False)
    return result

def main():
    start = time.time()
    print("="*60)
    print("  PHASE 9 — LONG-SHORT RL AGENT")
    print(f"  Action Space: [-1, 1] (SHORT / FLAT / LONG)")
    print(f"  Device: {DEVICE}")
    print("="*60)

    mdl, ps, ss, ns = load_transformer()
    print(f"  Transformer loaded")

    print(f"\n  Pre-computing training predictions (2019-2023)...")
    train_data = precompute_from_labelled(mdl, ps, ss, ns, "2019-07-01", "2023-12-31")
    total_train = sum(len(d) for d in train_data.values())
    print(f"  -> {len(train_data)} tickers, {total_train:,} steps")

    print(f"\n{'='*40}")
    print(f"  TRAINING Long-Short PPO (500K steps)")
    print(f"{'='*40}")
    nan_count = 0
    for t, tdf in train_data.items():
        nc = tdf.isna().sum().sum()
        if nc > 0:
            log.info(f"  WARNING: {t} has {nc} NaN values, dropping")
            train_data[t] = tdf.dropna()
            nan_count += nc
    print(f"  Data validation: removed {nan_count} NaN entries")

    env = DummyVecEnv([lambda: LongShortTradingEnv(train_data, "train")])
    agent = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=min(2048, total_train),
        batch_size=64, n_epochs=10,
        gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.01,
        vf_coef=0.5, max_grad_norm=0.5,
        verbose=0, device="cpu",
        policy_kwargs={"net_arch": dict(pi=[128, 64], vf=[128, 64])},
    )
    agent.learn(total_timesteps=500_000, callback=LogCB())
    agent.save(str(RESULTS_DIR / "ppo_long_short"))
    env.close()
    train_time = time.time() - start
    print(f"  Training done ({train_time:.0f}s)")

    print(f"\n{'='*40}")
    print(f"  TESTING on 3 periods")
    print(f"{'='*40}")

    print(f"\n  Pre-computing 2024 predictions...")
    eval_2024 = precompute_from_labelled(mdl, ps, ss, ns, "2024-01-01", "2024-12-31")
    print(f"  -> {len(eval_2024)} tickers, {sum(len(d) for d in eval_2024.values()):,} steps")
    r0 = simulate(agent, eval_2024, "Jan-Dec 2024 (Test)")

    print(f"\n  Downloading 2016-2017 data...")
    oos_2016 = precompute_from_yfinance(mdl, ps, ss, ns,
        "2015-12-01", "2017-06-30", "2016-05-01", "2017-05-31")
    print(f"  -> {len(oos_2016)} tickers, {sum(len(d) for d in oos_2016.values()):,} steps")
    r1 = simulate(agent, oos_2016, "May 2016-May 2017 (OOS)")

    print(f"\n  Downloading 2024-2025 data...")
    oos_2024 = precompute_from_yfinance(mdl, ps, ss, ns,
        "2024-06-01", "2025-06-30", "2024-09-01", "2025-05-31")
    print(f"  -> {len(oos_2024)} tickers, {sum(len(d) for d in oos_2024.values()):,} steps")
    r2 = simulate(agent, oos_2024, "Sep 2024-May 2025 (OOS)")

    all_res = [r0, r1, r2]
    long_only_res = {
        "Jan-Dec 2024": {"ret": 78.62, "sharpe": 15.57, "dd": -0.39},
        "May 2016-May 2017": {"ret": 92.25, "sharpe": 16.99, "dd": -0.44},
        "Sep 2024-May 2025": {"ret": 35.14, "sharpe": 7.64, "dd": -0.38},
    }

    print(f"\n\n{'='*60}")
    print(f"  FINAL COMPARISON: Long-Short vs Long-Only")
    print(f"{'='*60}")
    print(f"\n  {'Period':<28s} {'L-S Return':>12s} {'L-O Return':>12s} {'L-S Short%':>12s}")
    print(f"  {'─'*28} {'─'*12} {'─'*12} {'─'*12}")
    lo_keys = list(long_only_res.keys())
    for r, lo_k in zip(all_res, lo_keys):
        lo = long_only_res[lo_k]
        print(f"  {r['period']:<28s} {r['total_ret']:>+11.2f}% {lo['ret']:>+11.2f}% {r['short_pct']:>11.1f}%")

    print(f"\n  {'Period':<28s} {'L-S Sharpe':>12s} {'L-O Sharpe':>12s} {'L-S MaxDD':>12s} {'L-O MaxDD':>12s}")
    print(f"  {'─'*28} {'─'*12} {'─'*12} {'─'*12} {'─'*12}")
    for r, lo_k in zip(all_res, lo_keys):
        lo = long_only_res[lo_k]
        print(f"  {r['period']:<28s} {r['sharpe']:>12.4f} {lo['sharpe']:>12.4f} {r['max_dd']:>11.2f}% {lo['dd']:>11.2f}%")

    with open(RESULTS_DIR / "long_short_results.json", "w") as f:
        json.dump({"long_short": all_res, "long_only": long_only_res}, f, indent=2, default=str)

    with open(RESULTS_DIR / "long_short_summary.txt", "w") as f:
        f.write("Long-Short vs Long-Only RL Agent\n" + "="*50 + "\n\n")
        for r in all_res:
            f.write(f"{r['period']}:\n")
            f.write(f"  Return: {r['total_ret']:+.2f}%, Sharpe: {r['sharpe']:.4f}\n")
            f.write(f"  LONG: {r['long_pct']:.1f}%, SHORT: {r['short_pct']:.1f}%, FLAT: {r['flat_pct']:.1f}%\n\n")

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  Complete. Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Results: {RESULTS_DIR}/")
    print(f"{'='*60}")

if __name__=="__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
    main()
