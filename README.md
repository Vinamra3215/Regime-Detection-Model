# Regime Detection Model

A multi-phase trading intelligence system — from HMM-based market regime labeling through Transformer classification with sentiment fusion, uncertainty quantification, and reinforcement learning-based trading.

---

## Project Vision

Build a production-grade trading system that:

- Detects the current **market regime** (Bullish / Bearish / Sideways)
- Fuses **price features + news sentiment** for robust predictions
- Quantifies **prediction uncertainty** (MC Dropout + calibration)
- Generates **confidence-weighted trading signals** with risk management
- Learns optimal **position sizing via RL** (PPO agent)

**Universe:** Nifty 50 stocks (NSE, India) · **Data:** 2019–2025

---

## Architecture Overview

```
Phase 1  →  HMM Regime Labeling                              ✅ COMPLETE
Phase 2  →  Transformer Regime Classifier (+ Stock Embeds)    ✅ COMPLETE
Phase 3  →  Sentiment Data Pipeline (FinBERT + RSS News)      ✅ COMPLETE
Phase 4  →  Sentiment-Enriched Dual-Stream Transformer        ✅ COMPLETE
Phase 5  →  Uncertainty Quantification (MC Dropout + Calib.)  ✅ COMPLETE
Phase 6  →  Trading Signal Generator                          ✅ COMPLETE
Phase 7  →  Position Sizing & Risk Management                 ✅ COMPLETE
Phase 8  →  Backtesting & Paper Trading                       ✅ COMPLETE
Phase 9  →  RL Trading Agent (PPO)                            ✅ COMPLETE
```

### System Pipeline

```
Market Data (OHLCV)                    Financial News (RSS)
       │                                       │
  Phase 1: HMM Labeling                Phase 3: FinBERT Scoring
  (Bull/Bear/Sideways labels)          (sentiment features)
       │                                       │
       └──────────── Phase 4 ──────────────────┘
                 Dual-Stream Transformer
              (price encoder + sentiment encoder
               + cross-attention fusion)
                       │
              Phase 5: MC Dropout
           (uncertainty quantification
            + temperature calibration)
                       │
              Phase 6: Signal Generator
           (regime + confidence → LONG/SHORT/FLAT)
                       │
              Phase 7: Position Sizer
           (stop-losses, trailing stops,
            confidence-based sizing)
                       │
        ┌──────────────┴──────────────┐
  Phase 8: Backtester            Phase 9: RL Agent
  (walk-forward with             (PPO learns optimal
   Zerodha costs)                 position sizing)
```

---

## Phase 1 — HMM Regime Labeling ✅

Downloads 6 years of OHLCV data for **50 Nifty 50 stocks**, engineers 15+ technical features, and applies a **3-state Gaussian HMM** per ticker to label each trading day as Bull / Bear / Sideways.

| Regime | Avg % of Days |
|---|---|
| 🟢 Bull | 36.4% |
| 🔴 Bear | 21.8% |
| 🟡 Sideways | 41.8% |

**Go/No-Go:** All 5 validation metrics passed (return separation, persistence, Sharpe, confidence, SMA baseline).

```
Phase_1/
├── config.py               # Constants, tickers, paths
├── data_download.py        # yfinance downloader with caching
├── feature_engineering.py  # 15+ technical indicators
├── hmm_labeler.py          # GaussianHMM training, state mapping, smoothing
├── visualize.py            # Interactive Plotly charts
├── evaluate.py             # 10-metric evaluation + era breakdown
├── main.py                 # CLI pipeline orchestrator
└── requirements.txt
```

---

## Phase 2 — Transformer Regime Classifier ✅

**Transformer encoder** on 60-day sliding windows with **per-stock learnable embeddings** (16-dim) and dual prediction heads (regime + transition).

| Feature | Details |
|---|---|
| Architecture | d_model=64, 2 layers, 4 heads, ff_dim=128 |
| Stock Embeddings | 16-dim per stock (50 stocks) |
| Loss | Focal Loss (γ=2.0) + label smoothing (0.1) |
| Schedule | 5-epoch warmup → cosine decay (LR=3e-5) |
| Split | Train: 2019–2022, Val: 2023, Test: 2024 |

```
Phase_2/
├── config.py       ├── dataset.py      ├── model.py
├── train.py        ├── evaluate.py     ├── predict.py
└── requirements.txt
```

---

## Phase 3 — Sentiment Data Pipeline ✅

Collects financial news from **Indian RSS feeds** (MoneyControl, Economic Times, LiveMint, Google News) and scores with **FinBERT**. Produces per-ticker daily sentiment features + market-wide indicators (VIX, breadth, Nifty returns).

```
Phase_3/
├── config.py               ├── news_collector.py
├── sentiment_scorer.py     ├── market_features.py
├── feature_engineering.py  ├── evaluate.py
├── main.py                 └── requirements.txt
```

---

## Phase 4 — Sentiment-Enriched Transformer ✅

**Dual-stream architecture** fusing price features with sentiment via **cross-attention**:

1. **Price Encoder** — Transformer on 18 technical features (Phase 2 pretrained)
2. **Sentiment Encoder** — Small Transformer on 12 market/sentiment features
3. **Cross-Attention Fusion** — Gated residual price→sentiment attention
4. **Staged Training** — Freeze price encoder first, then fine-tune end-to-end

```
Phase_4/
├── config.py       ├── dataset.py      ├── model.py
├── train.py        ├── evaluate.py     ├── predict.py
├── main.py         └── requirements.txt
```

---

## Phase 5 — Uncertainty Quantification ✅

**MC Dropout** (N stochastic passes) + **temperature scaling** on Phase 4 model — no retraining needed. Enables selective trading by rejecting uncertain predictions.

| Component | Purpose |
|---|---|
| MC Dropout | Predictive mean + variance from N=30 forward passes |
| Temperature Scaling | Calibrate probabilities to match true frequencies |
| Selective Accuracy | Higher accuracy by rejecting low-confidence predictions |

```
Phase_5/
├── config.py       ├── dataset.py      ├── model_loader.py
├── mc_dropout.py   ├── calibration.py  ├── evaluate.py
├── predict.py      ├── main.py         └── requirements.txt
```

---

## Phase 6 — Trading Signal Generator ✅

Converts regime predictions + uncertainty into **LONG / SHORT / FLAT** signals with strength levels (STRONG/WEAK) based on confidence and uncertainty thresholds.

```
Phase_6/
├── config.py            ├── signal_generator.py
├── evaluate.py          ├── main.py
└── requirements.txt
```

---

## Phase 7 — Position Sizing & Risk Management ✅

**Daily rolling signals** with MC Dropout + **confidence-based position sizing**, stop-losses, trailing stops, and portfolio-level risk constraints.

```
Phase_7/
├── config.py            ├── daily_signals.py
├── position_sizer.py    ├── evaluate.py
├── main.py              └── requirements.txt
```

---

## Phase 8 — Backtesting & Paper Trading ✅

**Walk-forward backtester** with realistic **Zerodha transaction costs** (brokerage, STT, stamp duty, GST). Includes a **paper trading framework** that mimics Zerodha Kite API.

```
Phase_8/
├── config.py            ├── backtester.py
├── paper_trading.py     ├── evaluate.py
├── main.py              └── requirements.txt
```

---

## Phase 9 — RL Trading Agent ✅

**PPO-based reinforcement learning** agent that learns optimal position sizing from Transformer predictions, replacing rule-based signal generation.

| Component | Details |
|---|---|
| Environment | Custom Gym env: state = regime predictions + market context |
| Agent | PPO (stable-baselines3), continuous action space [0, 1] |
| Reward | Risk-adjusted returns with transaction cost penalties |
| Training | On historical data (2019–2023), evaluated on 2024 |

```
Phase_9/
├── config.py            ├── trading_env.py
├── train_rl.py          ├── evaluate.py
├── main.py              └── requirements.txt
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data | `yfinance`, RSS feeds (`feedparser`) |
| Features | `ta`, `numpy`, `pandas` |
| Regime Labeling | `hmmlearn` (GaussianHMM) |
| Deep Learning | `PyTorch` (Transformer encoder) |
| Sentiment | `FinBERT` (`transformers`, ProsusAI/finbert) |
| Uncertainty | MC Dropout, temperature scaling |
| RL Agent | `stable-baselines3`, `gymnasium` (PPO) |
| Backtesting | Custom engine with Zerodha cost model |
| Visualization | `plotly`, `tensorboard` |

---

## Design Philosophy

> **Regime-first.** Markets are non-stationary. By detecting regime *before* placing trades, the agent acts only when there is structural edge — either a clear trend or high-confidence prediction.

### Key Design Decisions

- **HMM for labeling** — Unsupervised, principled regime discovery with posterior probabilities
- **Stock embeddings** — Per-stock learned vectors capture stock-specific dynamics
- **Focal loss** — Prevents dominant Sideways class from overwhelming training
- **Dual-stream fusion** — Price + sentiment via cross-attention with gated residual
- **MC Dropout** — Epistemic uncertainty at inference time without retraining
- **Temperature calibration** — Trustworthy probabilities before acting
- **Staged training** — Pretrained encoder → freeze → fine-tune
- **Walk-forward validation** — Each phase validated on held-out time windows
- **Realistic costs** — Zerodha fee structure (STT, stamp duty, GST, brokerage)
- **RL position sizing** — PPO learns optimal allocation from regime + uncertainty signals
