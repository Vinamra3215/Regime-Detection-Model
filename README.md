# Regime Detection Model

A Self-Improving RL Trading Agent built in structured phases — starting from HMM-based market regime labeling, through Transformer classification with sentiment fusion, to uncertainty-aware production predictions.

---

## Project Vision

Build a production-grade, risk-aware trading system that:

- Detects the current **market regime** (Bullish / Bearish / Sideways)
- Fuses **price features + news sentiment** for robust predictions
- Quantifies **prediction uncertainty** (MC Dropout + calibration)
- Only trades when the model is **confident and calibrated**

**Universe:** Nifty 50 stocks (NSE, India) · **Data:** 2019–2025

---

## Architecture Roadmap

```
Phase 1  →  HMM Regime Labeling                              ✅ COMPLETE
Phase 2  →  Transformer Regime Classifier (+ Stock Embeds)    ✅ COMPLETE
Phase 3  →  Sentiment Data Pipeline (FinBERT + RSS news)      ✅ COMPLETE
Phase 4  →  Sentiment-Enriched Dual-Stream Transformer        ✅ COMPLETE
Phase 5  →  Uncertainty Quantification (MC Dropout + Calib.)  ✅ COMPLETE
Phase 6  →  Safe RL Trading Agent (PPO/SAC + CVaR)            🔜 NEXT
```

---

## Phase 1 — HMM Regime Labeling ✅

### What It Does

Downloads 6 years of OHLCV data for all **50 Nifty 50 stocks**, engineers 15+ technical features, and applies a **3-state Gaussian HMM** per ticker to label each trading day as:

| Label | Meaning |
|---|---|
| 🟢 Bull | Trending upward, positive mean return |
| 🔴 Bear | Trending downward, negative mean return |
| 🟡 Sideways | Choppy/ranging, near-zero mean return |

State → regime mapping is principled: states sorted by **mean log return** so label assignment is automatic and data-driven.

### Results (49/50 stocks labelled)

| Regime | Avg % of Trading Days |
|---|---|
| 🟢 Bull | 36.4% |
| 🔴 Bear | 21.8% |
| 🟡 Sideways | 41.8% |

### Evaluation — Go/No-Go for Phase 2

| Metric | Pass Rate | Result |
|---|---|---|
| Return Separation (Bull > Sideways > Bear, p < 0.05) | 71.4% | ✅ PASS |
| Regime Persistence (avg duration ≥ 10 days) | 100.0% | ✅ PASS |
| Regime-Filtered Strategy Sharpe > Buy & Hold | 89.8% | ✅ PASS |
| Posterior Confidence (avg max prob ≥ 0.60) | 100.0% | ✅ PASS |
| HMM Strategy Sharpe > SMA Crossover Baseline | — | ✅ PASS |

**Era-based validation across 5 market periods:**

| Era | Period | Dominant Regime |
|---|---|---|
| Pre-COVID | Jan 2019 – Jan 2020 | Bull |
| COVID Crash | Feb 2020 – Apr 2020 | Bear |
| Recovery | May 2020 – Dec 2021 | Bull |
| 2022 Bear | Jan 2022 – Dec 2022 | Bear/Sideways |
| 2023 Rally | Jan 2023 – Dec 2024 | Bull |

**Verdict: 🟢 GO — Phase 1 labels are reliable.**

### File Structure

```
Phase_1/
├── config.py               # All constants, tickers, paths
├── data_download.py        # yfinance downloader with caching
├── feature_engineering.py  # 15+ technical indicators (ATR, RSI, MACD, ADX, BB)
├── hmm_labeler.py          # GaussianHMM training, state mapping, smoothing
├── visualize.py            # 5 interactive Plotly HTML charts
├── evaluate.py             # 10-metric evaluation: SMA baseline + 5 era breakdown
├── main.py                 # CLI pipeline orchestrator
└── requirements.txt
```

### Usage

```bash
cd Phase_1
pip install -r requirements.txt
python main.py                         # full pipeline
python main.py --force-download        # force fresh data download
python main.py --ticker RELIANCE.NS    # single stock
python evaluate.py                     # run evaluation metrics
```

---

## Phase 2 — Transformer Regime Classifier ✅

### What It Does

Builds a **Transformer encoder** that takes 60-day sliding windows of Phase 1's technical features and predicts:

- **Head 1 — Regime:** Next-day regime (Bull / Bear / Sideways) via 3-class classification
- **Head 2 — Transition:** Binary flag for regime change within 5 days

Key innovation: **Per-stock learnable embeddings** (16-dim) allow the model to capture stock-specific regime patterns while sharing Transformer weights across all 50 stocks.

### Model Architecture

```
Input (60-day window × 18 features)
       │
       ├── Per-Stock Embedding (16-dim, concat per timestep)
       ▼
  Linear Projection → d_model=64
       │
  Positional Encoding (learnable)
       │
  2 × Transformer Encoder Layers (n_head=4, ff_dim=128, pre-LN, GELU)
       │
  [Sentiment Fusion Placeholder — activated in Phase 4]
       │
  Global Pooling (last-token ⊕ mean-pool → project)
       │
       ├── Regime Head → 3-class softmax
       └── Transition Head → sigmoid
```

### Key Features

| Feature | Details |
|---|---|
| **Focal Loss** | γ=2.0, per-class α weights — focuses on hard/minority samples |
| **Label Smoothing** | 0.1 — prevents overconfident predictions |
| **Transition Head** | BCEWithLogitsLoss, pos_weight=6.0 (~13.5% positive rate) |
| **LR Schedule** | 5-epoch linear warmup → cosine decay (LR=3e-5) |
| **Regularization** | Dropout=0.35, weight decay=1e-3, gradient clipping |
| **Early Stopping** | Patience=20 on validation accuracy |
| **Time-Based Split** | Train: 2019–2022, Val: 2023, Test: 2024 |

### Go/No-Go Thresholds

| Metric | Threshold |
|---|---|
| Overall Test Accuracy | ≥ 75% |
| High-Confidence Accuracy (prob > 0.7) | ≥ 80% |
| Min Per-Class F1 | ≥ 0.65 |
| Transition Recall | ≥ 60% |

### File Structure

```
Phase_2/
├── config.py           # Hyperparameters, paths, stock embedding config
├── dataset.py          # Sliding-window dataset with stock ID tracking
├── model.py            # Transformer encoder + stock embeddings + dual heads
├── train.py            # Training: focal loss, warmup+cosine LR, early stopping
├── evaluate.py         # Metrics, confusion matrix, calibration, Go/No-Go
├── predict.py          # Inference for single/batch stock prediction
└── requirements.txt
```

### Usage

```bash
cd Phase_2
pip install -r requirements.txt
python train.py                         # full training (100 epochs)
python train.py --smoke-test            # quick 2-epoch validation
python evaluate.py                      # Go/No-Go evaluation
python predict.py --ticker RELIANCE.NS  # single stock prediction
python predict.py --all                 # all Nifty 50
```

---

## Phase 3 — Sentiment Data Pipeline ✅

### What It Does

Collects financial news from **Indian financial RSS feeds** and scores them with **FinBERT** (ProsusAI/finbert) to produce daily sentiment features per ticker.

### Data Sources

| Source | Feeds |
|---|---|
| **MoneyControl** | Markets, Business, Stocks RSS |
| **Economic Times** | Markets, Stocks, Companies RSS |
| **LiveMint** | Market, Companies RSS |
| **Google News** | Per-ticker RSS search |

### Sentiment Features

| Feature Type | Details |
|---|---|
| **Per-Ticker News** | FinBERT scores: positive, negative, neutral, compound per article |
| **Market-Wide** | India VIX (fear gauge), Nifty 50 returns (momentum), market breadth (advancers vs decliners) |
| **Sector-Level** | Aggregated sentiment by sector |
| **Temporal** | Rolling averages, sentiment momentum, volatility of sentiment |

### Pipeline

```
RSS Feeds → News Collection → FinBERT Scoring → Market Features (VIX, breadth)
    → Feature Engineering (daily per-ticker vectors) → Quality Evaluation
```

### File Structure

```
Phase_3/
├── config.py               # Paths, FinBERT params, feature specs
├── news_collector.py       # RSS feed collection from 4 Indian sources
├── sentiment_scorer.py     # FinBERT sentiment scoring engine
├── market_features.py      # VIX, Nifty returns, breadth, sector sentiment
├── feature_engineering.py  # Per-ticker daily sentiment feature vectors
├── evaluate.py             # Coverage, quality, correlation, Go/No-Go
├── main.py                 # Pipeline orchestrator
└── requirements.txt
```

### Usage

```bash
cd Phase_3
pip install -r requirements.txt
python main.py                  # full pipeline: collect → score → engineer → evaluate
python main.py --skip-collect   # skip news collection, use cached data
python evaluate.py              # standalone quality evaluation
```

---

## Phase 4 — Sentiment-Enriched Transformer ✅

### What It Does

Extends Phase 2's Transformer into a **dual-stream architecture** that fuses price features with sentiment features via **cross-attention**:

1. **Price Encoder** — Transformer on 18 technical features (initialized from Phase 2 pretrained weights)
2. **Sentiment Encoder** — Small Transformer on market/sentiment features from Phase 3
3. **Cross-Attention Fusion** — Price stream attends to sentiment stream via gated residual connection
4. **Dual Heads** — Regime classification (3-class) + Transition detection (binary)

### Architecture

```
Price Features (60 × 18)          Sentiment Features (60 × 12)
       │                                    │
  Price Encoder                      Sentiment Encoder
  (Phase 2 pretrained)             (small Transformer)
       │                                    │
       └──────── Cross-Attention ───────────┘
                 (gated residual)
                       │
                 Global Pooling
                       │
              ├── Regime Head (3-class)
              └── Transition Head (binary)
```

### Training Strategy

| Stage | Description |
|---|---|
| **Stage 1** | Freeze pretrained price encoder, train only sentiment encoder + fusion + heads |
| **Stage 2** | Unfreeze all, fine-tune end-to-end with reduced LR |

### File Structure

```
Phase_4/
├── config.py           # Dual-stream config, fusion params, staged training
├── dataset.py          # 5-tuple dataset: (X_price, X_sentiment, y_regime, y_transition, stock_id)
├── model.py            # Dual-stream Transformer with cross-attention fusion
├── train.py            # Staged training: freeze → unfreeze, pretrained init
├── evaluate.py         # Full evaluation with sentiment vs price-only comparison
├── predict.py          # Dual-stream inference module
├── main.py             # Pipeline orchestrator (train → evaluate → predict)
└── requirements.txt
```

### Usage

```bash
cd Phase_4
pip install -r requirements.txt
python main.py                  # full pipeline: train → evaluate → predict
python train.py                 # training only
python train.py --smoke-test    # quick 2-epoch validation
python evaluate.py              # evaluation and Go/No-Go
python predict.py --all         # predict all Nifty 50
```

---

## Phase 5 — Uncertainty Quantification ✅

### What It Does

Adds **uncertainty-aware inference** on top of the Phase 4 model using **MC Dropout** and **temperature scaling** — no retraining needed. The system knows when it doesn't know, enabling selective trading.

### Key Components

| Component | Description |
|---|---|
| **MC Dropout** | N stochastic forward passes with dropout enabled → predictive mean + variance |
| **Temperature Scaling** | Post-hoc calibration so predicted probabilities match true frequencies |
| **Selective Accuracy** | Only trade when uncertainty is below threshold → higher accuracy on acted predictions |
| **Transition Smoothing** | Smoothed transition probabilities reduce false regime-change signals |

### Uncertainty Pipeline

```
Phase 4 Model (frozen)
       │
  Temperature Calibration (on validation set)
       │
  MC Dropout Inference (N=30 forward passes per sample)
       │
  ├── Predictive Mean (regime probabilities)
  ├── Predictive Uncertainty (entropy / mutual information)
  ├── Calibrated Confidence
  └── Smoothed Transition Probability
       │
  Uncertainty-Aware Evaluation
       │
  ├── Selective Accuracy (reject uncertain predictions)
  ├── Coverage vs Accuracy tradeoff
  ├── Reliability Diagram (calibration plot)
  └── Go/No-Go for Phase 6
```

### File Structure

```
Phase_5/
├── config.py           # MC Dropout params, calibration settings, thresholds
├── dataset.py          # Reuses Phase 4 scalers, 5-tuple output
├── model_loader.py     # Loads Phase 4 model via importlib (avoids path conflicts)
├── mc_dropout.py       # MC Dropout stochastic inference engine
├── calibration.py      # Temperature scaling calibration
├── evaluate.py         # Uncertainty-aware metrics, selective accuracy, ECE
├── predict.py          # Production prediction with uncertainty estimates
├── main.py             # Pipeline: calibrate → MC inference → evaluate → Go/No-Go
└── requirements.txt
```

### Usage

```bash
cd Phase_5
pip install -r requirements.txt
python main.py                  # full pipeline: calibrate → infer → evaluate
python predict.py --all         # production predictions with uncertainty
python evaluate.py              # standalone uncertainty evaluation
```

---

## Future Work

### Phase 6 — Safe RL Trading Agent 🔜

- **PPO / SAC** agent using regime predictions + uncertainty as state features
- **CVaR-constrained reward** for risk-aware position sizing
- Action space: Long / Short / Flat with position sizing
- Walk-forward backtesting with transaction costs and slippage
- Production deployment: FastAPI backend + React dashboard

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data | `yfinance`, RSS feeds (`feedparser`) |
| Features | `ta`, `numpy`, `pandas` |
| Regime Labeling (Phase 1) | `hmmlearn` (GaussianHMM) |
| Classifier (Phase 2) | `PyTorch` (Transformer encoder) |
| Sentiment (Phase 3) | `FinBERT` (`transformers`, ProsusAI/finbert) |
| Fusion Model (Phase 4) | `PyTorch` (dual-stream Transformer + cross-attention) |
| Uncertainty (Phase 5) | MC Dropout, temperature scaling |
| Visualization | `plotly`, `tensorboard` |
| Infra | SLURM (GPU cluster), `Docker`, CI/CD |

---

## Design Philosophy

> **Regime-first.** Markets are non-stationary. Trading in the wrong regime destroys returns via transaction costs and noise. By detecting regime *before* placing trades, the agent acts only when there is structural edge.

### Key Design Decisions

- **HMM for labeling** — Unsupervised, principled, industry-standard for latent regime discovery
- **State smoothing** — 3-day minimum run to avoid single-day noise labels
- **Posterior probabilities** — Every sample carries `prob_Bull`, `prob_Bear`, `prob_Sideways` for soft downstream consumption
- **Stock embeddings** — Learnable per-stock vectors capture stock-specific dynamics while sharing temporal weights
- **Focal loss** — Prevents dominant Sideways class from overwhelming training
- **Dual-head architecture** — Regime + transition heads share encoder for complementary learning
- **Sentiment fusion via cross-attention** — Price stream attends to sentiment, gated to avoid noise when sentiment is uninformative
- **Staged training** — Freeze pretrained price encoder first, then fine-tune end-to-end
- **MC Dropout uncertainty** — No retraining needed; quantifies epistemic uncertainty at inference time
- **Temperature calibration** — Ensures predicted probabilities are trustworthy before acting on them
- **Walk-forward validation** — Each phase validated on held-out time windows before proceeding
