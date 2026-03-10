# Regime Detection Model

A Self-Improving RL Trading Agent built in structured phases — starting from HMM-based market regime labeling and progressively extending to a full Safe RL trading system.

---

## Project Vision

The overall goal is to build a production-grade, risk-aware RL trading agent that:

- Detects the current **market regime** (Bullish / Bearish / Sideways/Choppy)
- Uses regime context to decide **whether to trade** and in which direction
- Optimizes **risk-adjusted return** (CVaR) rather than raw profit
- Learns to **adapt** to new market conditions via self-improving RL

**Universe:** Nifty 50 stocks (NSE, India)

---

## Architecture Roadmap

```
Phase 1  →  HMM Regime Labeling                              ✅ COMPLETE
Phase 2  →  Transformer Regime Classifier (+ Stock Embeds)    ✅ COMPLETE
Phase 3  →  Add Sentiment (FinBERT + news)                    🔜 NEXT
Phase 4  →  Uncertainty + Transition Detection heads
Phase 5  →  Safe RL Trading Agent (PPO/SAC + CVaR)
```

---

## Phase 1 — HMM Regime Labeling ✅

### What It Does

Downloads 6 years (2019–2025) of OHLCV data for all **50 Nifty 50 stocks**, engineers 15+ technical features, and applies a **3-state Gaussian HMM** per ticker to label each trading day as:

| Label | Color | Meaning |
|---|---|---|
| 🟢 Bull | Green | Trending upward, positive mean return |
| 🔴 Bear | Red | Trending downward, negative mean return |
| 🟡 Sideways | Yellow | Choppy/ranging, near-zero mean return |

State → regime mapping is principled: states sorted by **mean log return** so label assignment is automatic and data-driven.

### Results (49/50 stocks labelled)

| Regime | Avg % of Trading Days |
|---|---|
| 🟢 Bull | 36.4% |
| 🔴 Bear | 21.8% |
| 🟡 Sideways | 41.8% |

### Evaluation Results

Validation run across **49 Nifty 50 stocks** — all 5 hard requirements passed:

| Metric | Pass Rate | Result |
|---|---|---|
| Return Separation (Bull > Sideways > Bear, p < 0.05) | 71.4% | ✅ PASS |
| Regime Persistence (avg duration ≥ 10 days) | 100.0% | ✅ PASS |
| Regime-Filtered Strategy Sharpe > Buy & Hold | 89.8% | ✅ PASS |
| Posterior Confidence (avg max prob ≥ 0.60) | 100.0% | ✅ PASS |
| HMM Strategy Sharpe > SMA Crossover Baseline | - | ✅ PASS |

**Era-based validation across 5 market periods:**

| Era | Period | Dominant Regime |
|---|---|---|
| Pre-COVID | Jan 2019 – Jan 2020 | Bull |
| COVID Crash | Feb 2020 – Apr 2020 | Bear |
| Recovery | May 2020 – Dec 2021 | Bull |
| 2022 Bear | Jan 2022 – Dec 2022 | Bear/Sideways |
| 2023 Rally | Jan 2023 – Dec 2024 | Bull |

**Verdict: 🟢 GO — Phase 1 labels are reliable. Proceeding to Phase 2.**

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
└── requirements.txt        # Python dependencies
```

### Generated Plots

| Plot | Description |
|---|---|
| `regime_chart_<TICKER>.html` | Price + regime shading + posterior probabilities + returns |
| `regime_distribution.html` | Stacked bar: Bull/Bear/Sideways % per stock |
| `return_distribution_by_regime.html` | Violin plot validating return ordering by regime |
| `hmm_transition_heatmap.html` | How often each regime transitions to another |
| `regime_timeline.html` | 15-stock simultaneous regime timeline |

### Usage

**Local run:**
```bash
cd Phase_1
pip install -r requirements.txt
python main.py                     # full pipeline (cached data if available)
python main.py --force-download    # force fresh download
python main.py --ticker RELIANCE.NS  # single stock only
python evaluate.py                 # run all evaluation metrics
```

**SLURM (GPU cluster):**
```bash
sbatch run_phase1.slurm            # submit job to btech/small partition
squeue -u $USER                    # monitor job status
cat logs/phase1_<JOBID>.out        # view output
```

---

## Phase 2 — Transformer Regime Classifier ✅

### What It Does

Builds a **Transformer encoder** that takes sliding windows of Phase 1's technical features and predicts:

- **Head 1 — Regime Classification:** Next-day regime (Bull / Bear / Sideways) via 3-class classification
- **Head 2 — Transition Detection:** Binary flag for whether a regime change occurs within 5 days

Key innovation: **Per-stock learnable embeddings** allow the model to capture stock-specific regime patterns while sharing the core Transformer weights across all 50 stocks.

### Model Architecture

```
Input (60-day window × 18 features)
       │
       ├── Per-Stock Embedding (16-dim, concatenated to each timestep)
       │
       ▼
  Linear Projection → d_model=128
       │
  Positional Encoding (learnable)
       │
  4 × Transformer Encoder Layers
       │  (n_head=4, ff_dim=256, pre-LN, GELU, dropout=0.35)
       │
  [Sentiment Fusion Placeholder — Phase 4]
       │
  Global Pooling (last-token ⊕ mean-pool → project to d_model)
       │
       ├── Regime Head → 3-class softmax (Bull/Bear/Sideways)
       └── Transition Head → sigmoid (regime change within 5 days)
```

### Key Features

| Feature | Details |
|---|---|
| **18 Technical Features** | log returns, rolling vol, ATR, RSI, MACD, Bollinger, ADX, volume ratio, MA distance, linear regression slopes |
| **Stock Embeddings** | 16-dim learnable embedding per stock (50 stocks), concatenated to features at each timestep |
| **Focal Loss** | γ=2.0, per-class α from inverse-frequency weighting — focuses training on hard/minority samples |
| **Label Smoothing** | 0.1 — prevents overconfident predictions |
| **Transition Head** | BCEWithLogitsLoss, pos_weight=6.0 to handle ~13.5% positive rate |
| **LR Schedule** | 5-epoch linear warmup → cosine decay (LR=3e-5) |
| **Regularization** | Dropout=0.35, weight decay=1e-4, gradient clipping (max_norm=1.0) |
| **Early Stopping** | Patience=20 epochs on validation accuracy |
| **Time-Based Split** | Train: 2019–2022, Val: 2023, Test: 2024 (no data leakage) |

### Go / No-Go Thresholds

| Metric | Threshold | Purpose |
|---|---|---|
| Overall Test Accuracy | ≥ 75% | Basic model quality |
| High-Confidence Accuracy (prob > 0.7) | ≥ 80% | The model knows what it knows |
| Min Per-Class F1 | ≥ 0.65 | No class is neglected |
| Transition Recall | ≥ 60% | Catches regime changes |

### File Structure

```
Phase_2/
├── config.py           # Hyperparameters, paths, stock embedding config
├── dataset.py          # Sliding-window dataset with stock ID tracking
├── model.py            # Transformer encoder + stock embeddings + dual heads
├── train.py            # Training loop: focal loss, warmup+cosine LR, early stopping
├── evaluate.py         # Comprehensive metrics, confusion matrix, calibration, Go/No-Go
├── predict.py          # Inference module for single/batch stock prediction
└── requirements.txt    # Python dependencies
```

### Usage

**Local run:**
```bash
cd Phase_2
pip install -r requirements.txt
python train.py                         # full training (100 epochs)
python train.py --smoke-test            # quick 2-epoch validation
python train.py --lr 1e-4 --epochs 50   # custom hyperparameters
python evaluate.py                      # run full Go/No-Go evaluation
python predict.py --ticker RELIANCE.NS  # predict single stock regime
python predict.py --all                 # predict all Nifty 50
```

**SLURM (GPU cluster):**
```bash
sbatch run_phase2.slurm
squeue -u $USER
cat logs/phase2_<JOBID>.out
```

---

## Future Phases

### Phase 3 — Sentiment Integration 🔜

- **FinBERT** fine-tuned on Indian financial news for sentiment scoring
- News data aggregated per ticker per day (headline + article body)
- Sentiment features fused with price features via cross-attention in the existing Fusion Placeholder module
- Expected improvement: better transition detection by capturing news-driven regime shifts

### Phase 4 — Uncertainty & Enhanced Transition Detection

- Monte Carlo Dropout and/or ensemble-based **uncertainty quantification**
- Calibrated confidence scores for selective trading (only trade when model is confident)
- Enhanced transition head with lookahead labels and multi-horizon prediction

### Phase 5 — Safe RL Trading Agent

- **PPO / SAC** agent using regime predictions as state features
- **CVaR-constrained reward** for risk-aware position sizing
- Action space: Long / Short / Flat with position sizing
- Walk-forward backtesting with transaction costs and slippage
- Production deployment: FastAPI backend + React dashboard

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data | `yfinance` |
| Features | `ta`, `numpy`, `pandas` |
| Regime Model (Phase 1) | `hmmlearn` (GaussianHMM) |
| Classifier (Phase 2) | `torch` (Transformer encoder) |
| Visualization | `plotly`, `tensorboard` |
| Sentiment (Phase 3) | `FinBERT`, `transformers` |
| RL Agent (Phase 5) | `PPO / SAC`, CVaR reward |
| Backend (Phase 5) | `FastAPI`, `PostgreSQL` |
| Frontend (Phase 5) | `React` |
| Infra | `Docker`, CI/CD, SLURM |

---

## Design Philosophy

> **Regime-first.** Markets are non-stationary. Trading in the wrong regime (choppy/sideways) destroys returns via transaction costs and noise. By detecting regime *before* placing trades, the agent acts only when there is structural edge — either a clear uptrend or downtrend.

### Key Design Decisions

- **HMM for labeling**: Unsupervised, principled, industry-standard for latent regime discovery
- **State smoothing**: 3-day minimum run to avoid single-day noise labels
- **Posterior probabilities**: Every row carries `prob_Bull`, `prob_Bear`, `prob_Sideways` — giving the downstream model soft regime information rather than hard labels
- **Stock embeddings**: Learnable per-stock vectors let the Transformer capture stock-specific regime dynamics while sharing temporal pattern weights
- **Focal loss**: Prevents the dominant Sideways class from overwhelming training; focuses learning on hard Bear/transition samples
- **Dual-head architecture**: Regime + transition heads share the encoder, providing complementary learning signals
- **Walk-forward validation**: Each phase validated on held-out time windows before proceeding
