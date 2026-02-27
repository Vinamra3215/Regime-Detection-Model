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
Phase 1  →  HMM Regime Labeling          ✅ COMPLETE
Phase 2  →  Transformer Regime Classifier (price features only)
Phase 3  →  Add Sentiment (FinBERT + news)
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

### Evaluation Results (Go / No-Go for Phase 2)

Validation run across **49 Nifty 50 stocks** — all 4 hard requirements passed:

| Metric | Pass Rate | Result |
|---|---|---|
| Return Separation (Bull > Sideways > Bear, p < 0.05) | 71.4% | ✅ PASS |
| Regime Persistence (avg duration ≥ 10 days) | 100.0% | ✅ PASS |
| Regime-Filtered Strategy Sharpe > Buy & Hold | 89.8% | ✅ PASS |
| Posterior Confidence (avg max prob ≥ 0.60) | 100.0% | ✅ PASS |

**Verdict: 🟢 GO — Phase 1 labels are reliable. Proceeding to Phase 2.**

Sample Sharpe improvements via regime filtering:

| Stock | Strategy Sharpe | Buy & Hold Sharpe |
|---|---|---|
| SBIN | 1.97 | 0.56 |
| HINDUNILVR | 1.31 | 0.29 |
| INFY | 1.49 | 0.70 |
| BHARTIARTL | 1.49 | 0.97 |

### File Structure

```
Phase_1/
├── config.py               # All constants, tickers, paths
├── data_download.py        # yfinance downloader with caching
├── feature_engineering.py  # 15+ technical indicators (ATR, RSI, MACD, ADX, BB, etc.)
├── hmm_labeler.py          # GaussianHMM training, state mapping, smoothing
├── visualize.py            # 5 interactive Plotly HTML charts
├── evaluate.py             # 8-metric Go/No-Go evaluation with Plotly dashboard
├── main.py                 # CLI pipeline orchestrator
├── requirements.txt        # Python dependencies
└── outputs/
    ├── data/raw/           # Downloaded OHLCV CSVs
    ├── data/labelled/      # Regime-labelled CSVs (with posterior probs)
    ├── data/models/        # Trained HMM .pkl files
    └── plots/              # Interactive HTML Plotly charts
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

```bash
cd Phase_1

# Install dependencies
pip install -r requirements.txt

# Run full pipeline (uses cached data if available)
python main.py

# Force fresh data download
python main.py --force-download

# Single stock only
python main.py --ticker RELIANCE.NS

# Skip plot generation
python main.py --skip-plots
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data | `yfinance` |
| Features | `ta`, `numpy`, `pandas` |
| Regime Model | `hmmlearn` (GaussianHMM) |
| Visualization | `plotly` |
| ML (Phase 2+) | `torch`, Transformer encoder |
| Sentiment (Phase 3+) | `FinBERT` |
| RL Agent (Phase 5) | `PPO / SAC`, CVaR reward |
| Backend (Phase 5) | `FastAPI`, `PostgreSQL` |
| Frontend (Phase 5) | `React` |
| Infra | `Docker`, CI/CD |

---

## Design Philosophy

> **Regime-first.** Markets are non-stationary. Trading in the wrong regime (choppy/sideways) destroys returns via transaction costs and noise. By detecting regime *before* placing trades, the agent acts only when there is structural edge — either a clear uptrend or downtrend.

### Key Design Decisions

- **HMM for labeling**: Unsupervised, principled, industry-standard for latent regime discovery
- **State smoothing**: 3-day minimum run to avoid single-day noise labels
- **Posterior probabilities**: Every row carries `prob_Bull`, `prob_Bear`, `prob_Sideways` — giving the downstream RL agent soft regime information rather than hard labels
- **Walk-forward validation**: Each phase validated on held-out time windows before proceeding
