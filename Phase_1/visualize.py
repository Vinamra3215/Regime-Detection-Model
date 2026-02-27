import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from pathlib import Path

from config import REGIME_COLORS, REGIME_ORDER, PLOTS_DIR

log = logging.getLogger(__name__)


def plot_regime_chart(df: pd.DataFrame, ticker: str, save: bool = True) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.04,
        subplot_titles=[
            f"{ticker} — Close Price with Regime Labels",
            "Regime Posterior Probabilities",
            "Daily Log Return"
        ]
    )

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        mode="lines",
        line=dict(color="#E0E0E0", width=1.5),
        name="Close Price",
        hovertemplate="%{x|%Y-%m-%d}<br>Close: ₹%{y:.2f}<extra></extra>"
    ), row=1, col=1)

    _add_regime_shading(fig, df, row=1)

    prob_cols = {
        "prob_Bull":     ("#00C853", "P(Bull)"),
        "prob_Bear":     ("#DD2C00", "P(Bear)"),
        "prob_Sideways": ("#FFD600", "P(Sideways)"),
    }
    for col, (color, label) in prob_cols.items():
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col],
                mode="lines",
                line=dict(color=color, width=1.2),
                name=label,
                hovertemplate=f"{label}: %{{y:.3f}}<extra></extra>"
            ), row=2, col=1)

    if "log_return_1d" in df.columns:
        colors = df["Regime"].map(REGIME_COLORS).fillna("#888")
        fig.add_trace(go.Bar(
            x=df.index, y=df["log_return_1d"],
            marker_color=colors,
            name="Log Return",
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}<br>Return: %{y:.4f}<extra></extra>"
        ), row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=800,
        title=dict(text=f"<b>{ticker}</b> — HMM Regime Detection", x=0.5, font=dict(size=18)),
        legend=dict(orientation="h", y=1.02, x=0),
        hovermode="x unified",
        paper_bgcolor="#0D0D0D",
        plot_bgcolor="#1A1A1A",
        font=dict(family="Inter, Arial", color="#E0E0E0"),
        xaxis_rangeslider_visible=False,
    )
    fig.update_yaxes(gridcolor="#2A2A2A", zerolinecolor="#444")
    fig.update_xaxes(gridcolor="#2A2A2A")

    if save:
        path = PLOTS_DIR / f"regime_chart_{ticker.replace('^', 'INDEX_')}.html"
        fig.write_html(str(path))
        log.info(f"Saved: {path}")

    return fig


def _add_regime_shading(fig: go.Figure, df: pd.DataFrame, row: int = 1):
    if "Regime" not in df.columns:
        return

    current_regime = None
    block_start    = None
    intervals      = list(zip(df.index, df["Regime"]))
    intervals.append((None, None))

    for date, regime in intervals:
        if regime != current_regime:
            if current_regime is not None and block_start is not None:
                color = REGIME_COLORS.get(current_regime, "#888")
                fig.add_vrect(
                    x0=str(block_start), x1=str(prev_date),
                    fillcolor=color, opacity=0.12,
                    layer="below", line_width=0,
                    row=row, col=1,
                )
            current_regime = regime
            block_start    = date
        prev_date = date


def plot_regime_distribution(labelled: dict[str, pd.DataFrame], save: bool = True) -> go.Figure:
    rows = []
    for ticker, df in labelled.items():
        if "Regime" not in df.columns:
            continue
        counts = df["Regime"].value_counts(normalize=True) * 100
        row    = {"Ticker": ticker.replace(".NS", "")}
        for r in REGIME_ORDER:
            row[r] = counts.get(r, 0.0)
        rows.append(row)

    if not rows:
        return go.Figure()

    dist_df = pd.DataFrame(rows).sort_values("Bull", ascending=True)

    fig = go.Figure()
    for regime in REGIME_ORDER:
        fig.add_trace(go.Bar(
            y=dist_df["Ticker"],
            x=dist_df[regime],
            name=regime,
            orientation="h",
            marker_color=REGIME_COLORS[regime],
            hovertemplate=f"<b>%{{y}}</b><br>{regime}: %{{x:.1f}}%<extra></extra>"
        ))

    fig.update_layout(
        barmode="stack",
        template="plotly_dark",
        height=max(600, len(rows) * 18),
        title=dict(text="<b>Regime Distribution — Nifty 50 Stocks</b>", x=0.5, font=dict(size=16)),
        xaxis_title="% of Trading Days",
        legend=dict(orientation="h", y=1.02),
        paper_bgcolor="#0D0D0D",
        plot_bgcolor="#1A1A1A",
        font=dict(family="Inter, Arial", color="#E0E0E0"),
        xaxis=dict(range=[0, 100], gridcolor="#2A2A2A"),
        yaxis=dict(gridcolor="#2A2A2A"),
        margin=dict(l=120),
    )

    if save:
        path = PLOTS_DIR / "regime_distribution.html"
        fig.write_html(str(path))
        log.info(f"Saved: {path}")

    return fig


def plot_return_distribution(labelled: dict[str, pd.DataFrame],
                             tickers_sample: int = 10,
                             save: bool = True) -> go.Figure:
    all_rows       = []
    sample_tickers = list(labelled.keys())[:tickers_sample]

    for ticker in sample_tickers:
        df = labelled[ticker]
        if "Regime" not in df.columns or "log_return_1d" not in df.columns:
            continue
        sub = df[["log_return_1d", "Regime"]].dropna()
        sub["Ticker"] = ticker.replace(".NS", "")
        all_rows.append(sub)

    if not all_rows:
        return go.Figure()

    combined = pd.concat(all_rows, ignore_index=True)

    fig = go.Figure()
    for regime in REGIME_ORDER:
        sub = combined[combined["Regime"] == regime]
        fig.add_trace(go.Violin(
            y=sub["log_return_1d"],
            name=regime,
            box_visible=True,
            meanline_visible=True,
            fillcolor=REGIME_COLORS[regime],
            opacity=0.7,
            line_color="white",
            hoverinfo="y+name",
        ))

    fig.update_layout(
        template="plotly_dark",
        title=dict(text="<b>Daily Return Distribution by Regime</b>", x=0.5, font=dict(size=16)),
        yaxis_title="Log Return (1d)",
        yaxis=dict(zeroline=True, zerolinecolor="#555", gridcolor="#2A2A2A",
                   range=[-0.12, 0.12]),
        paper_bgcolor="#0D0D0D",
        plot_bgcolor="#1A1A1A",
        font=dict(family="Inter, Arial", color="#E0E0E0"),
        height=550,
        legend=dict(orientation="h"),
    )

    if save:
        path = PLOTS_DIR / "return_distribution_by_regime.html"
        fig.write_html(str(path))
        log.info(f"Saved: {path}")

    return fig


def plot_transition_heatmap(labelled: dict[str, pd.DataFrame],
                            reference_ticker: str = None,
                            save: bool = True) -> go.Figure:
    if reference_ticker is None:
        reference_ticker = list(labelled.keys())[0]

    df = labelled.get(reference_ticker)
    if df is None or "Regime" not in df.columns:
        return go.Figure()

    regimes   = REGIME_ORDER
    trans_mat = pd.DataFrame(0.0, index=regimes, columns=regimes)

    prev = None
    for r in df["Regime"].values:
        if prev is not None:
            trans_mat.loc[prev, r] += 1
        prev = r

    row_sums  = trans_mat.sum(axis=1)
    trans_pct = trans_mat.div(row_sums, axis=0).fillna(0) * 100

    fig = go.Figure(go.Heatmap(
        z=trans_pct.values,
        x=regimes,
        y=regimes,
        colorscale="Plasma",
        zmin=0, zmax=100,
        text=[[f"{v:.1f}%" for v in row] for row in trans_pct.values],
        texttemplate="%{text}",
        textfont=dict(size=14, color="white"),
        hovertemplate="From: %{y}<br>To: %{x}<br>Prob: %{z:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"<b>Regime Transition Matrix — {reference_ticker.replace('.NS','')}</b>",
            x=0.5, font=dict(size=16)
        ),
        xaxis_title="Next Regime",
        yaxis_title="Current Regime",
        height=450,
        paper_bgcolor="#0D0D0D",
        plot_bgcolor="#1A1A1A",
        font=dict(family="Inter, Arial", color="#E0E0E0"),
    )

    if save:
        path = PLOTS_DIR / "hmm_transition_heatmap.html"
        fig.write_html(str(path))
        log.info(f"Saved: {path}")

    return fig


def plot_regime_timeline(labelled: dict[str, pd.DataFrame],
                         top_n: int = 15,
                         save: bool = True) -> go.Figure:
    tickers = list(labelled.keys())[:top_n]

    fig = go.Figure()

    for ticker in tickers:
        df    = labelled[ticker]
        if "Regime" not in df.columns:
            continue
        label = ticker.replace(".NS", "")
        fig.add_trace(go.Scatter(
            x=df.index,
            y=[label] * len(df),
            mode="markers",
            marker=dict(
                symbol="square",
                size=4,
                color=[REGIME_COLORS.get(r, "#888") for r in df["Regime"]],
                opacity=0.9,
            ),
            name=label,
            showlegend=False,
            hovertemplate=f"<b>{label}</b><br>%{{x|%Y-%m-%d}}<br>Regime: %{{text}}<extra></extra>",
            text=df["Regime"].values,
        ))

    for regime, color in REGIME_COLORS.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=10, color=color, symbol="square"),
            name=regime,
            showlegend=True,
        ))

    fig.update_layout(
        template="plotly_dark",
        title=dict(text=f"<b>Regime Timeline — Top {top_n} Nifty 50 Stocks</b>", x=0.5, font=dict(size=16)),
        height=max(500, top_n * 30),
        paper_bgcolor="#0D0D0D",
        plot_bgcolor="#1A1A1A",
        font=dict(family="Inter, Arial", color="#E0E0E0"),
        legend=dict(orientation="h", y=1.02),
        xaxis=dict(gridcolor="#2A2A2A"),
        yaxis=dict(gridcolor="#2A2A2A"),
        hovermode="closest",
    )

    if save:
        path = PLOTS_DIR / "regime_timeline.html"
        fig.write_html(str(path))
        log.info(f"Saved: {path}")

    return fig


def generate_all_plots(labelled: dict[str, pd.DataFrame],
                       per_ticker_limit: int = 10):
    log.info("\n📊 Generating Plotly visualisations...")

    for ticker in list(labelled.keys())[:per_ticker_limit]:
        try:
            plot_regime_chart(labelled[ticker], ticker)
        except Exception as e:
            log.error(f"Error plotting {ticker}: {e}")

    try:
        plot_regime_distribution(labelled)
    except Exception as e:
        log.error(f"Error in regime distribution plot: {e}")

    try:
        plot_return_distribution(labelled)
    except Exception as e:
        log.error(f"Error in return distribution plot: {e}")

    try:
        plot_transition_heatmap(labelled)
    except Exception as e:
        log.error(f"Error in transition heatmap: {e}")

    try:
        plot_regime_timeline(labelled)
    except Exception as e:
        log.error(f"Error in regime timeline: {e}")

    log.info(f"\n✅ All plots saved to: {PLOTS_DIR}")
    _print_plot_list()


def _print_plot_list():
    plots = sorted(PLOTS_DIR.glob("*.html"))
    print(f"\n{'─'*60}")
    print(f"  📁 Generated {len(plots)} plot files in: {PLOTS_DIR}")
    print(f"{'─'*60}")
    for p in plots:
        print(f"  • {p.name}")
    print(f"{'─'*60}")
