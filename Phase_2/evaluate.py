
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import logging
import sys
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_recall_fscore_support, accuracy_score
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    DEVICE, CHECKPOINT_DIR, PLOTS_DIR, NIFTY_50_TICKERS,
    IDX_TO_REGIME, NUM_CLASSES, BATCH_SIZE,
    MIN_TEST_ACCURACY, MIN_HIGH_CONF_ACCURACY,
    MIN_PER_CLASS_F1, MIN_TRANSITION_RECALL,
    HIGH_CONF_THRESHOLD,
)
from dataset import create_dataloaders
from model import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PASS = "✅ PASS"
FAIL = "❌ FAIL"


def load_best_model() -> tuple[nn.Module, dict]:
    ckpt_path = CHECKPOINT_DIR / "best_model.pt"
    if not ckpt_path.exists():
        log.error(f"No checkpoint found at {ckpt_path}. Run train.py first.")
        sys.exit(1)

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    num_features = checkpoint["num_features"]

    model = build_model(num_features).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    log.info(f"Loaded best model from epoch {checkpoint['epoch']} "
             f"(val_acc={checkpoint['val_acc']:.4f})")

    return model, checkpoint


@torch.no_grad()
def collect_predictions(model: nn.Module, loader) -> dict:
    model.eval()

    all_probs       = []
    all_labels      = []
    all_trans_probs = []
    all_trans_labels = []

    for X, y_regime, y_transition, stock_ids in loader:
        X = X.to(DEVICE)
        stock_ids = stock_ids.to(DEVICE)
        output = model(X, stock_ids=stock_ids)

        all_probs.append(output["regime_probs"].cpu())
        all_labels.append(y_regime)
        all_trans_probs.append(output["transition_prob"].cpu().squeeze())
        all_trans_labels.append(y_transition)

    return {
        "probs":        torch.cat(all_probs).numpy(),
        "labels":       torch.cat(all_labels).numpy(),
        "preds":        torch.cat(all_probs).argmax(dim=1).numpy(),
        "trans_probs":  torch.cat(all_trans_probs).numpy(),
        "trans_labels": torch.cat(all_trans_labels).numpy(),
    }


def evaluate_regime(results: dict) -> dict:
    labels = results["labels"]
    preds  = results["preds"]
    probs  = results["probs"]

    accuracy = accuracy_score(labels, preds)

    class_names = [IDX_TO_REGIME[i] for i in range(NUM_CLASSES)]
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(NUM_CLASSES)), zero_division=0
    )

    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))

    max_probs = probs.max(axis=1)
    high_conf_mask = max_probs >= HIGH_CONF_THRESHOLD
    high_conf_acc = None
    high_conf_count = int(high_conf_mask.sum())
    if high_conf_count > 10:
        high_conf_acc = accuracy_score(labels[high_conf_mask], preds[high_conf_mask])

    min_f1 = f1.min()

    return {
        "accuracy":          accuracy,
        "precision":         {class_names[i]: precision[i] for i in range(NUM_CLASSES)},
        "recall":            {class_names[i]: recall[i] for i in range(NUM_CLASSES)},
        "f1":                {class_names[i]: f1[i] for i in range(NUM_CLASSES)},
        "support":           {class_names[i]: int(support[i]) for i in range(NUM_CLASSES)},
        "confusion_matrix":  cm,
        "min_f1":            min_f1,
        "high_conf_acc":     high_conf_acc,
        "high_conf_count":   high_conf_count,
        "high_conf_frac":    high_conf_mask.mean(),
        "class_names":       class_names,
    }


def evaluate_transition(results: dict) -> dict:
    trans_labels = results["trans_labels"]
    trans_probs  = results["trans_probs"]
    trans_preds  = (trans_probs >= 0.5).astype(int)

    accuracy = accuracy_score(trans_labels, trans_preds)

    tp = ((trans_preds == 1) & (trans_labels == 1)).sum()
    fp = ((trans_preds == 1) & (trans_labels == 0)).sum()
    fn = ((trans_preds == 0) & (trans_labels == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "pos_rate":  trans_labels.mean(),
    }


def evaluate_calibration(results: dict, n_bins: int = 10) -> dict:
    probs  = results["probs"]
    labels = results["labels"]
    preds  = results["preds"]

    max_probs = probs.max(axis=1)
    correct   = (preds == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (max_probs >= lo) & (max_probs < hi)
        if mask.sum() == 0:
            continue

        avg_conf = max_probs[mask].mean()
        avg_acc  = correct[mask].mean()
        count    = int(mask.sum())
        ece     += abs(avg_conf - avg_acc) * count

        bin_data.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "avg_conf": avg_conf,
            "avg_acc": avg_acc,
            "count": count,
        })

    ece /= len(max_probs)
    return {"ece": ece, "bins": bin_data}


def print_report(regime_metrics: dict, trans_metrics: dict,
                 calibration: dict) -> bool:

    print("\n" + "═" * 70)
    print("  📊  PHASE 2 EVALUATION — GO / NO-GO FOR PHASE 3")
    print("═" * 70)

    acc = regime_metrics["accuracy"]
    acc_pass = acc >= MIN_TEST_ACCURACY
    print(f"\n  ── REGIME CLASSIFICATION ─────────────────────────────────────────")
    print(f"  Overall Test Accuracy:     {acc:.4f} ({acc*100:.1f}%)  "
          f"{'  ' + PASS if acc_pass else '  ' + FAIL} (threshold: {MIN_TEST_ACCURACY*100:.0f}%)")

    hc_acc = regime_metrics["high_conf_acc"]
    hc_pass = hc_acc is not None and hc_acc >= MIN_HIGH_CONF_ACCURACY
    if hc_acc is not None:
        print(f"  High-Confidence Accuracy:  {hc_acc:.4f} ({hc_acc*100:.1f}%)  "
              f"{'  ' + PASS if hc_pass else '  ' + FAIL} "
              f"(threshold: {MIN_HIGH_CONF_ACCURACY*100:.0f}%, "
              f"n={regime_metrics['high_conf_count']}, "
              f"{regime_metrics['high_conf_frac']*100:.1f}% of test)")
    else:
        hc_pass = False
        print(f"  High-Confidence Accuracy:  N/A (too few high-confidence predictions)  {FAIL}")

    min_f1 = regime_metrics["min_f1"]
    f1_pass = min_f1 >= MIN_PER_CLASS_F1
    print(f"  Min Per-Class F1:          {min_f1:.4f}  "
          f"{'  ' + PASS if f1_pass else '  ' + FAIL} (threshold: {MIN_PER_CLASS_F1})")

    print(f"\n  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("  " + "─" * 55)
    for cls in regime_metrics["class_names"]:
        p = regime_metrics["precision"][cls]
        r = regime_metrics["recall"][cls]
        f1_val = regime_metrics["f1"][cls]
        s = regime_metrics["support"][cls]
        print(f"  {cls:<12} {p:>10.4f} {r:>10.4f} {f1_val:>10.4f} {s:>10}")

    cm = regime_metrics["confusion_matrix"]
    print(f"\n  Confusion Matrix (rows=true, cols=predicted):")
    print(f"  {'':>12}  {'Bear':>8} {'Sideways':>8} {'Bull':>8}")
    for i, cls in enumerate(regime_metrics["class_names"]):
        print(f"  {cls:>12}  {cm[i,0]:>8} {cm[i,1]:>8} {cm[i,2]:>8}")

    print(f"\n  ── TRANSITION DETECTION ──────────────────────────────────────────")
    trans_recall = trans_metrics["recall"]
    trans_pass = trans_recall >= MIN_TRANSITION_RECALL
    print(f"  Transition Recall:    {trans_recall:.4f}  "
          f"{'  ' + PASS if trans_pass else '  ' + FAIL} (threshold: {MIN_TRANSITION_RECALL})")
    print(f"  Transition Precision: {trans_metrics['precision']:.4f}")
    print(f"  Transition F1:        {trans_metrics['f1']:.4f}")
    print(f"  Transition Rate:      {trans_metrics['pos_rate']:.3f} (% of days with upcoming transition)")

    print(f"\n  ── CALIBRATION ───────────────────────────────────────────────────")
    print(f"  Expected Calibration Error (ECE): {calibration['ece']:.4f}")

    hard_checks = [
        ("Test Accuracy ≥ 75%",          acc_pass),
        ("High-Conf Accuracy ≥ 80%",     hc_pass),
        ("Min Per-Class F1 ≥ 0.65",     f1_pass),
        ("Transition Recall ≥ 60%",      trans_pass),
    ]

    all_pass = all(p for _, p in hard_checks)

    print(f"\n  ── GO / NO-GO CHECKLIST ──────────────────────────────────────────")
    for label, passed in hard_checks:
        print(f"  {'  ' + PASS if passed else '  ' + FAIL}  {label}")

    print("\n" + "═" * 70)
    if all_pass:
        print("  VERDICT: 🟢  GO — Transformer model meets all thresholds. Proceed to Phase 3.")
    else:
        passed_count = sum(1 for _, p in hard_checks if p)
        print(f"  VERDICT: 🔴  NO-GO — {passed_count}/4 checks passed. "
              f"Improve model before proceeding.")
    print("═" * 70 + "\n")

    return all_pass


def plot_evaluation_dashboard(regime_metrics: dict, trans_metrics: dict,
                              calibration: dict, results: dict):

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Confusion Matrix",
            "Per-Class F1 Scores",
            "Calibration Plot",
            "Confidence Distribution",
        ],
        specs=[
            [{"type": "heatmap"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "histogram"}],
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    class_names = regime_metrics["class_names"]

    cm = regime_metrics["confusion_matrix"]
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    fig.add_trace(go.Heatmap(
        z=cm_pct, x=class_names, y=class_names,
        colorscale="Plasma", zmin=0, zmax=100,
        text=[[f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)" for j in range(3)] for i in range(3)],
        texttemplate="%{text}",
        textfont=dict(size=12, color="white"),
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z:.1f}%<extra></extra>",
        showscale=False,
    ), row=1, col=1)

    f1_values = [regime_metrics["f1"][cls] for cls in class_names]
    colors = ["#00C853", "#FFD600", "#DD2C00"]
    fig.add_trace(go.Bar(
        x=class_names, y=f1_values,
        marker_color=colors,
        text=[f"{v:.3f}" for v in f1_values],
        textposition="outside",
    ), row=1, col=2)
    fig.add_hline(y=MIN_PER_CLASS_F1, line_dash="dash", line_color="#FF6D00",
                  annotation_text=f"Threshold ({MIN_PER_CLASS_F1})", row=1, col=2)

    if calibration["bins"]:
        confs = [b["avg_conf"] for b in calibration["bins"]]
        accs  = [b["avg_acc"] for b in calibration["bins"]]
        fig.add_trace(go.Scatter(
            x=confs, y=accs, mode="markers+lines",
            marker=dict(color="#00C853", size=8),
            name="Model",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash", color="#888"),
            name="Perfect",
        ), row=2, col=1)

    max_probs = results["probs"].max(axis=1)
    for i, (cls, color) in enumerate(zip(class_names, colors)):
        mask = results["labels"] == i
        fig.add_trace(go.Histogram(
            x=max_probs[mask], nbinsx=30,
            marker_color=color, opacity=0.6,
            name=cls,
        ), row=2, col=2)

    fig.update_layout(
        template="plotly_dark",
        height=800,
        title=dict(text="<b>Phase 2 — Transformer Evaluation Dashboard</b>",
                   x=0.5, font=dict(size=18)),
        paper_bgcolor="#0D0D0D",
        plot_bgcolor="#1A1A1A",
        font=dict(family="Inter, Arial", color="#E0E0E0"),
        showlegend=False,
    )
    fig.update_yaxes(gridcolor="#2A2A2A", zerolinecolor="#555")
    fig.update_xaxes(gridcolor="#2A2A2A")

    path = PLOTS_DIR / "phase2_evaluation_dashboard.html"
    fig.write_html(str(path))
    log.info(f"Saved dashboard: {path}")


def main():
    print("\n" + "═" * 70)
    print("  🔬  PHASE 2 — TRANSFORMER MODEL EVALUATION")
    print("═" * 70 + "\n")

    model, checkpoint = load_best_model()

    _, _, test_loader, _ = create_dataloaders()

    log.info("Running inference on test set...")
    results = collect_predictions(model, test_loader)
    log.info(f"Collected {len(results['labels'])} predictions")

    regime_metrics = evaluate_regime(results)
    trans_metrics  = evaluate_transition(results)
    calibration    = evaluate_calibration(results)

    go_decision = print_report(regime_metrics, trans_metrics, calibration)

    log.info("Generating evaluation dashboard...")
    plot_evaluation_dashboard(regime_metrics, trans_metrics, calibration, results)
    print(f"  📊 Dashboard saved: {PLOTS_DIR}/phase2_evaluation_dashboard.html\n")

    sys.exit(0 if go_decision else 1)


if __name__ == "__main__":
    main()

