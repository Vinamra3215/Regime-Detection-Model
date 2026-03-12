
import numpy as np
import pandas as pd
import logging
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, precision_recall_fscore_support
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    IDX_TO_REGIME, NUM_CLASSES, RESULTS_DIR, PLOTS_DIR,
    HIGH_CONF_THRESHOLD, UNCERTAINTY_THRESHOLD, ABSTAIN_THRESHOLD,
    TRANSITION_SMOOTH_WINDOW, TRANSITION_THRESHOLD, MIN_REGIME_DURATION,
    MIN_TEST_ACCURACY, MIN_HIGH_CONF_ACCURACY, MIN_PER_CLASS_F1,
    MIN_TRANSITION_RECALL, MIN_COVERAGE_RATE,
)

log = logging.getLogger(__name__)

PASS = "✅ PASS"
FAIL = "❌ FAIL"


def compute_standard_metrics(preds, labels, probs):
    acc = accuracy_score(labels, preds)
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])

    report = classification_report(
        labels, preds,
        target_names=["Bear", "Sideways", "Bull"],
        zero_division=0
    )

    return {
        "accuracy": acc,
        "per_class_f1": per_class_f1,
        "macro_f1": macro_f1,
        "min_f1": float(per_class_f1.min()),
        "confusion_matrix": cm,
        "report": report,
    }


def compute_selective_metrics(preds, labels, confidence, uncertainty,
                               conf_threshold, unc_threshold):
    conf_mask = confidence >= conf_threshold
    unc_mask  = uncertainty < unc_threshold
    combined_mask = conf_mask & unc_mask

    total = len(preds)
    selected = combined_mask.sum()
    coverage = selected / total if total > 0 else 0

    if selected == 0:
        return {
            "selective_accuracy": 0.0,
            "coverage": 0.0,
            "n_selected": 0,
            "n_total": total,
        }

    sel_acc = accuracy_score(labels[combined_mask], preds[combined_mask])
    sel_f1  = f1_score(labels[combined_mask], preds[combined_mask],
                       average="macro", zero_division=0)

    return {
        "selective_accuracy": sel_acc,
        "selective_f1": sel_f1,
        "coverage": coverage,
        "n_selected": int(selected),
        "n_total": total,
    }


def compute_ece(probs, labels, n_bins=15):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies  = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > low) & (confidences <= high)
        n_in_bin = mask.sum()

        if n_in_bin > 0:
            avg_conf = confidences[mask].mean()
            avg_acc  = accuracies[mask].mean()
            ece += (n_in_bin / len(labels)) * abs(avg_acc - avg_conf)
            bin_data.append({
                "bin": f"({low:.2f}, {high:.2f}]",
                "count": int(n_in_bin),
                "avg_confidence": float(avg_conf),
                "avg_accuracy": float(avg_acc),
                "gap": float(abs(avg_acc - avg_conf)),
            })

    return float(ece), bin_data


def compute_transition_metrics(trans_probs, trans_labels, trans_uncertainty,
                                threshold=TRANSITION_THRESHOLD,
                                smooth_window=TRANSITION_SMOOTH_WINDOW):
    smoothed = pd.Series(trans_probs).rolling(
        smooth_window, center=True, min_periods=1
    ).mean().values

    raw_preds = (trans_probs >= threshold).astype(int)
    smooth_preds = (smoothed >= threshold).astype(int)

    unc_mask = trans_uncertainty < np.percentile(trans_uncertainty, 75)
    filtered_preds = smooth_preds.copy()
    filtered_preds[~unc_mask] = 0

    def _metrics(preds, name):
        tp = ((preds == 1) & (trans_labels == 1)).sum()
        fp = ((preds == 1) & (trans_labels == 0)).sum()
        fn = ((preds == 0) & (trans_labels == 1)).sum()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
        return {
            f"{name}_recall": float(recall),
            f"{name}_precision": float(precision),
            f"{name}_f1": float(f1),
        }

    results = {}
    results.update(_metrics(raw_preds, "raw"))
    results.update(_metrics(smooth_preds, "smooth"))
    results.update(_metrics(filtered_preds, "filtered"))

    return results


def run_go_no_go(standard, selective, transition, ece):
    log.info("\n" + "=" * 60)
    log.info("GO / NO-GO — Phase 5 → Phase 6")
    log.info("=" * 60)

    checks = {}

    val = standard["accuracy"]
    passed = val >= MIN_TEST_ACCURACY
    checks["Test Accuracy"] = {"value": val, "threshold": MIN_TEST_ACCURACY,
                                "result": PASS if passed else FAIL}
    log.info(f"  1. Test Accuracy: {val*100:.1f}% "
             f"(≥{MIN_TEST_ACCURACY*100:.0f}%) → {checks['Test Accuracy']['result']}")

    val = selective["selective_accuracy"]
    passed = val >= MIN_HIGH_CONF_ACCURACY
    checks["High-Conf Accuracy (uncertainty-aware)"] = {
        "value": val, "threshold": MIN_HIGH_CONF_ACCURACY,
        "result": PASS if passed else FAIL
    }
    log.info(f"  2. High-Conf Accuracy: {val*100:.1f}% "
             f"(≥{MIN_HIGH_CONF_ACCURACY*100:.0f}%) → {checks['High-Conf Accuracy (uncertainty-aware)']['result']}")

    val = standard["min_f1"]
    passed = val >= MIN_PER_CLASS_F1
    checks["Min Per-Class F1"] = {"value": val, "threshold": MIN_PER_CLASS_F1,
                                   "result": PASS if passed else FAIL}
    log.info(f"  3. Min Per-Class F1: {val:.3f} "
             f"(≥{MIN_PER_CLASS_F1}) → {checks['Min Per-Class F1']['result']}")

    val = transition.get("smooth_recall", transition.get("raw_recall", 0))
    passed = val >= MIN_TRANSITION_RECALL
    checks["Transition Recall"] = {"value": val, "threshold": MIN_TRANSITION_RECALL,
                                    "result": PASS if passed else FAIL}
    log.info(f"  4. Transition Recall: {val*100:.1f}% "
             f"(≥{MIN_TRANSITION_RECALL*100:.0f}%) → {checks['Transition Recall']['result']}")

    val = selective["coverage"]
    passed = val >= MIN_COVERAGE_RATE
    checks["Coverage Rate"] = {"value": val, "threshold": MIN_COVERAGE_RATE,
                                "result": PASS if passed else FAIL}
    log.info(f"  5. Coverage Rate: {val*100:.1f}% "
             f"(≥{MIN_COVERAGE_RATE*100:.0f}%) → {checks['Coverage Rate']['result']}")

    total_pass = sum(1 for c in checks.values() if c["result"] == PASS)
    total = len(checks)
    verdict = "GO" if total_pass >= 4 else "NO-GO"

    log.info(f"\n  {'🟢' if verdict == 'GO' else '🔴'} VERDICT: {verdict} "
             f"({total_pass}/{total})")
    log.info("=" * 60)

    return {"checks": checks, "verdict": verdict,
            "passed": total_pass, "total": total}


def create_dashboard(standard, selective, transition, ece, ece_bins,
                     uncertainty_metrics, go_results):
    log.info("\nCreating evaluation dashboard...")

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "Confusion Matrix",
            "Calibration (Reliability Diagram)",
            "Uncertainty Distribution by Correctness",
            "Selective Accuracy vs Coverage",
            "Go/No-Go Results",
            "Per-Class F1 Scores",
        ],
        specs=[
            [{"type": "heatmap"}, {"type": "scatter"}],
            [{"type": "histogram"}, {"type": "scatter"}],
            [{"type": "table"}, {"type": "bar"}],
        ],
    )

    cm = standard["confusion_matrix"]
    regime_names = ["Bear", "Sideways", "Bull"]
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    text = [[f"{cm[i][j]}<br>({cm_pct[i][j]:.1f}%)"
             for j in range(3)] for i in range(3)]

    fig.add_trace(
        go.Heatmap(z=cm_pct, x=regime_names, y=regime_names,
                   text=text, texttemplate="%{text}", colorscale="Blues",
                   showscale=False),
        row=1, col=1,
    )

    if ece_bins:
        bin_confs = [b["avg_confidence"] for b in ece_bins]
        bin_accs  = [b["avg_accuracy"] for b in ece_bins]
        fig.add_trace(
            go.Scatter(x=bin_confs, y=bin_accs, mode="markers+lines",
                       name="Model", marker=dict(size=10)),
            row=1, col=2,
        )
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                       name="Perfect", line=dict(dash="dash", color="gray")),
            row=1, col=2,
        )

    preds = uncertainty_metrics["predicted_class"]
    labels = uncertainty_metrics["labels"]
    mi = uncertainty_metrics["mutual_information"]
    correct = preds == labels

    fig.add_trace(
        go.Histogram(x=mi[correct], name="Correct", opacity=0.7,
                     marker_color="#27ae60", nbinsx=40),
        row=2, col=1,
    )
    fig.add_trace(
        go.Histogram(x=mi[~correct], name="Incorrect", opacity=0.7,
                     marker_color="#e74c3c", nbinsx=40),
        row=2, col=1,
    )

    thresholds = np.linspace(0.01, 0.50, 30)
    sel_accs = []
    coverages = []
    for t in thresholds:
        mask = mi < t
        if mask.sum() > 0:
            sel_accs.append(accuracy_score(labels[mask], preds[mask]))
            coverages.append(mask.sum() / len(labels))
        else:
            sel_accs.append(0)
            coverages.append(0)

    fig.add_trace(
        go.Scatter(x=coverages, y=sel_accs, mode="lines+markers",
                   name="Selective Acc", marker=dict(size=5)),
        row=2, col=2,
    )
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[MIN_HIGH_CONF_ACCURACY, MIN_HIGH_CONF_ACCURACY],
                   mode="lines", name="80% Target",
                   line=dict(dash="dash", color="red")),
        row=2, col=2,
    )

    checks = go_results.get("checks", {})
    header_vals = ["Check", "Value", "Threshold", "Result"]
    cell_vals = [[], [], [], []]
    for name, data in checks.items():
        cell_vals[0].append(name)
        cell_vals[1].append(f"{data['value']:.3f}")
        cell_vals[2].append(f"{data['threshold']:.3f}")
        cell_vals[3].append(data["result"])

    fig.add_trace(
        go.Table(
            header=dict(values=header_vals,
                        fill_color="#2c3e50", font=dict(color="white")),
            cells=dict(values=cell_vals,
                       fill_color=[[("#d5f5e3" if r == PASS else "#fadbd8")
                                    for r in cell_vals[3]]]),
        ),
        row=3, col=1,
    )

    f1s = standard["per_class_f1"]
    colors = ["#e74c3c", "#f39c12", "#27ae60"]
    fig.add_trace(
        go.Bar(x=regime_names, y=f1s, marker_color=colors, name="F1"),
        row=3, col=2,
    )
    fig.add_trace(
        go.Scatter(x=regime_names, y=[MIN_PER_CLASS_F1]*3, mode="lines",
                   name="Threshold", line=dict(dash="dash", color="red")),
        row=3, col=2,
    )

    fig.update_layout(
        title=f"Phase 5 — Uncertainty-Aware Evaluation | "
              f"Verdict: {go_results['verdict']} ({go_results['passed']}/{go_results['total']})",
        height=1200, width=1200, showlegend=True,
    )
    fig.update_xaxes(title_text="Predicted", row=1, col=1)
    fig.update_yaxes(title_text="True", row=1, col=1)
    fig.update_xaxes(title_text="Confidence", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)
    fig.update_xaxes(title_text="Epistemic Uncertainty (MI)", row=2, col=1)
    fig.update_xaxes(title_text="Coverage", row=2, col=2)
    fig.update_yaxes(title_text="Accuracy", row=2, col=2)

    out_path = PLOTS_DIR / "phase5_evaluation_dashboard.html"
    fig.write_html(str(out_path))
    log.info(f"Dashboard saved: {out_path}")

    return out_path


def save_summary(standard, selective, transition, ece, go_results, temperature):
    summary = {
        "standard_metrics": {
            "test_accuracy": float(standard["accuracy"]),
            "macro_f1": float(standard["macro_f1"]),
            "min_f1": float(standard["min_f1"]),
            "per_class_f1": {
                IDX_TO_REGIME[i]: float(f)
                for i, f in enumerate(standard["per_class_f1"])
            },
        },
        "uncertainty_metrics": {
            "selective_accuracy": float(selective["selective_accuracy"]),
            "selective_f1": float(selective.get("selective_f1", 0)),
            "coverage": float(selective["coverage"]),
            "n_selected": selective["n_selected"],
            "n_total": selective["n_total"],
        },
        "transition_metrics": {k: float(v) for k, v in transition.items()},
        "calibration": {
            "ece": float(ece),
            "temperature": float(temperature),
        },
        "go_no_go": go_results,
    }

    out_path = RESULTS_DIR / "phase5_evaluation_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    txt_path = RESULTS_DIR / "phase5_evaluation_summary.txt"
    with open(txt_path, "w") as f:
        f.write("Phase 5 — Uncertainty-Aware Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Test Accuracy:           {standard['accuracy']*100:.1f}%\n")
        f.write(f"Selective Accuracy:      {selective['selective_accuracy']*100:.1f}% "
                f"(coverage: {selective['coverage']*100:.1f}%)\n")
        f.write(f"Min F1:                  {standard['min_f1']:.3f}\n")
        f.write(f"Macro F1:                {standard['macro_f1']:.3f}\n")
        f.write(f"ECE:                     {ece:.4f}\n")
        f.write(f"Temperature:             {temperature:.4f}\n\n")

        f.write("Per-Class F1:\n")
        for i, f1 in enumerate(standard["per_class_f1"]):
            f.write(f"  {IDX_TO_REGIME[i]:>10s}: {f1:.3f}\n")

        f.write(f"\nTransition (smoothed): R={transition.get('smooth_recall',0)*100:.1f}% "
                f"P={transition.get('smooth_precision',0)*100:.1f}%\n")

        f.write(f"\nGo/No-Go: {go_results['verdict']} "
                f"({go_results['passed']}/{go_results['total']})\n")

        for name, data in go_results["checks"].items():
            f.write(f"  {data['result']}  {name}: {data['value']:.3f} "
                    f"(threshold: {data['threshold']:.3f})\n")

    log.info(f"Summary saved: {out_path}")
    log.info(f"Summary (txt): {txt_path}")

    return summary

