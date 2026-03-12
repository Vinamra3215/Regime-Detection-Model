
import logging
import time
import torch
import numpy as np
from datetime import datetime

from config import (
    DEVICE, MC_SAMPLES,
    HIGH_CONF_THRESHOLD, UNCERTAINTY_THRESHOLD,
    RESULTS_DIR, CHECKPOINT_DIR,
    IDX_TO_REGIME,
)

log = logging.getLogger(__name__)


def main():
    start_time = time.time()

    print("=" * 60)
    print("  REGIME DETECTION MODEL — Phase 5")
    print("  Uncertainty & Transition Detection")
    print(f"  Device: {DEVICE}")
    print(f"  MC Samples: {MC_SAMPLES}")
    print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    print("\n" + "=" * 40)
    print("  STEP 1/5: Load Phase 4 Model")
    print("=" * 40)
    from mc_dropout import load_phase4_model
    model, num_price, num_sent = load_phase4_model()
    print(f"  -> Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"  -> Price features: {num_price}, Sentiment features: {num_sent}")

    print("\n" + "=" * 40)
    print("  STEP 2/5: Create DataLoaders")
    print("=" * 40)
    from dataset import create_dataloaders
    train_loader, val_loader, test_loader = create_dataloaders()
    print(f"  -> Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, "
          f"Test: {len(test_loader.dataset)}")

    print("\n" + "=" * 40)
    print("  STEP 3/5: Temperature Calibration")
    print("=" * 40)
    from calibration import TemperatureScaler, collect_logits
    val_logits, val_labels = collect_logits(model, val_loader)
    print(f"  -> Collected {len(val_logits)} validation logits")

    scaler = TemperatureScaler()
    temperature = scaler.calibrate(val_logits, val_labels)
    print(f"  -> Temperature: {temperature:.4f}")
    torch.save({"temperature": scaler.temperature.item()}, CHECKPOINT_DIR / "temperature.pt")

    print("\n" + "=" * 40)
    print("  STEP 4/5: MC Dropout Inference (Test Set)")
    print("=" * 40)
    from mc_dropout import mc_predict_loader
    mc_metrics, labels, trans_labels, trans_probs, trans_unc = mc_predict_loader(
        model, test_loader, n_samples=MC_SAMPLES
    )

    preds = mc_metrics["predicted_class"]
    confidence = mc_metrics["confidence"]
    mean_probs = mc_metrics["mean_probs"]
    mutual_info = mc_metrics["mutual_information"]
    pred_entropy = mc_metrics["predictive_entropy"]
    pred_variance = mc_metrics["prediction_variance"]

    mean_logits = torch.tensor(np.log(mean_probs + 1e-10))
    scaler_cpu = scaler.cpu()
    calibrated_probs = torch.softmax(scaler_cpu(mean_logits), dim=-1).detach().numpy()
    calibrated_conf = calibrated_probs.max(axis=1)

    print(f"  -> Processed {len(labels)} test samples")
    print(f"  -> Mean epistemic uncertainty: {mutual_info.mean():.4f}")
    print(f"  -> Mean confidence: {confidence.mean():.4f}")
    print(f"  -> Calibrated mean confidence: {calibrated_conf.mean():.4f}")

    print("\n" + "=" * 40)
    print("  STEP 5/5: Uncertainty-Aware Evaluation")
    print("=" * 40)
    from evaluate import (
        compute_standard_metrics, compute_selective_metrics,
        compute_ece, compute_transition_metrics,
        run_go_no_go, create_dashboard, save_summary,
    )

    standard = compute_standard_metrics(preds, labels, calibrated_probs)
    print(f"\n  Standard Metrics:")
    print(f"  Test Accuracy:    {standard['accuracy']*100:.1f}%")
    print(f"  Macro F1:         {standard['macro_f1']:.3f}")
    print(f"  Min Per-Class F1: {standard['min_f1']:.3f}")
    print(f"  Per-Class F1:     Bear={standard['per_class_f1'][0]:.3f} | "
          f"Sideways={standard['per_class_f1'][1]:.3f} | "
          f"Bull={standard['per_class_f1'][2]:.3f}")
    print(f"\n{standard['report']}")

    selective = compute_selective_metrics(
        preds, labels, calibrated_conf, mutual_info,
        HIGH_CONF_THRESHOLD, UNCERTAINTY_THRESHOLD,
    )
    print(f"  Selective Metrics (conf>={HIGH_CONF_THRESHOLD}, MI<{UNCERTAINTY_THRESHOLD}):")
    print(f"  Selective Accuracy: {selective['selective_accuracy']*100:.1f}%")
    print(f"  Coverage:           {selective['coverage']*100:.1f}% "
          f"({selective['n_selected']}/{selective['n_total']})")

    selective_no_unc = compute_selective_metrics(
        preds, labels, calibrated_conf, mutual_info,
        HIGH_CONF_THRESHOLD, 999.0,
    )
    print(f"\n  High-Conf Only (no uncertainty filter):")
    print(f"  Accuracy: {selective_no_unc['selective_accuracy']*100:.1f}% "
          f"(coverage: {selective_no_unc['coverage']*100:.1f}%)")

    ece, ece_bins = compute_ece(calibrated_probs, labels)
    print(f"\n  Calibration: ECE = {ece:.4f}")

    transition = compute_transition_metrics(trans_probs, trans_labels, trans_unc)
    print(f"\n  Transition Detection:")
    print(f"  Raw:      R={transition['raw_recall']*100:.1f}% P={transition['raw_precision']*100:.1f}%")
    print(f"  Smoothed: R={transition['smooth_recall']*100:.1f}% P={transition['smooth_precision']*100:.1f}%")

    go_results = run_go_no_go(standard, selective, transition, ece)

    correct = (preds == labels)
    print(f"\n  Uncertainty by Correctness:")
    print(f"  Correct:   MI={mutual_info[correct].mean():.4f} (n={correct.sum()})")
    print(f"  Incorrect: MI={mutual_info[~correct].mean():.4f} (n={(~correct).sum()})")

    unc_for_plot = {"predicted_class": preds, "labels": labels, "mutual_information": mutual_info}
    create_dashboard(standard, selective, transition, ece, ece_bins, unc_for_plot, go_results)
    save_summary(standard, selective, transition, ece, go_results, temperature)

    import pandas as pd
    pd.DataFrame({
        "true_regime": [IDX_TO_REGIME[l] for l in labels],
        "pred_regime": [IDX_TO_REGIME[p] for p in preds],
        "confidence": calibrated_conf,
        "epistemic_uncertainty": mutual_info,
        "predictive_entropy": pred_entropy,
        "pred_variance": pred_variance,
        "correct": correct,
    }).to_csv(RESULTS_DIR / "detailed_predictions.csv", index=False)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  Phase 5 complete. Elapsed: {elapsed:.1f}s")
    print(f"  Results: {RESULTS_DIR}/")
    print(f"  Verdict: {go_results['verdict']} ({go_results['passed']}/{go_results['total']})")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
    main()

