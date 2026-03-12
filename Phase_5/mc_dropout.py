
import numpy as np
import torch
import logging

from config import DEVICE, MC_SAMPLES, PHASE4_CHECKPOINT
from model_loader import load_phase4_modules

log = logging.getLogger(__name__)

_, _p4_model_mod = load_phase4_modules()
build_model = _p4_model_mod.build_model


def load_phase4_model():
    checkpoint = torch.load(PHASE4_CHECKPOINT, map_location=DEVICE, weights_only=False)
    num_price = checkpoint["num_price_features"]
    num_sent  = checkpoint["num_sent_features"]

    model = build_model(num_price, num_sent)
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)

    log.info(f"Loaded Phase 4 model (epoch {checkpoint['epoch']}, "
             f"val_acc={checkpoint['val_acc']:.4f})")
    log.info(f"  Price features: {num_price}, Sentiment features: {num_sent}")
    return model, num_price, num_sent


def enable_mc_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


@torch.no_grad()
def mc_predict_batch(model, x_price, x_sentiment, stock_ids, n_samples=MC_SAMPLES):
    model.eval()
    enable_mc_dropout(model)

    regime_samples = []
    transition_samples = []

    for _ in range(n_samples):
        output = model(x_price, x_sentiment, stock_ids=stock_ids)
        regime_samples.append(output["regime_probs"].cpu())
        transition_samples.append(output["transition_prob"].squeeze(-1).cpu())

    return torch.stack(regime_samples, dim=0).numpy(), torch.stack(transition_samples, dim=0).numpy()


def compute_uncertainty_metrics(regime_probs_samples):
    n_samples, batch_size, num_classes = regime_probs_samples.shape
    mean_probs = regime_probs_samples.mean(axis=0)
    predicted_class = mean_probs.argmax(axis=1)
    confidence = mean_probs.max(axis=1)

    eps = 1e-10
    predictive_entropy = -np.sum(mean_probs * np.log(mean_probs + eps), axis=1)
    individual_entropies = -np.sum(regime_probs_samples * np.log(regime_probs_samples + eps), axis=2)
    expected_entropy = individual_entropies.mean(axis=0)
    mutual_information = np.clip(predictive_entropy - expected_entropy, 0, None)
    prediction_variance = regime_probs_samples.var(axis=0).mean(axis=1)

    return {
        "mean_probs": mean_probs,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "predictive_entropy": predictive_entropy,
        "expected_entropy": expected_entropy,
        "mutual_information": mutual_information,
        "prediction_variance": prediction_variance,
    }


@torch.no_grad()
def mc_predict_loader(model, loader, n_samples=MC_SAMPLES):
    model.eval()
    enable_mc_dropout(model)

    all_regime_samples = []
    all_trans_samples  = []
    all_labels = []
    all_trans_labels = []

    log.info(f"  Running MC Dropout with {n_samples} samples...")

    for batch_idx, (x_price, x_sent, y_regime, y_trans, stock_ids) in enumerate(loader):
        x_price   = x_price.to(DEVICE)
        x_sent    = x_sent.to(DEVICE)
        stock_ids = stock_ids.to(DEVICE)

        regime_samples, trans_samples = mc_predict_batch(model, x_price, x_sent, stock_ids, n_samples)

        all_regime_samples.append(regime_samples)
        all_trans_samples.append(trans_samples)
        all_labels.append(y_regime.numpy())
        all_trans_labels.append(y_trans.numpy())

        if (batch_idx + 1) % 50 == 0:
            log.info(f"    Processed {(batch_idx+1) * loader.batch_size} samples...")

    all_regime = np.concatenate(all_regime_samples, axis=1)
    all_trans  = np.concatenate(all_trans_samples, axis=1)
    labels = np.concatenate(all_labels)
    trans_labels = np.concatenate(all_trans_labels)

    metrics = compute_uncertainty_metrics(all_regime)
    mean_trans = all_trans.mean(axis=0)
    trans_uncertainty = all_trans.var(axis=0)

    return metrics, labels, trans_labels, mean_trans, trans_uncertainty

