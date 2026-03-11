
import argparse
import logging
import math
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from config import (
    DEVICE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE,
    CHECKPOINT_DIR, TENSORBOARD_LOG_DIR, NIFTY_50_TICKERS,
    BATCH_SIZE, IDX_TO_REGIME, NUM_CLASSES, WARMUP_EPOCHS,
    USE_FOCAL_LOSS, FOCAL_GAMMA, LABEL_SMOOTHING,
    TRANSITION_LOSS_WEIGHT, TRANSITION_POS_WEIGHT,
    USE_PRETRAINED_PRICE_ENCODER, PHASE2_CHECKPOINT,
    FREEZE_PRICE_ENCODER_EPOCHS,
)
from dataset import create_dataloaders
from model import build_model, load_phase2_weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


class FocalLoss(nn.Module):

    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0,
                 reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, targets):
        num_classes = logits.size(-1)

        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.zeros_like(logits)
                smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1),
                                        1.0 - self.label_smoothing)
        else:
            smooth_targets = None

        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        if smooth_targets is not None:
            p_t = (probs * smooth_targets).sum(dim=-1)
            focal_weight = (1 - p_t) ** self.gamma
            loss = -(smooth_targets * log_probs).sum(dim=-1)
        else:
            p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            focal_weight = (1 - p_t) ** self.gamma
            loss = F.nll_loss(log_probs, targets, reduction="none")

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = alpha_t * focal_weight * loss
        else:
            loss = focal_weight * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs,
                                 steps_per_epoch):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EarlyStopping:
    def __init__(self, patience=PATIENCE, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
            return False


def set_price_encoder_frozen(model, frozen=True):
    prefix_list = ["price_input_proj.", "price_pos_encoder.", "price_transformer.",
                   "stock_embedding."]
    count = 0
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in prefix_list):
            param.requires_grad = not frozen
            count += 1
    status = "frozen" if frozen else "unfrozen"
    log.info(f"  Price encoder: {count} parameters {status}")


def train_one_epoch(model, loader, optimizer, regime_criterion,
                    transition_criterion, device, epoch, scheduler=None,
                    max_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    regime_correct = 0
    total_samples = 0
    transition_correct = 0

    for X_price, X_sent, y_regime, y_trans, stock_ids in loader:
        X_price  = X_price.to(device)
        X_sent   = X_sent.to(device)
        y_regime = y_regime.to(device)
        y_trans  = y_trans.to(device).unsqueeze(1)
        stock_ids = stock_ids.to(device)

        optimizer.zero_grad()

        output = model(X_price, X_sent, stock_ids=stock_ids)

        loss_regime = regime_criterion(output["regime_logits"], y_regime)
        loss_trans  = transition_criterion(output["transition_logit"], y_trans)
        loss = loss_regime + TRANSITION_LOSS_WEIGHT * loss_trans

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * X_price.size(0)
        preds = output["regime_logits"].argmax(dim=1)
        regime_correct += (preds == y_regime).sum().item()
        trans_preds = (output["transition_prob"] > 0.5).float()
        transition_correct += (trans_preds == y_trans).sum().item()
        total_samples += X_price.size(0)

    return {
        "loss": total_loss / total_samples,
        "regime_acc": regime_correct / total_samples,
        "trans_acc": transition_correct / total_samples,
    }


@torch.no_grad()
def validate(model, loader, regime_criterion, transition_criterion, device):
    model.eval()
    total_loss = 0.0
    regime_correct = 0
    total_samples = 0
    transition_correct = 0
    all_probs = []
    all_labels = []

    for X_price, X_sent, y_regime, y_trans, stock_ids in loader:
        X_price  = X_price.to(device)
        X_sent   = X_sent.to(device)
        y_regime = y_regime.to(device)
        y_trans  = y_trans.to(device).unsqueeze(1)
        stock_ids = stock_ids.to(device)

        output = model(X_price, X_sent, stock_ids=stock_ids)

        loss_regime = regime_criterion(output["regime_logits"], y_regime)
        loss_trans  = transition_criterion(output["transition_logit"], y_trans)
        loss = loss_regime + TRANSITION_LOSS_WEIGHT * loss_trans

        total_loss += loss.item() * X_price.size(0)
        preds = output["regime_logits"].argmax(dim=1)
        regime_correct += (preds == y_regime).sum().item()
        trans_preds = (output["transition_prob"] > 0.5).float()
        transition_correct += (trans_preds == y_trans).sum().item()
        total_samples += X_price.size(0)
        all_probs.append(output["regime_probs"].cpu())
        all_labels.append(y_regime.cpu())

    all_probs  = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    all_preds  = all_probs.argmax(dim=1)

    per_class_acc = {}
    for cls_idx in range(NUM_CLASSES):
        mask = all_labels == cls_idx
        if mask.sum() > 0:
            per_class_acc[IDX_TO_REGIME[cls_idx]] = \
                (all_preds[mask] == cls_idx).float().mean().item()

    max_probs = all_probs.max(dim=1).values
    high_conf_mask = max_probs > 0.7
    high_conf_acc = None
    if high_conf_mask.sum() > 10:
        high_conf_acc = (all_preds[high_conf_mask] == all_labels[high_conf_mask]).float().mean().item()

    return {
        "loss": total_loss / total_samples,
        "regime_acc": regime_correct / total_samples,
        "trans_acc": transition_correct / total_samples,
        "per_class_acc": per_class_acc,
        "high_conf_acc": high_conf_acc,
        "high_conf_frac": high_conf_mask.float().mean().item(),
    }


def train(
    tickers=NIFTY_50_TICKERS,
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    smoke_test=False,
):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 65)
    print("  PHASE 4 — SENTIMENT-ENRICHED REGIME CLASSIFIER TRAINING")
    print(f"       {len(tickers)} tickers | {epochs} epochs | lr={lr}")
    print(f"       Device: {DEVICE}")
    print(f"       Fusion: ENABLED (cross-attention)")
    print("=" * 65 + "\n")

    log.info("Creating dataloaders...")
    (train_loader, val_loader, test_loader, class_weights,
     num_price_features, num_sent_features) = create_dataloaders(
        tickers=tickers, batch_size=batch_size
    )

    model = build_model(num_price_features, num_sent_features).to(DEVICE)

    if USE_PRETRAINED_PRICE_ENCODER and PHASE2_CHECKPOINT.exists():
        log.info(f"Loading Phase 2 pretrained weights from {PHASE2_CHECKPOINT}")
        model = load_phase2_weights(model, PHASE2_CHECKPOINT)
    else:
        log.info("Training from scratch (no Phase 2 checkpoint)")

    if FREEZE_PRICE_ENCODER_EPOCHS > 0:
        log.info(f"Freezing price encoder for first {FREEZE_PRICE_ENCODER_EPOCHS} epochs")
        set_price_encoder_frozen(model, frozen=True)

    if USE_FOCAL_LOSS:
        regime_criterion = FocalLoss(
            alpha=class_weights.to(DEVICE),
            gamma=FOCAL_GAMMA,
            label_smoothing=LABEL_SMOOTHING,
        )
        log.info(f"Using Focal Loss (gamma={FOCAL_GAMMA}, "
                 f"smoothing={LABEL_SMOOTHING})")
    else:
        regime_criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(DEVICE),
            label_smoothing=LABEL_SMOOTHING,
        )

    transition_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([TRANSITION_POS_WEIGHT]).to(DEVICE)
    )

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=WEIGHT_DECAY,
    )
    steps_per_epoch = len(train_loader)
    scheduler = get_warmup_cosine_scheduler(
        optimizer, WARMUP_EPOCHS, epochs, steps_per_epoch
    )

    writer = SummaryWriter(log_dir=str(TENSORBOARD_LOG_DIR / timestamp))
    early_stopping = EarlyStopping(patience=PATIENCE)

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        if epoch == FREEZE_PRICE_ENCODER_EPOCHS + 1 and FREEZE_PRICE_ENCODER_EPOCHS > 0:
            log.info(f"\n  Unfreezing price encoder at epoch {epoch}")
            set_price_encoder_frozen(model, frozen=False)
            optimizer = optim.AdamW(
                model.parameters(), lr=lr * 0.3,
                weight_decay=WEIGHT_DECAY,
            )
            scheduler = get_warmup_cosine_scheduler(
                optimizer, 3, epochs - epoch + 1, steps_per_epoch
            )

        train_metrics = train_one_epoch(
            model, train_loader, optimizer,
            regime_criterion, transition_criterion, DEVICE, epoch,
            scheduler=scheduler,
        )

        val_metrics = validate(
            model, val_loader, regime_criterion, transition_criterion, DEVICE
        )

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
        writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("Accuracy/train", train_metrics["regime_acc"], epoch)
        writer.add_scalar("Accuracy/val", val_metrics["regime_acc"], epoch)
        writer.add_scalar("Accuracy/val_trans", val_metrics["trans_acc"], epoch)
        writer.add_scalar("LR", current_lr, epoch)

        if val_metrics["high_conf_acc"] is not None:
            writer.add_scalar("Accuracy/val_high_conf",
                              val_metrics["high_conf_acc"], epoch)

        for regime, acc in val_metrics["per_class_acc"].items():
            writer.add_scalar(f"PerClass/{regime}", acc, epoch)

        hc_str = (f"{val_metrics['high_conf_acc']:.3f}"
                  if val_metrics['high_conf_acc'] else "N/A")
        per_cls_str = " | ".join(
            f"{r}={a:.2f}" for r, a in val_metrics["per_class_acc"].items()
        )

        frozen_tag = " [frozen]" if epoch <= FREEZE_PRICE_ENCODER_EPOCHS else ""
        print(
            f"  Epoch {epoch:3d}/{epochs} │ "
            f"Loss: {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} │ "
            f"Acc: {train_metrics['regime_acc']:.3f}/{val_metrics['regime_acc']:.3f} │ "
            f"HC: {hc_str} │ "
            f"Trans: {val_metrics['trans_acc']:.3f} │ "
            f"LR: {current_lr:.2e} │ "
            f"{elapsed:.1f}s{frozen_tag}"
        )

        if epoch % 10 == 0 or epoch == 1:
            print(f"         Per-class: {per_cls_str}")

        if val_metrics["regime_acc"] > best_val_acc:
            best_val_acc = val_metrics["regime_acc"]
            best_epoch = epoch

            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_acc": best_val_acc,
                "num_price_features": num_price_features,
                "num_sent_features": num_sent_features,
                "val_metrics": val_metrics,
            }
            torch.save(checkpoint, CHECKPOINT_DIR / "best_model.pt")
            log.info(f"  New best model saved (val_acc={best_val_acc:.4f})")

        if early_stopping(val_metrics["regime_acc"]):
            log.info(f"\n  Early stopping at epoch {epoch}. "
                     f"Best epoch: {best_epoch} (val_acc={best_val_acc:.4f})")
            break

        if smoke_test and epoch >= 2:
            log.info("  Smoke test complete (2 epochs).")
            break

    final_checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "num_price_features": num_price_features,
        "num_sent_features": num_sent_features,
    }
    torch.save(final_checkpoint, CHECKPOINT_DIR / "final_model.pt")
    writer.close()

    print("\n" + "=" * 65)
    print(f"  TRAINING COMPLETE")
    print(f"  Best epoch:        {best_epoch}")
    print(f"  Best val accuracy: {best_val_acc:.4f}")
    print(f"  Phase 2 baseline:  0.621")
    improvement = (best_val_acc - 0.621) * 100
    sign = "+" if improvement >= 0 else ""
    print(f"  Improvement:       {sign}{improvement:.1f}%")
    print(f"  Checkpoints:       {CHECKPOINT_DIR}")
    print("=" * 65 + "\n")

    return best_val_acc, best_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 Training")
    parser.add_argument("--tickers", nargs="+", default=NIFTY_50_TICKERS)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    train(
        tickers=args.tickers,
        epochs=args.epochs,
        smoke_test=args.smoke_test,
    )

