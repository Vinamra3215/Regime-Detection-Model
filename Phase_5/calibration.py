
import torch
import torch.nn as nn
import logging

from config import DEVICE, CALIBRATION_LR, CALIBRATION_ITERS

log = logging.getLogger(__name__)


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature.to(logits.device)

    def calibrate(self, val_logits, val_labels):
        val_logits = val_logits.to(DEVICE)
        val_labels = val_labels.to(DEVICE)
        self.to(DEVICE)

        nll = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=CALIBRATION_LR,
                                       max_iter=CALIBRATION_ITERS)

        def closure():
            optimizer.zero_grad()
            loss = nll(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        final_loss = nll(self.forward(val_logits), val_labels).item()
        log.info(f"  Calibrated temperature: {self.temperature.item():.4f} (NLL: {final_loss:.4f})")
        return self.temperature.item()


@torch.no_grad()
def collect_logits(model, loader):
    model.eval()
    all_logits = []
    all_labels = []

    for x_price, x_sent, y_regime, y_trans, stock_ids in loader:
        x_price   = x_price.to(DEVICE)
        x_sent    = x_sent.to(DEVICE)
        stock_ids = stock_ids.to(DEVICE)
        output = model(x_price, x_sent, stock_ids=stock_ids)
        all_logits.append(output["regime_logits"].cpu())
        all_labels.append(y_regime)

    return torch.cat(all_logits), torch.cat(all_labels)

