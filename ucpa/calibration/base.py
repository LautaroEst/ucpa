import torch
from torch import nn

class BaseCalibrator(nn.Module):

    def calibrate(self, logits):
        logits_device = logits.device
        model = self.to(logits_device)
        model.eval()
        with torch.no_grad():
            calibrated_logits = self(logits)
        return calibrated_logits