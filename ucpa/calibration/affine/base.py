
import torch
from torch import nn, optim
from .psrcal import AffineCalLogLoss, AffineCalBrier



class AffineCalibrator(nn.Module):

    MODELS = [
        "vector scaling", # Two parameters (scaling and shift) per class
        "bias only", # One parameter per class (shift)
    ]

    PSRS = [
        "log-loss",
        "brier"
    ]

    def __init__(
        self,
        num_classes,
        model="vector scaling", 
        psr="log-loss", 
    ):
        super().__init__()

        if model == "vector scaling" and psr == "log-loss":
            calmodel = AffineCalLogLoss(num_classes, bias=True, scale=True)
        elif model == "vector scaling" and psr == "brier":
            calmodel = AffineCalBrier(num_classes, bias=True, scale=True)
        elif model == "bias only" and psr == "log-loss":
            calmodel = AffineCalLogLoss(num_classes, bias=True, scale=False)
        elif model == "bias only" and psr == "brier":
            calmodel = AffineCalBrier(num_classes, bias=True, scale=False)
        else:
            raise ValueError(f"Calibration method with {model} model and {psr} PSR not supported.")
        self.calmodel = calmodel
        
    def forward(self, logits):
        scores = torch.log_softmax(logits, dim=1)
        cal_scores = self.calmodel.calibrate(scores)
        cal_logprobs = torch.log_softmax(cal_scores, dim=1)
        return cal_logprobs

    def train_calibrator(self, logits, labels):
        scores = torch.log_softmax(logits, dim=1)
        self.calmodel.train(scores, labels)

    def calibrate(self, logits):
        with torch.no_grad():
            cal_logits = self(logits)
        return cal_logits