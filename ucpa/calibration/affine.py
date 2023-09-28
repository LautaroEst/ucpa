
import torch
from torch import nn, optim
from .losses import LogLoss, Brier
from .base import BaseCalibrator


class AffineCalibrator(BaseCalibrator):

    MODELS = [
        "temperature scaling", # Only one parameter (scaling)
        "vector scaling", # Two parameters (scaling and shift) per class
        "matrix scaling", # K x (K+1) parameters, where K is the number of classes
        "bias only", # One parameter per class (shift)
    ]

    PSRS = [
        "log-loss",
        "normalized log-loss",
        "brier",
        "normalized brier"
    ]

    def __init__(
        self,
        num_classes,
        model="temperature scaling", 
        psr="log-loss", 
        maxiters=100, 
        lr=1e-4, 
        tolerance=1e-6,
    ):
        super().__init__()

        if psr == "log-loss":
            loss_fn = LogLoss(norm=False)
        elif psr == "normalized log-loss":
            loss_fn = LogLoss(norm=True)
        elif psr == "brier":
            loss_fn = Brier(norm=False)
        elif psr == "normalized brier":
            loss_fn = Brier(norm=True)
        else:
            raise ValueError(f"PSR {psr} not supported")
        
        if model == "matrix scaling":
            self.scale = nn.Parameter(torch.eye(n=num_classes, dtype=torch.float))
            self.bias = nn.Parameter(torch.zeros(1,num_classes))
        elif model == "vector scaling":
            self.scale = nn.Parameter(torch.ones(num_classes,dtype=torch.float))
            self.bias = nn.Parameter(torch.zeros(1,num_classes))
        elif model == "temperature scaling":
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.bias = torch.zeros(1,num_classes)
        elif model == "bias only":
            self.scale = torch.tensor(1.0)
            self.bias = nn.Parameter(torch.zeros(1,num_classes))
        else:
            raise ValueError(f"Calibration model {model} not supported")
        
        self.model = model
        self.num_classes = num_classes
        self.loss_fn = loss_fn
        self.maxiters = maxiters
        self.lr = lr
        self.tolerance_change = tolerance

    def forward(self, logits):
        if self.model == "matrix scaling":
            return torch.matmul(logits, self.scale) + self.bias
        return logits * self.scale + self.bias

    def train_calibrator(self, logits, labels):
        self.train()
        optimizer = optim.LBFGS(
            self.parameters(),
            lr=self.lr,
            max_iter=self.maxiters,
            tolerance_change=self.tolerance_change
        )
        logits = logits.cpu()
        labels = labels.cpu()
        def closure():
            optimizer.zero_grad()
            cal_logits = self(logits)
            loss = self.loss_fn(cal_logits, labels)
            loss.backward()
            return loss
        optimizer.step(closure)