
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold
import torch
from torch import nn, optim
from .losses import LogLoss, Brier

class _AffineModel(nn.Module):

    def __init__(self, num_classes, scale="vector", bias=True):
        super().__init__()
        self.num_classes = num_classes
        if scale == "matrix":
            self.scale = nn.Parameter(torch.eye(n=num_classes, dtype=torch.float))
        elif scale == "vector":
            self.scale = nn.Parameter(torch.ones(num_classes,dtype=torch.float))
        elif scale == "scalar":
            self.scale = nn.Parameter(torch.tensor(1.0))
        elif scale == "none":
            self.scale = torch.tensor(1.0)
        else:
            raise ValueError(f"Scale {scale} not valid.")

        if bias:
            self.bias = nn.Parameter(torch.zeros(1,num_classes))
        else:
            self.bias = torch.zeros(1,num_classes)

    def forward(self, logits):
        if self.scale == "matrix":
            return torch.matmul(logits, self.scale) + self.bias
        return logits * self.scale + self.bias


class _AffineCalibrationTrainer:

    def __init__(self, num_classes, scale, bias, loss_fn, maxiters=100, lr=1e-4, tolerance=1e-6):
        self.model = _AffineModel(num_classes, scale, bias)
        self.loss_fn = loss_fn
        self.maxiters = maxiters
        self.lr = lr
        self.tolerance_change = tolerance

    def fit(self, logits, labels):
        self.model.train()
        optimizer = optim.LBFGS(
            self.model.parameters(),
            lr=self.lr,
            max_iter=self.maxiters,
            tolerance_change=self.tolerance_change
        )
        logits = logits.cpu()
        labels = labels.cpu()
        def closure():
            optimizer.zero_grad()
            cal_logits = self.model(logits)
            loss = self.loss_fn(cal_logits, labels)
            loss.backward()
            return loss
        optimizer.step(closure)
    
    def calibrate(self,logits):
        device = logits.device
        model = self.model.to(device)
        model.eval()
        with torch.no_grad():
            calibrated_logits = model(logits)
        return calibrated_logits        



class AffineCalibrator:

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
        model="temperature scaling", 
        psr="log-loss", 
        maxiters=100, 
        lr=1e-4, 
        tolerance=1e-6,
        random_state=None
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
            scale = "matrix"
            bias = True
        elif model == "vector scaling":
            scale = "vector"
            bias = True
        elif model == "temperature scaling":
            scale = "scalar"
            bias = False
        elif model == "bias only":
            scale = "none"
            bias = True
        else:
            raise ValueError(f"Calibration model {model} not supported")

        self.model = model
        self.psr = psr
        self.scale = scale
        self.bias = bias
        self.loss_fn = loss_fn
        self.maxiters = maxiters
        self.lr = lr
        self.tolerance = tolerance
        self.seed = random_state
        self.trainer = None

    def train(self, logits, labels):
        num_classes = logits.size(1)
        trainer = _AffineCalibrationTrainer(
            num_classes=num_classes,
            scale=self.scale,
            bias=self.bias,
            loss_fn=self.loss_fn,
            maxiters=self.maxiters,
            lr=self.lr,
            tolerance=self.tolerance
        )
        trainer.fit(logits, labels)
        self.trainer = trainer

    def calibrate(self, logits):
        if self.trainer is None:
            raise RuntimeError("Train the calibrator before use it.")
        cal_logits = self.trainer.calibrate(logits)
        return cal_logits