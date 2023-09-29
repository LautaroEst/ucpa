import numpy as np
import torch
from torch import nn

class UCPACalibrator(nn.Module):

    """ 
    Implementation of the UCPA/SUCPA iterative and naive methods. 
    For the case in which max_iters == 1, the model implements the
    naive approach. Otherwise, the training is iterative.
    In the case priors are provided to the .train() method, model
    incorporate information from the priors (SUCPA). Otherwise, it
    assumes them uniform (UCPA).
    """

    def __init__(self, num_classes, max_iters=10, tolerance=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.beta = nn.Parameter(torch.zeros(1, num_classes), requires_grad=False)

    def forward(self, logits):
        cal_logits = logits + self.beta
        return cal_logits

    def train_calibrator(self, logits, priors=None):
        self.train()
        num_samples, num_classes = logits.shape
        log_num_samples = torch.log(torch.tensor(num_samples))
        log_priors = torch.log(torch.ones(1, num_classes) / num_classes) if priors is None else torch.log(priors).view(1,-1)
        gamma_old = torch.zeros(num_samples, 1)
        for _ in range(self.max_iters):
            beta = log_priors - torch.logsumexp(logits + gamma_old, dim=0, keepdim=True) + log_num_samples
            gamma_new = - torch.logsumexp(logits + beta, dim=1, keepdim=True)
            if max(torch.abs(gamma_new - gamma_old)) < self.tolerance:
                break
            gamma_old = gamma_new
        self.beta.data = beta

    def calibrate(self, logits):
        logits_device = logits.device
        model = self.to(logits_device)
        model.eval()
        with torch.no_grad():
            calibrated_logits = self(logits)
        return calibrated_logits