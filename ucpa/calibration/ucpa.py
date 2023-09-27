from scipy.special import logsumexp, softmax, log_softmax
import numpy as np

class UCPACalibrator:

    def __init__(self, max_iters=10, tolerance=1e-6):
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.exp_beta = None

    def train(self, logits):
        num_classes = logits.shape[1]
        priors = np.ones(num_classes) / num_classes
        probs_train = softmax(logits,axis=1)
        c_old = np.ones((probs_train.shape[0],1))
        for _ in range(self.max_iters):
            exp_beta = priors / (probs_train / c_old).mean(axis=0)
            c_new = (probs_train * exp_beta).sum(axis=1, keepdims=True)
            if np.max(np.abs(c_new - c_old)) < self.tolerance:
                break
        self.exp_beta = exp_beta

    def calibrate(self, logits):
        logits = log_softmax(logits, axis=1) + np.log(self.exp_beta) - logsumexp(logits, axis=1, keepdims=True)
        return logits
