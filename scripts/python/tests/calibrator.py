
import argparse
import json
import os
import pickle
import numpy as np
from ucpa.calibration import AffineCalibrator
from psrcal import AffineCalLogLoss
import torch




def parse_args():
    """ Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_directory", type=str, default="")
    args = parser.parse_args()
    return args



def main():

    args = parse_args()
    exp_id = "16283375850316419363298856249698691175"
    with open(os.path.join(args.root_directory,f"../efficient-reestimation/results/calibrated/logloss_100boots/{exp_id}.pkl"), "rb") as f:
        exp_results = pickle.load(f)

    with open(os.path.join(args.root_directory,f"../efficient-reestimation/results/train_test/{exp_id}/config.json"), "r") as f:
        main_config = json.load(f)

    with open(os.path.join(args.root_directory,f"../efficient-reestimation/results/train_test/{exp_id}/train.pkl"), "rb") as f:
        train_results = pickle.load(f)

    test_probs = exp_results[0]["test_probs_original"]
    test_logits = np.log(test_probs)
    test_labels = exp_results[0]["test_labels"]

    train_logits = np.log(train_results["train_probs"].copy())
    train_labels = train_results["train_labels"].copy()

    model = AffineCalibrator(num_classes=2, model="vector scaling", psr="log-loss")
    model.train_calibrator(torch.from_numpy(train_logits), torch.from_numpy(train_labels))
    cal_test_logits = model.calibrate(torch.from_numpy(test_logits)).numpy()
    cal_test_probs = np.exp(cal_test_logits)
    cal_test_probs /= np.sum(cal_test_probs, axis=1, keepdims=True)
    print(cal_test_probs)
    print((np.argmax(cal_test_probs, axis=1) == test_labels).mean())

    trnf = torch.log_softmax(torch.as_tensor(train_logits, dtype=torch.float32),dim=1)
    trnt = torch.as_tensor(train_labels, dtype=torch.int64)
    tstf = torch.log_softmax(torch.as_tensor(test_logits, dtype=torch.float32),dim=1)
    model = AffineCalLogLoss(trnf, trnt, bias=True, scale=True, priors=None)
    model.train()
    test_post_cal = torch.softmax(model.calibrate(tstf),dim=1).detach().numpy()
    print(test_post_cal)
    print((np.argmax(test_post_cal, axis=1) == test_labels).mean())



if __name__ == "__main__":
    main()