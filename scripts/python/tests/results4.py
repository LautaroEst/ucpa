
import numpy as np
import torch
import torch.nn.functional as F

def main():
    dataset = "tony_zhao_agnews"
    model = "gpt2-xl"
    seed = 4
    num_shots = 1
    logits = np.load(f"results/paper_results/{dataset}/{model}/{str(seed)}/{str(num_shots)}_shot/test.logits.npy")
    labels = np.load(f"results/paper_results/{dataset}/{model}/{str(seed)}/{str(num_shots)}_shot/test.labels.npy")
    print(compute_metric(logits, labels, metric="norm_cross_entropy"))



def compute_metric(logits, labels, metric="cross_entropy"):
    
    logits = torch.from_numpy(logits)
    labels = torch.from_numpy(labels)

    if metric == "cross_entropy":
        score = F.cross_entropy(logits, labels, reduction="mean")
    elif metric == "norm_cross_entropy":
        import pdb; pdb.set_trace()
        score = F.cross_entropy(logits, labels, reduction="mean")
        priors = torch.bincount(labels,minlength=logits.shape[1]) / logits.shape[0]
        dummy_score = - (priors * torch.log(priors)).sum()
        score = score / dummy_score
    elif metric == "accuracy":
        score = torch.mean((torch.max(logits,dim=1).indices == labels).type(torch.float))
    elif metric == "error_rate":
        score = torch.mean((torch.max(logits,dim=1).indices == labels).type(torch.float))
        score = 1 - score
    else:
        raise ValueError(f"Metric {metric} not supported.")
    
    score = score.item()
    return score

if __name__ == "__main__":
    main()