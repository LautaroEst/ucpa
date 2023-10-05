

# Levantar las logits de los resultados y ver qué hay
import numpy as np
import os
import pickle

def main():
    model = "gpt2-xl"
    dataset = "tony_zhao_sst2"
    num_shots = 1
    metrics = ["norm_cross_entropy", "accuracy"]
    seed = "82033"#, "12782", "1263", "987", "12299", "9203", "4", "20343", "43", "92374"]
    prompts = np.load(os.path.join("results/paper_results",dataset,model,seed,f"{num_shots}_shot","test.prompts.npy"))
    print(prompts[:10])

    # "dataset": "sst2", "model": "gpt2-xl", "n_shots": 1, "random_state": 58776
    original_results_id = "31323818017343297470302527315948969317"
    with open(f"../efficient-reestimation/results/train_test/{original_results_id}/test.pkl", "rb") as f:
        original_results = pickle.load(f)
    print()
    print(original_results["test_queries"][:10])

if __name__ == "__main__":
    main()