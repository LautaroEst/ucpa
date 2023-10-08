
import pandas as pd


def main():
    df_original = pd.read_csv("../efficient-reestimation/results/gpt2-xl_trec_sst2_logloss_100boots.csv")
    df_original = df_original[
        df_original["prob_type"].str.contains(
            "original|noscalecal|reest_train|reestwithpriors|reestiterative_|reestiterativewithpriors"
        )
    ][~df_original["prob_type"].str.contains("150")]
    df_original = df_original.groupby(["dataset","n_shots","prob_type"]).agg({f"score:{metric}": ["mean","std"] for metric in ["cross-entropy","accuracy"]})
    df_new = pd.read_csv("results/paper_results/results.csv")
    df_new = df_new.groupby(["dataset","num_shots","method","num_samples"]).agg({f"metric:{metric}": ["mean","std"] for metric in ["norm_cross_entropy","error_rate"]})
    import pdb; pdb.set_trace()
    print()
    print()



if __name__ == "__main__":
    main()