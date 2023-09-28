

import pandas as pd

def main():
    df = pd.DataFrame({
        "a": [0,1,2,3,4,5,3,1,2,2,4,23,4],
        "b": [0,1,2,3,4,5,7,1,2,2,4,23,4],
        "c": [93,5,7,9,3,65,3,4,6,2,7,8,56]
    })
    df = df.groupby(["a","b"]).agg(["mean", "std"]).loc[(3,slice(None)),(slice(None),"mean")]
    ids = df.index.get_level_values("b").values
    values = df.values
    print(ids, values)


if __name__ == "__main__":
    main()