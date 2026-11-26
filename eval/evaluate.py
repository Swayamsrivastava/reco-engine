import json, os, numpy as np, pandas as pd

te = pd.read_parquet("data/processed/ranker_test_scored.parquet")

def ndcg10(col):
    out = []
    for _, g in te.groupby("userId", sort=False):
        y = g.label.values
        if y.sum() == 0:
            continue
        ys = y[np.argsort(-g[col].values, kind="stable")]
        dcg = sum(ys[i] / np.log2(i + 2) for i in range(min(10, len(ys))))
        idcg = sum(1 / np.log2(i + 2) for i in range(int(min(y.sum(), 10))))
        out.append(dcg / idcg)
    return float(np.mean(out))

new, retrieval = ndcg10("score_wd"), ndcg10("ret_score")
champ = "eval/champion.json"
old = json.load(open(champ))["ndcg"] if os.path.exists(champ) else retrieval   # first run: compare against retrieval-only
json.dump({"new_ndcg": new, "old_ndcg": old, "retrieval_ndcg": retrieval}, open("eval/metrics.json", "w"), indent=2)
print(f"new NDCG@10={new:.4f}  current champion={old:.4f}")