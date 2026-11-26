import numpy as np, pandas as pd
from scipy import stats

te = pd.read_parquet("data/processed/ranker_test_scored.parquet")
MODELS = {"popularity": "log_pop_all", "retrieval-only": "ret_score",
          "wide&deep": "score_wd", "DCN-V2": "score_dcn"}
W = 1 / np.log2(np.arange(1, 11) + 1)             # position bias for ranks 1..10

def expected_ctr_per_user(col):
    out = {}
    for u, g in te.groupby("userId", sort=False):
        top = g.label.values[np.argsort(-g[col].values, kind="stable")[:10]]
        p = np.where(top == 1, 0.60, 0.02) * W[:len(top)]   # click probability per slot
        out[u] = p.sum() / 10                                # expected CTR for this user
    return pd.Series(out)

ctr = pd.DataFrame({name: expected_ctr_per_user(col) for name, col in MODELS.items()})
print("users:", len(ctr))
print("\nExpected CTR (mean over the same users):")
print(ctr.mean().round(4).to_string())

rng = np.random.default_rng(0)
def compare(a, b):
    d = (ctr[a] - ctr[b]).values
    boot = [rng.choice(d, len(d)).mean() for _ in range(5000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    t, p = stats.ttest_rel(ctr[a], ctr[b])
    print(f"{a:>15} - {b:<15} diff={d.mean():+.4f}  95% CI=({lo:+.4f}, {hi:+.4f})  p={p:.4f}",
          "*" if p < 0.05 else "")

print("\nPaired differences (* = significant at 0.05):")
compare("wide&deep", "retrieval-only")
compare("DCN-V2", "retrieval-only")
compare("wide&deep", "DCN-V2")
compare("retrieval-only", "popularity")