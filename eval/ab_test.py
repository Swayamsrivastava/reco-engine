import hashlib, numpy as np, pandas as pd
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

te = pd.read_parquet("data/processed/ranker_test_scored.parquet")
rng = np.random.default_rng(42)

def arm_of(user_id):                       # stable 50/50 split: same user always gets the same arm
    return int(hashlib.md5(str(user_id).encode()).hexdigest(), 16) % 2

def simulate(col_a, name_a, col_b, name_b):
    rows = []
    for u, g in te.groupby("userId", sort=False):
        arm = arm_of(u)
        col, name = (col_a, name_a) if arm == 0 else (col_b, name_b)
        top = g.nlargest(10, col)                                   # the 10 items this arm would show
        for rank, label in enumerate(top.label.values, start=1):
            p = (0.60 if label == 1 else 0.02) / np.log2(rank + 1)  # position bias
            rows.append((u, name, rank, rng.random() < p))
    return pd.DataFrame(rows, columns=["userId", "arm", "rank", "click"])

def report(df, a, b):
    t = df.groupby("arm").click.agg(["sum", "count"])
    users = df.groupby("arm").userId.nunique()
    z, p = proportions_ztest(t.loc[[a, b], "sum"].values, t.loc[[a, b], "count"].values)
    print(f"\n=== {a}  vs  {b} ===")
    for arm in (a, b):
        lo, hi = proportion_confint(t.loc[arm, "sum"], t.loc[arm, "count"], method="wilson")
        print(f"{arm:<16} users={users[arm]:>4}  impressions={t.loc[arm,'count']:>5}  "
              f"clicks={t.loc[arm,'sum']:>4}  CTR={t.loc[arm,'sum']/t.loc[arm,'count']:.4f}  "
              f"95% CI=({lo:.4f}, {hi:.4f})")
    print(f"z = {z:.3f}   p-value = {p:.4f}   ->",
          "significant at 0.05" if p < 0.05 else "NOT significant at 0.05")

report(simulate("ret_score", "retrieval-only", "score_wd", "wide&deep"), "retrieval-only", "wide&deep")
report(simulate("score_wd", "wide&deep", "score_dcn", "DCN-V2"), "wide&deep", "DCN-V2")