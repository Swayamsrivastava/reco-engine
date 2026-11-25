import numpy as np, pandas as pd

KS = [10, 50, 100, 300]
U = np.load("retrieval/user_embeddings.npy"); I = np.load("retrieval/item_embeddings.npy")
user_ids = np.load("retrieval/user_ids.npy");  item_ids = np.load("retrieval/item_ids.npy")
u_pos = {u: i for i, u in enumerate(user_ids)}; i_pos = {m: i for i, m in enumerate(item_ids)}

train = pd.read_parquet("data/processed/train.parquet", columns=["userId", "movieId", "label"])
val   = pd.read_parquet("data/processed/val.parquet",   columns=["userId", "movieId", "label"])
val = val[(val.label == 1) & val.userId.isin(user_ids) & val.movieId.isin(item_ids)]

rng = np.random.default_rng(0)
users = rng.choice(val.userId.unique(), size=min(2000, val.userId.nunique()), replace=False)

seen  = train[train.userId.isin(users)].groupby("userId").movieId.apply(list).to_dict()
truth = val[val.userId.isin(users)].groupby("userId").movieId.apply(set).to_dict()

pop = np.zeros(len(item_ids), dtype="float32")
for m, c in train[train.label == 1].movieId.value_counts().items():
    if m in i_pos: pop[i_pos[m]] = c

def recalls(order, rel):
    return [len(set(order[:k]) & rel) / len(rel) for k in KS]

res = {"two-tower": [], "popularity": []}
for u in users:
    rel = {i_pos[m] for m in truth[u]}
    seen_idx = [i_pos[m] for m in seen.get(u, []) if m in i_pos]   # never recommend seen items
    for name, scores in [("two-tower", I @ U[u_pos[u]]), ("popularity", pop.copy())]:
        scores = scores.copy(); scores[seen_idx] = -np.inf
        top = np.argpartition(-scores, max(KS))[:max(KS)]
        order = top[np.argsort(-scores[top])]
        res[name].append(recalls(order, rel))

print("users evaluated:", len(users))
for name, r in res.items():
    print(name, {f"Recall@{k}": round(float(v), 4) for k, v in zip(KS, np.mean(r, axis=0))})