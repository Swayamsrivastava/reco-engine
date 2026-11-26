import numpy as np, pandas as pd, faiss

N_CAND, K_FETCH, BATCH = 300, 2000, 500
MAX_USERS = {"train": 6000, "test": 3000}
rng = np.random.default_rng(0)

U = np.load("retrieval/user_embeddings.npy"); I = np.load("retrieval/item_embeddings.npy")
user_ids = np.load("retrieval/user_ids.npy");  item_ids = np.load("retrieval/item_ids.npy")
u_pos = {u: i for i, u in enumerate(user_ids)}
i_pos = {m: i for i, m in enumerate(item_ids)}
index = faiss.read_index("retrieval/items_flat.faiss")

cols = ["userId", "movieId", "label"]
train = pd.read_parquet("data/processed/train.parquet", columns=cols)
val   = pd.read_parquet("data/processed/val.parquet",   columns=cols)
test  = pd.read_parquet("data/processed/test.parquet",  columns=cols)

def build_table(label_df, seen_df, max_users, name):
    pos = label_df[(label_df.label == 1) & label_df.userId.isin(user_ids) & label_df.movieId.isin(item_ids)]
    truth = pos.groupby("userId").movieId.apply(set).to_dict()
    users = np.array(sorted(truth))
    if len(users) > max_users:
        users = rng.choice(users, max_users, replace=False)
    seen = seen_df[seen_df.userId.isin(users)].groupby("userId").movieId.apply(list).to_dict()

    rows, hits, total = [], 0, 0
    k_fetch = min(K_FETCH, index.ntotal)
    for s in range(0, len(users), BATCH):
        ub = users[s:s + BATCH]
        scores, idx = index.search(U[[u_pos[u] for u in ub]], k_fetch)   # stage 1: FAISS
        for r, u in enumerate(ub):
            seen_idx = {i_pos[m] for m in seen.get(u, []) if m in i_pos}  # drop already-seen movies
            keep = [(sc, ix) for sc, ix in zip(scores[r], idx[r]) if ix not in seen_idx][:N_CAND]
            rel = {i_pos[m] for m in truth[u]}
            hits += len(rel & {ix for _, ix in keep}); total += len(rel)
            for rank, (sc, ix) in enumerate(keep):
                rows.append((u, item_ids[ix], ix, sc, rank, int(ix in rel)))
    df = pd.DataFrame(rows, columns=["userId", "movieId", "item_idx", "ret_score", "ret_rank", "label"])
    print(f"[{name}] users={len(users)} rows={len(df)} positive_rate={df.label.mean():.4f} "
          f"candidate_recall@{N_CAND}={hits / total:.4f}")
    return df

# ---- feature tables (all computed from TRAIN data only) ----
item_f = pd.read_parquet("data/processed/item_features.parquet").set_index("movieId")
user_f = pd.read_parquet("data/processed/user_features.parquet").set_index("userId")
aff    = pd.read_parquet("data/processed/user_genre_affinity.parquet")
movies = pd.read_parquet("data/processed/movies.parquet")

genres = sorted({g for gl in movies.genres_list for g in gl})
g_idx = {g: i for i, g in enumerate(genres)}
G = np.zeros((len(item_ids), len(genres)), dtype="float32")          # movie x genre (0/1)
mg = dict(zip(movies.movieId, movies.genres_list))
for i, m in enumerate(item_ids):
    for g in mg.get(m, []):
        G[i, g_idx[g]] = 1
A = np.zeros((len(user_ids), len(genres)), dtype="float32")          # user x genre (share of likes)
aff = aff[aff.userId.isin(u_pos) & aff.genre.isin(g_idx)]
A[aff.userId.map(u_pos).values, aff.genre.map(g_idx).values] = aff.share.values

gm_mean = item_f.avg_rating_smooth.mean(); year_med = item_f.year.median()
FEATURES = ["ret_score", "ret_rank", "genre_aff", "log_pop_all", "log_pop_recent",
            "avg_rating_smooth", "year", "log_n_ratings", "user_avg_rating",
            "user_rating_std", "rating_gap"]

def add_features(df):
    ui, ii = df.userId.map(u_pos).values, df.item_idx.values
    df["genre_aff"] = (A[ui] * G[ii]).sum(axis=1)                    # cross feature: user taste x movie genres
    it = item_f.reindex(df.movieId.values); uf = user_f.reindex(df.userId.values)
    df["log_pop_all"]    = np.log1p(it.pop_all.fillna(0).values)
    df["log_pop_recent"] = np.log1p(it.pop_recent.fillna(0).values)
    df["avg_rating_smooth"] = it.avg_rating_smooth.fillna(gm_mean).values
    df["year"] = it.year.fillna(year_med).values
    df["log_n_ratings"]   = np.log1p(uf.n_ratings.fillna(0).values)
    df["user_avg_rating"] = uf.user_avg_rating.fillna(gm_mean).values
    df["user_rating_std"] = uf.user_rating_std.fillna(0).values
    df["rating_gap"] = df.avg_rating_smooth - df.user_avg_rating
    return df[["userId", "movieId", "label"] + FEATURES]

# ranker TRAIN table: labels from the validation window, seen = train
add_features(build_table(val, train, MAX_USERS["train"], "ranker_train")) \
    .to_parquet("data/processed/ranker_train.parquet")
# ranker TEST table: labels from the test window, seen = train + validation
add_features(build_table(test, pd.concat([train, val]), MAX_USERS["test"], "ranker_test")) \
    .to_parquet("data/processed/ranker_test.parquet")
print("done. n_dense_features =", len(FEATURES))