import time, numpy as np, pandas as pd, faiss, tensorflow as tf
from serving import cache

D, N_CAND = "data/processed", 300

class Recommender:
    def __init__(self, ranker_dir="model_repo/ranker_wd/1/model.savedmodel"):
        # --- retrieval artifacts ---
        self.U = np.load("retrieval/user_embeddings.npy"); self.I = np.load("retrieval/item_embeddings.npy")
        self.user_ids = np.load("retrieval/user_ids.npy");  self.item_ids = np.load("retrieval/item_ids.npy")
        self.u_pos = {int(u): i for i, u in enumerate(self.user_ids)}
        self.i_pos = {int(m): i for i, m in enumerate(self.item_ids)}
        self.index = faiss.read_index("retrieval/items_flat.faiss")

        # --- movie info and what each user has already rated ---
        movies = pd.read_parquet(f"{D}/movies.parquet")
        self.title = {int(m): t for m, t in zip(movies.movieId, movies.title)}
        mg = {int(m): gl for m, gl in zip(movies.movieId, movies.genres_list)}
        rated = pd.concat([pd.read_parquet(f"{D}/{s}.parquet", columns=["userId", "movieId"])
                           for s in ("train", "val", "test")])
        self.seen = rated.groupby("userId").movieId.apply(set).to_dict()

        # --- ranker feature tables (same ones used in training) ---
        self.item_f = pd.read_parquet(f"{D}/item_features.parquet").set_index("movieId")
        self.user_f = pd.read_parquet(f"{D}/user_features.parquet").set_index("userId")
        self.gm_mean = self.item_f.avg_rating_smooth.mean(); self.year_med = self.item_f.year.median()
        genres = sorted({g for gl in movies.genres_list for g in gl})
        g_idx = {g: i for i, g in enumerate(genres)}
        self.G = np.zeros((len(self.item_ids), len(genres)), dtype="float32")
        for i, m in enumerate(self.item_ids):
            for g in mg.get(int(m), []):
                self.G[i, g_idx[g]] = 1
        self.A = np.zeros((len(self.user_ids), len(genres)), dtype="float32")
        aff = pd.read_parquet(f"{D}/user_genre_affinity.parquet")
        aff = aff[aff.userId.isin(list(self.u_pos)) & aff.genre.isin(list(g_idx))]
        self.A[aff.userId.map(self.u_pos).values, aff.genre.map(g_idx).values] = aff.share.values

        # --- popularity list for cold start ---
        pop = pd.read_parquet(f"{D}/popularity_global.parquet").sort_values(["pop_recent", "pop_all"], ascending=False)
        self.pop_list = [int(m) for m in pop.movieId]

        # --- ranker (the same SavedModel Triton will serve later) ---
        self._model = tf.saved_model.load(ranker_dir)
        self.infer = self._model.signatures["serving_default"]
        self.in_key = list(self.infer.structured_input_signature[1].keys())[0]

    def _describe(self, ids):
        return [{"movieId": int(m), "title": self.title.get(int(m), "?")} for m in ids]

    # ---- cold-start strategies ----
    def popular(self, k, exclude):
        return [m for m in self.pop_list if m not in exclude][:k]

    def content_based(self, liked, k, exclude):
        idx = [self.i_pos[m] for m in liked if m in self.i_pos]
        if not idx:
            return self.popular(k, exclude)
        v = self.I[idx].mean(axis=0, keepdims=True)
        v = (v / (np.linalg.norm(v) + 1e-9)).astype("float32")   # average taste of the liked movies
        _, rows = self.index.search(v, min(self.index.ntotal, k + len(exclude) + 50))
        out = [int(self.item_ids[r]) for r in rows[0] if int(self.item_ids[r]) not in exclude]
        return out[:k]

    # ---- full two-stage pipeline ----
    def _features(self, ui, user_id, sc, ix):
        n, mids = len(ix), self.item_ids[ix]
        genre_aff = (self.A[ui] * self.G[ix]).sum(axis=1)
        it = self.item_f.reindex(mids)
        if user_id in self.user_f.index:
            u = self.user_f.loc[user_id]
            nr, ua = u.n_ratings, u.user_avg_rating
            us = 0.0 if pd.isna(u.user_rating_std) else u.user_rating_std
        else:
            nr, ua, us = 0, self.gm_mean, 0.0
        avg = it.avg_rating_smooth.fillna(self.gm_mean).values
        return np.column_stack([
            sc, np.arange(n), genre_aff,
            np.log1p(it.pop_all.fillna(0).values), np.log1p(it.pop_recent.fillna(0).values),
            avg, it.year.fillna(self.year_med).values,
            np.full(n, np.log1p(nr)), np.full(n, ua), np.full(n, us), avg - ua]).astype("float32")

    def two_stage(self, user_id, k):
        st = {}
        t = time.perf_counter()
        ui, seen = self.u_pos[user_id], self.seen.get(user_id, set())
        scores, rows = self.index.search(self.U[ui:ui + 1], min(2000, self.index.ntotal))   # stage 1: FAISS
        cand = [(s, r) for s, r in zip(scores[0], rows[0]) if int(self.item_ids[r]) not in seen][:N_CAND]
        sc = np.array([c[0] for c in cand], dtype="float32"); ix = np.array([c[1] for c in cand])
        st["faiss_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        X = self._features(ui, user_id, sc, ix)
        st["features_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        out = self.infer(**{self.in_key: tf.constant(X)})                                    # stage 2: ranker
        s = list(out.values())[0].numpy().ravel()
        st["ranker_ms"] = (time.perf_counter() - t) * 1000
        ids = [int(self.item_ids[ix[o]]) for o in np.argsort(-s)[:k]]
        return ids, {name: round(v, 2) for name, v in st.items()}

    # ---- the router ----
    def recommend(self, user_id, k=10, liked=None):
        t0 = time.time()
        liked = sorted(set(liked or []))
        extra = ",".join(map(str, liked))
        hit = cache.get_cached(user_id, k, extra)
        if hit is not None:
            return {"source": "cache", "ms": round((time.time() - t0) * 1000, 2), "items": hit}
        stages = None
        if user_id in self.u_pos and len(self.seen.get(user_id, ())) >= 5:
            source = "two_stage"
            ids, stages = self.two_stage(user_id, k)
        elif liked:
            source, ids = "cold_content", self.content_based(liked, k, set(liked))
        else:
            source, ids = "cold_popular", self.popular(k, set())
        items = self._describe(ids)
        cache.set_cached(user_id, k, extra, items)
        res = {"source": source, "ms": round((time.time() - t0) * 1000, 2), "items": items}
        if stages:
            res["stages"] = stages
        return res