import faiss, time, numpy as np

emb = np.load("retrieval/item_embeddings.npy")
item_ids = np.load("retrieval/item_ids.npy")
d = emb.shape[1]

# exact search (IP = inner product, which equals cosine because vectors have length 1)
flat = faiss.IndexFlatIP(d); flat.add(emb)
faiss.write_index(flat, "retrieval/items_flat.faiss")

# approximate search (HNSW graph)
hnsw = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
hnsw.hnsw.efConstruction = 200
hnsw.add(emb); hnsw.hnsw.efSearch = 128
faiss.write_index(hnsw, "retrieval/items_hnsw.faiss")

# benchmark: speed, and how many of HNSW's results match the exact ones
q = emb[np.random.default_rng(0).choice(len(emb), size=min(1000, len(emb)), replace=False)]
K = 300
for name, idx in [("flat", flat), ("hnsw", hnsw)]:
    t = time.time(); idx.search(q, K)
    print(name, round((time.time() - t) / len(q) * 1000, 3), "ms/query")
_, exact = flat.search(q, K); _, approx = hnsw.search(q, K)
print("HNSW recall vs exact:", round(float(np.mean([len(set(a) & set(e)) / K for a, e in zip(approx, exact)])), 4))

# sanity test: movies closest to the most popular movie in the index
movies = __import__("pandas").read_parquet("data/processed/movies.parquet")[["movieId", "title"]]
title = dict(zip(movies.movieId, movies.title))
pop_row = 0
_, rows = flat.search(emb[pop_row:pop_row + 1], 6)
print("Query movie:", title[int(item_ids[pop_row])])
for r in rows[0][1:]:
    print("  similar:", title[int(item_ids[r])])