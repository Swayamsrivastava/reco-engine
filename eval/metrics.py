import numpy as np

def recall_at_k(recs, relevant, k):
    relevant = set(relevant)
    return len(set(recs[:k]) & relevant) / len(relevant) if relevant else 0.0

def ndcg_at_k(recs, relevant, k):
    rel = set(relevant)
    dcg = sum(1 / np.log2(i + 2) for i, r in enumerate(recs[:k]) if r in rel)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / idcg if idcg else 0.0