import os
os.environ["RANKER_BACKEND"] = "onnx"
import time, numpy as np, tensorflow as tf
from serving.recommender import Recommender

rec = Recommender()
users = [u for u in rec.u_pos if len(rec.seen.get(u, ())) >= 5][:200]

Xs = []                                        # candidate feature matrices, built exactly like the pipeline does
for u in users:
    ui, seen = rec.u_pos[u], rec.seen.get(u, set())
    scores, rows = rec.index.search(rec.U[ui:ui + 1], min(2000, rec.index.ntotal))
    cand = [(s, r) for s, r in zip(scores[0], rows[0]) if int(rec.item_ids[r]) not in seen][:300]
    sc = np.array([c[0] for c in cand], dtype="float32"); ix = np.array([c[1] for c in cand])
    Xs.append(rec._features(ui, u, sc, ix))
print("requests:", len(Xs), " candidates per request:", Xs[0].shape)

def tf_score(X):
    out = rec.infer(**{rec.in_key: tf.constant(X)})
    return list(out.values())[0].numpy().ravel()

def onnx_score(X):
    return rec._ort.run(None, {rec._ort_in: X})[0].ravel()

for name, fn in [("TensorFlow SavedModel", tf_score), ("ONNX Runtime", onnx_score)]:
    for X in Xs[:30]:
        fn(X)                                  # warm-up
    times = []
    for _ in range(3):
        for X in Xs:
            t = time.perf_counter(); fn(X); times.append((time.perf_counter() - t) * 1000)
    print(f"{name:<22} p50 = {np.percentile(times, 50):6.2f} ms   p95 = {np.percentile(times, 95):6.2f} ms")

print("max score difference:", max(float(np.abs(tf_score(X) - onnx_score(X)).max()) for X in Xs[:50]))