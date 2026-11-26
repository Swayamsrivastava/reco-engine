import numpy as np
from fastapi import FastAPI, Query
from serving import cache
from serving.recommender import Recommender

app = FastAPI(title="Two-Stage Recommender API",
              description="Retrieval (two-tower + FAISS) -> ranking (Wide & Deep), with Redis cache and cold-start fallbacks.")
rec = Recommender()
LAT, STAGES = [], []          # per-request latency log, used by /stats

@app.get("/health")
def health():
    return {"status": "ok", "cache_backend": cache.BACKEND, "model_version": cache.MODEL_VERSION}

@app.get("/recommend")
def recommend(user_id: int,
              k: int = Query(10, ge=1, le=50),
              liked: str = Query("", description="comma-separated movieIds a new user likes (cold start)")):
    liked_ids = [int(x) for x in liked.split(",") if x.strip()]
    res = rec.recommend(user_id, k, liked_ids)
    LAT.append((res["source"], res["ms"]))
    if "stages" in res:
        STAGES.append(res["stages"])
    return res

def _pct(values):
    v = np.array(values)
    return {"n": len(v), "p50": round(float(np.percentile(v, 50)), 2), "p95": round(float(np.percentile(v, 95)), 2)}

@app.get("/stats")
def stats():
    h, m = cache.STATS["hits"], cache.STATS["misses"]
    out = {"cache_backend": cache.BACKEND, "cache_hits": h, "cache_misses": m,
           "cache_hit_rate": round(h / (h + m), 3) if h + m else None,
           "latency_ms_by_source": {}, "latency_ms_by_stage": {}}
    for src in sorted({s for s, _ in LAT}):
        out["latency_ms_by_source"][src] = _pct([ms for s, ms in LAT if s == src])
    for name in ("faiss_ms", "features_ms", "ranker_ms"):
        vals = [st[name] for st in STAGES]
        if vals:
            out["latency_ms_by_stage"][name] = _pct(vals)
    return out

@app.post("/cache/flush")
def flush():
    cache.flush(); LAT.clear(); STAGES.clear()
    return {"flushed": True}