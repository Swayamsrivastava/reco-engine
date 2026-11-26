import json, time

MODEL_VERSION = "v1"      # bump when you deploy a new model: old cache entries are ignored
TTL = 3600                # entries expire after 1 hour
STATS = {"hits": 0, "misses": 0}
_mem = {}

try:
    import redis
    _r = redis.Redis(host="localhost", port=6379, decode_responses=True, socket_connect_timeout=1)
    _r.ping()
    BACKEND = "redis"
except Exception:
    _r = None
    BACKEND = "in-memory dict (Redis not reachable)"

def _key(user_id, k, extra):
    return f"rec:{MODEL_VERSION}:{user_id}:{k}:{extra}"

def get_cached(user_id, k, extra=""):
    key = _key(user_id, k, extra)
    if _r is not None:
        v = _r.get(key)
    else:
        e = _mem.get(key)
        v = e[1] if e and e[0] > time.time() else None
    STATS["hits" if v is not None else "misses"] += 1
    return json.loads(v) if v is not None else None

def set_cached(user_id, k, extra, items):
    key, v = _key(user_id, k, extra), json.dumps(items)
    if _r is not None:
        _r.setex(key, TTL, v)
    else:
        _mem[key] = (time.time() + TTL, v)

def flush():
    if _r is not None:
        for key in _r.scan_iter(f"rec:{MODEL_VERSION}:*"):
            _r.delete(key)
    _mem.clear()
    STATS.update(hits=0, misses=0)