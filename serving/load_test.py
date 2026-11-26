import json, urllib.request, numpy as np

BASE = "http://localhost:9000"
def call(path, method="GET"):
    req = urllib.request.Request(BASE + path, method=method)
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())

call("/cache/flush", "POST")
rng = np.random.default_rng(1)
users = np.load("retrieval/user_ids.npy")[:150]
for u in rng.choice(users, size=400):
    call(f"/recommend?user_id={int(u)}&k=10")
for i in range(20):
    call(f"/recommend?user_id={999000 + i}&k=10&liked=1,260,296")   # new users with 3 liked movies
for i in range(20):
    call(f"/recommend?user_id={999500 + i}&k=10")                   # brand-new users
print(json.dumps(call("/stats"), indent=2))