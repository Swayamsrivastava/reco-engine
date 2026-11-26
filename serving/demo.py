from serving import cache
from serving.recommender import Recommender

rec = Recommender()
cache.flush()
print("cache backend:", cache.BACKEND)
known = next(iter(rec.u_pos))               # a user the model was trained on

tests = [("known user, first call", known, None),
         ("known user, same call again", known, None),
         ("new user, 3 liked movies (Toy Story, Star Wars IV, Pulp Fiction)", 999999991, [1, 260, 296]),
         ("brand-new user, no history", 999999992, None)]
for label, uid, liked in tests:
    r = rec.recommend(uid, k=5, liked=liked)
    print(f"\n{label}\n  source={r['source']}  time={r['ms']} ms")
    for it in r["items"]:
        print("   -", it["title"])
print("\ncache stats:", cache.STATS)