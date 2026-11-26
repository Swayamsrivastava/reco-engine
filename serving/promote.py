import json, os, shutil

src = "model_repo/ranker_wd/1/model.savedmodel"      # the newly trained candidate
live = "model_repo/live/ranker"
os.makedirs(live, exist_ok=True)
n = max([int(d) for d in os.listdir(live) if d.isdigit()], default=0) + 1
os.makedirs(f"{live}/{n}")
shutil.copytree(src, f"{live}/{n}/model.savedmodel")          # versioned copy of the winning model

m = json.load(open("eval/metrics.json"))
json.dump({"version": n, "ndcg": m["new_ndcg"]}, open("eval/champion.json", "w"))
open("model_repo/live/VERSION", "w").write(f"v{n}")             # cache key version: old cached results are ignored
print(f"promoted ranker version {n}, NDCG@10={m['new_ndcg']:.4f}")