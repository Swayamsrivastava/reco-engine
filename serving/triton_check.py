import time, numpy as np, tritonclient.http as httpclient

X = np.load("/tmp/check_X.npy")[:300]            # 300 candidates, like one request
stored = np.load("/tmp/check_scores.npy")[:300]
client = httpclient.InferenceServerClient("localhost:8000")
print("model ready:", client.is_model_ready("ranker"))

def score(x):
    inp = httpclient.InferInput("features", list(x.shape), "FP32")
    inp.set_data_from_numpy(x)
    res = client.infer("ranker", [inp], outputs=[httpclient.InferRequestedOutput("score")])
    return res.as_numpy("score").ravel()

out = score(X)
print("max |Triton - stored score|:", float(np.abs(out - stored).max()))
times = []
for _ in range(300):
    t = time.perf_counter(); score(X); times.append((time.perf_counter() - t) * 1000)
print("p50 =", round(float(np.percentile(times, 50)), 2), "ms  p95 =", round(float(np.percentile(times, 95)), 2), "ms")