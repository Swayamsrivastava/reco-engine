import numpy as np, pandas as pd, tensorflow as tf
import tensorflow_recommenders as tfrs
from sklearn.metrics import roc_auc_score

tf.keras.utils.set_random_seed(42)
FEATURES = ["ret_score", "ret_rank", "genre_aff", "log_pop_all", "log_pop_recent",
            "avg_rating_smooth", "year", "log_n_ratings", "user_avg_rating",
            "user_rating_std", "rating_gap"]

tr = pd.read_parquet("data/processed/ranker_train.parquet")
te = pd.read_parquet("data/processed/ranker_test.parquet")

# hold out 15% of USERS (not rows) to monitor training
users = tr.userId.unique(); rng = np.random.default_rng(0); rng.shuffle(users)
hold = tr.userId.isin(set(users[:int(0.15 * len(users))]))
Xtr, ytr = tr.loc[~hold, FEATURES].values.astype("float32"), tr.loc[~hold, "label"].values.astype("float32")
Xva, yva = tr.loc[hold, FEATURES].values.astype("float32"),  tr.loc[hold, "label"].values.astype("float32")
Xte = te[FEATURES].values.astype("float32")

def make_norm():
    n = tf.keras.layers.Normalization(axis=-1); n.adapt(Xtr); return n

def build_wide_deep():
    inp = tf.keras.Input(shape=(len(FEATURES),), name="features")
    x = make_norm()(inp)
    # wide part: linear model plus a few hand-made feature crosses
    crosses = tf.keras.layers.Lambda(lambda t: tf.stack(
        [t[:, 0] * t[:, 2], t[:, 2] * t[:, 3], t[:, 0] * t[:, 3]], axis=1))(x)
    wide = tf.keras.layers.Dense(1)(tf.keras.layers.Concatenate()([x, crosses]))
    # deep part: small neural network
    deep = tf.keras.layers.Dense(128, activation="relu")(x)
    deep = tf.keras.layers.Dense(64, activation="relu")(deep)
    deep = tf.keras.layers.Dense(1)(deep)
    out = tf.keras.layers.Activation("sigmoid", name="score")(tf.keras.layers.Add()([wide, deep]))
    return tf.keras.Model(inp, out)

def build_dcn():
    inp = tf.keras.Input(shape=(len(FEATURES),), name="features")
    x0 = make_norm()(inp); x = x0
    for _ in range(2):
        x = tfrs.layers.dcn.Cross()(x0, x)          # learns feature crosses automatically
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="score")(x)
    return tf.keras.Model(inp, out)

def train(name, builder):
    model = builder()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(name="auc")])
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=3,
                                            restore_best_weights=True)
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=30, batch_size=2048,
              callbacks=[stop], verbose=2)
    model.save(f"model_repo/{name}/1/model.savedmodel")
    return model

wd  = train("ranker_wd",  build_wide_deep)
dcn = train("ranker_dcn", build_dcn)

te["score_wd"]  = wd.predict(Xte,  batch_size=8192, verbose=0).ravel()
te["score_dcn"] = dcn.predict(Xte, batch_size=8192, verbose=0).ravel()
te.to_parquet("data/processed/ranker_test_scored.parquet")   # reused by the A/B test

# ---------- metrics (per user, over that user's 300 candidates) ----------
def evaluate(score):
    d = te[["userId", "label"]].copy(); d["s"] = score
    nd, rc = [], []
    for _, g in d.groupby("userId", sort=False):
        y = g.label.values
        if y.sum() == 0: continue
        ys = y[np.argsort(-g.s.values, kind="stable")]
        dcg = sum(ys[i] / np.log2(i + 2) for i in range(min(10, len(ys))))
        idcg = sum(1 / np.log2(i + 2) for i in range(int(min(y.sum(), 10))))
        nd.append(dcg / idcg); rc.append(ys[:10].sum() / y.sum())
    return np.mean(nd), np.mean(rc), len(nd)

print("\nTest users with at least one liked candidate:", evaluate(te.ret_score)[2])
print(f"{'method':<18}{'NDCG@10':>9}{'Recall@10':>11}{'AUC':>8}")
for name, s in [("popularity", te.log_pop_all.values), ("retrieval order", te.ret_score.values),
                ("wide & deep", te.score_wd.values), ("DCN-V2", te.score_dcn.values)]:
    n, r, _ = evaluate(s)
    print(f"{name:<18}{n:>9.4f}{r:>11.4f}{roc_auc_score(te.label, s):>8.4f}")