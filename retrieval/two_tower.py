import numpy as np, pandas as pd, tensorflow as tf
import tensorflow_recommenders as tfrs

EMB, BATCH, EPOCHS = 64, 4096, 30     # 30 epochs because the small dataset has few ratings

train = pd.read_parquet("data/processed/train.parquet", columns=["userId", "movieId", "label"])
pos = train[train.label == 1]                       # only "liked" interactions

user_ids = np.sort(pos.userId.unique())
item_ids = np.sort(pos.movieId.unique())
u_map = pd.Series(np.arange(len(user_ids)), index=user_ids)
i_map = pd.Series(np.arange(len(item_ids)), index=item_ids)

u_idx = u_map.loc[pos.userId.values].values.astype("int32")   # user id -> 0..n-1
i_idx = i_map.loc[pos.movieId.values].values.astype("int32")  # movie id -> 0..n-1

# how often each movie appears (corrects popularity bias in the negatives)
counts = np.bincount(i_idx, minlength=len(item_ids))
prob = (counts / counts.sum()).astype("float32")
p = prob[i_idx]

def tower(n):
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(n, EMB),
        tf.keras.layers.Dense(EMB),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1)),  # length 1, so dot product = cosine
    ])

class TwoTower(tfrs.Model):
    def __init__(self):
        super().__init__()
        self.user_model = tower(len(user_ids))
        self.item_model = tower(len(item_ids))
        self.task = tfrs.tasks.Retrieval(temperature=0.05)

    def compute_loss(self, features, training=False):
        u = self.user_model(features["u"])
        i = self.item_model(features["i"])
        return self.task(u, i, candidate_sampling_probability=features["p"])

ds = (tf.data.Dataset.from_tensor_slices({"u": u_idx, "i": i_idx, "p": p})
      .shuffle(500_000).batch(BATCH).prefetch(tf.data.AUTOTUNE))

model = TwoTower()
model.compile(optimizer=tf.keras.optimizers.Adam(0.005))
model.fit(ds, epochs=EPOCHS)

# save embeddings and the id maps (row i of each array = ids[i])
np.save("retrieval/user_embeddings.npy", model.user_model(tf.range(len(user_ids))).numpy().astype("float32"))
np.save("retrieval/item_embeddings.npy", model.item_model(tf.range(len(item_ids))).numpy().astype("float32"))
np.save("retrieval/user_ids.npy", user_ids)
np.save("retrieval/item_ids.npy", item_ids)
print("saved", len(user_ids), "users,", len(item_ids), "items")