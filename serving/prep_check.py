import numpy as np, pandas as pd

FEATURES = ["ret_score", "ret_rank", "genre_aff", "log_pop_all", "log_pop_recent",
            "avg_rating_smooth", "year", "log_n_ratings", "user_avg_rating",
            "user_rating_std", "rating_gap"]
te = pd.read_parquet("data/processed/ranker_test_scored.parquet")
np.save("/tmp/check_X.npy", te[FEATURES].values.astype("float32")[:2000])
np.save("/tmp/check_scores.npy", te.score_wd.values.astype("float32")[:2000])
print("saved", len(te[:2000]), "rows")