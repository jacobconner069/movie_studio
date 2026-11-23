"""Honest offline evaluation for both models.

1. Rating model: compares the gradient-boosting model against simple baselines
   with the same 5-fold cross-validation.
2. Recommender: leave-one-out style hold-out. For a sample of users, one movie
   they liked (rating >= 4) is hidden, the item matrix is rebuilt WITHOUT that
   rating, and the user's remaining liked movies seed the recommender. We then
   check whether the hidden movie shows up in the top-K, and compare to a
   "recommend the most-rated movies" popularity baseline.

Usage:
    python evaluate.py                 # prints a Markdown report, saves artifacts/evaluation.json
    python evaluate.py --write-docs    # also writes docs/RESULTS.md (commit this for your README)
    MOVIES_OFFLINE=1 python evaluate.py   # synthetic smoke test only; do NOT publish those numbers
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_validate

import core

LIKE_THRESHOLD = 4.0
SEED_SIZE = 5


def compare_rating_models(seed: int = 42) -> pd.DataFrame:
    t = core.movie_table()
    X, y = core.featurize(t), t["avg_rating"]
    models = {
        "Predict the mean (baseline)": DummyRegressor(),
        "Ridge regression": RidgeCV(alphas=np.logspace(-2, 3, 12)),
        "Random forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                               n_jobs=-1, random_state=seed),
        "Gradient boosting (used in app)": HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=20,
            l2_regularization=1.0, random_state=seed),
    }
    cv = KFold(5, shuffle=True, random_state=seed)
    rows = []
    for name, m in models.items():
        r = cross_validate(m, X, y, cv=cv, scoring=("r2", "neg_root_mean_squared_error",
                                                    "neg_mean_absolute_error"))
        rows.append({"model": name,
                     "rmse": -r["test_neg_root_mean_squared_error"].mean(),
                     "mae": -r["test_neg_mean_absolute_error"].mean(),
                     "r2": r["test_r2"].mean(), "r2_std": r["test_r2"].std()})
    return pd.DataFrame(rows)


def evaluate_recommender(k: int = 10, n_users: int = 300, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    t = core.movie_table()
    _, ratings = core.load_raw()
    ratings = ratings[ratings.movieId.isin(t.movieId)]
    eligible = set(t.loc[t.n_ratings >= core.REC_MIN_RATINGS, "movieId"])

    liked = ratings[ratings.rating >= LIKE_THRESHOLD]
    by_user = liked.groupby("userId")["movieId"].apply(list)
    by_user = by_user[by_user.apply(lambda m: len(m) >= SEED_SIZE + 1)]
    users = rng.choice(by_user.index.values, size=min(n_users, len(by_user)), replace=False)

    cases = []  # (userId, held_out_movie, seed_movies)
    for u in users:
        movies = by_user[u]
        held_pool = [m for m in movies if m in eligible]
        if not held_pool:
            continue
        held = int(rng.choice(held_pool))
        rest = [m for m in movies if m != held]
        seeds = [int(m) for m in rng.choice(rest, size=SEED_SIZE, replace=False)]
        cases.append((u, held, seeds))

    # Rebuild the similarity matrices with every held-out rating removed (no leakage).
    held_pairs = pd.MultiIndex.from_tuples([(u, h) for u, h, _ in cases])
    keep = ~pd.MultiIndex.from_frame(ratings[["userId", "movieId"]]).isin(held_pairs)
    matrices = core.build_item_matrix(t, ratings[keep])

    pop_order = t[t.n_ratings >= core.REC_MIN_RATINGS].sort_values("n_ratings", ascending=False)
    hits_model = hits_pop = 0
    rec_items, pop_items = set(), set()
    for _, held, seeds in cases:
        rec = core.recommend(seeds, n=k, matrices=matrices, table=t).movieId.tolist()
        pop = pop_order[~pop_order.movieId.isin(seeds)].movieId.head(k).tolist()
        hits_model += held in rec
        hits_pop += held in pop
        rec_items.update(rec)
        pop_items.update(pop)

    n = len(cases)
    n_catalog = int((t.n_ratings >= core.REC_MIN_RATINGS).sum())
    return {"k": k, "n_cases": n, "seed_size": SEED_SIZE,
            "hit_rate_model": hits_model / n, "hit_rate_popularity": hits_pop / n,
            "catalog_coverage_model": len(rec_items) / n_catalog,
            "catalog_coverage_popularity": len(pop_items) / n_catalog}


def to_markdown(models: pd.DataFrame, rec: dict) -> str:
    lines = ["## Rating model (5-fold CV on movie-level average rating)", "",
             "| Model | RMSE | MAE | R² |", "|---|---|---|---|"]
    for r in models.itertuples():
        lines.append(f"| {r.model} | {r.rmse:.3f} | {r.mae:.3f} | {r.r2:.3f} ± {r.r2_std:.3f} |")
    lines += ["", f"## Recommender (hold-out hit rate @ {rec['k']}, {rec['n_cases']} users)", "",
              "| Method | Hit rate | Catalog coverage |", "|---|---|---|",
              f"| Item-based CF (used in app) | {rec['hit_rate_model']:.1%} | "
              f"{rec['catalog_coverage_model']:.1%} |",
              f"| Most-popular baseline | {rec['hit_rate_popularity']:.1%} | "
              f"{rec['catalog_coverage_popularity']:.1%} |", "",
              f"*Protocol: for each user, hide one liked movie (rating >= {LIKE_THRESHOLD}), "
              f"seed with {rec['seed_size']} other liked movies, rebuild similarities without the "
              "hidden rating, and check whether the hidden movie appears in the top-K.*"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-docs", action="store_true", help="write docs/RESULTS.md")
    args = ap.parse_args()
    models, rec = compare_rating_models(), evaluate_recommender()
    md = to_markdown(models, rec)
    print(md)
    core.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (core.ARTIFACTS / "evaluation.json").write_text(
        json.dumps({"models": models.to_dict("records"), "recommender": rec}, indent=2))
    if args.write_docs:
        docs = core.ROOT / "docs"
        docs.mkdir(exist_ok=True)
        (docs / "RESULTS.md").write_text(md)
        print(f"Wrote {docs / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
