"""Shared data, model, and recommender layer for Movie Studio.

Both interfaces (the Dash dashboard and the FastAPI service) import from here,
so there is exactly one implementation of feature engineering, prediction,
and recommendation.
"""
import io
import json
import os
import ssl
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path

import certifi
import joblib
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parent
OFFLINE = os.getenv("MOVIES_OFFLINE") == "1"
ARTIFACTS = ROOT / ("artifacts_offline" if OFFLINE else "artifacts")
MODEL_PATH = ARTIFACTS / "model.joblib"
METRICS_PATH = ARTIFACTS / "metrics.json"
DATA_DIR = ROOT / "data_cache" / "ml-latest-small"
DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

MIN_RATINGS = 10        # movies with fewer ratings are dropped from the catalog (noisy averages)
REC_MIN_RATINGS = 20    # a movie needs at least this many ratings to be recommended
SHRINK = 25             # shrinkage strength for item-item similarity: sim * co / (co + SHRINK)
GENRES = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama",
          "Fantasy", "Film-Noir", "Horror", "IMAX", "Musical", "Mystery", "Romance", "Sci-Fi",
          "Thriller", "War", "Western"]
FEATURES = ["year", "log_n_ratings", "n_genres"] + GENRES

def synthetic(n_movies = 1500, n_users=600, seed=0):
    rng = np.random.default_rng(seed)
    years = rng.integers(1950, 2019, n_movies)
    gens = [rng.choice(GENRES, size=rng.integers(1,4), replace=False) for _ in range(n_movies)]
    movies = pd.DataFrame({"movieId": np.arange(1, n_movies + 1), 
        "title": [f"Synthetic Movie {i} ({y})" for i,y in enumerate(years, 1)],
        "genres": ["|".join(g) for g in gens]})
    pop_raw = rng.lognormal(0, 1, n_movies)
    quality = rng.normal(3.2, 0.4, n_movies) + 0.12 * np.log(pop_raw)
    pop = pop_raw / pop_raw.sum()
    aff = rng.normal(0, 0.6, (n_users, len(GENRES)))
    gidx = [[GENRES.index(x) for x in g] for g in gens]
    recs = []
    for u in range(n_users):
        k = int(rng.integers(40, 200))
        ms = rng.choice(n_movies, size=k, replace=False, p=pop)
        base = np.array([quality[m] + aff[u, gidx[m]].mean() for m in ms])
        r = np.clip(np.round((base + rng.normal(0, 0.6, k))*2) / 2, 0.5, 5)
        recs.append(pd.DataFrame({"userId": u + 1, "movieId": ms + 1, "rating": r}))
    return movies, pd.concat(recs, ignore_index=True)

def ensure_download():
    if(DATA_DIR / "movies.csv").exists():
        return
    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading MovieLens (small, ~1MB) from grouplens.org ...")
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(DATA_URL, timeout=60, context=ctx) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(DATA_DIR.parent)

@lru_cache(maxsize=1)
def load_raw():
    if OFFLINE:
        return synthetic()
    ensure_download()
    return (pd.read_csv(DATA_DIR / "movies.csv"), pd.read_csv(DATA_DIR / "ratings.csv", 
            usecols = ["userId", "movieId", "rating"]))

@lru_cache(maxsize=1)
def movie_table() -> pd.DataFrame:
    movies, ratings = load_raw()
    stats = ratings.groupby("movieId")["rating"].agg(avg_rating="mean", n_ratings="size").reset_index()
    t = movies.merge(stats, on="movieId")
    t = t[t.n_ratings >= MIN_RATINGS].copy()
    t["year"] = t["title"].str.extract(r"\((\d{4})\)\s*$")[0]
    t = t.dropna(subset=["year"])
    t["year"] = t["year"].astype(int)
    t["name"] = t["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
    gl = t["genres"].str.split("|")
    for g in GENRES:
        t[g] = gl.apply(lambda lst, g=g: int(g in lst))
    t["avg_rating"] = t["avg_rating"].round(3)
    return t.reset_index(drop=True)

def featurize(df: pd.DataFrame) -> pd.DataFrame:
    X = df[["year"] + GENRES].copy()
    X["log_n_ratings"] = np.log1p(df["n_ratings"])
    X["n_genres"] = df[GENRES].sum(axis=1)
    return X[FEATURES]

@lru_cache(maxsize=1)
def get_bundle():
    if not MODEL_PATH.exists():
        import train
        train.main()
    return joblib.load(MODEL_PATH)

@lru_cache(maxsize=1)
def get_metrics() -> dict:
    get_bundle()
    return json.loads(METRICS_PATH.read_text())

def predict(rows: list) -> pd.DataFrame:
    """Predict average rating plus an 80% interval for each movie description.

    Each row is ``{"year": int, "n_ratings": int, "genres": [str, ...]}``.
    Returns a DataFrame with ``predicted_rating``, ``low_80``, ``high_80``.
    """
    if not rows:
        return pd.DataFrame(columns=["predicted_rating", "low_80", "high_80"])
    b = get_bundle()
    df = pd.DataFrame({"year": [r["year"] for r in rows], "n_ratings": [r["n_ratings"] for r in rows]})
    for g in GENRES:
        df[g] = [int(g in r["genres"]) for r in rows]
    X = featurize(df)
    mid, lo, hi = (b[k].predict(X) for k in ("mid", "low", "high"))
    lo, hi = np.minimum(lo, mid), np.maximum(hi, mid)
    out = pd.DataFrame({"predicted_rating": mid, "low_80": lo, "high_80": hi})
    return out.clip(0.5, 5.0).round(2)

def build_item_matrix(table: pd.DataFrame, ratings: pd.DataFrame):
    """Build the item-item similarity inputs from an arbitrary ratings frame.

    Kept separate from the cached ``item_matrix()`` so the recommender can be
    evaluated on a train split without leaking held-out ratings.

    Returns ``(Cn, B, mid)``:
      Cn  - row-normalised, user-mean-centred rating matrix (movies x users)
      B   - binary "who rated what" matrix, used for co-rating counts
      mid - Index mapping matrix rows to movieIds
    """
    r = ratings[ratings.movieId.isin(table.movieId)].copy()
    r["c"] = r["rating"] - r.groupby("userId")["rating"].transform("mean")
    mid = pd.Index(table.movieId)
    uid = pd.Index(r.userId.unique())
    rows, cols = mid.get_indexer(r.movieId), uid.get_indexer(r.userId)
    C = sparse.csr_matrix((r.c.values, (rows, cols)), shape=(len(mid), len(uid)))
    B = sparse.csr_matrix((np.ones(len(r)), (rows, cols)), shape=C.shape)
    norms = np.sqrt(np.asarray(C.multiply(C).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return (sparse.diags(1 / norms) @ C).tocsr(), B, mid


@lru_cache(maxsize=1)
def item_matrix():
    _, ratings = load_raw()
    return build_item_matrix(movie_table(), ratings)


def score_items(Cn, B, seed_idx) -> np.ndarray:
    """Average shrunk cosine similarity of every movie to the seed rows."""
    total = np.zeros(Cn.shape[0])
    for i in seed_idx:
        sim = (Cn @ Cn[i].T).toarray().ravel()
        co = (B @ B[i].T).toarray().ravel()
        total += sim * co / (co + SHRINK)
    return total / len(seed_idx)


def recommend(movie_ids, n=10, genre=None, matrices=None, table=None) -> pd.DataFrame:
    """Item-based recommendations for one or more liked movieIds.

    ``matrices`` / ``table`` can be injected (used by evaluate.py); by default
    the cached full-data versions are used.
    """
    Cn, B, mid = matrices if matrices is not None else item_matrix()
    t = table if table is not None else movie_table()
    cols = ["movieId", "title", "genres", "year", "avg_rating", "n_ratings", "similarity"]
    idx = mid.get_indexer(list(movie_ids))
    idx = idx[idx >= 0]
    if len(idx) == 0:
        return pd.DataFrame(columns=cols)
    total = score_items(Cn, B, idx)
    total[idx] = -np.inf
    out = t.assign(similarity=total.round(4))
    out = out[(out.n_ratings >= REC_MIN_RATINGS) & np.isfinite(out.similarity)]
    if genre:
        out = out[out[genre] == 1]
    return out.nlargest(n, "similarity")[cols]
