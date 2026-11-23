"""FastAPI service exposing the same model and recommender as the dashboard.

Run:  uvicorn api:app --reload        (interactive docs at /docs)
"""
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

import core

app = FastAPI(
    title="Movie Studio API",
    description="Rating prediction with 80% intervals and item-based recommendations "
                "on MovieLens latest-small.",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    year: int = Field(2010, ge=1900, le=2030, description="Release year")
    n_ratings: int = Field(200, ge=1, description="Expected number of ratings (audience size)")
    genres: List[str] = Field(..., min_length=1, examples=[["Drama", "Crime"]])

    @field_validator("genres")
    @classmethod
    def genres_must_be_known(cls, v):
        bad = sorted(set(v) - set(core.GENRES))
        if bad:
            raise ValueError(f"unknown genres {bad}; valid options: {core.GENRES}")
        return v


class PredictResponse(BaseModel):
    predicted_rating: float
    low_80: float
    high_80: float


class RecommendRequest(BaseModel):
    movie_ids: List[int] = Field(..., min_length=1, description="movieIds the user likes")
    n: int = Field(10, ge=1, le=50)
    genre: Optional[str] = None

    @field_validator("genre")
    @classmethod
    def genre_must_be_known(cls, v):
        if v is not None and v not in core.GENRES:
            raise ValueError(f"unknown genre {v!r}")
        return v


@app.get("/health")
def health():
    return {"status": "ok", "n_movies": len(core.movie_table())}


@app.get("/metrics")
def metrics():
    """Held-out evaluation metrics of the deployed rating model."""
    m = core.get_metrics()
    return {k: v for k, v in m.items() if k != "scatter"}


@app.get("/movies/search")
def search_movies(q: str = Query(..., min_length=2), limit: int = Query(10, ge=1, le=50)):
    """Find movieIds by title substring (use these ids with /recommend)."""
    t = core.movie_table()
    hits = t[t.title.str.contains(q, case=False, regex=False)]
    hits = hits.sort_values("n_ratings", ascending=False).head(limit)
    return hits[["movieId", "title", "genres", "avg_rating", "n_ratings"]].to_dict("records")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    row = {"year": req.year, "n_ratings": req.n_ratings, "genres": req.genres}
    return core.predict([row]).iloc[0].to_dict()


@app.post("/recommend")
def recommend(req: RecommendRequest):
    rec = core.recommend(req.movie_ids, n=req.n, genre=req.genre)
    if rec.empty:
        raise HTTPException(404, "None of the supplied movie_ids are in the catalog.")
    return rec.to_dict("records")
