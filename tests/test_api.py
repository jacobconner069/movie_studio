from fastapi.testclient import TestClient

import api
import core

client = TestClient(api.app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["n_movies"] > 0


def test_metrics_has_no_scatter_payload():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "rmse" in r.json() and "scatter" not in r.json()


def test_predict_ok():
    r = client.post("/predict", json={"year": 2005, "n_ratings": 200, "genres": ["Drama"]})
    body = r.json()
    assert r.status_code == 200
    assert body["low_80"] <= body["predicted_rating"] <= body["high_80"]


def test_predict_rejects_unknown_genre_and_bad_year():
    assert client.post("/predict", json={"year": 2005, "n_ratings": 5, "genres": ["Nope"]}).status_code == 422
    assert client.post("/predict", json={"year": 1500, "n_ratings": 5, "genres": ["Drama"]}).status_code == 422


def test_search_then_recommend():
    title = core.movie_table().title.iloc[0][:6]
    hits = client.get("/movies/search", params={"q": title}).json()
    assert hits
    r = client.post("/recommend", json={"movie_ids": [hits[0]["movieId"]], "n": 5})
    assert r.status_code == 200 and len(r.json()) == 5


def test_recommend_unknown_ids_404():
    assert client.post("/recommend", json={"movie_ids": [-1]}).status_code == 404
