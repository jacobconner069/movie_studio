import core

SAMPLE = {"year": 2005, "n_ratings": 200, "genres": ["Drama", "Crime"]}


def test_movie_table_schema():
    t = core.movie_table()
    assert len(t) > 0
    assert {"name", "year", "avg_rating", "n_ratings", *core.GENRES} <= set(t.columns)
    assert (t.n_ratings >= core.MIN_RATINGS).all()
    assert t.avg_rating.between(0.5, 5).all()


def test_predict_interval_order():
    r = core.predict([SAMPLE, {**SAMPLE, "year": 1990}])
    assert len(r) == 2
    assert ((0.5 <= r.low_80) & (r.low_80 <= r.predicted_rating)
            & (r.predicted_rating <= r.high_80) & (r.high_80 <= 5)).all()


def test_recommend():
    t = core.movie_table()
    liked = int(t.sort_values("n_ratings", ascending=False).movieId.iloc[0])
    rec = core.recommend([liked], n=5)
    assert len(rec) == 5
    assert liked not in set(rec.movieId)
    assert rec.similarity.is_monotonic_decreasing


def test_recommend_genre_filter_and_unknown_id():
    t = core.movie_table()
    liked = int(t.movieId.iloc[0])
    rec = core.recommend([liked], n=5, genre="Drama")
    assert rec.genres.str.contains("Drama").all()
    assert core.recommend([-1], n=5).empty


def test_dashboard_callbacks():
    import dashboard as d
    scatter, bar, rows, msg = d.explore(None, [d.YMIN, d.YMAX], 50)
    assert len(rows) <= 60 and "movies match" in msg
    out, fig = d.predict_cb(2010, 200, ["Drama"])
    assert len(fig.data) == 4
    table, text = d.recommend_cb([int(d.T.movieId.iloc[0])], None, 5)
    assert len(table) == 5

def test_predict_empty_input():
    assert core.predict([]).empty


def test_build_item_matrix_matches_cached():
    t = core.movie_table()
    _, ratings = core.load_raw()
    Cn, B, mid = core.build_item_matrix(t, ratings)
    Cn2, B2, mid2 = core.item_matrix()
    assert Cn.shape == Cn2.shape and (mid == mid2).all()
    assert abs(Cn - Cn2).max() < 1e-12


def test_item_vectors_are_unit_length_or_zero():
    Cn, _, _ = core.item_matrix()
    norms = (Cn.multiply(Cn)).sum(axis=1).A1 ** 0.5
    assert ((abs(norms - 1) < 1e-9) | (norms == 0)).all()
