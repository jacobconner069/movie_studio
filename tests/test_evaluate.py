import evaluate


def test_recommender_eval_runs_and_is_well_formed():
    r = evaluate.evaluate_recommender(k=10, n_users=40, seed=1)
    assert r["n_cases"] > 0
    for key in ("hit_rate_model", "hit_rate_popularity",
                "catalog_coverage_model", "catalog_coverage_popularity"):
        assert 0.0 <= r[key] <= 1.0


def test_heldout_rating_is_removed_from_similarity_matrices():
    """Guards against leakage: a rating hidden for evaluation must not be in the matrices."""
    import core
    t = core.movie_table()
    _, ratings = core.load_raw()
    r = ratings[ratings.movieId.isin(t.movieId)]
    u, m = int(r.userId.iloc[0]), int(r.movieId.iloc[0])
    keep = ~((r.userId == u) & (r.movieId == m))
    _, B_full, mid = core.build_item_matrix(t, r)
    _, B_cut, _ = core.build_item_matrix(t, r[keep])
    assert B_full.sum() - B_cut.sum() == 1
