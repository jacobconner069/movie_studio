"""Train and evaluate the rating model; writes model.joblib and metrics.json."""
import json

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
import core

def main():
    t = core.movie_table()
    x, y = core.featurize(t), t["avg_rating"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    params = dict(max_iter=250, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=20,
                  l2_regularization=1.0, random_state=42)
    mid = HistGradientBoostingRegressor(**params).fit(x_train, y_train)
    low = HistGradientBoostingRegressor(loss="quantile", quantile=0.1, **params).fit(x_train, y_train)
    high = HistGradientBoostingRegressor(loss="quantile", quantile=0.9, **params).fit(x_train, y_train)

    pred = mid.predict(x_test)
    lo, hi = np.minimum(low.predict(x_test), pred), np.maximum(high.predict(x_test), pred)
    cv = cross_val_score(HistGradientBoostingRegressor(**params), x, y, cv=5, scoring="r2")
    imp = permutation_importance(mid, x_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))

    idx = np.random.default_rng(0).choice(len(x_test), size=min(1200, len(x_test)), replace=False)
    top = sorted(zip(core.FEATURES, map(float, imp.importances_mean)), key=lambda kv: kv[1])[-12:]
    metrics={"n_movies": int(len(t)), "n_train": int(len(x_train)), "n_test": int(len(x_test)),
             "rmse": rmse, "mae": float(mean_absolute_error(y_test, pred)),
             "baseline_rmse": float(np.sqrt(np.mean((y_test - y_train.mean())**2))),
             "r2_test": float(r2_score(y_test, pred)),
             "r2_cv_mean": float(cv.mean()), "r2_cv_std": float(cv.std()),
             "interval_coverage_80": float(np.mean((y_test >= lo) & (y_test <= hi))),
             "importance": dict(top), "scatter": {"actual": y_test.values[idx].tolist(),
                                                  "pred": pred[idx].tolist()},}
    core.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"mid": mid, "low": low, "high": high}, core.MODEL_PATH)
    core.METRICS_PATH.write_text(json.dumps(metrics))
    print(f"Saved model to {core.MODEL_PATH}\n"
          f"  test R2 = {metrics['r2_test']:.3f}\n"
          f"  test RMSE = {rmse:.3f} (baseline {metrics['baseline_rmse']:.3f})\n"
          f"  80% interval coverage = {metrics['interval_coverage_80']:.1%}")

if __name__ == "__main__":
    main()
