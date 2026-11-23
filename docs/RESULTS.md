## Rating model (5-fold CV on movie-level average rating)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Predict the mean (baseline) | 0.505 | 0.405 | -0.001 ± 0.001 |
| Ridge regression | 0.427 | 0.338 | 0.286 ± 0.033 |
| Random forest | 0.396 | 0.305 | 0.383 ± 0.058 |
| Gradient boosting (used in app) | 0.391 | 0.300 | 0.399 ± 0.051 |

## Recommender (hold-out hit rate @ 10, 300 users)

| Method | Hit rate | Catalog coverage |
|---|---|---|
| Item-based CF (used in app) | 11.0% | 38.6% |
| Most-popular baseline | 10.3% | 1.0% |

*Protocol: for each user, hide one liked movie (rating >= 4.0), seed with 5 other liked movies, rebuild similarities without the hidden rating, and check whether the hidden movie appears in the top-K.*
