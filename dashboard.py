"""Plotly Dash dashboard: Explore, Predict rating, Recommend, About the model."""
import json
import os

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html

import core

T = core.movie_table()
M = core.get_metrics()
YMIN, YMAX = int(T.year.min()), int(T.year.max())
TITLE_OPTIONS = [{"label": r.title, "value": int(r.movieId)} 
                 for r in T.sort_values("n_ratings", ascending=False).itertuples()]
GENRE_OPTIONS = [{"label": g, "value": g} for g in core.GENRES]
TABLE_STYLE = dict(style_cell={"textAlign": "left", "padding": "6px", "fontFamily": "sans-serif"},
                   style_header={"fontWeight": "bold"}, page_size=15)

app = Dash(__name__, title="Movie Studio")
server = app.server

def card(title, value):
    return html.Div([html.Div(title, style={"fontSize": 12, "color": "#666"}),
                     html.Div(value, style={"fontSize": 22, "fontWeight": "bold"})],
                     style={"border": "1px solid #ddd", "borderRadius": 8, "padding": "10px 16px",
                            "minWidth": 150})

def empty_fig(msg):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font={"size": 16})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig

explore_tab = html.Div([
    html.P("Filter the MovieLens catalog; every chart and the table update instantly."),
    html.Div([
        html.Div([html.Label("Genres (any of)"),
                  dcc.Dropdown(id="ex-genres", options=GENRE_OPTIONS, multi=True,
                                placeholder="All genres")], style={"flex": 2, "minWidth": 260}),
        html.Div([html.Label("Release Years"),
                  dcc.RangeSlider(id="ex-years", min=YMIN, max=YMAX, value=[YMIN, YMAX], step=1,
                                  marks={y: str(y) for y in range((YMIN // 20+1)*20, YMAX + 1, 20)},
                                  tooltip={"placement": "bottom"})], style={"flex": 3, "minWidth": 300}),
        html.Div([html.Label("Minimum number of ratings"),
                  dcc.Slider(id="ex-minr", min=10, max=300, step=10, value=50, marks={10:"10",
                            100: "100", 200: "200", 300: "300"}, tooltip={"placement": "bottom"})],
                            style={"flex": 2, "minWidth": 220}),],
    style={"display": "flex", "gap": 24, "flexWrap": "wrap"}), html.H4(id="ex-count"),
    dcc.Graph(id="ex-scatter"), 
    html.Div([dcc.Graph(id="ex-genre-bar", style={"flex": 1}),
            html.Div([html.H4("Top-rated movies in this selection"),
                    dash_table.DataTable(id="ex-table", columns=[{"name": "Title", "id": "name"},
                        {"name": "Year", "id": "year"}, {"name": "Avg rating", "id": "avg_rating"},
                        {"name": "# ratings", "id": "n_ratings"}], **TABLE_STYLE)], style={"flex": 1})],
            style={"display": "flex", "gap": 24, "flexWrap": "wrap"}),])

predict_tab = html.Div([
    html.P("Describe a movie and the model predicts its average rating (0.5-5 scale) with an 80% "
           "prediction interval."),
    html.Div([
        html.Div([html.Label("Release Year"),
                  dcc.Input(id="pr-year", type="number", min=1900, max=2030, value=2010, debounce=True)]),
        html.Div([html.Label("Expected audience size (# ratings)"),
                  dcc.Input(id="pr-n", type="number", min=1, value=200, debounce=True)]),],
    style={"display": "flex", "gap": 32, "marginBottom": 8}),
    html.Label("Genres"),
    dcc.Checklist(id="pr-genres", options=GENRE_OPTIONS, value=["Drama"], inline=True,
                  labelStyle={"marginRight": 14}),
    html.Div(id="pr-out", style={"margin": "16px 0"}),
    dcc.Graph(id="pr-whatif"),])

recommend_tab = html.Div([
    html.P("Pick movies you like to form recommendations from item-item collaborative filtering."),
    html.Div([html.Div([html.Label("Movies you like"),
                        dcc.Dropdown(id="rc-movies", options=TITLE_OPTIONS, multi=True,
                                     placeholder="Search a title")], style={"flex": 3, "minWidth": 300}),
    html.Div([html.Label("Restrict to genre"), dcc.Dropdown(id="rc-genre", options=GENRE_OPTIONS,
            placeholder="Any")], style={"flex": 1, "minWidth": 160}),
    html.Div([html.Label("How many"), dcc.Slider(id="rc-n", min=5, max=25, step=5, value=10,
            marks={5: "5", 15: "15", 25: "25",})], style={"flex": 1, "minWidth": 160}),],
    style={"display": "flex", "gap": 24, "flexWrap": "wrap"}),
    html.Div(id="rc-msg", style={"margin": "12px 0"}),
    html.Div([dash_table.DataTable(id="rc-table", columns=[{"name": "Title", "id": "title"},
            {"name": "Genres", "id": "genres"}, {"name": "Avg rating", "id": "avg_rating"},
            {"name": "# ratings", "id": "n_ratings"}, {"name": "Similarity", "id": "similarity"}],
            **TABLE_STYLE)]),])

imp = M["importance"]
EVAL_PATH = core.ARTIFACTS / "evaluation.json"
EVAL = json.loads(EVAL_PATH.read_text()) if EVAL_PATH.exists() else None


def recommender_eval_block():
    """Hold-out recommender results; only shown after `python evaluate.py` has been run."""
    if EVAL is None:
        return html.P("Run `python evaluate.py` to add recommender hold-out results here.",
                      style={"color": "#666"})
    r = EVAL["recommender"]
    return html.Div([
        html.H4(f"Recommender hold-out test (hit rate @ {r['k']}, {r['n_cases']} users)"),
        html.Div([card("Item-based CF", f"{r['hit_rate_model']:.1%}"),
                  card("Most-popular baseline", f"{r['hit_rate_popularity']:.1%}"),
                  card("Catalog coverage (CF)", f"{r['catalog_coverage_model']:.1%}"),
                  card("Catalog coverage (popular)", f"{r['catalog_coverage_popularity']:.1%}")],
                 style={"display": "flex", "gap": 12, "flexWrap": "wrap"})])


model_tab = html.Div([
    html.P(f"Gradient boosted trees trained on {M['n_train']:,} movies, evaluated on {M['n_test']:,} "
           "held-out movies."),
    html.Div([card("Test R^2", f"{M['r2_test']:.3f}"),
              card("5-fold CV R^2", f"{M['r2_cv_mean']:.3f} ± {M['r2_cv_std']:.3f}"),
              card("Test RMSE", f"{M['rmse']:.3f} stars"),
              card("Baseline RMSE (predict mean)", f"{M['baseline_rmse']:.3f}"),
              card("80% interval coverage", f"{M['interval_coverage_80']:.1%}")],
              style={"display": "flex", "gap": 12, "flexWrap": "wrap"}),
    html.Div([dcc.Graph(style={"flex": 1}, figure=px.scatter(
        x=M["scatter"]["actual"], y=M["scatter"]["pred"], opacity=0.5,
        labels={"x": "Actual avg rating", "y": "Predicted avg rating"},
        title="Held out predictions vs actual").add_shape(
        type="line", x0=1, y0=1, x1=5, y1=5, line={"dash": "dash"})),
        dcc.Graph(style={"flex": 1}, figure=px.bar(x=list(imp.values()), y=list(imp.keys()),
        orientation="h", labels={"x": "Permutation importance (drop in R² when shuffled)", "y": ""},
        title="What drives predicted ratings")),], style={"display": "flex", "flexWrap": "wrap"}),
    recommender_eval_block(),])

app.layout = html.Div([html.H2("Movie Studio: explore, predict, recommend"),
    dcc.Tabs([dcc.Tab(label="Explore", children=explore_tab),
            dcc.Tab(label="Predict rating", children=predict_tab),
            dcc.Tab(label="Recommend", children=recommend_tab),
            dcc.Tab(label="About the model", children=model_tab)]),],
    style={"maxWidth": 1200, "margin": "0 auto", "padding": 16, "fontFamily": "sans-serif"})

@app.callback(Output("ex-scatter", "figure"), Output("ex-genre-bar", "figure"),
              Output("ex-table", "data"), Output("ex-count", "children"),
               Input("ex-genres", "value"), Input("ex-years", "value"), Input("ex-minr", "value"))
def explore(genres, years, minr):
    d = T[T.year.between(years[0], years[1]) & (T.n_ratings >= minr)]
    if genres:
        d = d[d[genres].sum(axis=1)>0]
    if d.empty:
        return empty_fig("No movies match"), empty_fig("No movies match"), [], "0 movies"
    scatter = px.scatter(d, x="year", y="avg_rating", size="n_ratings", size_max=18, opacity=0.55,
                         hover_name="name", hover_data={"n_ratings": True, "year": True},
                         labels={"avg_rating": "Average rating", "year": "Release year"},
                         title="Average rating by release year")
    g = [(x, d.loc[d[x] == 1, "avg_rating"].mean(), 
          int(d[x].sum())) for x in core.GENRES if d[x].sum() > 0]
    g.sort(key=lambda r: r[1])
    bar = px.bar(x=[r[1] for r in g], y=[r[0] for r in g], orientation="h", 
                 hover_data={"n_movies": [r[2] for r in g]}, labels={"x": "Mean of movie averages",
                "y": ""}, title="Average rating by genre")
    bar.update_xaxes(range=[max(0, min(r[1] for r in g) - 0.3), 5])
    top = d.nlargest(60, "avg_rating")[["name", "year", "avg_rating", "n_ratings"]]
    return scatter, bar, top.to_dict("records"), f"{len(d):,} movies match (avg {d.avg_rating.mean():.2f})"

@app.callback(Output("pr-out", "children"), Output("pr-whatif", "figure"), Input("pr-year", "value"),
              Input("pr-n", "value"), Input("pr-genres", "value"))
def predict_cb(year, n, genres):
    if year is None or n is None or not genres:
        return "Enter a year, audience size, and at least one genre.", empty_fig("Waiting for valid inputs")
    row = {"year": int(np.clip(year, 1900, 2030)), "n_ratings": max(1, int(n)), "genres": genres}
    p = core.predict([row]).iloc[0]
    years = np.arange(YMIN, YMAX + 1)
    grid = core.predict([{**row, "year": int(y)} for y in years])
    fig = go.Figure([go.Scatter(x=years, y=grid.high_80, line={"width": 0}, showlegend=False,
                                hoverinfo="skip"),
                    go.Scatter(x=years, y=grid.low_80, fill="tonexty", line={"width": 0},
                               name="80% interval"),
                    go.Scatter(x=years, y=grid.predicted_rating, name="Predicted rating"),
                    go.Scatter(x=[row["year"]], y=[p.predicted_rating], mode="markers",
                               marker={"size": 12}, name="Your movie")])
    fig.update_layout(title="What if: same movie, different release year", xaxis_title="Release year",
                      yaxis_title="Predicted average rating")
    out = html.Div([html.Span(f"{p.predicted_rating:.2f} / 5",
                              style={"fontSize": 34, "fontWeight": "bold"}),
                    html.Span(f"   80% interval: {p.low_80:.2f} to {p.high_80:.2f}   "
                              f"(typical movie in dataset: {T.avg_rating.mean():.2f})")])
    return out, fig

@app.callback(Output("rc-table", "data"), Output("rc-msg", "children"), Input("rc-movies", "value"),
              Input("rc-genre", "value"), Input("rc-n", "value"))
def recommend_cb(ids, genre, n):
    if not ids:
        return [], "Select at least one movie to get recommendations."
    rec = core.recommend(ids, n, genre)
    if rec.empty:
        return [], "No recommendations found for that combination."
    return rec.to_dict("records"), f"Because you like {len(ids)} movie(s):"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8050)), debug=False)
