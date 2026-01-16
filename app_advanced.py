import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from src.utils import load_joblib

DATA_PATH = "data/processed/spotify_processed.csv"
MODEL_DIR = "reports/models"

# Load data to build dropdown options + defaults
df = pd.read_csv(DATA_PATH)
X_full = df.drop(columns=["is_popular"])

# Load pipelines (each pipeline contains its own preprocessing)
MODELS = {
    "lasso": load_joblib("reports/models/lasso.joblib"),
    "cart_lasso": load_joblib("reports/models/cart_lasso.joblib"),
    "rf_lasso": load_joblib("reports/models/rf_lasso.joblib"),
    "xgb_lasso": load_joblib("reports/models/xgb_lasso.joblib"),
    "cart_pca": load_joblib("reports/models/cart_pca.joblib"),
    "rf_pca": load_joblib("reports/models/rf_pca.joblib"),
    "xgb_pca": load_joblib("reports/models/xgb_pca.joblib"),
}

model_options = [
    {"label": "LASSO", "value": "lasso"},
    {"label": "CART + LASSO", "value": "cart_lasso"},
    {"label": "Random Forest + LASSO", "value": "rf_lasso"},
    {"label": "XGBoost + LASSO", "value": "xgb_lasso"},
    {"label": "CART + PCA", "value": "cart_pca"},
    {"label": "Random Forest + PCA", "value": "rf_pca"},
    {"label": "XGBoost + PCA", "value": "xgb_pca"},
]

# Spotify colors
SPOTIFY_GREEN = "#1DB954"
SPOTIFY_BLACK = "#191414"
SPOTIFY_GRAY = "#282828"

external_stylesheets = [
    {"href": "https://cdnjs.cloudflare.com/ajax/libs/semantic-ui/2.4.1/semantic.min.css", "rel": "stylesheet"}
]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Build genre options from data (no mapping table needed)
genre_col = "track_genre"
genre_options = []
if genre_col in X_full.columns:
    genres = sorted(X_full[genre_col].dropna().astype(str).unique().tolist())
    genre_options = [{"label": g, "value": g} for g in genres]

def slider(id_, min_, max_, step_, value_, marks=None):
    return dcc.Slider(
        id=id_, min=min_, max=max_, step=step_, value=value_,
        marks=marks, tooltip={"always_visible": True}
    )

app.layout = html.Div(
    style={"backgroundColor": SPOTIFY_BLACK, "color": "white", "padding": "30px"},
    children=[
        html.H1("🎵 Song Popularity Prediction Dashboard 🎵",
                style={"color": SPOTIFY_GREEN, "textAlign": "center", "marginBottom": "30px"}),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "30px", "marginBottom": "30px"},
            children=[
                html.Div(style={"padding": "20px", "backgroundColor": SPOTIFY_GRAY, "borderRadius": "10px"},
                         children=[
                             html.Label("Track Genre", style={"color": SPOTIFY_GREEN}),
                             dcc.Dropdown(
                                 id="track_genre",
                                 options=genre_options,
                                 value=genre_options[0]["value"] if genre_options else None,
                                 searchable=True,
                                 style={"backgroundColor": "#C0C0C0", "color": "black", "marginBottom": "20px"},
                             ),

                             html.Label("Explicit", style={"color": SPOTIFY_GREEN}),
                             dcc.Dropdown(
                                 id="explicit",
                                 options=[{"label": "No", "value": 0}, {"label": "Yes", "value": 1}],
                                 value=0,
                                 style={"backgroundColor": "#C0C0C0", "color": "black", "marginBottom": "20px"},
                             ),

                             html.Label("Time Signature", style={"color": SPOTIFY_GREEN}),
                             dcc.Dropdown(
                                 id="time_signature",
                                 options=[{"label": str(i), "value": i} for i in range(3, 8)],
                                 value=4,
                                 style={"backgroundColor": "#C0C0C0", "color": "black", "marginBottom": "20px"},
                             ),

                             html.Label("Key", style={"color": SPOTIFY_GREEN}),
                             dcc.Dropdown(
                                 id="key",
                                 options=[{"label": str(i), "value": i} for i in range(0, 12)],
                                 value=1,
                                 style={"backgroundColor": "#C0C0C0", "color": "black", "marginBottom": "20px"},
                             ),

                             html.Label("Mode", style={"color": SPOTIFY_GREEN}),
                             dcc.Dropdown(
                                 id="mode",
                                 options=[{"label": "Major", "value": 1}, {"label": "Minor", "value": 0}],
                                 value=1,
                                 style={"backgroundColor": "#C0C0C0", "color": "black", "marginBottom": "20px"},
                             ),
                         ]),

                html.Div(style={"padding": "20px", "backgroundColor": SPOTIFY_GRAY, "borderRadius": "10px"},
                         children=[
                             html.Label("Duration (mm:ss)", style={"color": SPOTIFY_GREEN}),
                             html.Div(style={"display": "flex", "gap": "12px", "marginBottom": "20px"}, children=[
                                 dcc.Input(id="duration_min", type="number", value=3, min=0, max=20, step=1,
                                           style={"width": "80px", "backgroundColor": SPOTIFY_GRAY, "color": "white"}),
                                 dcc.Input(id="duration_sec", type="number", value=30, min=0, max=59, step=1,
                                           style={"width": "80px", "backgroundColor": SPOTIFY_GRAY, "color": "white"}),
                             ]),

                             html.Label("Danceability", style={"color": SPOTIFY_GREEN}),
                             slider("danceability", 0, 1, 0.01, 0.65, marks={i/10: str(i/10) for i in range(11)}),
                             html.Br(),

                             html.Label("Energy", style={"color": SPOTIFY_GREEN}),
                             slider("energy", 0, 1, 0.01, 0.55, marks={i/10: str(i/10) for i in range(11)}),
                             html.Br(),

                             html.Label("Loudness (dB)", style={"color": SPOTIFY_GREEN}),
                             slider("loudness", -60, 5, 0.1, -6.5, marks={-60+i*6.5: str(round(-60+i*6.5,1)) for i in range(11)}),
                         ]),
            ],
        ),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "30px", "marginBottom": "20px"},
            children=[
                html.Div(style={"padding": "20px", "backgroundColor": SPOTIFY_GRAY, "borderRadius": "10px"},
                         children=[
                             html.Label("Speechiness", style={"color": SPOTIFY_GREEN}),
                             slider("speechiness", 0, 1, 0.01, 0.15, marks={i/10: str(i/10) for i in range(11)}),
                             html.Br(),

                             html.Label("Acousticness", style={"color": SPOTIFY_GREEN}),
                             slider("acousticness", 0, 1, 0.01, 0.20, marks={i/10: str(i/10) for i in range(11)}),
                             html.Br(),

                             html.Label("Instrumentalness", style={"color": SPOTIFY_GREEN}),
                             slider("instrumentalness", 0, 1, 0.01, 0.00, marks={i/10: str(i/10) for i in range(11)}),
                         ]),

                html.Div(style={"padding": "20px", "backgroundColor": SPOTIFY_GRAY, "borderRadius": "10px"},
                         children=[
                             html.Label("Valence", style={"color": SPOTIFY_GREEN}),
                             slider("valence", 0, 1, 0.01, 0.60, marks={i/10: str(i/10) for i in range(11)}),
                             html.Br(),

                             html.Label("Tempo (BPM)", style={"color": SPOTIFY_GREEN}),
                             slider("tempo", 30, 250, 1, 120, marks={30+i*22: str(30+i*22) for i in range(11)}),
                             html.Br(),

                             html.Label("Liveness", style={"color": SPOTIFY_GREEN}),
                             slider("liveness", 0, 1, 0.01, 0.20, marks={i/10: str(i/10) for i in range(11)}),
                         ]),
            ],
        ),

        html.Hr(style={"borderColor": SPOTIFY_GREEN}),

        html.Label("Select Model", style={"color": SPOTIFY_GREEN}),
        dcc.Dropdown(
            options=model_options, id="model_choice", value="lasso",
            style={"backgroundColor": "#C0C0C0", "color": "black", "marginBottom": "20px"}
        ),

        html.Button(
            "Predict Popularity",
            id="predict_button",
            n_clicks=0,
            style={
                "backgroundColor": SPOTIFY_GREEN,
                "color": "white",
                "fontWeight": "bold",
                "padding": "15px 30px",
                "borderRadius": "30px",
                "border": "none",
                "cursor": "pointer",
                "display": "block",
                "margin": "0 auto",
            },
        ),

        html.H2(id="prediction_output", style={"marginTop": "30px", "textAlign": "center", "color": SPOTIFY_GREEN}),
    ]
)

@app.callback(
    Output("prediction_output", "children"),
    Input("predict_button", "n_clicks"),
    State("explicit", "value"),
    State("track_genre", "value"),
    State("time_signature", "value"),
    State("key", "value"),
    State("mode", "value"),
    State("duration_min", "value"),
    State("duration_sec", "value"),
    State("danceability", "value"),
    State("energy", "value"),
    State("loudness", "value"),
    State("speechiness", "value"),
    State("acousticness", "value"),
    State("instrumentalness", "value"),
    State("valence", "value"),
    State("tempo", "value"),
    State("liveness", "value"),
    State("model_choice", "value"),
)
def update_prediction(n_clicks, explicit, track_genre, time_signature, key, mode,
                      duration_min, duration_sec, danceability, energy, loudness, speechiness,
                      acousticness, instrumentalness, valence, tempo, liveness, model_choice):
    if n_clicks == 0:
        return ""

    duration_ms = (duration_min * 60 + duration_sec) * 1000

    # Start from medians/modes so missing columns won't break the pipeline
    row = {}
    for c in X_full.columns:
        if pd.api.types.is_numeric_dtype(X_full[c]):
            row[c] = float(X_full[c].median())
        else:
            row[c] = str(X_full[c].mode(dropna=True).iloc[0]) if not X_full[c].mode(dropna=True).empty else ""

    # Override with UI inputs (only if the columns exist)
    overrides = {
        "explicit": explicit,
        "track_genre": track_genre,
        "time_signature": time_signature,
        "key": key,
        "mode": mode,
        "duration_ms": duration_ms,
        "danceability": danceability,
        "energy": energy,
        "loudness": loudness,
        "speechiness": speechiness,
        "acousticness": acousticness,
        "instrumentalness": instrumentalness,
        "valence": valence,
        "tempo": tempo,
        "liveness": liveness,
    }
    for k, v in overrides.items():
        if k in X_full.columns:
            row[k] = v

    X_one = pd.DataFrame([row], columns=X_full.columns)

    model = MODELS.get(model_choice)
    if model is None:
        return "Invalid model choice."

    score = float(model.predict_proba(X_one)[:, 1][0])
    return f"Predicted Popularity Probability: {score:.4f}"

if __name__ == "__main__":
    app.run(debug=True)