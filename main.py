from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware

from api import app as api_app          # existing FastAPI app (has /health, /predict)
from app_advanced import app as dash_app # Dash app instance

root = FastAPI(title="Spotify ML App")

# Put API under /api
root.mount("/api", api_app)

# Mount Dash at /
root.mount("/", WSGIMiddleware(dash_app.server))

app = root  # <- uvicorn will run "main:app"