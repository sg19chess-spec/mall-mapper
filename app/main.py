from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api.routes import router

app = FastAPI(title="Indoor Mall Mapping — 5-Agent Geospatial Production System")
app.include_router(router)

_STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/")
def root():
    return {
        "service": "mall_mapper",
        "ui": "/ui",
        "endpoints": ["/run", "/status/{job_id}", "/jobs/{job_id}/trail", "/geojson/{floor}",
                      "/review-queue", "/audit/{feature_id}", "/rerun/{feature_id}"],
    }


@app.get("/ui")
def ui():
    return FileResponse(_STATIC_DIR / "index.html")
