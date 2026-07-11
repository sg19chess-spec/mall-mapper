from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(title="Indoor Mall Mapping — 5-Agent Geospatial Production System")
app.include_router(router)


@app.get("/")
def root():
    return {
        "service": "mall_mapper",
        "endpoints": ["/run", "/status/{job_id}", "/geojson/{floor}", "/review-queue",
                      "/audit/{feature_id}", "/rerun/{feature_id}"],
    }
