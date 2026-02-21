"""
backend/main.py
---------------
FastAPI application entry point.

Run with:
    uvicorn backend.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.database import seed_database
from backend.routers.districts import router as districts_router
from backend.routers.predictions import router as predictions_router
from backend.routers.optimizer import router as optimizer_router

app = FastAPI(
    title="SchemeImpactNet API",
    description="MNREGA district-level forecasting and budget optimization",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    seed_database()

app.include_router(districts_router)
app.include_router(predictions_router)
app.include_router(optimizer_router)

@app.get("/")
def root():
    return {"project": "SchemeImpactNet", "version": "1.0.0", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok"}
