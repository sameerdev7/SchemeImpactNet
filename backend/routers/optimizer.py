"""routers/predictions.py — Model prediction endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional
from backend.database import get_db
from backend import crud

router = APIRouter(prefix="/predictions", tags=["Predictions"])


@router.get("/")
def get_predictions(
    state: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    year: Optional[int] = Query(None),
    db: Session = Depends(get_db)
):
    return crud.get_predictions(db, state, district, year)
