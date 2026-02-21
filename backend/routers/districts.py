"""routers/districts.py — District data endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional, List
from backend.database import get_db
from backend import crud

router = APIRouter(prefix="/districts", tags=["Districts"])


@router.get("/states")
def list_states(db: Session = Depends(get_db)):
    return crud.get_states(db)


@router.get("/list")
def list_districts(state: str = Query(...), db: Session = Depends(get_db)):
    return crud.get_districts(db, state)


@router.get("/history")
def district_history(
    state: str = Query(...),
    district: str = Query(...),
    db: Session = Depends(get_db)
):
    return crud.get_district_history(db, state, district)


@router.get("/top")
def top_districts(
    state: Optional[str] = Query(None),
    metric: str = Query("person_days_lakhs"),
    n: int = Query(10),
    db: Session = Depends(get_db)
):
    return crud.get_top_districts(db, state, metric, n)


@router.get("/trend")
def yearly_trend(
    state: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    return crud.get_yearly_trend(db, state)


@router.get("/stats")
def stats(db: Session = Depends(get_db)):
    return crud.get_stats(db)
