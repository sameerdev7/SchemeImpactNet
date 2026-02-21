"""
crud.py
-------
Database query functions. All queries return plain dicts/lists
so FastAPI routers stay thin.
"""

import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Optional, List


# ── Districts ─────────────────────────────────────────────────────────────────

def get_states(db: Session) -> List[str]:
    rows = db.execute(text("SELECT DISTINCT state FROM district_data ORDER BY state")).fetchall()
    return [r[0] for r in rows]


def get_districts(db: Session, state: str) -> List[str]:
    rows = db.execute(
        text("SELECT DISTINCT district FROM district_data WHERE state=:s ORDER BY district"),
        {"s": state}
    ).fetchall()
    return [r[0] for r in rows]


def get_district_history(db: Session, state: str, district: str) -> List[dict]:
    rows = db.execute(text("""
        SELECT state, district, financial_year, person_days_lakhs,
               expenditure_lakhs, avg_wage_rate,
               expenditure_per_personday, demand_fulfillment_rate
        FROM district_data
        WHERE state=:s AND district=:d
        ORDER BY financial_year
    """), {"s": state, "d": district}).fetchall()
    return [dict(r._mapping) for r in rows]


def get_top_districts(db: Session, state: Optional[str], metric: str, n: int) -> List[dict]:
    valid = {"person_days_lakhs", "expenditure_lakhs", "expenditure_per_personday"}
    if metric not in valid:
        metric = "person_days_lakhs"
    where = "WHERE state=:s" if state else ""
    params = {"s": state} if state else {}
    rows = db.execute(text(f"""
        SELECT state, district,
               AVG(person_days_lakhs) as avg_persondays,
               AVG(expenditure_lakhs) as avg_expenditure,
               AVG(expenditure_per_personday) as avg_efficiency
        FROM district_data
        {where}
        GROUP BY state, district
        ORDER BY AVG({metric}) DESC
        LIMIT :n
    """), {**params, "n": n}).fetchall()
    return [dict(r._mapping) for r in rows]


def get_yearly_trend(db: Session, state: Optional[str]) -> List[dict]:
    where = "WHERE state=:s" if state else ""
    params = {"s": state} if state else {}
    rows = db.execute(text(f"""
        SELECT financial_year,
               SUM(person_days_lakhs)  as total_persondays,
               SUM(expenditure_lakhs)  as total_expenditure,
               AVG(avg_wage_rate)      as avg_wage
        FROM district_data
        {where}
        GROUP BY financial_year
        ORDER BY financial_year
    """), params).fetchall()
    return [dict(r._mapping) for r in rows]


def get_stats(db: Session) -> dict:
    r = db.execute(text("""
        SELECT
            COUNT(DISTINCT district)          as total_districts,
            COUNT(DISTINCT state)             as total_states,
            MIN(financial_year)||' – '||MAX(financial_year) as year_range,
            SUM(person_days_lakhs)            as total_persondays_lakhs,
            SUM(expenditure_lakhs)            as total_expenditure_lakhs
        FROM district_data
    """)).fetchone()
    base = dict(r._mapping)

    # COVID spike
    pre  = db.execute(text("SELECT AVG(person_days_lakhs) FROM district_data WHERE financial_year=2019")).scalar()
    post = db.execute(text("SELECT AVG(person_days_lakhs) FROM district_data WHERE financial_year=2020")).scalar()
    base["covid_spike_pct"] = round((post - pre) / pre * 100, 2) if pre else 0.0
    return base


# ── Predictions ───────────────────────────────────────────────────────────────

def get_predictions(
    db: Session,
    state: Optional[str],
    district: Optional[str],
    year: Optional[int]
) -> List[dict]:
    clauses, params = [], {}
    if state:
        clauses.append("state=:state"); params["state"] = state
    if district:
        clauses.append("district=:district"); params["district"] = district
    if year:
        clauses.append("financial_year=:year"); params["year"] = year
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = db.execute(text(f"""
        SELECT state, district, financial_year,
               person_days_lakhs, predicted_persondays, prediction_error
        FROM predictions {where}
        ORDER BY state, district, financial_year
    """), params).fetchall()
    return [dict(r._mapping) for r in rows]


# ── Optimizer ─────────────────────────────────────────────────────────────────

def get_optimizer_results(db: Session, state: Optional[str]) -> List[dict]:
    where = "WHERE state=:s" if state else ""
    params = {"s": state} if state else {}
    rows = db.execute(text(f"""
        SELECT * FROM optimizer {where}
        ORDER BY persondays_gain DESC
    """), params).fetchall()
    return [dict(r._mapping) for r in rows]
