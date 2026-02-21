"""routers/optimizer.py — Budget optimizer endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional
from backend.database import get_db
from backend import crud
from backend.schemas import OptimizerRequest, OptimizerResponse

router = APIRouter(prefix="/optimizer", tags=["Optimizer"])


@router.get("/results")
def get_optimizer_results(
    state: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    return crud.get_optimizer_results(db, state)


@router.post("/run", response_model=OptimizerResponse)
def run_optimizer_live(req: OptimizerRequest, db: Session = Depends(get_db)):
    """
    Run LP optimizer live with custom parameters.
    Reads predictions from DB, runs scipy LP, returns results.
    """
    import numpy as np
    from scipy.optimize import linprog
    from sqlalchemy import text

    # Get latest year predictions + budget
    state_clause = "AND p.state=:s" if req.state else ""
    params = {"s": req.state} if req.state else {}

    rows = db.execute(text(f"""
        SELECT p.state, p.district,
               p.predicted_persondays,
               o.budget_allocated_lakhs,
               o.persondays_per_lakh
        FROM predictions p
        JOIN optimizer o ON p.district = o.district AND p.state = o.state
        WHERE p.financial_year = (SELECT MAX(financial_year) FROM predictions)
        {state_clause}
    """), params).fetchall()

    if not rows:
        return OptimizerResponse(
            scope=req.state or "All-India",
            total_budget_lakhs=0, sq_persondays_total=0,
            opt_persondays_total=0, gain_lakhs=0, gain_pct=0, districts=[]
        )

    import pandas as pd
    df = pd.DataFrame([dict(r._mapping) for r in rows]).dropna()

    budgets    = df["budget_allocated_lakhs"].values * req.budget_scale
    efficiency = df["persondays_per_lakh"].values
    total_bud  = budgets.sum()

    lb = budgets * req.min_fraction
    ub = budgets * req.max_fraction

    res = linprog(-efficiency, A_ub=[np.ones(len(df))],
                  b_ub=[total_bud], bounds=list(zip(lb, ub)), method="highs")

    opt_budgets   = res.x if res.success else budgets
    sq_total      = float((efficiency * budgets).sum())
    opt_total     = float((efficiency * opt_budgets).sum())

    districts_out = []
    for i, row in df.iterrows():
        orig = budgets[df.index.get_loc(i)]
        opt  = opt_budgets[df.index.get_loc(i)]
        sq_pd  = float(efficiency[df.index.get_loc(i)] * orig)
        opt_pd = float(efficiency[df.index.get_loc(i)] * opt)
        districts_out.append({
            "state": row["state"],
            "district": row["district"],
            "budget_allocated_lakhs": round(orig, 2),
            "optimized_budget": round(opt, 2),
            "budget_change": round(opt - orig, 2),
            "budget_change_pct": round((opt - orig) / orig * 100, 2),
            "sq_persondays": round(sq_pd, 3),
            "opt_persondays": round(opt_pd, 3),
            "persondays_gain": round(opt_pd - sq_pd, 3),
            "persondays_gain_pct": round((opt_pd - sq_pd) / sq_pd * 100, 2) if sq_pd else 0,
            "persondays_per_lakh": round(float(efficiency[df.index.get_loc(i)]), 4),
        })

    gain = opt_total - sq_total
    return OptimizerResponse(
        scope=req.state or "All-India",
        total_budget_lakhs=round(total_bud, 2),
        sq_persondays_total=round(sq_total, 2),
        opt_persondays_total=round(opt_total, 2),
        gain_lakhs=round(gain, 2),
        gain_pct=round(gain / sq_total * 100, 2) if sq_total else 0,
        districts=districts_out
    )
