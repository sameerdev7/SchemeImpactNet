"""
schemas.py
----------
Pydantic schemas for API request/response validation.

V3 update: expenditure_lakhs, expenditure_per_personday, demand_fulfillment_rate
removed from DistrictSummary — synthetic columns dropped in leak-free pipeline.
"""

from pydantic import BaseModel
from typing import Optional, List


class DistrictSummary(BaseModel):
    state: str
    district: str
    financial_year: int
    person_days_lakhs: float
    avg_wage_rate: float

    class Config:
        from_attributes = True


class PredictionOut(BaseModel):
    state: str
    district: str
    financial_year: int
    person_days_lakhs: float
    predicted_persondays: float
    prediction_error: float

    class Config:
        from_attributes = True


class OptimizerOut(BaseModel):
    state: str
    district: str
    budget_allocated_lakhs: float
    optimized_budget: float
    budget_change: float
    budget_change_pct: float
    sq_persondays: float
    opt_persondays: float
    persondays_gain: float
    persondays_gain_pct: float
    persondays_per_lakh: float

    class Config:
        from_attributes = True


class OptimizerRequest(BaseModel):
    state: Optional[str] = None
    budget_scale: float = 1.0
    min_fraction: float = 0.40
    max_fraction: float = 2.50


class OptimizerResponse(BaseModel):
    scope: str
    total_budget_lakhs: float
    sq_persondays_total: float
    opt_persondays_total: float
    gain_lakhs: float
    gain_pct: float
    districts: List[OptimizerOut]


class StatsOut(BaseModel):
    total_districts: int
    total_states: int
    year_range: str
    total_persondays_lakhs: float
    total_expenditure_lakhs: float   # kept for API compat, always 0.0 in V3
    covid_spike_pct: float