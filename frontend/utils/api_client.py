"""
utils/api_client.py
--------------------
Centralized, cached API wrappers matching the exact backend endpoints
defined in backend/routers/districts.py, predictions.py, optimizer.py
and backend/schemas.py.
"""

import requests
import pandas as pd
import streamlit as st

API = "http://localhost:8000"
TIMEOUT = 8


@st.cache_data(ttl=300)
def _get(endpoint: str, params: dict | None = None):
    """Raw cached GET — returns JSON or None on any error."""
    try:
        r = requests.get(f"{API}{endpoint}", params=params or {}, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return None
    except requests.exceptions.Timeout:
        return None
    except Exception:
        return None


def _df(data) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        return pd.DataFrame([data])
    return pd.DataFrame()


# ── Health ─────────────────────────────────────────────────────────────────────
def is_online() -> bool:
    try:
        requests.get(f"{API}/health", timeout=3)
        return True
    except Exception:
        return False


# ── /districts/* ───────────────────────────────────────────────────────────────
def fetch_stats() -> dict:
    """GET /districts/stats → StatsOut dict"""
    return _get("/districts/stats") or {}


def fetch_states() -> list[str]:
    """GET /districts/states → list[str]"""
    return _get("/districts/states") or []


def fetch_districts(state: str) -> list[str]:
    """GET /districts/list?state=... → list[str]"""
    return _get("/districts/list", {"state": state}) or []


def fetch_district_history(state: str, district: str) -> pd.DataFrame:
    """GET /districts/history?state=...&district=... → DistrictSummary list"""
    return _df(_get("/districts/history", {"state": state, "district": district}))


def fetch_top_districts(
    state: str | None = None,
    metric: str = "person_days_lakhs",
    n: int = 12,
) -> pd.DataFrame:
    """GET /districts/top → top N districts by metric"""
    params = {"metric": metric, "n": n}
    if state:
        params["state"] = state
    return _df(_get("/districts/top", params))


def fetch_yearly_trend(state: str | None = None) -> pd.DataFrame:
    """GET /districts/trend → yearly aggregates"""
    params = {"state": state} if state else {}
    return _df(_get("/districts/trend", params))


# ── /predictions/* ─────────────────────────────────────────────────────────────
def fetch_predictions(
    state: str | None = None,
    district: str | None = None,
    year: int | None = None,
) -> pd.DataFrame:
    """GET /predictions/ → PredictionOut list"""
    params = {}
    if state:    params["state"]    = state
    if district: params["district"] = district
    if year:     params["year"]     = year
    return _df(_get("/predictions/", params))


# ── /optimizer/* ───────────────────────────────────────────────────────────────
def fetch_optimizer_results(state: str | None = None) -> pd.DataFrame:
    """GET /optimizer/results → OptimizerOut list (pre-computed allocation)"""
    params = {"state": state} if state else {}
    return _df(_get("/optimizer/results", params))


def run_optimizer_live(
    state: str | None = None,
    budget_scale: float = 1.0,
    min_fraction: float = 0.40,
    max_fraction: float = 2.50,
) -> dict | None:
    """POST /optimizer/run → OptimizerResponse (live LP run)"""
    payload = {
        "state":        state,
        "budget_scale": budget_scale,
        "min_fraction": min_fraction,
        "max_fraction": max_fraction,
    }
    try:
        r = requests.post(f"{API}/optimizer/run", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach API — start: `uvicorn backend.main:app --port 8000`")
        return None
    except Exception as e:
        st.error(f"Optimizer error: {e}")
        return None
