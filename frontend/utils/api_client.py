"""
utils/api_client.py
-------------------
Centralized API calls with caching. All pages import from here.
"""

import requests
import pandas as pd
import streamlit as st

API = "http://localhost:8000"


@st.cache_data(ttl=300)
def get(endpoint: str, params: dict = None):
    try:
        r = requests.get(f"{API}{endpoint}", params=params or {}, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error [{endpoint}]: {e}")
        return None


def post(endpoint: str, payload: dict):
    try:
        r = requests.post(f"{API}{endpoint}", json=payload, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error [{endpoint}]: {e}")
        return None


def to_df(data) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        return pd.DataFrame([data])
    return pd.DataFrame()


# ── Convenience wrappers ──────────────────────────────────────────────────────

def fetch_stats():
    return get("/districts/stats")

def fetch_states():
    return get("/districts/states") or []

def fetch_trend(state=None):
    params = {"state": state} if state else {}
    return to_df(get("/districts/trend", params))

def fetch_top(state=None, metric="person_days_lakhs", n=10):
    return to_df(get("/districts/top", {"state": state, "metric": metric, "n": n}))

def fetch_history(state, district):
    return to_df(get("/districts/history", {"state": state, "district": district}))

def fetch_predictions(state=None, district=None, year=None):
    params = {}
    if state:    params["state"]    = state
    if district: params["district"] = district
    if year:     params["year"]     = year
    return to_df(get("/predictions/", params))

def fetch_optimizer(state=None):
    params = {"state": state} if state else {}
    return to_df(get("/optimizer/results", params))
