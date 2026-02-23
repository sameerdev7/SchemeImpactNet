"""
utils/geo_loader.py
-------------------
Loads India district GeoJSON and joins with backend data.

GeoJSON source (free, no auth):
  https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson

The GeoJSON has properties: {ST_NM (state), DISTRICT (district)}
We normalize both to lowercase + strip for fuzzy joining.

Usage:
    from frontend.utils.geo_loader import load_geojson, join_geodata
    geojson = load_geojson()
    joined  = join_geodata(geojson, df)
"""

import json
import re
import requests
import streamlit as st
import pandas as pd

GEOJSON_URL = (
    "https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson"
)

# Fallback: local path if downloaded manually
LOCAL_PATH = "data/india_district.geojson"


def _normalize(s: str) -> str:
    """Lowercase, strip punctuation, collapse spaces."""
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9 ]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


@st.cache_data(ttl=3600)
def load_geojson() -> dict:
    """Download or load GeoJSON. Returns raw dict."""
    import os
    if os.path.exists(LOCAL_PATH):
        with open(LOCAL_PATH) as f:
            return json.load(f)
    try:
        r = requests.get(GEOJSON_URL, timeout=15)
        r.raise_for_status()
        data = r.json()
        # Cache locally
        os.makedirs("data", exist_ok=True)
        with open(LOCAL_PATH, "w") as f:
            json.dump(data, f)
        return data
    except Exception as e:
        st.warning(f"Could not load GeoJSON: {e}. Map will be unavailable.")
        return None


def build_lookup(geojson: dict) -> dict:
    """
    Build a dict: normalized_district_name → feature index
    for fast joining.
    """
    lookup = {}
    for i, feat in enumerate(geojson.get("features", [])):
        props = feat.get("properties", {})
        district_raw = props.get("DISTRICT", props.get("district", ""))
        key = _normalize(district_raw)
        lookup[key] = i
    return lookup


def join_geodata(geojson: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'geo_match' column to df indicating whether the district
    was matched in GeoJSON. Returns df with geo-compatible district key.
    
    Plotly choropleth needs a 'feature_key' that matches GeoJSON feature id.
    We set featureidkey='properties.DISTRICT' and match on district name.
    """
    if geojson is None:
        df["geo_match"] = False
        return df

    geo_districts = set()
    for feat in geojson.get("features", []):
        props = feat.get("properties", {})
        name = props.get("DISTRICT", props.get("district", ""))
        geo_districts.add(_normalize(name))

    df = df.copy()
    df["district_norm"] = df["district"].apply(_normalize)
    df["geo_match"] = df["district_norm"].isin(geo_districts)

    matched = df["geo_match"].sum()
    total   = len(df)
    st.caption(f"📍 GeoJSON match: {matched}/{total} districts ({matched/total*100:.0f}%)")

    return df


def get_state_fallback_geojson() -> dict:
    """
    State-level GeoJSON fallback if district matching is too low.
    Uses a reliable public source.
    """
    STATE_URL = (
        "https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson"
    )
    try:
        r = requests.get(STATE_URL, timeout=10)
        return r.json()
    except Exception:
        return None
