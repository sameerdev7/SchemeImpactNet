"""
utils/gemini_utils.py
---------------------
Shared Gemini setup, data context builder, and model caller.
Used by gemini_insights.py, home.py, and every page's AI summary widget.
"""

import json
import streamlit as st
import google.generativeai as genai


import re

def strip_markdown(text: str) -> str:
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)   # **bold**
    text = re.sub(r'\*(.+?)\*',     r'\1', text)   # *italic*
    text = re.sub(r'__(.+?)__',     r'\1', text)   # __bold__
    text = re.sub(r'_(.+?)_',       r'\1', text)   # _italic_
    text = re.sub(r'^\s*#{1,6}\s+', '', text, flags=re.MULTILINE)  # headings
    text = re.sub(r'^\s*[-*•]\s+',  '', text, flags=re.MULTILINE)  # bullets
    text = re.sub(r'^\s*\d+\.\s+',  '', text, flags=re.MULTILINE)  # numbered lists
    text = re.sub(r'`(.+?)`',       r'\1', text)   # inline code
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

from utils.api_client import (
    fetch_stats, fetch_predictions, fetch_optimizer_results, fetch_yearly_trend,
)

MODEL_NAME = "gemini-2.5-flash-lite"

# ── Preset questions (used by insights page) ──────────────────────────────────
PRESET_QUESTIONS = [
    {
        "label": "Which districts are predicted to see the steepest employment decline?",
        "key": "declining",
        "icon": "📉",
    },
    {
        "label": "Which districts offer the best return on additional budget investment?",
        "key": "roi",
        "icon": "💰",
    },
    {
        "label": "What does the model predict for national employment in the next cycle?",
        "key": "forecast",
        "icon": "🔭",
    },
    {
        "label": "Which states should be prioritised for budget reallocation and why?",
        "key": "realloc",
        "icon": "⚖️",
    },
    {
        "label": "What is the predicted COVID recovery trajectory across districts?",
        "key": "covid",
        "icon": "🦠",
    },
    {
        "label": "Which districts are most underfunded relative to their predicted demand?",
        "key": "underfunded",
        "icon": "🚨",
    },
    {
        "label": "What are the top 5 efficiency leaders and what can we learn from them?",
        "key": "efficiency",
        "icon": "🏆",
    },
    {
        "label": "Summarise the overall model prediction results in plain language.",
        "key": "summary",
        "icon": "📋",
    },
]

# ── Per-page summary prompts ──────────────────────────────────────────────────
PAGE_SUMMARY_PROMPTS = {
    "overview": "In 3–4 sentences, summarise the key takeaways from the national MNREGA employment trend data shown. Focus on the most important patterns, anomalies, and what they imply for policy. Use specific numbers.",
    "districts": "In 3–4 sentences, give a sharp analytical summary of this district's MNREGA performance trajectory. What is the trend, how did COVID affect it, and what does the model predict? Be specific.",
    "predictions": "In 3–4 sentences, summarise what the model predictions reveal. Comment on accuracy, any notable over/under-predictions, and what the forecasts imply for the next cycle.",
    "optimizer": "In 3–4 sentences, explain the budget optimiser results in plain language. What is the headline gain, which districts benefit most, and is the reallocation realistic to implement?",
    "insights": "In 3–4 sentences, provide a crisp executive summary of the strategic insights. What are the 2–3 most urgent actions a policymaker should take based on this data?",
    "spatial": "In 3–4 sentences, describe what the spatial distribution of predicted employment reveals. Are there regional clusters of high or low performance? What geographic patterns stand out?",
}


def get_gemini_key() -> str | None:
    """Get key from session state (set once in sidebar)."""
    return st.session_state.get("gemini_api_key", "")


def configure_gemini(api_key: str):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(MODEL_NAME)


@st.cache_data(ttl=300, show_spinner=False)
def build_context(state_param: str | None) -> dict:
    """Build a structured data context dict from live API data."""
    stats    = fetch_stats()
    pred_df  = fetch_predictions(state=state_param)
    opt_df   = fetch_optimizer_results(state=state_param)
    trend_df = fetch_yearly_trend(state=state_param)

    ctx: dict = {}

    ctx["scope"] = state_param or "All India"
    ctx["overview"] = {
        "total_districts": stats.get("total_districts"),
        "total_states": stats.get("total_states"),
        "year_range": stats.get("year_range"),
        "total_persondays_lakhs": round(stats.get("total_persondays_lakhs", 0), 1),
        "covid_spike_pct": stats.get("covid_spike_pct"),
    }

    if not trend_df.empty:
        ctx["yearly_trend"] = (
            trend_df[["financial_year","total_persondays","avg_wage"]]
            .round(2).to_dict(orient="records")
        )

    if not pred_df.empty:
        ly  = int(pred_df["financial_year"].max())
        prv = ly - 1
        lat = pred_df[pred_df["financial_year"] == ly]
        prv_df = pred_df[pred_df["financial_year"] == prv]

        ctx["model"] = {
            "algorithm": "GradientBoostingRegressor",
            "latest_predicted_year": ly,
            "walk_forward_r2": 0.91,
            "note": "2022 West Bengal anomaly excluded from CV",
        }

        if not prv_df.empty:
            mg = lat.merge(
                prv_df[["state","district","person_days_lakhs"]]
                .rename(columns={"person_days_lakhs":"prev"}),
                on=["state","district"], how="inner",
            )
            mg["chg"] = (mg["predicted_persondays"] - mg["prev"]).round(2)
            mg["chg_pct"] = (mg["chg"] / mg["prev"] * 100).round(1)

            ctx["predictions"] = {
                "n_improving": int((mg["chg"] >= 0).sum()),
                "n_declining":  int((mg["chg"] < 0).sum()),
                "top_improving": mg.nlargest(5, "chg")[
                    ["state","district","prev","predicted_persondays","chg","chg_pct"]
                ].to_dict(orient="records"),
                "top_declining": mg.nsmallest(5, "chg")[
                    ["state","district","prev","predicted_persondays","chg","chg_pct"]
                ].to_dict(orient="records"),
                "national_predicted_total": round(float(lat["predicted_persondays"].sum()), 1),
                "national_actual_prev": round(float(prv_df["person_days_lakhs"].sum()), 1),
            }

    if not opt_df.empty and "persondays_gain" in opt_df.columns:
        sq   = float(opt_df["sq_persondays"].sum())
        gain = float(opt_df["persondays_gain"].sum())
        ctx["optimizer"] = {
            "total_budget_lakhs": round(float(opt_df.get("budget_allocated_lakhs", opt_df["sq_persondays"]).sum()), 0),
            "status_quo_persondays": round(sq, 1),
            "gain_lakhs": round(gain, 2),
            "gain_pct": round(gain / sq * 100, 2) if sq else 0,
            "top_gain": opt_df.nlargest(5, "persondays_gain")[
                ["state","district","persondays_gain","persondays_per_lakh","budget_change_pct"]
            ].round(3).to_dict(orient="records"),
            "top_cut": opt_df.nsmallest(5, "persondays_gain")[
                ["state","district","persondays_gain","persondays_per_lakh","budget_change_pct"]
            ].round(3).to_dict(orient="records"),
            "by_state": (
                opt_df.groupby("state")["persondays_gain"]
                .sum().nlargest(8).round(2).to_dict()
            ),
            "underfunded": opt_df[
                opt_df["budget_allocated_lakhs"] < opt_df["budget_allocated_lakhs"].quantile(0.33)
            ].nlargest(5, "persondays_per_lakh")[
                ["state","district","persondays_per_lakh","budget_allocated_lakhs"]
            ].round(3).to_dict(orient="records") if "budget_allocated_lakhs" in opt_df.columns else [],
        }

    return ctx


def call_gemini(api_key: str, prompt: str, temperature: float = 0.35) -> str:
    """Call Gemini and return the text response."""
    try:
        m = configure_gemini(api_key)
        resp = m.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=1024,
            ),
        )
        return strip_markdown(resp.text)
    except Exception as e:
        return f"⚠️ Gemini error: {e}"


def base_prompt(ctx: dict) -> str:
    return f"""You are a senior policy analyst specialising in India's MNREGA rural employment scheme.
Scope: {ctx.get('scope', 'All India')}

Live data from SchemeImpactNet (GradientBoostingRegressor, walk-forward CV R²≈0.91):
{json.dumps(ctx, indent=2)}

Rules:
- Person-days in lakhs (1 lakh = 100,000). Budget in ₹ lakhs.
- 2020: COVID surge (reverse migration drove demand spike).
- 2022: West Bengal data anomaly (-93% to -98%) — not a real decline.
- The LP optimizer reallocates budget across districts at zero additional cost.
- Base every claim on the numbers above. Name specific districts and states.
- Be direct, analytical, and avoid generic statements.

"""


def preset_prompt(ctx: dict, question_key: str) -> str:
    base = base_prompt(ctx)
    prompts = {
        "declining": base + "Which districts are predicted to see the steepest employment decline? Name the top 5, give exact predicted change figures, identify any state-level patterns, and suggest specific interventions. (~300 words)",
        "roi": base + "Which districts offer the best return on additional budget investment based on efficiency (persondays_per_lakh) scores? Name top districts, explain why their efficiency is high, and estimate the employment gain from a 10% budget increase. (~300 words)",
        "forecast": base + "What does the model predict for national employment in the next cycle? Compare predicted vs previous actual totals, identify which states drive the change, and assess confidence given model performance. (~300 words)",
        "realloc": base + "Which states should be prioritised for budget reallocation and why? Use the optimizer state-level data, name the top 3 states for increase and top 3 for reduction, with the employment gain rationale. (~300 words)",
        "covid": base + "What is the predicted COVID recovery trajectory? Has employment normalised post-2020 surge, or are certain districts still at elevated levels? What does this imply for future demand planning? (~300 words)",
        "underfunded": base + "Which districts are most underfunded relative to their predicted demand and efficiency scores? Name specific districts, show the gap between their efficiency and their budget allocation, and recommend reallocation amounts. (~300 words)",
        "efficiency": base + "Who are the top 5 efficiency leaders (highest persondays_per_lakh)? What structural factors likely explain their high efficiency? What can other districts learn and replicate? (~300 words)",
        "summary": base + "Summarise the overall model prediction results in plain language for a non-technical policymaker. Cover: what the model predicts nationally, which regions face challenges, and the 3 most important numbers to know. (~300 words)",
    }
    return prompts.get(question_key, base + "Provide a strategic analysis of the MNREGA data.")


def page_summary_prompt(ctx: dict, page_key: str, extra_context: str = "") -> str:
    base = base_prompt(ctx)
    instruction = PAGE_SUMMARY_PROMPTS.get(page_key, "Summarise the key insights from this page's data in 3–4 sentences.")
    extra = f"\nAdditional page context:\n{extra_context}\n" if extra_context else ""
    return base + extra + "\n" + instruction + "\nRespond in 3–4 sentences only. Be precise and use specific numbers."
