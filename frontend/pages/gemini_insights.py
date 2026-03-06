# pages/gemini_insights.py — Gemini AI Policy Insights

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st

from theme import inject_theme, page_header, section_label
from utils.api_client import fetch_states
from utils.gemini_utils import build_context, call_gemini

inject_theme()
page_header(
    "◈ Gemini 2.5 Flash Lite",
    "AI Policy Insights",
    "Model-grounded analysis — all insights derived from GBR predictions and LP optimizer results",
)

# ── Read key from sidebar session state (set in app.py) ───────────────────────
api_key = st.session_state.get("gemini_api_key", "")

if not api_key:
    st.markdown("""
<div style="background:#FFF7ED; border:1px solid #FED7AA; border-left:3px solid #FB923C;
            border-radius:8px; padding:1.2rem 1.4rem; margin:1rem 0;">
  <p style="font-family:'DM Mono',monospace; font-size:0.6rem; letter-spacing:2px;
            text-transform:uppercase; color:#FB923C; margin:0 0 6px 0;">Setup Required</p>
  <p style="font-family:'Source Serif 4',serif; font-size:0.9rem; color:#431407;
            margin:0; line-height:1.7;">
    Enter your Gemini API key in the sidebar to unlock AI insights.
    Free tier available at
    <a href="https://aistudio.google.com/apikey" target="_blank"
       style="color:#FB923C;">aistudio.google.com/apikey</a>
  </p>
</div>
""", unsafe_allow_html=True)
    st.stop()

# ── Scope selector ────────────────────────────────────────────────────────────
states = fetch_states()
if not states:
    st.error("API offline — run `uvicorn backend.main:app --port 8000`")
    st.stop()

cs, _ = st.columns([1, 2])
with cs:
    scope = st.selectbox("State Scope", ["All India"] + states)
state_param = None if scope == "All India" else scope

st.markdown("---")
section_label("Ask a Question")

st.markdown("""
<p style="font-family:'Source Serif 4',serif; font-size:0.88rem; color:#78716C;
          margin:0 0 1rem 0; line-height:1.6;">
  Ask anything about the MNREGA data. All answers are grounded exclusively in the
  GBR model predictions and LP optimizer output — no external knowledge used.
</p>
""", unsafe_allow_html=True)

# ── Free-text question input ──────────────────────────────────────────────────
question = st.text_area(
    "Your question",
    placeholder="e.g. Which districts are predicted to see the steepest employment decline?\n"
                "e.g. What does the LP optimizer recommend for budget reallocation in Rajasthan?\n"
                "e.g. Summarise the overall model predictions in plain language.",
    height=100,
    key="user_question",
    label_visibility="collapsed",
)

ask_col, clear_col, _ = st.columns([1, 1, 4])
with ask_col:
    ask_clicked = st.button("Ask Gemini", key="ask_btn", use_container_width=True)
with clear_col:
    if st.button("Clear", key="clear_btn", use_container_width=True):
        if "last_answer" in st.session_state:
            del st.session_state["last_answer"]
        if "last_question" in st.session_state:
            del st.session_state["last_question"]
        st.rerun()

# ── Generate answer ───────────────────────────────────────────────────────────
if ask_clicked and question.strip():
    cache_key = f"qa_{hash(question.strip())}_{state_param}"
    if cache_key not in st.session_state:
        with st.spinner("Analysing with Gemini 2.5 Flash Lite..."):
            ctx = build_context(state_param)

            # Build a grounded prompt from the user's free-text question
            prompt = f"""You are a policy analyst assistant for India's MNREGA rural employment scheme.
You have access to outputs from a Gradient Boosting Regressor (GBR, R²≈0.91) and a 
SciPy LP budget optimizer. Answer ONLY based on the data context provided below.
Do not use any external knowledge or make up figures not present in the data.
Be concise, specific, and cite actual numbers from the data.

Data context:
{ctx}

Question ({scope}):
{question.strip()}

Answer in plain prose, 3-6 sentences. No markdown formatting."""

            answer = call_gemini(api_key, prompt)
            st.session_state[cache_key] = answer
            st.session_state["last_answer"]   = answer
            st.session_state["last_question"] = question.strip()
    else:
        st.session_state["last_answer"]   = st.session_state[cache_key]
        st.session_state["last_question"] = question.strip()

elif ask_clicked and not question.strip():
    st.warning("Please enter a question first.")

# ── Display answer ────────────────────────────────────────────────────────────
if st.session_state.get("last_answer"):
    st.markdown("---")
    section_label(f"Answer · {scope}")

    st.markdown(f"""
<div style="background:#F5F5F4; border-radius:6px; padding:0.7rem 1rem; margin-bottom:1rem;">
  <p style="font-family:'DM Mono',monospace; font-size:0.68rem; color:#57534E; margin:0;">
    ❓ {st.session_state['last_question']}
  </p>
</div>
<div style="background:#FFFFFF; border:1px solid #E7E5E4; border-left:3px solid #FB923C;
            border-radius:8px; padding:1.5rem 1.75rem; margin-bottom:1rem;">
  <p style="font-family:'DM Mono',monospace; font-size:0.52rem; letter-spacing:3px;
            text-transform:uppercase; color:#FB923C; margin:0 0 1rem 0;">
    ◈ Gemini 2.5 Flash Lite · {scope} · Model-grounded
  </p>
  <p style="font-family:'Source Serif 4',serif; font-size:0.92rem; color:#1C1917;
            line-height:1.8; margin:0;">
    {st.session_state['last_answer']}
  </p>
</div>
""", unsafe_allow_html=True)

    st.download_button(
        "⬇ Export Answer",
        data=f"Question: {st.session_state['last_question']}\nScope: {scope}\n\n{st.session_state['last_answer']}",
        file_name=f"mnrega_insight_{scope.replace(' ', '_')}.txt",
        mime="text/plain",
        key="dl",
    )

    with st.expander("🔍 Data context sent to Gemini"):
        st.json(build_context(state_param))
