"""
utils/ai_summary.py
--------------------
Drop-in AI summary widget for the bottom of any page.

Usage:
    from utils.ai_summary import render_ai_summary
    render_ai_summary("overview", state_param=state_param)

    # With extra context string:
    render_ai_summary("districts", state_param=state_param,
                      extra=f"District: {district}, State: {state}, Latest PD: {latest_pd}")
"""

import streamlit as st
from utils.gemini_utils import build_context, call_gemini, page_summary_prompt, get_gemini_key


def render_ai_summary(page_key: str, state_param: str | None = None, extra: str = ""):
    """
    Renders a collapsible AI summary card at the bottom of a page.
    Only shows if a Gemini key is present in session state.
    """
    api_key = get_gemini_key()
    if not api_key:
        return  # silently skip if no key — don't nag the user on every page

    st.markdown("---")
    st.markdown("""
<p style="font-family:'DM Mono',monospace; font-size:0.58rem; letter-spacing:3px;
          text-transform:uppercase; color:#A8A29E; margin:0 0 10px 0;
          padding-bottom:8px; border-bottom:1px solid #F5F5F4;">
  ✦ AI Summary · Gemini 2.5 Flash Lite
</p>
""", unsafe_allow_html=True)

    col_btn, col_space = st.columns([1, 3])
    with col_btn:
        generate = st.button("Generate Summary", key=f"ai_summary_{page_key}")

    if generate or st.session_state.get(f"ai_summary_done_{page_key}"):
        cache_key = f"ai_summary_text_{page_key}_{state_param}"

        if cache_key not in st.session_state or generate:
            with st.spinner("Analysing with Gemini..."):
                ctx = build_context(state_param)
                prompt = page_summary_prompt(ctx, page_key, extra)
                text = call_gemini(api_key, prompt, temperature=0.3)
                st.session_state[cache_key] = text
                st.session_state[f"ai_summary_done_{page_key}"] = True

        text = st.session_state.get(cache_key, "")
        if text:
            st.markdown(f"""
<div style="background:#FFFBF5; border:1px solid #FED7AA; border-left:3px solid #FB923C;
            border-radius:8px; padding:1.1rem 1.4rem; margin-top:0.5rem;">
  <p style="font-family:'Source Serif 4',serif; font-size:0.9rem; color:#1C1917;
            line-height:1.8; margin:0;">{text}</p>
</div>
""", unsafe_allow_html=True)
