"""
components/ui_cards.py
UI presentation helpers and styled components for Streamlit laptop cards,
match percentage breakdowns, confidence indicators, and detailed explanations (Phase 3).
"""

import streamlit as st
import pandas as pd


def format_inr(price: float) -> str:
    """Format numerical price into clean INR currency format (e.g. ₹89,990)."""
    return f"₹{int(price):,}"


def render_score_badge(score: float):
    """Render a visual color-coded badge for match score."""
    if score >= 85:
        color = "#10B981"  # Emerald green
        label = "Excellent Match"
    elif score >= 70:
        color = "#3B82F6"  # Blue
        label = "Great Match"
    elif score >= 50:
        color = "#F59E0B"  # Amber
        label = "Fair Match"
    else:
        color = "#EF4444"  # Red
        label = "Basic Match"

    return f"""
    <div style="background-color: {color}20; border: 1px solid {color}; border-radius: 8px; padding: 5px 10px; display: inline-block;">
        <span style="color: {color}; font-weight: 700; font-size: 1.05rem;">★ {score:.1f}% Match</span>
        <span style="color: #64748B; font-size: 0.8rem; margin-left: 4px;">({label})</span>
    </div>
    """


def render_confidence_badge(confidence: float, label: str = ""):
    """Renders a transparent Recommendation Confidence indicator (Phase 3)."""
    if confidence >= 85:
        color = "#059669"
        tier = "High"
    elif confidence >= 70:
        color = "#2563EB"
        tier = "Moderate"
    else:
        color = "#D97706"
        tier = "Preliminary"

    return f"""
    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-left: 3px solid {color}; border-radius: 6px; padding: 4px 8px; display: inline-block; margin-left: 8px;">
        <span style="font-size: 0.82rem; color: #475569; font-weight: 600;">🛡️ Recommendation Confidence: </span>
        <span style="font-size: 0.82rem; font-weight: 700; color: {color};">{confidence:.1f}% ({tier})</span>
    </div>
    """


def render_laptop_card(laptop: pd.Series, rank: int = None, show_actions: bool = True):
    """
    Renders a well-structured laptop specification card inside a Streamlit container
    with Phase 3 Evaluation metrics, sub-scores, and plain-English explainability.
    """
    with st.container(border=True):
        header_col, price_col = st.columns([3, 1])

        with header_col:
            rank_prefix = f"#{rank} " if rank else ""
            st.markdown(f"### {rank_prefix}{laptop['brand']} {laptop['model']}")
            badge_html = ""
            if "match_score" in laptop and pd.notna(laptop["match_score"]):
                badge_html += render_score_badge(laptop["match_score"])
            if "confidence_score" in laptop and pd.notna(laptop["confidence_score"]):
                badge_html += render_confidence_badge(
                    laptop["confidence_score"],
                    laptop.get("confidence_label", ""),
                )
            if badge_html:
                st.markdown(badge_html, unsafe_allow_html=True)

        with price_col:
            st.markdown(
                f"<div style='text-align: right;'>"
                f"<span style='font-size: 1.6rem; font-weight: 800; color: #1E293B;'>{format_inr(laptop['price'])}</span>"
                f"<br><span style='font-size: 0.8rem; color: #64748B;'>Incl. all taxes</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Plain-English Explanation (Phase 3 Requirement 1)
        if "detailed_explanation" in laptop and pd.notna(laptop["detailed_explanation"]):
            st.info(f"💡 **Recommendation Rationale:** {laptop['detailed_explanation']}", icon="🎯")
        elif "recommendation_reason" in laptop and laptop["recommendation_reason"]:
            st.info(f"🎯 **Why this laptop:** {laptop['recommendation_reason']}", icon="💡")

        # Phase 3: Requirement Match Percentage Sub-Scores
        if "budget_match_pct" in laptop:
            with st.expander("📊 View Requirement Match Percentage Breakdown (Phase 3)", expanded=False):
                s1, s2, s3, s4 = st.columns(4)
                with s1:
                    b_pct = laptop.get("budget_match_pct", 100.0)
                    st.metric("💰 Budget Match", f"{b_pct:.1f}%")
                    st.progress(min(b_pct / 100.0, 1.0))
                with s2:
                    r_pct = laptop.get("ram_match_pct", 100.0)
                    st.metric("🧠 RAM Match", f"{r_pct:.1f}%", help=f"{laptop['ram']}GB RAM")
                    st.progress(min(r_pct / 100.0, 1.0))
                with s3:
                    g_pct = laptop.get("gpu_match_pct", 100.0)
                    st.metric("🎮 GPU Suitability", f"{g_pct:.1f}%", help=laptop["gpu"])
                    st.progress(min(g_pct / 100.0, 1.0))
                with s4:
                    u_pct = laptop.get("usage_match_pct", laptop.get("domain_score", 80.0))
                    st.metric("🎯 Workload Match", f"{u_pct:.1f}%")
                    st.progress(min(u_pct / 100.0, 1.0))

        st.divider()

        # Key Hardware Specs in 4 columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"**⚡ Processor:**\n{laptop['processor']}")
            st.markdown(f"**🧠 RAM:**\n{laptop['ram']} GB")
        with col2:
            st.markdown(f"**💾 Storage:**\n{laptop['storage']}")
            st.markdown(f"**🎮 GPU:**\n{laptop['gpu']}")
        with col3:
            st.markdown(f"**🖥️ Display:**\n{laptop['display']}")
            st.markdown(f"**🔋 Battery:**\n{laptop['battery']}")
        with col4:
            st.markdown(f"**⚖️ Weight:**\n{laptop['weight']} kg")
            st.markdown(f"**🏷️ GPU Category:**\n{laptop['gpu_type']}")

        # Form Factor Portability & Battery Callouts (Phase 3)
        if "portability_suitability" in laptop or "battery_suitability" in laptop:
            p_text = laptop.get("portability_suitability", "")
            b_text = laptop.get("battery_suitability", "")
            if p_text or b_text:
                st.caption(f"🎒 **Portability:** {p_text} &nbsp;|&nbsp; 🔋 **Battery Profile:** {b_text}")

        # Benchmark score progress bars
        st.markdown(
            "<p style='font-size: 0.85rem; font-weight: 600; margin-top: 10px; margin-bottom: 4px; color: #475569;'>PERFORMANCE BENCHMARK RATINGS</p>",
            unsafe_allow_html=True,
        )
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.caption(f"🎮 Gaming: {int(laptop['gaming_score'])}/100")
            st.progress(laptop["gaming_score"] / 100.0)
        with m_col2:
            st.caption(f"💻 Coding: {int(laptop['programming_score'])}/100")
            st.progress(laptop["programming_score"] / 100.0)
        with m_col3:
            st.caption(f"🤖 AI / ML: {int(laptop['ai_ml_score'])}/100")
            st.progress(laptop["ai_ml_score"] / 100.0)
        with m_col4:
            st.caption(f"🎬 Video: {int(laptop['video_editing_score'])}/100")
            st.progress(laptop["video_editing_score"] / 100.0)
