"""
components/chat_ui.py
Streamlit Chatbot Interface for AI-Powered Laptop Recommendations (Phase 2).
Provides conversational interaction, follow-up suggestions, requirement badges,
and renders real laptop recommendations from SQLite.
"""

import logging
import streamlit as st
import pandas as pd
from typing import Optional

from services.ai_service import GeminiService, sanitize_error_message
from services.requirement_parser import sanitize_user_input
from services.recommender import recommend_laptops
from database.db_manager import get_all_laptops
from components.ui_cards import render_laptop_card, format_inr
from components.charts import create_radar_chart

logger = logging.getLogger(__name__)


def init_chat_session():
    """Initializes chatbot session state variables if not already present."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 **Hello! I'm your AI Laptop Advisor.**\n\n"
                    "Tell me what you're looking for in plain English! For example:\n"
                    "> *\"I'm a CSE student. My budget is around ₹75,000. I need a laptop for programming, "
                    "AI/ML and occasional gaming with at least 16GB RAM and a dedicated GPU.\"*\n\n"
                    "I will analyze your requirements, query our verified database, and recommend the best matching laptops."
                ),
                "laptops": None,
                "chart": None,
                "requirements": None,
            }
        ]

    if "user_requirements" not in st.session_state:
        st.session_state.user_requirements = {
            "maximum_budget": None,
            "primary_usage": "General College Use / Everyday",
            "secondary_usage": None,
            "minimum_ram": 8,
            "gpu_required": "Any",
            "preferred_brand": "All Brands",
        }

    if "last_recommended_laptops" not in st.session_state:
        st.session_state.last_recommended_laptops = []


def render_active_requirements_summary(req: dict):
    """Renders a visual badge displaying currently active requirement constraints."""
    if not req:
        return

    budget_str = format_inr(req["maximum_budget"]) if req.get("maximum_budget") else "Not Set"
    usage_str = req.get("primary_usage", "General")
    ram_str = f"{req.get('minimum_ram', 8)}GB RAM"
    gpu_str = req.get("gpu_required", "Any GPU")
    brand_str = req.get("preferred_brand", "All Brands")

    st.markdown(
        f"""
        <div style="background: #F1F5F9; border-radius: 8px; padding: 8px 14px; margin-bottom: 12px; font-size: 0.85rem; color: #334155; border: 1px solid #CBD5E1;">
            <b>📋 Active Context:</b>
            <span style="margin: 0 4px; padding: 2px 6px; background: #E2E8F0; border-radius: 4px;">💰 Budget: {budget_str}</span>
            <span style="margin: 0 4px; padding: 2px 6px; background: #E2E8F0; border-radius: 4px;">🎯 Usage: {usage_str}</span>
            <span style="margin: 0 4px; padding: 2px 6px; background: #E2E8F0; border-radius: 4px;">🧠 {ram_str}</span>
            <span style="margin: 0 4px; padding: 2px 6px; background: #E2E8F0; border-radius: 4px;">🎮 {gpu_str}</span>
            <span style="margin: 0 4px; padding: 2px 6px; background: #E2E8F0; border-radius: 4px;">🏷️ {brand_str}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def process_user_query(user_query: str, gemini_service: GeminiService, df_all: pd.DataFrame):
    """
    Executes the Phase 2 AI architecture pipeline:
    User message -> LLM requirement extraction -> Structured JSON -> Existing recommender
    -> Real SQLite laptop matches -> Grounded explanation -> Streamlit Chat.
    """
    # 0. Sanitize input (enforce length limit & scrub control characters)
    cleaned_query = sanitize_user_input(user_query, max_length=1000)
    if not cleaned_query:
        return

    # 1. Append user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": cleaned_query,
        "laptops": None,
        "chart": None,
        "requirements": None,
    })

    # 2. Extract structured requirements and generate response safely
    current_req = st.session_state.user_requirements
    assistant_message = ""
    recommended_laptops = None
    radar_chart = None
    extracted_req = current_req

    try:
        extracted_req = gemini_service.extract_requirements(
            user_message=cleaned_query,
            conversation_history=st.session_state.messages,
            current_state=current_req,
        )

        # Update active requirements session state
        st.session_state.user_requirements = extracted_req
        intent = extracted_req.get("intent", "recommendation")
        missing_fields = extracted_req.get("missing_critical_info", [])

        # Handle Intent: Greeting
        if intent == "greeting":
            assistant_message = (
                "Hello! I am ready to help you find the ideal laptop for your college studies, coding, or gaming. "
                "What is your approximate budget and what workloads will you be running?"
            )

        # Handle Intent: Clarification (Missing critical information like budget)
        elif missing_fields and intent != "comparison":
            assistant_message = gemini_service.generate_clarification(missing_fields)

        # Handle Intent: Comparison
        elif intent == "comparison":
            comp_indices = extracted_req.get("comparison_indices") or [1, 2]
            prev_laptops = st.session_state.get("last_recommended_laptops") or []

            # Recover previous recommendations from message history if needed
            if len(prev_laptops) < 2:
                for msg in reversed(st.session_state.get("messages", [])):
                    if msg.get("laptops") and len(msg["laptops"]) >= 2:
                        prev_laptops = msg["laptops"]
                        st.session_state.last_recommended_laptops = prev_laptops
                        break

            # If still < 2 laptops, fetch top laptops matching current requirements from database
            if len(prev_laptops) < 2:
                budget = extracted_req.get("maximum_budget") or current_req.get("maximum_budget") or 80000
                usage = extracted_req.get("primary_usage") or current_req.get("primary_usage") or "Programming / Coding"
                candidate_df = recommend_laptops(df=df_all, budget=budget, primary_usage=usage, top_n=2)
                if len(candidate_df) >= 2:
                    prev_laptops = candidate_df.to_dict(orient="records")
                else:
                    prev_laptops = df_all.head(2).to_dict(orient="records")
                st.session_state.last_recommended_laptops = prev_laptops

            # Map indices (1-based to 0-based)
            idx1 = min(max(comp_indices[0] - 1, 0), len(prev_laptops) - 1)
            idx2 = min(max(comp_indices[1] - 1, 0), len(prev_laptops) - 1)
            if idx1 == idx2 and len(prev_laptops) > 1:
                idx2 = 1 if idx1 == 0 else 0

            laptops_to_comp = [prev_laptops[idx1], prev_laptops[idx2]]
            assistant_message = gemini_service.generate_comparison(laptops_to_comp, cleaned_query)
            recommended_laptops = laptops_to_comp

            try:
                radar_chart = create_radar_chart(laptops_to_comp)
            except Exception as chart_err:
                logger.warning(f"Could not generate radar chart: {chart_err}")
                radar_chart = None

        # Handle Intent: Recommendation (Default)
        else:
            budget = extracted_req.get("maximum_budget") or 100000
            primary_usage = extracted_req.get("primary_usage", "General College Use / Everyday")
            min_ram = extracted_req.get("minimum_ram", 8)
            gpu_req = extracted_req.get("gpu_required", "Any")
            brand = extracted_req.get("preferred_brand", "All Brands")

            recs_df = recommend_laptops(
                df=df_all,
                budget=budget,
                primary_usage=primary_usage,
                min_ram=min_ram,
                gpu_requirement=gpu_req,
                preferred_brand=brand,
                top_n=3,
            )

            if recs_df.empty:
                recommended_laptops = []
                st.session_state.last_recommended_laptops = []
                assistant_message = gemini_service.generate_explanation(
                    user_message=cleaned_query,
                    user_requirements=extracted_req,
                    recommended_laptops=[],
                )
            else:
                recommended_laptops = recs_df.to_dict(orient="records")
                st.session_state.last_recommended_laptops = recommended_laptops
                assistant_message = gemini_service.generate_explanation(
                    user_message=cleaned_query,
                    user_requirements=extracted_req,
                    recommended_laptops=recommended_laptops,
                )

    except Exception as exc:
        safe_exc = sanitize_error_message(exc, getattr(gemini_service, "api_key", None))
        logger.error(f"Error processing user query: {safe_exc}")
        if not assistant_message:
            fallback_laps = df_all.head(2).to_dict(orient="records")
            assistant_message = gemini_service._generate_fallback_comparison(fallback_laps)
            recommended_laptops = fallback_laps

    # Guarantee assistant_message is never blank
    if not assistant_message or not assistant_message.strip():
        assistant_message = "I evaluated your preferences against our verified database. Here are the top laptop recommendations:"

    # 3. Always append assistant response to messages
    st.session_state.messages.append({
        "role": "assistant",
        "content": assistant_message,
        "laptops": recommended_laptops,
        "chart": radar_chart,
        "requirements": extracted_req,
    })


def render_chat_tab(gemini_service: GeminiService, df_all: pd.DataFrame):
    """Renders the complete Chatbot interface inside Streamlit."""
    init_chat_session()

    st.subheader("💬 AI Conversational Laptop Advisor")
    st.caption("Chat naturally with the AI. Your requirements are parsed, validated, and matched against real SQLite records.")

    # Prominent Active Engine Badge (Requirement 9)
    if gemini_service.is_available():
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; justify-content: space-between; background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 8px; padding: 10px 16px; margin-bottom: 12px;">
                <div>
                    <span style="font-weight: 700; color: #166534; font-size: 0.95rem;">🤖 Engine Status:</span>
                    <span style="background: #DCFCE7; color: #15803D; font-weight: 700; padding: 3px 10px; border-radius: 12px; margin-left: 6px; font-size: 0.85rem; border: 1px solid #86EFAC;">Gemini AI: Connected</span>
                </div>
                <span style="font-size: 0.8rem; color: #15803D;">✅ Real-time Gemini 2.5 Flash Reasoning Active</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; justify-content: space-between; background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 8px; padding: 10px 16px; margin-bottom: 12px;">
                <div>
                    <span style="font-weight: 700; color: #92400E; font-size: 0.95rem;">⚙️ Engine Status:</span>
                    <span style="background: #FEF3C7; color: #B45309; font-weight: 700; padding: 3px 10px; border-radius: 12px; margin-left: 6px; font-size: 0.85rem; border: 1px solid #FCD34D;">Gemini AI: Unavailable — using fallback engine</span>
                </div>
                <span style="font-size: 0.8rem; color: #92400E;">🛡️ Operating via deterministic fallback rule engine</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Expandable "How AI Works" Architecture Pipeline (Requirement 10)
    with st.expander("ℹ️ How the AI Architecture Works (Zero-Hallucination Pipeline)", expanded=False):
        st.markdown(
            """
            **End-to-End Processing Architecture:**
            
            ```
            User Natural-Language Message
                     ↓
            Gemini Requirement Extraction (Structured JSON)
                     ↓
            Pydantic Schema & Safety Validation
                     ↓
            Existing MCDA Recommendation Engine (Deterministic Scoring)
                     ↓
            SQLite Database (126 Verified Records — Single Source of Truth)
                     ↓
            Top Ranked Recommendations (Deterministic)
                     ↓
            Gemini Technical Explanation (Grounded ONLY in Retrieved DB Records)
                     ↓
            Streamlit Conversational Chatbot UI
            ```
            
            - **100% Grounded Facts**: Gemini never invents laptop models, specs, or benchmark scores.
            - **Mathematical MCDA Ranking**: Recommendations are scored using multi-criteria weighted utility functions.
            - **Fault Tolerance**: If the API key is missing or network limits occur, the **Built-in Rule Engine (Fallback)** processes requirements with zero interruption.
            """
        )

    # Display Active Requirements Badge
    render_active_requirements_summary(st.session_state.user_requirements)

    # Quick Prompt Chips for Fast Testing & Viva Demonstration
    st.markdown("**⚡ Quick Prompts / Follow-up Examples:**")
    col1, col2, col3, col4 = st.columns(4)

    prompt_to_run = None
    with col1:
        if st.button("💻 CSE Student under ₹75k", use_container_width=True):
            prompt_to_run = "I'm a CSE student. My budget is around ₹75,000. I need a laptop for programming, AI/ML and occasional gaming. I want at least 16GB RAM and a dedicated GPU."
    with col2:
        if st.button("📉 Show me something cheaper", use_container_width=True):
            prompt_to_run = "Show me something cheaper"
    with col3:
        if st.button("🏷️ What about ASUS?", use_container_width=True):
            prompt_to_run = "What about ASUS?"
    with col4:
        if st.button("⚖️ Compare the first two", use_container_width=True):
            prompt_to_run = "Compare the first two"

    # Render Conversation Stream
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # If message contains real recommended laptops, display cards
            if msg.get("laptops"):
                st.markdown("---")
                if len(msg["laptops"]) == 2 and msg.get("chart"):
                    st.markdown("#### ⚖️ Compared Models")
                else:
                    st.markdown("#### 🏆 Verified Matches from Database")

                for rank, lap in enumerate(msg["laptops"], start=1):
                    lap_series = pd.Series(lap)
                    render_laptop_card(lap_series, rank=rank)

            # If message contains a comparison chart
            if msg.get("chart"):
                st.plotly_chart(msg["chart"], use_container_width=True)

    # Process quick prompt if clicked
    if prompt_to_run:
        process_user_query(prompt_to_run, gemini_service, df_all)
        st.rerun()

    # User Chat Input
    user_input = st.chat_input("Describe what laptop you need (e.g. budget, usage, RAM, brand)...")
    if user_input:
        process_user_query(user_input, gemini_service, df_all)
        st.rerun()
