"""
components/what_if_ui.py
Interactive What-If Analysis Interface (Phase 3).
Allows users to modify constraints and immediately observe before-and-after recommendation changes.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any

from services.recommender import run_what_if_analysis
from components.ui_cards import render_laptop_card, format_inr


def render_what_if_tab(
    df_all: pd.DataFrame,
    current_budget: float,
    current_usage: str,
    current_ram: int,
    current_gpu: str,
    current_brand: str,
):
    """
    Renders the What-If Analysis simulator.
    """
    st.subheader("🔮 Interactive What-If Scenario Simulator")
    st.caption(
        "Tweak a single requirement below to immediately see how the algorithm's recommendations shift. "
        "Observe price deltas, hardware upgrades, and score improvements."
    )

    base_params = {
        "budget": current_budget,
        "primary_usage": current_usage,
        "min_ram": current_ram,
        "gpu_requirement": current_gpu,
        "preferred_brand": current_brand,
    }

    # Simulation Controls
    with st.expander("🛠️ Configure 'What-If' Parameter Modifications", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            sim_budget = st.slider(
                "Hypothetical Budget (₹)",
                min_value=30000,
                max_value=350000,
                value=int(current_budget),
                step=5000,
                format="₹%d",
                help="Test what happens if you increase or decrease your budget ceiling.",
            )
            sim_usage = st.selectbox(
                "Hypothetical Primary Usage",
                options=[
                    "Programming / Coding",
                    "AI / Data Science",
                    "Gaming",
                    "Video Editing / Content Creation",
                    "General College Use / Everyday",
                ],
                index=[
                    "Programming / Coding",
                    "AI / Data Science",
                    "Gaming",
                    "Video Editing / Content Creation",
                    "General College Use / Everyday",
                ].index(current_usage) if current_usage in [
                    "Programming / Coding",
                    "AI / Data Science",
                    "Gaming",
                    "Video Editing / Content Creation",
                    "General College Use / Everyday",
                ] else 0,
            )

        with col2:
            sim_ram = st.selectbox(
                "Hypothetical Minimum RAM",
                options=[8, 16, 32],
                index=[8, 16, 32].index(current_ram) if current_ram in [8, 16, 32] else 1,
                format_func=lambda x: f"{x} GB RAM",
            )
            sim_gpu = st.selectbox(
                "Hypothetical GPU Requirement",
                options=["Any", "Dedicated GPU", "Integrated GPU (Thin & Light / Battery)"],
                index=["Any", "Dedicated GPU", "Integrated GPU (Thin & Light / Battery)"].index(current_gpu) if current_gpu in ["Any", "Dedicated GPU", "Integrated GPU (Thin & Light / Battery)"] else 0,
            )

        with col3:
            brands = ["All Brands", "Apple", "Asus", "Dell", "HP", "Lenovo", "Acer", "MSI", "Samsung", "LG", "Gigabyte"]
            sim_brand = st.selectbox(
                "Hypothetical Brand Preference",
                options=brands,
                index=brands.index(current_brand) if current_brand in brands else 0,
            )

            # Quick Preset Scenario Buttons
            st.markdown("**Quick Preset Scenarios:**")
            quick_cols = st.columns(2)
            with quick_cols[0]:
                if st.button("➕ Add ₹15k Budget", use_container_width=True):
                    sim_budget = min(int(current_budget) + 15000, 350000)
            with quick_cols[1]:
                if st.button("🚀 Upgrade to 32GB", use_container_width=True):
                    sim_ram = 32

    modified_params = {
        "budget": sim_budget,
        "primary_usage": sim_usage,
        "min_ram": sim_ram,
        "gpu_requirement": sim_gpu,
        "preferred_brand": sim_brand,
    }

    # Run What-If Analysis
    analysis = run_what_if_analysis(df_all, base_params, modified_params)

    # Narrative Summary
    if analysis["parameter_diffs"]:
        diffs_text = " • ".join(analysis["parameter_diffs"])
        st.info(f"🔄 **Modifications Applied:** {diffs_text}\n\n**Verdict:** {analysis['verdict']}", icon="💡")
    else:
        st.info("ℹ️ Modify one or more parameters above to simulate recommendation shifts.", icon="💡")

    st.markdown("---")

    # Side-by-Side Before vs After Layout
    col_before, col_after = st.columns(2)

    with col_before:
        st.markdown("#### 🔵 BEFORE (Active Baseline)")
        st.caption(
            f"Budget: ₹{int(current_budget):,} | Usage: {current_usage} | {current_ram}GB RAM | {current_gpu} | {current_brand}"
        )
        if analysis["base_top"]:
            render_laptop_card(pd.Series(analysis["base_top"]), rank=1)
        else:
            st.warning("No matches under current baseline constraints.")

    with col_after:
        st.markdown("#### 🟢 AFTER (Simulated Scenario)")
        st.caption(
            f"Budget: ₹{int(sim_budget):,} | Usage: {sim_usage} | {sim_ram}GB RAM | {sim_gpu} | {sim_brand}"
        )
        if analysis["mod_top"]:
            render_laptop_card(pd.Series(analysis["mod_top"]), rank=1)
        else:
            st.warning("No matches found under the modified constraints. Try easing budget or RAM requirements.")
