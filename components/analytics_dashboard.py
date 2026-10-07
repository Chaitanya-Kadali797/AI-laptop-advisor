"""
components/analytics_dashboard.py
Recommendation Evaluation Dashboard (Phase 3).
Displays aggregate evaluation metrics, satisfaction rates, and Plotly analytics.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

from components.ui_cards import format_inr
from components.charts import create_radar_chart, DEFAULT_HOVERLABEL, apply_hoverlabel_style


def render_evaluation_dashboard(
    df_all: pd.DataFrame,
    recommendations: pd.DataFrame,
    budget: float,
    primary_usage: str,
    min_ram: int,
    gpu_requirement: str,
    preferred_brand: str,
    alternatives: Dict[str, Any],
):
    """
    Renders the Phase 3 Recommendation Evaluation Dashboard.
    """
    st.subheader("📊 Recommendation Evaluation & Analytics Dashboard")
    st.caption(
        "Quantitative evaluation of the recommendation pool, constraint satisfaction, and multi-attribute distribution."
    )

    total_db_count = len(df_all)
    matching_count = len(recommendations)
    satisfaction_rate = (matching_count / total_db_count * 100.0) if total_db_count > 0 else 0.0

    if not recommendations.empty:
        avg_score = recommendations["match_score"].mean()
        best_score = recommendations["match_score"].max()
        avg_price = recommendations["price"].mean()
    else:
        avg_score = 0.0
        best_score = 0.0
        avg_price = 0.0

    # KPI Metric Cards
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    with kpi1:
        st.metric("📚 Database Size", f"{total_db_count} models")
    with kpi2:
        st.metric("🎯 Eligible Matches", f"{matching_count} models")
    with kpi3:
        st.metric("🏆 Best Match Score", f"{best_score:.1f}%")
    with kpi4:
        st.metric("⭐ Average Score", f"{avg_score:.1f}%")
    with kpi5:
        st.metric("💵 Average Price", format_inr(avg_price) if avg_price > 0 else "N/A")
    with kpi6:
        st.metric("✅ Candidate Rate", f"{satisfaction_rate:.1f}%")

    st.divider()

    if recommendations.empty:
        st.warning("⚠️ No laptops in the database satisfy all active constraints. Adjust your budget or hardware filters to see analytics.")
        return

    # Visual Analytics Grid (2x2)
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("##### 📈 Sub-Score Breakdown (Top 5 Recommendations)")
        top5 = recommendations.head(5).copy()
        top5["Label"] = top5["brand"] + " " + top5["model"].apply(lambda x: x[:16] + ".." if len(x) > 16 else x)

        score_data = []
        for _, row in top5.iterrows():
            score_data.extend([
                {"Laptop": row["Label"], "Criterion": "Budget Match", "Score": row.get("budget_match_pct", 100.0)},
                {"Laptop": row["Label"], "Criterion": "RAM Match", "Score": row.get("ram_match_pct", 100.0)},
                {"Laptop": row["Label"], "Criterion": "GPU Suitability", "Score": row.get("gpu_match_pct", 100.0)},
                {"Laptop": row["Label"], "Criterion": "Workload Rating", "Score": row.get("usage_match_pct", row["domain_score"])},
            ])

        score_df = pd.DataFrame(score_data)
        fig_sub = px.bar(
            score_df,
            x="Criterion",
            y="Score",
            color="Laptop",
            barmode="group",
            range_y=[0, 105],
            template="plotly_white",
        )
        fig_sub.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
            hoverlabel=dict(
                bgcolor="#FFFFFF",
                bordercolor="#222222",
                font=dict(
                    color="#000000",
                    size=14,
                    family="Arial, sans-serif",
                ),
            ),
        )
        apply_hoverlabel_style(fig_sub)
        st.plotly_chart(fig_sub, use_container_width=True)

    with chart_col2:
        st.markdown("##### 🏷️ Brand Distribution in Matching Pool")
        brand_counts = recommendations["brand"].value_counts().reset_index()
        brand_counts.columns = ["Brand", "Count"]
        fig_brand = px.pie(
            brand_counts,
            values="Count",
            names="Brand",
            hole=0.45,
            template="plotly_white",
        )
        fig_brand.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            hoverlabel=dict(
                bgcolor="#FFFFFF",
                bordercolor="#222222",
                font=dict(
                    color="#000000",
                    size=14,
                    family="Arial, sans-serif",
                ),
            ),
        )
        apply_hoverlabel_style(fig_brand)
        st.plotly_chart(fig_brand, use_container_width=True)

    # Section: Alternative Standout Recommendations (Phase 3 Requirement 5)
    st.markdown("---")
    st.markdown("### 🏆 Alternative Standout Laptops Across Decision Axes")
    st.caption("Key options in your matching pool optimized for specific user trade-offs:")

    alt_cols = st.columns(len(alternatives))
    for col, (category_title, laptop_item) in zip(alt_cols, alternatives.items()):
        if laptop_item:
            with col:
                with st.container(border=True):
                    st.markdown(f"**{category_title}**")
                    st.markdown(f"##### {laptop_item['brand']} {laptop_item['model']}")
                    st.markdown(f"**{format_inr(laptop_item['price'])}** | Match: **{laptop_item['match_score']:.1f}%**")
                    st.caption(f"⚡ {laptop_item['processor']}")
                    st.caption(f"🧠 {laptop_item['ram']}GB | 🎮 {laptop_item['gpu']}")
                    st.caption(f"🎯 Workload: {int(laptop_item['domain_score'])}/100")

    # Radar profile comparison of the distinct alternatives
    distinct_alternatives = []
    seen_models = set()
    for cat in ["Best Overall", "Best Budget Option", "Best Performance Option"]:
        lap = alternatives.get(cat)
        if lap and lap["model"] not in seen_models:
            seen_models.add(lap["model"])
            distinct_alternatives.append(lap)

    if len(distinct_alternatives) >= 2:
        st.markdown("##### 🕸️ Alternative Trade-off Radar Analysis")
        radar_fig = create_radar_chart(distinct_alternatives)
        st.plotly_chart(radar_fig, use_container_width=True)
