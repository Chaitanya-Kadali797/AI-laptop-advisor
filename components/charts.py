"""
components/charts.py
Plotly visualization charts for laptop score comparisons, price-performance analysis,
and radar charts for multi-domain benchmark evaluation.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Reusable explicit high-contrast hover label styling (White background, Black text, Dark border)
DEFAULT_HOVERLABEL = dict(
    bgcolor="#FFFFFF",
    bordercolor="#222222",
    font=dict(
        color="#000000",
        size=14,
        family="Arial, sans-serif",
    ),
)


def apply_hoverlabel_style(fig):
    """
    Explicitly applies high-contrast hoverlabel styling (white background, black text,
    and dark border) to layout, all traces, and layout.template to ensure
    no template, theme, or trace defaults can override text contrast.
    """
    if fig is None:
        return None

    hover_cfg = dict(
        bgcolor="#FFFFFF",
        bordercolor="#222222",
        font=dict(
            color="#000000",
            size=14,
            family="Arial, sans-serif",
        ),
    )

    fig.update_layout(hoverlabel=hover_cfg)
    fig.update_traces(hoverlabel=hover_cfg)

    if hasattr(fig, "layout") and hasattr(fig.layout, "template") and hasattr(fig.layout.template, "layout"):
        fig.layout.template.layout.hoverlabel = hover_cfg

    return fig


def create_price_vs_performance_chart(df: pd.DataFrame, primary_usage: str):
    """
    Creates a scatter plot showing Laptop Price vs Performance Score
    to easily identify the best-value laptops.
    """
    if df.empty:
        return None

    hover_data = {
        "price": ":,.0f",
        "processor": True,
        "ram": True,
        "gpu": True,
    }
    if "match_score" in df.columns:
        hover_data["match_score"] = ":.1f"

    fig = px.scatter(
        df,
        x="price",
        y="domain_score" if "domain_score" in df.columns else "programming_score",
        color="brand",
        size="ram",
        hover_name="model",
        hover_data=hover_data,
        labels={
            "price": "Price (₹)",
            "domain_score": f"{primary_usage} Score (out of 100)",
            "programming_score": "Programming Score (out of 100)",
            "brand": "Brand",
            "ram": "RAM (GB)",
        },
        title=f"💡 Price vs. Performance Analysis ({primary_usage})",
        template="plotly_white",
    )

    fig.update_layout(
        font=dict(family="Inter, sans-serif", size=12),
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
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
    fig.update_traces(marker=dict(line=dict(width=1, color="DarkSlateGrey")))
    apply_hoverlabel_style(fig)
    return fig


def create_radar_chart(laptops: list[dict]):
    """
    Generates a multi-attribute Radar chart comparing domain capabilities
    (Gaming, Programming, AI/ML, Video Editing) for up to 4 laptops.
    """
    categories = ["Gaming", "Programming", "AI / ML", "Video Editing"]

    fig = go.Figure()

    colors = ["#2563EB", "#10B981", "#F59E0B", "#EF4444"]

    for idx, laptop in enumerate(laptops):
        values = [
            laptop["gaming_score"],
            laptop["programming_score"],
            laptop["ai_ml_score"],
            laptop["video_editing_score"],
        ]
        # Close the loop
        values.append(values[0])
        radar_categories = categories + [categories[0]]

        color = colors[idx % len(colors)]
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=radar_categories,
                fill="toself",
                name=f"{laptop['brand']} {laptop['model']}",
                line=dict(color=color, width=2),
                opacity=0.65,
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10),
            )
        ),
        showlegend=True,
        title="📊 Multi-Domain Capability Comparison (Radar Analysis)",
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
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
    apply_hoverlabel_style(fig)
    return fig


def create_domain_scores_bar_chart(laptops: list[dict]):
    """
    Creates a grouped bar chart comparing performance metrics of selected laptops.
    """
    data = []
    for lap in laptops:
        name = f"{lap['brand']} {lap['model']}"
        data.extend([
            {"Laptop": name, "Metric": "Gaming", "Score": lap["gaming_score"]},
            {"Laptop": name, "Metric": "Programming", "Score": lap["programming_score"]},
            {"Laptop": name, "Metric": "AI / ML", "Score": lap["ai_ml_score"]},
            {"Laptop": name, "Metric": "Video Editing", "Score": lap["video_editing_score"]},
        ])

    df = pd.DataFrame(data)
    fig = px.bar(
        df,
        x="Metric",
        y="Score",
        color="Laptop",
        barmode="group",
        title="📈 Benchmark Scores Breakdown",
        template="plotly_white",
        range_y=[0, 100],
    )
    fig.update_layout(
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
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
    apply_hoverlabel_style(fig)
    return fig
