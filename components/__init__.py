"""UI components package for AI Laptop Advisor."""
from .charts import create_price_vs_performance_chart, create_radar_chart, create_domain_scores_bar_chart
from .ui_cards import render_laptop_card, format_inr, render_score_badge, render_confidence_badge
from .chat_ui import render_chat_tab, init_chat_session
from .analytics_dashboard import render_evaluation_dashboard
from .what_if_ui import render_what_if_tab

__all__ = [
    "create_price_vs_performance_chart",
    "create_radar_chart",
    "create_domain_scores_bar_chart",
    "render_laptop_card",
    "format_inr",
    "render_score_badge",
    "render_confidence_badge",
    "render_chat_tab",
    "init_chat_session",
    "render_evaluation_dashboard",
    "render_what_if_tab",
]
