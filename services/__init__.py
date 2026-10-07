"""Services package for AI Laptop Advisor."""
from .recommender import (
    recommend_laptops,
    calculate_performance_score,
    get_alternative_recommendations,
    run_what_if_analysis,
    calculate_recommendation_confidence,
    generate_detailed_explanation,
)
from .requirement_parser import (
    UserRequirements,
    parse_requirements_rule_based,
    update_requirements_state,
    sanitize_user_input,
    validate_and_sanitize_requirements,
)
from .ai_service import GeminiService, get_api_key
from .evaluator import evaluate_recommendation_system, TEST_SCENARIOS

__all__ = [
    "recommend_laptops",
    "calculate_performance_score",
    "get_alternative_recommendations",
    "run_what_if_analysis",
    "calculate_recommendation_confidence",
    "generate_detailed_explanation",
    "UserRequirements",
    "parse_requirements_rule_based",
    "update_requirements_state",
    "sanitize_user_input",
    "validate_and_sanitize_requirements",
    "GeminiService",
    "get_api_key",
    "evaluate_recommendation_system",
    "TEST_SCENARIOS",
]
