"""Database package for AI Laptop Advisor."""
from .db_manager import init_db, get_all_laptops, get_unique_brands, get_budget_bounds

__all__ = ["init_db", "get_all_laptops", "get_unique_brands", "get_budget_bounds"]
