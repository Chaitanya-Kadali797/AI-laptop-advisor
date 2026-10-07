"""
services/evaluator.py
Evaluation & Testing Module (Phase 3).
Executes a battery of 24 rigorous predefined user scenarios covering diverse workloads,
budget limits, hardware configurations, brand affinities, and edge cases.
Calculates quantitative metrics: Budget Compliance, RAM Satisfaction, GPU Compliance, and Availability.
"""

import pandas as pd
from typing import Dict, List, Any
from services.recommender import recommend_laptops


TEST_SCENARIOS: List[Dict[str, Any]] = [
    {
        "id": "TC-01",
        "name": "First-Year CSE Student",
        "category": "Programming",
        "budget": 60000,
        "primary_usage": "Programming / Coding",
        "min_ram": 16,
        "gpu_requirement": "Any",
        "preferred_brand": "All Brands",
        "is_feasible": True,
    },
    {
        "id": "TC-02",
        "name": "Deep Learning / AI Researcher",
        "category": "AI/ML",
        "budget": 150000,
        "primary_usage": "AI / Data Science",
        "min_ram": 32,
        "gpu_requirement": "Dedicated GPU",
        "preferred_brand": "All Brands",
        "is_feasible": True,
    },
    {
        "id": "TC-03",
        "name": "Competitive Esports Gamer",
        "category": "Gaming",
        "budget": 120000,
        "primary_usage": "Gaming",
        "min_ram": 16,
        "gpu_requirement": "Dedicated GPU",
        "preferred_brand": "All Brands",
        "is_feasible": True,
    },
    {
        "id": "TC-04",
        "name": "Professional 4K Video Editor",
        "category": "Video Editing",
        "budget": 160000,
        "primary_usage": "Video Editing / Content Creation",
        "min_ram": 32,
        "gpu_requirement": "Dedicated GPU",
        "preferred_brand": "All Brands",
        "is_feasible": True,
    },
    {
        "id": "TC-05",
        "name": "Ultra-Budget Student Daily",
        "category": "Budget",
        "budget": 35000,
        "primary_usage": "General College Use / Everyday",
        "min_ram": 8,
        "gpu_requirement": "Any",
        "preferred_brand": "All Brands",
        "is_feasible": True,
    },
    {
        "id": "TC-06",
        "name": "Premium Ultrabook Executive (Apple)",
        "category": "Premium",
        "budget": 180000,
        "primary_usage": "Programming / Coding",
        "min_ram": 16,
        "gpu_requirement": "Integrated GPU (Thin & Light / Battery)",
        "preferred_brand": "Apple",
        "is_feasible": True,
    },
    {
        "id": "TC-07",
        "name": "Mid-Range Coding + Occasional Gaming (Asus)",
        "category": "Programming",
        "budget": 75000,
        "primary_usage": "Programming / Coding",
        "min_ram": 16,
        "gpu_requirement": "Dedicated GPU",
        "preferred_brand": "Asus",
        "is_feasible": True,
    },
    {
        "id": "TC-08",
        "name": "Entry AI/ML on a Tight Budget",
        "category": "AI/ML",
        "budget": 70000,
        "primary_usage": "AI / Data Science",
        "min_ram": 16,
        "gpu_requirement": "Dedicated GPU",
        "preferred_brand": "All Brands",
        "is_feasible": True,
    },
    {
        "id": "TC-09",
        "name": "Content Creator on Lenovo",
        "category": "Video Editing",
        "budget": 120000,
        "primary_usage": "Video Editing / Content Creation",
        "min_ram": 16,
        "gpu_requirement": "Dedicated GPU",
        "preferred_brand": "Lenovo",
        "is_feasible": True,
    },
    {
        "id": "TC-10",
        "name": "Lightweight Campus Developer (HP)",
        "category": "Programming",
        "budget": 90000,
        "primary_usage": "Programming / Coding",
        "min_ram": 16,
        "gpu_requirement": "Any",
        "preferred_brand": "HP",
        "is_feasible": True,
    },
    {
        "id": "TC-11",
        "name": "High-Performance Workstation (Dell)",
        "category": "AI/ML",
        "budget": 250000,
        "primary_usage": "AI / Data Science",
        "min_ram": 32,
        "gpu_requirement": "Dedicated GPU",
        "preferred_brand": "Dell",
        "is_feasible": True,
    },
    {
        "id": "TC-12",
        "name": "Budget Esports Gamer (Acer)",
        "category": "Gaming",
        "budget": 65000,
        "primary_usage": "Gaming",
        "min_ram": 16,
        "gpu_requirement": "Dedicated GPU",
        "preferred_brand": "Acer",
        "is_feasible": True,
    },
    {
        "id": "TC-13",
        "name": "Silent Battery-Focused Coder",
        "category": "Programming",
        "budget": 100000,
        "primary_usage": "Programming / Coding",
        "min_ram": 16,
        "gpu_requirement": "Integrated GPU (Thin & Light / Battery)",
        "preferred_brand": "All Brands",
        "is_feasible": True,
    },
    {
        "id": "TC-14",
        "name": "MSI Dedicated Gamer",
        "category": "Gaming",
        "budget": 90000,
        "primary_usage": "Gaming",
        "min_ram": 16,
        "gpu_requirement": "Dedicated GPU",
        "preferred_brand": "MSI",
        "is_feasible": True,
    },
    {
        "id": "TC-15",
        "name": "Apple Silicon Swift Developer",
        "category": "Programming",
        "budget": 110000,
        "primary_usage": "Programming / Coding",
        "min_ram": 16,
        "gpu_requirement": "Any",
        "preferred_brand": "Apple",
        "is_feasible": True,
    },
    {
        "id": "TC-16",
        "name": "Samsung AMOLED Productivity",
        "category": "General",
        "budget": 70000,
        "primary_usage": "General College Use / Everyday",
        "min_ram": 16,
        "gpu_requirement": "Integrated GPU (Thin & Light / Battery)",
        "preferred_brand": "Samsung",
        "is_feasible": True,
    },
    {
        "id": "TC-17",
        "name": "Extreme Compute Multitasker",
        "category": "AI/ML",
        "budget": 350000,
        "primary_usage": "AI / Data Science",
        "min_ram": 32,
        "gpu_requirement": "Dedicated GPU",
        "preferred_brand": "All Brands",
        "is_feasible": True,
    },
    {
        "id": "TC-18",
        "name": "LG Gram Ultralight Campus Commuter",
        "category": "Programming",
        "budget": 125000,
        "primary_usage": "Programming / Coding",
        "min_ram": 16,
        "gpu_requirement": "Integrated GPU (Thin & Light / Battery)",
        "preferred_brand": "LG",
        "is_feasible": True,
    },
    {
        "id": "TC-19",
        "name": "Gigabyte RTX Content Creator",
        "category": "Video Editing",
        "budget": 120000,
        "primary_usage": "Video Editing / Content Creation",
        "min_ram": 16,
        "gpu_requirement": "Dedicated GPU",
        "preferred_brand": "Gigabyte",
        "is_feasible": True,
    },
    {
        "id": "TC-20",
        "name": "Sub-50k Student Productivity",
        "category": "Budget",
        "budget": 50000,
        "primary_usage": "General College Use / Everyday",
        "min_ram": 8,
        "gpu_requirement": "Any",
        "preferred_brand": "All Brands",
        "is_feasible": True,
    },
    # --- EDGE CASES (Impossible / Hostile Inputs) ---
    {
        "id": "TC-21",
        "name": "Edge Case: Impossible 18k Dedicated GPU",
        "category": "Edge Case",
        "budget": 18000,
        "primary_usage": "Gaming",
        "min_ram": 8,
        "gpu_requirement": "Dedicated GPU",
        "preferred_brand": "All Brands",
        "is_feasible": False,
    },
    {
        "id": "TC-22",
        "name": "Edge Case: 32GB RAM Under 40k",
        "category": "Edge Case",
        "budget": 40000,
        "primary_usage": "Programming / Coding",
        "min_ram": 32,
        "gpu_requirement": "Any",
        "preferred_brand": "All Brands",
        "is_feasible": False,
    },
    {
        "id": "TC-23",
        "name": "Edge Case: Unsupported Brand (Sony)",
        "category": "Edge Case",
        "budget": 80000,
        "primary_usage": "Programming / Coding",
        "min_ram": 16,
        "gpu_requirement": "Any",
        "preferred_brand": "Sony",
        "is_feasible": False,
    },
    {
        "id": "TC-24",
        "name": "Edge Case: Negative Budget (-5000)",
        "category": "Edge Case",
        "budget": -5000,
        "primary_usage": "Programming / Coding",
        "min_ram": 16,
        "gpu_requirement": "Any",
        "preferred_brand": "All Brands",
        "is_feasible": False,
    },
]


def evaluate_recommendation_system(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Executes the full evaluation battery across all 24 scenarios.
    Calculates quantitative reliability metrics.
    """
    results = []
    feasible_count = 0
    feasible_success_count = 0

    total_recommendations_checked = 0
    budget_compliant_count = 0
    ram_compliant_count = 0
    gpu_compliant_count = 0

    total_scores = []
    total_confidences = []

    for sc in TEST_SCENARIOS:
        recs = recommend_laptops(
            df=df,
            budget=sc["budget"],
            primary_usage=sc["primary_usage"],
            min_ram=sc["min_ram"],
            gpu_requirement=sc["gpu_requirement"],
            preferred_brand=sc["preferred_brand"],
            top_n=5,
        )

        match_count = len(recs)
        is_feasible = sc["is_feasible"]

        if is_feasible:
            feasible_count += 1
            if match_count > 0:
                feasible_success_count += 1

        # Check compliance on returned laptops
        budget_target = max(sc["budget"], float(df["price"].min())) if sc["budget"] > 0 else float(df["price"].min())
        all_budget_ok = True
        all_ram_ok = True
        all_gpu_ok = True

        for _, row in recs.iterrows():
            total_recommendations_checked += 1

            # Budget test
            if row["price"] <= budget_target:
                budget_compliant_count += 1
            else:
                all_budget_ok = False

            # RAM test
            if row["ram"] >= sc["min_ram"]:
                ram_compliant_count += 1
            else:
                all_ram_ok = False

            # GPU test
            if sc["gpu_requirement"] == "Dedicated GPU":
                if row["gpu_type"] == "Dedicated":
                    gpu_compliant_count += 1
                else:
                    all_gpu_ok = False
            elif sc["gpu_requirement"] == "Integrated GPU (Thin & Light / Battery)":
                if row["gpu_type"] == "Integrated":
                    gpu_compliant_count += 1
                else:
                    all_gpu_ok = False
            else:
                gpu_compliant_count += 1

            total_scores.append(row["match_score"])
            total_confidences.append(row["confidence_score"])

        top_laptop = recs.iloc[0]["brand"] + " " + recs.iloc[0]["model"] if not recs.empty else "No Candidates"
        top_score = recs.iloc[0]["match_score"] if not recs.empty else 0.0

        results.append({
            "Scenario ID": sc["id"],
            "Scenario Name": sc["name"],
            "Category": sc["category"],
            "Budget": sc["budget"],
            "Min RAM": sc["min_ram"],
            "GPU Req": sc["gpu_requirement"],
            "Matches Found": match_count,
            "Top Pick": top_laptop,
            "Top Match Score": top_score,
            "Passed Constraints": (all_budget_ok and all_ram_ok and all_gpu_ok),
        })

    # Summary Metrics Calculation
    results_df = pd.DataFrame(results)

    budget_compliance_rate = (budget_compliant_count / total_recommendations_checked * 100.0) if total_recommendations_checked > 0 else 100.0
    ram_compliance_rate = (ram_compliant_count / total_recommendations_checked * 100.0) if total_recommendations_checked > 0 else 100.0
    gpu_compliance_rate = (gpu_compliant_count / total_recommendations_checked * 100.0) if total_recommendations_checked > 0 else 100.0
    availability_rate = (feasible_success_count / feasible_count * 100.0) if feasible_count > 0 else 100.0

    avg_score = sum(total_scores) / len(total_scores) if total_scores else 0.0
    avg_confidence = sum(total_confidences) / len(total_confidences) if total_confidences else 0.0

    return {
        "total_scenarios": len(TEST_SCENARIOS),
        "feasible_scenarios": feasible_count,
        "edge_case_scenarios": len(TEST_SCENARIOS) - feasible_count,
        "recommendations_checked": total_recommendations_checked,
        "budget_compliance_rate": round(budget_compliance_rate, 2),
        "ram_compliance_rate": round(ram_compliance_rate, 2),
        "gpu_compliance_rate": round(gpu_compliance_rate, 2),
        "availability_rate": round(availability_rate, 2),
        "average_match_score": round(avg_score, 1),
        "average_confidence": round(avg_confidence, 1),
        "scenario_results_df": results_df,
    }
