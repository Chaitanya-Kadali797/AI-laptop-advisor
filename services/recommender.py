"""
services/recommender.py
Rule-based Multi-Criteria Decision Analysis (MCDA) recommendation engine for laptop selection.
Phase 3 Enhanced: Includes requirement sub-scores, transparent recommendation confidence,
plain-English explainability, alternative recommendations, and What-If analysis.
"""

import pandas as pd
from typing import Optional, Dict, Any, Tuple


# Weight profiles for each primary usage category
USAGE_WEIGHTS = {
    "Gaming": {
        "gaming": 0.55,
        "programming": 0.15,
        "ai_ml": 0.10,
        "video_editing": 0.20,
    },
    "Programming / Coding": {
        "gaming": 0.10,
        "programming": 0.50,
        "ai_ml": 0.25,
        "video_editing": 0.15,
    },
    "AI / Data Science": {
        "gaming": 0.10,
        "programming": 0.25,
        "ai_ml": 0.50,
        "video_editing": 0.15,
    },
    "Video Editing / Content Creation": {
        "gaming": 0.15,
        "programming": 0.15,
        "ai_ml": 0.15,
        "video_editing": 0.55,
    },
    "General College Use / Everyday": {
        "gaming": 0.20,
        "programming": 0.30,
        "ai_ml": 0.20,
        "video_editing": 0.30,
    },
}


def calculate_performance_score(row: pd.Series, primary_usage: str) -> float:
    """
    Computes a weighted domain capability score out of 100
    based on the user's primary intended usage.
    """
    weights = USAGE_WEIGHTS.get(primary_usage, USAGE_WEIGHTS["General College Use / Everyday"])
    score = (
        row["gaming_score"] * weights["gaming"]
        + row["programming_score"] * weights["programming"]
        + row["ai_ml_score"] * weights["ai_ml"]
        + row["video_editing_score"] * weights["video_editing"]
    )
    return round(score, 1)


def generate_recommendation_reason(row: pd.Series, primary_usage: str) -> str:
    """
    Generates human-readable explanation bullet points of why this laptop was recommended.
    Crucial for college viva explanation.
    """
    reasons = []

    # Domain specific highlight
    if primary_usage == "Gaming":
        reasons.append(f"Strong Gaming Rating ({int(row['gaming_score'])}/100) with {row['gpu']}")
    elif primary_usage == "AI / Data Science":
        reasons.append(f"High AI/ML Compute Score ({int(row['ai_ml_score'])}/100) powered by {row['processor']}")
    elif primary_usage == "Programming / Coding":
        reasons.append(f"Optimal Coding Efficiency ({int(row['programming_score'])}/100) with {row['ram']}GB RAM")
    elif primary_usage == "Video Editing / Content Creation":
        reasons.append(f"Render Score ({int(row['video_editing_score'])}/100) & High-res {row['display'].split('(')[0]}")
    else:
        reasons.append(f"Balanced performance across everyday tasks ({int(row['programming_score'])}/100 coding)")

    # Form factor or battery highlight
    if row["weight"] <= 1.4:
        reasons.append(f"Ultralight portable design ({row['weight']} kg)")
    elif row["gpu_type"] == "Dedicated":
        reasons.append(f"Dedicated {row['gpu']} Graphics")

    return " • ".join(reasons)


def generate_detailed_explanation(
    row: pd.Series,
    budget: float,
    primary_usage: str,
    min_ram: int,
    gpu_requirement: str,
) -> str:
    """
    Generates a clear, student-friendly plain-English narrative explanation
    justifying why the specific laptop was selected.
    """
    parts = []
    price_str = f"₹{int(row['price']):,}"
    budget_str = f"₹{int(budget):,}"
    diff = budget - row['price']

    # Budget aspect
    if diff >= 5000:
        parts.append(f"Recommended because it is comfortably within your {budget_str} budget at {price_str} (saving ₹{int(diff):,})")
    else:
        parts.append(f"Recommended because it maximizes your {budget_str} budget at {price_str}")

    # RAM aspect
    if row["ram"] > min_ram:
        parts.append(f"provides {row['ram']}GB RAM exceeding your {min_ram}GB minimum requirement")
    else:
        parts.append(f"meets your {row['ram']}GB RAM requirement")

    # GPU aspect
    if row["gpu_type"] == "Dedicated":
        parts.append(f"includes dedicated {row['gpu']} graphics")
    else:
        parts.append(f"features power-efficient {row['gpu']} integrated graphics")

    # Workload score aspect
    if primary_usage == "AI / Data Science":
        parts.append(f"and delivers a high AI/ML benchmark score of {int(row['ai_ml_score'])}/100 and coding score of {int(row['programming_score'])}/100.")
    elif primary_usage == "Gaming":
        parts.append(f"and achieves a strong gaming benchmark rating of {int(row['gaming_score'])}/100.")
    elif primary_usage == "Programming / Coding":
        parts.append(f"and delivers an outstanding programming score of {int(row['programming_score'])}/100 with smooth multitasking.")
    elif primary_usage == "Video Editing / Content Creation":
        parts.append(f"and offers a video editing score of {int(row['video_editing_score'])}/100 with a {row['display'].split('(')[0].strip()} display.")
    else:
        parts.append(f"and provides well-balanced daily compute performance ({int(row['programming_score'])}/100).")

    return ", ".join(parts[:3]) + ", " + parts[-1]


def get_portability_suitability(weight: float) -> str:
    """Classifies physical form factor portability."""
    if weight <= 1.3:
        return f"Ultralight ({weight} kg) — Ideal for daily backpack carry"
    elif weight <= 1.7:
        return f"Portable ({weight} kg) — Standard student balance"
    elif weight <= 2.2:
        return f"Mainstream ({weight} kg) — Moderate carry weight"
    else:
        return f"Heavy Workstation ({weight} kg) — Best for desk setup"


def get_battery_suitability(battery_str: str) -> str:
    """Classifies battery endurance from specification text."""
    lower = battery_str.lower()
    if "18 hrs" in lower or "20 hrs" in lower or "22 hrs" in lower or "24 hrs" in lower:
        return f"{battery_str} — Exceptional all-day endurance"
    elif "12 hrs" in lower or "14 hrs" in lower or "15 hrs" in lower or "16 hrs" in lower:
        return f"{battery_str} — Full college day battery"
    elif "8 hrs" in lower or "9 hrs" in lower or "10 hrs" in lower:
        return f"{battery_str} — Good workday battery"
    else:
        return f"{battery_str} — Performance focused (carry charger)"


def calculate_recommendation_confidence(
    match_score: float,
    budget_specified: bool = True,
    usage_specified: bool = True,
    ram_specified: bool = True,
    gpu_specified: bool = True,
) -> Tuple[float, str]:
    """
    Computes a transparent Recommendation Confidence index (0–100%).
    NOT an ML probability: A deterministic metric combining input requirement
    completeness and candidate match fit.
    """
    completeness_score = 0.0
    if budget_specified:
        completeness_score += 30.0
    if usage_specified:
        completeness_score += 30.0
    if ram_specified:
        completeness_score += 20.0
    if gpu_specified:
        completeness_score += 20.0

    # Confidence is 40% input completeness + 60% candidate match quality
    conf = (0.40 * completeness_score) + (0.60 * match_score)
    confidence_val = min(max(round(conf, 1), 25.0), 98.5)

    if confidence_val >= 85.0:
        label = "High Confidence (Complete constraints & strong candidate fit)"
    elif confidence_val >= 70.0:
        label = "Moderate Confidence (Good candidate match with minor trade-offs)"
    else:
        label = "Preliminary Confidence (Partial requirements or constraint pressure)"

    return confidence_val, label


def recommend_laptops(
    df: pd.DataFrame,
    budget: float,
    primary_usage: str,
    min_ram: int = 8,
    gpu_requirement: str = "Any",
    preferred_brand: str = "All Brands",
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Filters and ranks laptops based on user constraints and calculates
    comprehensive Phase 3 evaluation metrics:
    - match_score (composite 0-100%)
    - budget_match_pct, ram_match_pct, gpu_match_pct, usage_match_pct
    - confidence_score & confidence_label
    - detailed_explanation narrative
    - portability and battery suitability descriptions
    """
    if df.empty:
        return pd.DataFrame()

    # Sanitize inputs gracefully for edge cases
    try:
        budget = float(budget)
        if budget <= 0 or pd.isna(budget):
            budget = float(df["price"].min())
    except (ValueError, TypeError):
        budget = float(df["price"].min())

    try:
        min_ram = int(min_ram)
    except (ValueError, TypeError):
        min_ram = 8

    filtered = df.copy()

    # 1. Budget Filter
    filtered = filtered[filtered["price"] <= budget]

    # 2. Minimum RAM Filter
    filtered = filtered[filtered["ram"] >= min_ram]

    # 3. GPU Requirement Filter
    if gpu_requirement == "Dedicated GPU":
        filtered = filtered[filtered["gpu_type"] == "Dedicated"]
    elif gpu_requirement == "Integrated GPU (Thin & Light / Battery)":
        filtered = filtered[filtered["gpu_type"] == "Integrated"]

    # 4. Brand Preference Filter
    if preferred_brand and preferred_brand != "All Brands":
        filtered = filtered[filtered["brand"].str.lower() == preferred_brand.lower()]

    if filtered.empty:
        return pd.DataFrame()

    # Calculate Phase 3 scores
    def compute_match_metrics(row):
        domain_score = calculate_performance_score(row, primary_usage)

        overall_avg = (
            row["gaming_score"]
            + row["programming_score"]
            + row["ai_ml_score"]
            + row["video_editing_score"]
        ) / 4.0

        price_ratio = row["price"] / max(budget, 1)
        budget_factor = 90.0 if price_ratio <= 1.0 else 50.0
        ram_bonus = min(row["ram"] / 32.0 * 100.0, 100.0)

        # Composite match score (0-100)
        final_score = (
            (domain_score * 0.55)
            + (overall_avg * 0.20)
            + (budget_factor * 0.15)
            + (ram_bonus * 0.10)
        )
        match_score = min(max(round(final_score, 1), 10.0), 99.5)

        # Phase 3 Sub-scores
        budget_match_pct = 100.0 if row["price"] <= budget else max(0.0, round((1.0 - (row["price"] - budget) / budget) * 100.0, 1))
        ram_match_pct = 100.0 if row["ram"] >= min_ram else round((row["ram"] / min_ram) * 100.0, 1)
        
        gpu_match_pct = 100.0
        if gpu_requirement == "Dedicated GPU" and row["gpu_type"] != "Dedicated":
            gpu_match_pct = 40.0
        elif gpu_requirement == "Integrated GPU (Thin & Light / Battery)" and row["gpu_type"] != "Integrated":
            gpu_match_pct = 40.0

        usage_match_pct = domain_score

        # Confidence calculation
        confidence_score, confidence_label = calculate_recommendation_confidence(
            match_score=match_score,
            budget_specified=budget is not None and budget > 0,
            usage_specified=primary_usage is not None,
            ram_specified=min_ram is not None,
            gpu_specified=gpu_requirement is not None and gpu_requirement != "Any",
        )

        reason = generate_recommendation_reason(row, primary_usage)
        detailed_exp = generate_detailed_explanation(row, budget, primary_usage, min_ram, gpu_requirement)
        portability_info = get_portability_suitability(row["weight"])
        battery_info = get_battery_suitability(row["battery"])

        return pd.Series([
            match_score,
            domain_score,
            overall_avg,
            budget_match_pct,
            ram_match_pct,
            gpu_match_pct,
            usage_match_pct,
            confidence_score,
            confidence_label,
            reason,
            detailed_exp,
            portability_info,
            battery_info,
        ])

    metrics_df = filtered.apply(compute_match_metrics, axis=1)
    metrics_df.columns = [
        "match_score",
        "domain_score",
        "overall_performance_score",
        "budget_match_pct",
        "ram_match_pct",
        "gpu_match_pct",
        "usage_match_pct",
        "confidence_score",
        "confidence_label",
        "recommendation_reason",
        "detailed_explanation",
        "portability_suitability",
        "battery_suitability",
    ]

    result = pd.concat([filtered, metrics_df], axis=1)
    result = result.sort_values(by=["match_score", "price"], ascending=[False, True])

    return result.head(top_n)


def get_alternative_recommendations(
    df: pd.DataFrame,
    budget: float,
    primary_usage: str,
    min_ram: int = 8,
    gpu_requirement: str = "Any",
    preferred_brand: str = "All Brands",
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Identifies standout alternative laptops across multiple decision axes:
    - Best Overall: Top match score
    - Best Budget Option: Lowest price with acceptable score
    - Best Performance Option: Highest domain score
    - Best for AI/ML: Highest AI/ML score
    - Best for Gaming: Highest gaming score
    - Best for Programming: Highest programming score
    """
    recs = recommend_laptops(
        df=df,
        budget=budget,
        primary_usage=primary_usage,
        min_ram=min_ram,
        gpu_requirement=gpu_requirement,
        preferred_brand=preferred_brand,
        top_n=20,
    )

    if recs.empty:
        return {}

    alternatives = {}

    # 1. Best Overall
    alternatives["Best Overall"] = recs.iloc[0].to_dict()

    # 2. Best Budget Option (lowest price with score >= 65)
    budget_candidates = recs[recs["match_score"] >= 65.0].sort_values(by="price", ascending=True)
    if not budget_candidates.empty:
        alternatives["Best Budget Option"] = budget_candidates.iloc[0].to_dict()
    else:
        alternatives["Best Budget Option"] = recs.sort_values(by="price", ascending=True).iloc[0].to_dict()

    # 3. Best Performance Option (highest domain score)
    best_perf = recs.sort_values(by="domain_score", ascending=False).iloc[0].to_dict()
    alternatives["Best Performance Option"] = best_perf

    # 4. Best for AI/ML
    best_ai = recs.sort_values(by="ai_ml_score", ascending=False).iloc[0].to_dict()
    alternatives["Best for AI/ML"] = best_ai

    # 5. Best for Gaming
    best_game = recs.sort_values(by="gaming_score", ascending=False).iloc[0].to_dict()
    alternatives["Best for Gaming"] = best_game

    # 6. Best for Programming
    best_code = recs.sort_values(by="programming_score", ascending=False).iloc[0].to_dict()
    alternatives["Best for Programming"] = best_code

    return alternatives


def run_what_if_analysis(
    df: pd.DataFrame,
    base_params: Dict[str, Any],
    modified_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Executes a side-by-side What-If analysis comparing recommendations
    under base parameters versus modified parameters.
    """
    base_recs = recommend_laptops(
        df=df,
        budget=base_params.get("budget", 80000),
        primary_usage=base_params.get("primary_usage", "Programming / Coding"),
        min_ram=base_params.get("min_ram", 16),
        gpu_requirement=base_params.get("gpu_requirement", "Any"),
        preferred_brand=base_params.get("preferred_brand", "All Brands"),
        top_n=3,
    )

    mod_recs = recommend_laptops(
        df=df,
        budget=modified_params.get("budget", 80000),
        primary_usage=modified_params.get("primary_usage", "Programming / Coding"),
        min_ram=modified_params.get("min_ram", 16),
        gpu_requirement=modified_params.get("gpu_requirement", "Any"),
        preferred_brand=modified_params.get("preferred_brand", "All Brands"),
        top_n=3,
    )

    base_top = base_recs.iloc[0].to_dict() if not base_recs.empty else None
    mod_top = mod_recs.iloc[0].to_dict() if not mod_recs.empty else None

    # Calculate parameter delta description
    diff_notes = []
    if base_params.get("budget") != modified_params.get("budget"):
        diff = modified_params.get("budget", 0) - base_params.get("budget", 0)
        sign = "+" if diff > 0 else ""
        diff_notes.append(f"Budget changed by {sign}₹{int(diff):,} (₹{int(base_params.get('budget', 0)):,} → ₹{int(modified_params.get('budget', 0)):,})")

    if base_params.get("min_ram") != modified_params.get("min_ram"):
        diff_notes.append(f"RAM constraint: {base_params.get('min_ram')}GB → {modified_params.get('min_ram')}GB")

    if base_params.get("gpu_requirement") != modified_params.get("gpu_requirement"):
        diff_notes.append(f"GPU requirement: {base_params.get('gpu_requirement')} → {modified_params.get('gpu_requirement')}")

    if base_params.get("primary_usage") != modified_params.get("primary_usage"):
        diff_notes.append(f"Primary usage: {base_params.get('primary_usage')} → {modified_params.get('primary_usage')}")

    if base_params.get("preferred_brand") != modified_params.get("preferred_brand"):
        diff_notes.append(f"Brand: {base_params.get('preferred_brand')} → {modified_params.get('preferred_brand')}")

    # Top pick comparison narrative
    verdict = ""
    if base_top and mod_top:
        price_diff = mod_top["price"] - base_top["price"]
        score_diff = mod_top["match_score"] - base_top["match_score"]
        same_model = base_top["model"] == mod_top["model"]

        if same_model:
            verdict = f"The top recommendation remains **{mod_top['brand']} {mod_top['model']}**, retaining its #1 rank."
        else:
            p_sign = "+" if price_diff > 0 else ""
            s_sign = "+" if score_diff > 0 else ""
            verdict = (
                f"Modifying constraints shifts your top match from **{base_top['brand']} {base_top['model']}** "
                f"to **{mod_top['brand']} {mod_top['model']}** ({p_sign}₹{int(price_diff):,} price change, "
                f"{s_sign}{score_diff:.1f}% match score delta)."
            )

    return {
        "base_recs": base_recs.to_dict(orient="records") if not base_recs.empty else [],
        "mod_recs": mod_recs.to_dict(orient="records") if not mod_recs.empty else [],
        "base_top": base_top,
        "mod_top": mod_top,
        "parameter_diffs": diff_notes,
        "verdict": verdict,
    }
