"""
test_chatbot.py
Production-Quality Automated Test Suite (Phase 4).
Covers 14 AI-specific test cases verifying:
- Natural budget expressions ("around ₹75k", "under 80 thousand", "1.5 lakh")
- Gaming + Programming dual workload
- AI/ML & Deep Learning requirements
- Missing budget detection & clarification questions
- Brand preferences (Asus, Apple, Lenovo)
- 16GB and 32GB RAM thresholds
- Dedicated GPU & preferred GPU model extraction (RTX 4060)
- Conversational follow-up chains
- Multi-model comparison intents
- Invalid & extreme requirements sanitization
- Zero-match handling
- Grounding & Anti-hallucination verification
- Fallback engine resilience & active engine status
"""

import sys
import io
import pandas as pd

# Ensure UTF-8 stdout on Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from database.db_manager import init_db, get_all_laptops
from services.requirement_parser import (
    UserRequirements,
    parse_requirements_rule_based,
    update_requirements_state,
    validate_and_sanitize_requirements,
)
from services.recommender import recommend_laptops
from services.ai_service import GeminiService


def run_tests():
    print("=" * 80)
    print("🚀 RUNNING PHASE 4 PRODUCTION-QUALITY AI & CHATBOT TEST BATTERY")
    print("=" * 80)

    # Test 0: Database integrity
    init_db()
    df_all = get_all_laptops()
    assert len(df_all) >= 120, f"Expected >= 120 laptops, got {len(df_all)}"
    print(f"✅ [Test 0] Database verified: {len(df_all)} laptops available.")

    ai_service = GeminiService()
    print(f"ℹ️ Engine Status: {ai_service.get_active_engine_name()} (Available: {ai_service.is_available()})")

    # TEST 1: Natural budget expression: "around ₹75k"
    print("\n" + "-" * 80)
    print("🧪 [Test 1] Natural Budget Expression: 'around ₹75k'")
    q1 = "I'm a CSE student. My budget is around ₹75k. Need a laptop for programming and machine learning with 16GB RAM."
    res1 = parse_requirements_rule_based(q1, {})
    assert res1.get("maximum_budget") == 75000.0, f"Expected 75000.0, got {res1.get('maximum_budget')}"
    assert res1.get("minimum_ram") == 16, f"Expected 16, got {res1.get('minimum_ram')}"
    assert res1.get("primary_usage") in ["Programming / Coding", "AI / Data Science"]
    print(f"   -> Extracted Budget: ₹{res1.get('maximum_budget'):,}, RAM: {res1.get('minimum_ram')}GB, Usage: {res1.get('primary_usage')}")
    print("✅ [Test 1] PASSED")

    # TEST 2: Natural budget expression: "under 80 thousand"
    print("\n" + "-" * 80)
    print("🧪 [Test 2] Natural Budget Expression: 'under 80 thousand'")
    q2 = "Looking for a fast coding machine under 80 thousand with good battery backup"
    res2 = parse_requirements_rule_based(q2, {})
    assert res2.get("maximum_budget") == 80000.0, f"Expected 80000.0, got {res2.get('maximum_budget')}"
    assert res2.get("battery_preference") == "High Battery Life", f"Expected High Battery Life, got {res2.get('battery_preference')}"
    print(f"   -> Extracted Budget: ₹{res2.get('maximum_budget'):,}, Battery: {res2.get('battery_preference')}")
    print("✅ [Test 2] PASSED")

    # TEST 3: Gaming + Programming dual workload
    print("\n" + "-" * 80)
    print("🧪 [Test 3] Dual Workload: Gaming + Programming")
    q3 = "Need a powerful machine for programming and gaming with at least 16GB RAM and a dedicated graphics card"
    res3 = parse_requirements_rule_based(q3, {})
    assert res3.get("primary_usage") in ["Programming / Coding", "Gaming"]
    assert res3.get("secondary_usage") in ["Programming / Coding", "Gaming"]
    assert res3.get("gpu_required") == "Dedicated GPU"
    print(f"   -> Primary: {res3.get('primary_usage')}, Secondary: {res3.get('secondary_usage')}, GPU: {res3.get('gpu_required')}")
    print("✅ [Test 3] PASSED")

    # TEST 4: AI/ML & Deep Learning requirements
    print("\n" + "-" * 80)
    print("🧪 [Test 4] AI/ML & Data Science Requirements")
    q4 = "I am training deep learning models and working with machine learning and data science. Budget 1.2 lakh."
    res4 = parse_requirements_rule_based(q4, {})
    assert res4.get("primary_usage") == "AI / Data Science", f"Expected AI / Data Science, got {res4.get('primary_usage')}"
    assert res4.get("maximum_budget") == 120000.0, f"Expected 120000.0, got {res4.get('maximum_budget')}"
    print(f"   -> Usage: {res4.get('primary_usage')}, Budget: ₹{res4.get('maximum_budget'):,}")
    print("✅ [Test 4] PASSED")

    # TEST 5: Missing budget detection & clarification trigger
    print("\n" + "-" * 80)
    print("🧪 [Test 5] Missing Budget Detection & Clarification Trigger")
    q5 = "I need a good laptop for college coding"
    res5 = parse_requirements_rule_based(q5, {})
    assert "maximum_budget" in res5.get("missing_critical_info", []), "Should flag missing maximum_budget"
    clarification = ai_service.generate_clarification(res5["missing_critical_info"])
    assert "budget" in clarification.lower()
    print(f"   -> Clarification Prompt: {clarification.splitlines()[0]}")
    print("✅ [Test 5] PASSED")

    # TEST 6: Brand preference extraction
    print("\n" + "-" * 80)
    print("🧪 [Test 6] Brand Preference Extraction (Apple, Asus, Lenovo)")
    q6_apple = "Looking for an Apple MacBook for software development"
    res6_apple = parse_requirements_rule_based(q6_apple, {})
    assert res6_apple.get("preferred_brand") == "Apple"

    q6_asus = "Show me Asus laptops under 70k"
    res6_asus = parse_requirements_rule_based(q6_asus, {})
    assert res6_asus.get("preferred_brand") == "Asus"
    assert res6_asus.get("maximum_budget") == 70000.0
    print(f"   -> Brands Detected: Apple={res6_apple.get('preferred_brand')}, Asus={res6_asus.get('preferred_brand')}")
    print("✅ [Test 6] PASSED")

    # TEST 7: 16GB and 32GB RAM thresholds
    print("\n" + "-" * 80)
    print("🧪 [Test 7] RAM Thresholds: 16GB & 32GB")
    q7_16 = "I need at least 16GB RAM for multitasking"
    res7_16 = parse_requirements_rule_based(q7_16, {})
    assert res7_16.get("minimum_ram") == 16

    q7_32 = "Must have 32GB RAM for large Docker containers"
    res7_32 = parse_requirements_rule_based(q7_32, {})
    assert res7_32.get("minimum_ram") == 32
    print(f"   -> Extracted RAM: {res7_16.get('minimum_ram')}GB, {res7_32.get('minimum_ram')}GB")
    print("✅ [Test 7] PASSED")

    # TEST 8: Dedicated GPU & Preferred GPU Model (RTX 4060)
    print("\n" + "-" * 80)
    print("🧪 [Test 8] Dedicated GPU & Preferred Model (RTX 4060)")
    q8 = "I want a laptop with an RTX 4060 graphics card and 16GB RAM under 95000"
    res8 = parse_requirements_rule_based(q8, {})
    assert res8.get("gpu_required") == "Dedicated GPU"
    assert res8.get("preferred_gpu") == "RTX 4060"
    assert res8.get("maximum_budget") == 95000.0
    print(f"   -> GPU Required: {res8.get('gpu_required')}, Model: {res8.get('preferred_gpu')}")
    print("✅ [Test 8] PASSED")

    # TEST 9: Conversational Follow-up Chain
    print("\n" + "-" * 80)
    print("🧪 [Test 9] Conversational Follow-up Chain (Cheaper -> Brand -> Integrated)")
    state9 = {"maximum_budget": 80000.0, "primary_usage": "Gaming", "minimum_ram": 16, "gpu_required": "Dedicated GPU", "preferred_brand": "All Brands"}
    
    # Step A: "Show me something cheaper"
    step_a = parse_requirements_rule_based("Show me something cheaper", state9)
    state9 = update_requirements_state(state9, step_a)
    assert state9["maximum_budget"] == 65000.0, f"Expected 65000, got {state9['maximum_budget']}"

    # Step B: "What about Lenovo?"
    step_b = parse_requirements_rule_based("What about Lenovo?", state9)
    state9 = update_requirements_state(state9, step_b)
    assert state9["preferred_brand"] == "Lenovo"
    assert state9["maximum_budget"] == 65000.0  # Preserved

    # Step C: "I don't need gaming anymore"
    step_c = parse_requirements_rule_based("I don't need gaming anymore", state9)
    state9 = update_requirements_state(state9, step_c)
    assert state9["gpu_required"] == "Any"
    print(f"   -> Final Follow-up State: Budget=₹{state9['maximum_budget']:,}, Brand={state9['preferred_brand']}, GPU={state9['gpu_required']}")
    print("✅ [Test 9] PASSED")

    # TEST 10: Comparison Intent & Conversational Follow-up Variations
    print("\n" + "-" * 80)
    print("🧪 [Test 10] Comparison Intent: Testing All Follow-up Phrases & Anti-Recursion Safety")
    comp_variations = [
        ("Compare the first two laptops", [1, 2]),
        ("compare the first two", [1, 2]),
        ("compare first two", [1, 2]),
        ("compare #1 and #2", [1, 2]),
        ("compare #1 and #3", [1, 3]),
        ("compare those two", [1, 2]),
        ("compare the two laptops", [1, 2]),
        ("compare 1 and 3", [1, 3]),
    ]
    for phrase, expected_indices in comp_variations:
        parsed = parse_requirements_rule_based(phrase, {})
        assert parsed.get("intent") == "comparison", f"Failed intent for '{phrase}'"
        assert parsed.get("comparison_indices") == expected_indices, f"Failed indices for '{phrase}': got {parsed.get('comparison_indices')}"
        print(f"   -> Follow-up phrase '{phrase}' correctly mapped to {expected_indices}")

    # Generate comparison between real DB models
    sample_pair = df_all.head(2).to_dict(orient="records")
    comp_text = ai_service.generate_comparison(sample_pair, "compare the first two")
    assert sample_pair[0]["model"] in comp_text or sample_pair[0]["brand"] in comp_text
    print(f"   -> Comparison generated ({len(comp_text)} chars) comparing {sample_pair[0]['brand']} and {sample_pair[1]['brand']}")

    # Verify Anti-Recursion Immunity: Mock Gemini failure during comparison
    from unittest.mock import MagicMock
    mock_service = GeminiService()
    mock_service.client = MagicMock()
    mock_service.client.models.generate_content.side_effect = Exception("Simulated Gemini API 429/Quota/Timeout")
    safe_comp = mock_service.generate_comparison(sample_pair, "compare the first two")
    assert "Side-by-Side Comparison" in safe_comp
    assert sample_pair[0]["model"] in safe_comp and sample_pair[1]["model"] in safe_comp
    print("   -> Anti-recursion immunity verified: Gemini exception gracefully handled with grounded table.")
    print("✅ [Test 10] PASSED")

    # TEST 11: Invalid & Extreme Requirements Sanitization
    print("\n" + "-" * 80)
    print("🧪 [Test 11] Invalid & Extreme Requirements Sanitization")
    dirty_req = {
        "maximum_budget": -5000.0,
        "minimum_ram": 0,
        "primary_usage": "Quantum Computing",
        "gpu_required": "Quantum Accelerator",
        "preferred_brand": "AlienwareX",
    }
    clean = validate_and_sanitize_requirements(dirty_req)
    assert clean["maximum_budget"] is None, "Negative budget must be sanitized to None"
    assert clean["minimum_ram"] == 8, "0 RAM must be sanitized to 8GB"
    assert clean["primary_usage"] == "General College Use / Everyday", "Invalid usage must default safely"
    assert clean["gpu_required"] == "Any", "Invalid GPU must default safely"
    assert clean["preferred_brand"] == "All Brands", "Invalid brand must default safely"
    print("   -> Sanitization properly repaired all invalid/extreme values.")
    print("✅ [Test 11] PASSED")

    # TEST 12: Zero Matches Handling (Impossible Budget + Dedicated GPU)
    print("\n" + "-" * 80)
    print("🧪 [Test 12] Zero Matches Handling (Impossible ₹18,000 + Dedicated GPU)")
    recs12 = recommend_laptops(
        df=df_all,
        budget=18000,
        primary_usage="Gaming",
        gpu_requirement="Dedicated GPU",
    )
    assert recs12.empty, "Expected 0 laptops matching ₹18,000 + Dedicated GPU"
    no_match_text = ai_service.generate_explanation("Budget 18000 dedicated gpu", {"maximum_budget": 18000, "gpu_required": "Dedicated GPU"}, [])
    assert "no laptops" in no_match_text.lower() or "constraints" in no_match_text.lower()
    print("   -> Graceful guidance generated without crashing.")
    print("✅ [Test 12] PASSED")

    # TEST 13: Grounding & Anti-Hallucination Verification
    print("\n" + "-" * 80)
    print("🧪 [Test 13] Grounding & Anti-Hallucination Verification")
    recs13 = recommend_laptops(
        df=df_all,
        budget=85000,
        primary_usage="Programming / Coding",
        min_ram=16,
        gpu_requirement="Dedicated GPU",
        top_n=3,
    )
    recs_list = recs13.to_dict(orient="records")
    explanation = ai_service.generate_explanation(
        user_message="Recommend laptops under 85k for coding with 16GB RAM",
        user_requirements={"maximum_budget": 85000, "primary_usage": "Programming / Coding"},
        recommended_laptops=recs_list,
    )
    # Ensure explanation cites the real recommended model from the database
    top_model = recs_list[0]["model"]
    assert top_model in explanation or recs_list[0]["brand"] in explanation, f"Explanation must cite top candidate {top_model}"
    print(f"   -> Top pick {recs_list[0]['brand']} {top_model} correctly cited in explanation.")
    print("✅ [Test 13] PASSED")

    # TEST 14: Engine Status & Fallback Resilience
    print("\n" + "-" * 80)
    print("🧪 [Test 14] Engine Status & Fallback Resilience")
    active_name = ai_service.get_active_engine_name()
    assert "Gemini AI" in active_name or "Built-in Rule Engine (Fallback)" in active_name
    masked_key = ai_service.get_masked_api_key()
    assert not (ai_service.api_key and ai_service.api_key in masked_key), "Raw API key must never be exposed"
    print(f"   -> Active Engine: {active_name}")
    print(f"   -> Masked Key Display: {masked_key}")
    print("✅ [Test 14] PASSED")

    print("\n" + "=" * 80)
    print("🎉 ALL 14 PHASE 4 AI & CHATBOT TESTS COMPLETED SUCCESSFULLY (100% PASS)!")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()
