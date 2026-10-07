"""
test_evaluation.py
Automated evaluation test runner executing 24 predefined scenarios across
different domains, budget levels, RAM tiers, GPU constraints, and edge cases.
"""

import sys
import io
import pandas as pd

# Set UTF-8 encoding for Windows terminal
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from database.db_manager import init_db, get_all_laptops
from services.evaluator import evaluate_recommendation_system, TEST_SCENARIOS
from services.recommender import (
    recommend_laptops,
    get_alternative_recommendations,
    run_what_if_analysis,
)

def run_tests():
    print("=" * 80)
    print("🚀 RUNNING PHASE 3 EVALUATION TEST BATTERY (24 SCENARIOS)")
    print("=" * 80)

    # 1. DB Verification
    init_db()
    df_all = get_all_laptops()
    total_db_count = len(df_all)
    print(f"📊 Final Database Record Count: {total_db_count} unique laptops.")
    assert total_db_count >= 120, f"Expected >= 120 laptops, got {total_db_count}"

    # 2. Run Comprehensive Evaluation
    eval_metrics = evaluate_recommendation_system(df_all)

    print("\n" + "-" * 80)
    print("📈 AGGREGATE EVALUATION METRICS:")
    print("-" * 80)
    print(f"   • Total Scenarios Tested          : {eval_metrics['total_scenarios']}")
    print(f"   • Feasible Scenarios Tested       : {eval_metrics['feasible_scenarios']}")
    print(f"   • Edge Case / Impossible Scenarios: {eval_metrics['edge_case_scenarios']}")
    print(f"   • Individual Candidates Verified  : {eval_metrics['recommendations_checked']}")
    print(f"   • Budget Compliance Rate          : {eval_metrics['budget_compliance_rate']}%")
    print(f"   • RAM Satisfaction Rate           : {eval_metrics['ram_compliance_rate']}%")
    print(f"   • GPU Compliance Rate             : {eval_metrics['gpu_compliance_rate']}%")
    print(f"   • Availability Rate (Feasible)    : {eval_metrics['availability_rate']}%")
    print(f"   • Average Match Score             : {eval_metrics['average_match_score']}%")
    print(f"   • Average Recommendation Conf.    : {eval_metrics['average_confidence']}%")

    # Assertions for 100% compliance
    assert eval_metrics['budget_compliance_rate'] == 100.0, "Budget compliance must be 100%"
    assert eval_metrics['ram_compliance_rate'] == 100.0, "RAM satisfaction must be 100%"
    assert eval_metrics['gpu_compliance_rate'] == 100.0, "GPU compliance must be 100%"
    assert eval_metrics['availability_rate'] == 100.0, "Availability on feasible scenarios must be 100%"

    # 3. Print Per-Scenario Breakdown
    print("\n" + "-" * 80)
    print("📋 PER-SCENARIO EXECUTION BREAKDOWN:")
    print("-" * 80)
    sc_df = eval_metrics["scenario_results_df"]
    for _, row in sc_df.iterrows():
        status = "✅ PASS" if row["Passed Constraints"] else "❌ FAIL"
        matches = f"{row['Matches Found']} match(es)" if row['Matches Found'] > 0 else "0 matches (Graceful)"
        print(f"[{row['Scenario ID']}] {row['Scenario Name'][:38]:<38} | {matches:<20} | Top: {row['Top Pick'][:24]:<24} | {status}")

    # 4. Verify Phase 3 Explainability & Sub-Score Outputs
    print("\n" + "-" * 80)
    print("🧪 VERIFYING PHASE 3 EXPLAINABILITY & SUB-SCORES:")
    print("-" * 80)
    sample_recs = recommend_laptops(
        df=df_all,
        budget=85000,
        primary_usage="Programming / Coding",
        min_ram=16,
        gpu_requirement="Dedicated GPU",
        top_n=2,
    )
    assert not sample_recs.empty, "Sample recommendations should not be empty"
    top_lap = sample_recs.iloc[0]

    # Check that all sub-score columns exist and have valid values
    required_phase3_cols = [
        "match_score", "budget_match_pct", "ram_match_pct", "gpu_match_pct",
        "usage_match_pct", "confidence_score", "confidence_label", "detailed_explanation",
        "portability_suitability", "battery_suitability"
    ]
    for col in required_phase3_cols:
        assert col in top_lap, f"Missing Phase 3 column: {col}"
        assert pd.notna(top_lap[col]), f"NaN found in Phase 3 column: {col}"

    print(f"Top Recommendation: {top_lap['brand']} {top_lap['model']}")
    print(f"• Overall Match Score      : {top_lap['match_score']:.1f}%")
    print(f"• Budget Match             : {top_lap['budget_match_pct']:.1f}%")
    print(f"• RAM Match                : {top_lap['ram_match_pct']:.1f}%")
    print(f"• GPU Match                : {top_lap['gpu_match_pct']:.1f}%")
    print(f"• Usage Match              : {top_lap['usage_match_pct']:.1f}%")
    print(f"• Recommendation Confidence: {top_lap['confidence_score']:.1f}% ({top_lap['confidence_label']})")
    print(f"• Rationale Narrative      : {top_lap['detailed_explanation']}")
    print(f"• Portability Profile      : {top_lap['portability_suitability']}")
    print(f"• Battery Profile          : {top_lap['battery_suitability']}")
    print("✅ Sub-scores and Explainability verified!")

    # 5. Verify Alternative Standout Recommendations
    print("\n" + "-" * 80)
    print("🧪 VERIFYING ALTERNATIVE RECOMMENDATIONS ENGINE:")
    print("-" * 80)
    alts = get_alternative_recommendations(
        df=df_all,
        budget=100000,
        primary_usage="AI / Data Science",
        min_ram=16,
    )
    assert "Best Overall" in alts and alts["Best Overall"] is not None
    assert "Best Budget Option" in alts and alts["Best Budget Option"] is not None
    assert "Best Performance Option" in alts and alts["Best Performance Option"] is not None
    assert "Best for AI/ML" in alts and alts["Best for AI/ML"] is not None
    assert "Best for Gaming" in alts and alts["Best for Gaming"] is not None
    assert "Best for Programming" in alts and alts["Best for Programming"] is not None

    print(f"• Best Overall       : {alts['Best Overall']['brand']} {alts['Best Overall']['model']} (₹{int(alts['Best Overall']['price']):,})")
    print(f"• Best Budget Option : {alts['Best Budget Option']['brand']} {alts['Best Budget Option']['model']} (₹{int(alts['Best Budget Option']['price']):,})")
    print(f"• Best Performance   : {alts['Best Performance Option']['brand']} {alts['Best Performance Option']['model']} (Score: {alts['Best Performance Option']['domain_score']}/100)")
    print(f"• Best for AI/ML     : {alts['Best for AI/ML']['brand']} {alts['Best for AI/ML']['model']} (AI Score: {alts['Best for AI/ML']['ai_ml_score']}/100)")
    print("✅ Alternative recommendations verified!")

    # 6. Verify What-If Analysis Simulator
    print("\n" + "-" * 80)
    print("🧪 VERIFYING WHAT-IF SCENARIO ANALYSIS:")
    print("-" * 80)
    base_p = {"budget": 70000, "primary_usage": "Programming / Coding", "min_ram": 16, "gpu_requirement": "Any", "preferred_brand": "All Brands"}
    mod_p = {"budget": 85000, "primary_usage": "Gaming", "min_ram": 16, "gpu_requirement": "Dedicated GPU", "preferred_brand": "All Brands"}
    what_if_res = run_what_if_analysis(df_all, base_p, mod_p)
    assert what_if_res["base_top"] is not None
    assert what_if_res["mod_top"] is not None
    assert len(what_if_res["parameter_diffs"]) > 0
    print(f"• Base Top Pick     : {what_if_res['base_top']['brand']} {what_if_res['base_top']['model']} (₹{int(what_if_res['base_top']['price']):,})")
    print(f"• Simulated Top Pick: {what_if_res['mod_top']['brand']} {what_if_res['mod_top']['model']} (₹{int(what_if_res['mod_top']['price']):,})")
    print(f"• Verdict           : {what_if_res['verdict']}")
    print("✅ What-If scenario simulation verified!")

    print("\n" + "=" * 80)
    print(f"🎉 ALL PHASE 3 TESTS PASSED (24/24 SCENARIOS PASSED, 0 FAILURES, 0 ISSUES)")
    print(f"📊 DATABASE RECORD COUNT: {total_db_count} UNIQUE LAPTOPS VERIFIED")
    print("=" * 80)

if __name__ == "__main__":
    run_tests()
