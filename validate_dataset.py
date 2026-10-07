"""
validate_dataset.py
Validation script for expanded laptop database.
Checks schema integrity, uniqueness, value ranges, brand distributions,
score consistency, and preservation of the original 34 records.
"""

import sys
import io
import pandas as pd

# UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from database.db_manager import init_db, get_all_laptops
from data.seed_data import LAPTOP_DATA

def run_validation():
    print("=" * 70)
    print("🔍 RUNNING COMPREHENSIVE DATABASE VALIDATION")
    print("=" * 70)

    # 1. Initialize and load
    init_db()
    df = get_all_laptops()
    total_count = len(df)
    seed_count = len(LAPTOP_DATA)

    print(f"📊 Total Records in SQLite: {total_count}")
    print(f"📊 Total Records in Seed Data: {seed_count}")
    assert total_count == seed_count, f"Mismatch: SQLite ({total_count}) vs Seed Data ({seed_count})"

    # 2. Schema check (16 columns)
    expected_columns = [
        "id", "brand", "model", "price", "processor", "ram", "storage",
        "gpu", "gpu_type", "display", "battery", "weight",
        "gaming_score", "programming_score", "ai_ml_score", "video_editing_score"
    ]
    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"
        assert df[col].isnull().sum() == 0, f"Found NULL values in column: {col}"
    print(f"✅ Schema Verified: All {len(expected_columns)} columns present with 0 NULL values.")

    # 3. Model uniqueness check
    unique_combinations = df.drop_duplicates(subset=["brand", "model"])
    dup_count = len(df) - len(unique_combinations)
    print(f"✅ Unique Brand+Model Configurations: {len(unique_combinations)}")
    if dup_count > 0:
        duplicates = df[df.duplicated(subset=["brand", "model"], keep=False)]
        print("❌ DUPLICATES FOUND:")
        print(duplicates[["brand", "model"]])
        assert dup_count == 0, f"Found {dup_count} duplicate configurations!"
    else:
        print("✅ Zero Duplicate Configurations Verified.")

    # 4. Brand Distribution
    print("\n📌 Brand Distribution:")
    brand_counts = df["brand"].value_counts()
    for brand, count in brand_counts.items():
        print(f"   - {brand:<10}: {count} models")
    assert len(brand_counts) >= 8, f"Expected at least 8 brands, got {len(brand_counts)}"

    # 5. Price Distribution
    min_price = df["price"].min()
    max_price = df["price"].max()
    median_price = df["price"].median()
    print(f"\n💰 Price Ranges:")
    print(f"   - Minimum: ₹{int(min_price):,}")
    print(f"   - Median:  ₹{int(median_price):,}")
    print(f"   - Maximum: ₹{int(max_price):,}")
    assert min_price <= 35000, "Should include budget laptops under ₹35k"
    assert max_price >= 200000, "Should include flagship laptops above ₹2L"

    b_sub50k = len(df[df["price"] < 50000])
    b_50to80k = len(df[(df["price"] >= 50000) & (df["price"] < 80000)])
    b_80to130k = len(df[(df["price"] >= 80000) & (df["price"] < 130000)])
    b_above130k = len(df[df["price"] >= 130000])

    print(f"   - Under ₹50,000      : {b_sub50k} laptops")
    print(f"   - ₹50,000 - ₹80,000  : {b_50to80k} laptops")
    print(f"   - ₹80,000 - ₹1,30,000: {b_80to130k} laptops")
    print(f"   - Above ₹1,30,000    : {b_above130k} laptops")

    # 6. GPU Type & RAM Distribution
    gpu_types = df["gpu_type"].value_counts().to_dict()
    print(f"\n🎮 GPU Architecture Distribution: {gpu_types}")
    assert "Dedicated" in gpu_types and "Integrated" in gpu_types

    ram_dist = df["ram"].value_counts().to_dict()
    print(f"🧠 RAM Configurations: {ram_dist}")
    assert 8 in ram_dist and 16 in ram_dist and 32 in ram_dist

    # 7. Benchmark Score Sanity Checks
    score_cols = ["gaming_score", "programming_score", "ai_ml_score", "video_editing_score"]
    for sc in score_cols:
        assert (df[sc] >= 10).all() and (df[sc] <= 100).all(), f"Scores in {sc} out of bounds"
    print("✅ Benchmark Scores Verified: All scores within 10–100 realistic range.")

    # 8. Check that all original 34 records are present
    original_models = [
        "MacBook Air M2 (13.6-inch)", "MacBook Air M3 (13.6-inch)", "MacBook Pro 14 M3 Pro",
        "MacBook Pro 16 M3 Max", "Vivobook 15 X1504", "Zenbook 14 OLED UX3405",
        "TUF Gaming A15", "ROG Zephyrus G14 (2024)", "ROG Strix G16",
        "IdeaPad Slim 3 15", "IdeaPad Slim 5 Gen 9", "LOQ 15 Gaming",
        "Legion Slim 5 16", "Legion Pro 5 16", "ThinkPad E14 Gen 5",
        "Inspiron 15 3520", "Inspiron 14 Plus 7440", "G15 5530 Gaming",
        "XPS 13 9340", "Alienware m16 R2", "HP 15s-fq5000",
        "Pavilion 14 Plus", "Victus 15 Gaming", "Omen 16 (2024)",
        "Spectre x360 14 2-in-1", "Aspire Lite AL15", "Swift Go 14 OLED",
        "Nitro V 15", "Predator Helios 16", "Modern 14 C12M",
        "Thin 15 B12UCX", "Katana 15 B13VFK", "Galaxy Book4",
        "Galaxy Book4 Pro 360"
    ]
    db_models = set(df["model"].tolist())
    for orig in original_models:
        assert orig in db_models, f"Original model missing: {orig}"
    print(f"✅ All {len(original_models)} original configurations verified present!")

    print("\n" + "=" * 70)
    print(f"🎉 VALIDATION SUCCESS: {total_count} UNIQUE LAPTOPS READY IN DATABASE")
    print("=" * 70)

if __name__ == "__main__":
    run_validation()
