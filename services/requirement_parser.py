"""
services/requirement_parser.py
Structured requirement schema, validation, state management, and fallback rule-based parser.
Ensures the database and recommendation engine remain the single source of truth.
"""

import re
from typing import Optional, Literal
from pydantic import BaseModel, Field


class UserRequirements(BaseModel):
    """Structured representation of extracted user requirements."""
    intent: Literal["recommendation", "comparison", "question", "greeting", "clarification"] = "recommendation"
    maximum_budget: Optional[float] = Field(default=None, description="Maximum budget in INR (₹)")
    primary_usage: Optional[str] = Field(
        default="General College Use / Everyday",
        description="One of: 'Programming / Coding', 'AI / Data Science', 'Gaming', 'Video Editing / Content Creation', 'General College Use / Everyday'"
    )
    secondary_usage: Optional[str] = Field(default=None, description="Secondary workload like gaming, coding, editing")
    minimum_ram: Optional[int] = Field(default=8, description="Minimum RAM in GB (8, 16, 32)")
    minimum_storage: Optional[str] = Field(default=None, description="Storage size like '512GB' or '1TB'")
    gpu_required: Optional[str] = Field(default="Any", description="'Any', 'Dedicated GPU', or 'Integrated GPU'")
    preferred_gpu: Optional[str] = Field(default=None, description="Specific preferred GPU model like 'RTX 4050', 'RTX 4060', 'RTX 3050', or null")
    preferred_brand: Optional[str] = Field(default="All Brands", description="Brand name or 'All Brands'")
    portability_preference: Optional[str] = Field(default="Standard", description="'High (Lightweight)' or 'Standard'")
    battery_preference: Optional[str] = Field(default="Standard", description="'High Battery Life' or 'Standard'")
    comparison_indices: Optional[list[int]] = Field(default=None, description="Indices of laptops to compare (e.g. [1, 2])")
    missing_critical_info: list[str] = Field(default_factory=list, description="List of missing fields like ['maximum_budget']")


# Valid usage categories recognized by the recommendation engine
USAGE_MAPPING = {
    "coding": "Programming / Coding",
    "programming": "Programming / Coding",
    "software": "Programming / Coding",
    "developer": "Programming / Coding",
    "cse": "Programming / Coding",
    "computer science": "Programming / Coding",
    "web development": "Programming / Coding",
    "ai": "AI / Data Science",
    "ml": "AI / Data Science",
    "machine learning": "AI / Data Science",
    "deep learning": "AI / Data Science",
    "data science": "AI / Data Science",
    "gaming": "Gaming",
    "games": "Gaming",
    "gamer": "Gaming",
    "video editing": "Video Editing / Content Creation",
    "video": "Video Editing / Content Creation",
    "editing": "Video Editing / Content Creation",
    "content creation": "Video Editing / Content Creation",
    "graphic design": "Video Editing / Content Creation",
    "general": "General College Use / Everyday",
    "everyday": "General College Use / Everyday",
    "college": "General College Use / Everyday",
    "studies": "General College Use / Everyday",
    "student": "General College Use / Everyday",
}

BRAND_LIST = ["Apple", "Asus", "Dell", "HP", "Lenovo", "Acer", "MSI", "Samsung", "LG", "Gigabyte"]


def sanitize_user_input(user_input: Optional[str], max_length: int = 1000) -> str:
    """
    Sanitizes raw user input for safe processing:
    1. Ensures string type and trims whitespace.
    2. Enforces maximum input length to prevent denial-of-service / token bloat.
    3. Strips null bytes (\\x00) and harmful non-printable control characters.
    4. Neutralizes prompt injection delimiter tags like </user_input> or <system>.
    """
    if not user_input or not isinstance(user_input, str):
        return ""

    # Remove null bytes and non-printable control characters (keep \n, \t, \r)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", user_input.strip())

    # Neutralize artificial prompt delimiters
    cleaned = re.sub(r"</?(?:user_input|system|prompt|assistant|instruction)>", "", cleaned, flags=re.IGNORECASE)

    # Enforce maximum length
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip()

    return cleaned


def validate_and_sanitize_requirements(req: dict) -> dict:
    """
    Validates and sanitizes requirement fields from Gemini or rule engine
    before passing to MCDA recommendation engine.
    Ensures safe numerical bounds and recognized categories.
    """
    sanitized = req.copy() if req else {}

    # 1. Budget validation
    if sanitized.get("maximum_budget") is not None:
        try:
            b = float(sanitized["maximum_budget"])
            if b <= 0:
                sanitized["maximum_budget"] = None
            else:
                sanitized["maximum_budget"] = float(min(max(b, 10000.0), 1000000.0))
        except (ValueError, TypeError):
            sanitized["maximum_budget"] = None

    # 2. RAM validation (clamp to valid database values: 8, 16, 32)
    if sanitized.get("minimum_ram") is not None:
        try:
            ram = int(sanitized["minimum_ram"])
            if ram <= 8:
                sanitized["minimum_ram"] = 8
            elif ram <= 16:
                sanitized["minimum_ram"] = 16
            else:
                sanitized["minimum_ram"] = 32
        except (ValueError, TypeError):
            sanitized["minimum_ram"] = 8
    else:
        sanitized["minimum_ram"] = 8

    # 3. Primary usage category validation
    valid_usages = [
        "Programming / Coding",
        "AI / Data Science",
        "Gaming",
        "Video Editing / Content Creation",
        "General College Use / Everyday",
    ]
    primary = sanitized.get("primary_usage")
    if primary not in valid_usages:
        if isinstance(primary, str) and primary.lower() in USAGE_MAPPING:
            sanitized["primary_usage"] = USAGE_MAPPING[primary.lower()]
        else:
            sanitized["primary_usage"] = "General College Use / Everyday"

    # 4. GPU Requirement validation
    valid_gpus = ["Any", "Dedicated GPU", "Integrated GPU"]
    gpu = sanitized.get("gpu_required")
    if gpu not in valid_gpus:
        if isinstance(gpu, str) and "dedicated" in gpu.lower():
            sanitized["gpu_required"] = "Dedicated GPU"
        elif isinstance(gpu, str) and "integrated" in gpu.lower():
            sanitized["gpu_required"] = "Integrated GPU"
        else:
            sanitized["gpu_required"] = "Any"

    # 5. Brand validation
    valid_brands = ["All Brands"] + BRAND_LIST
    brand = sanitized.get("preferred_brand")
    if brand not in valid_brands:
        matched = False
        if isinstance(brand, str):
            for vb in valid_brands:
                if vb.lower() == brand.lower():
                    sanitized["preferred_brand"] = vb
                    matched = True
                    break
        if not matched:
            sanitized["preferred_brand"] = "All Brands"

    # 6. Portability & Battery preferences
    if sanitized.get("portability_preference") not in ["High (Lightweight)", "Standard"]:
        sanitized["portability_preference"] = "Standard"
    if sanitized.get("battery_preference") not in ["High Battery Life", "Standard"]:
        sanitized["battery_preference"] = "Standard"

    return sanitized


def update_requirements_state(current_state: dict, new_extracted: dict) -> dict:
    """
    Merges newly extracted requirements into the existing conversation state.
    Preserves existing requirements while applying updates and overrides.
    """
    merged = current_state.copy() if current_state else {}

    # Handle explicit resets/modifications (e.g. "I don't need gaming anymore")
    for key, val in new_extracted.items():
        if val is not None and val != "":
            merged[key] = val

    # Validate and enforce defaults
    return validate_and_sanitize_requirements(merged)


def parse_requirements_rule_based(user_message: str, current_state: Optional[dict] = None) -> dict:
    """
    Deterministic rule-based extractor used as a reliable fallback or validation anchor.
    Accurately extracts budget, RAM, GPU, Brand, and workload from natural language.
    """
    clean_msg = sanitize_user_input(user_message)
    msg = clean_msg.lower()
    state = current_state.copy() if current_state else {}
    extracted = {}

    # 1. Intent Detection
    comparison_triggers = [
        "compare", "comparison", "vs", "versus", "difference between",
        "which one is better", "which is better", "side by side",
    ]
    if any(phrase in msg for phrase in comparison_triggers):
        extracted["intent"] = "comparison"

        # Check for explicitly numbered laptops: e.g. "#1 and #2", "1 and 3", "laptop 1 and laptop 2", "option 1 and option 3"
        digit_matches = re.findall(r"(?:#|no\.?|option\s*|laptop\s*)?(\b[1-9]\b)", msg)
        valid_indices = []
        for d_str in digit_matches:
            d = int(d_str)
            # ignore if part of ram or storage or price (e.g., 8gb, 1tb, 50k)
            if not re.search(rf"\b{d}\s*(?:gb|tb|k|lakh|lac)\b", msg):
                valid_indices.append(d)

        if len(valid_indices) >= 2:
            extracted["comparison_indices"] = [valid_indices[0], valid_indices[1]]
        elif "2 and 3" in msg or "second and third" in msg or "#2 and #3" in msg:
            extracted["comparison_indices"] = [2, 3]
        elif "1 and 3" in msg or "first and third" in msg or "#1 and #3" in msg:
            extracted["comparison_indices"] = [1, 3]
        elif any(phrase in msg for phrase in ["first two", "first 2", "top two", "top 2", "those two", "these two", "the two", "two laptops", "both"]):
            extracted["comparison_indices"] = [1, 2]
        else:
            extracted["comparison_indices"] = [1, 2]
    elif any(phrase in msg for phrase in ["hello", "hi", "hey", "good morning", "good evening"]) and len(msg.split()) <= 3:
        extracted["intent"] = "greeting"
        return extracted
    else:
        extracted["intent"] = "recommendation"

    # 2. Budget Extraction (e.g. "75,000", "75000", "75k", "around ₹75k", "under 80 thousand", "1.5 lakh")
    # Cheaper follow-up intent
    if any(phrase in msg for phrase in ["cheaper", "lower budget", "less expensive", "reduce budget", "under lower"]):
        current_budget = state.get("maximum_budget") or 80000
        extracted["maximum_budget"] = max(current_budget - 15000, 35000)
    else:
        # Check written number words (e.g. "eighty thousand", "seventy five thousand", "fifty thousand")
        word_numbers = {
            "one lakh": 100000.0, "1 lakh": 100000.0, "one lac": 100000.0,
            "ninety thousand": 90000.0, "eighty five thousand": 85000.0, "eighty thousand": 80000.0,
            "seventy five thousand": 75000.0, "seventy thousand": 70000.0, "sixty five thousand": 65000.0,
            "sixty thousand": 60000.0, "fifty five thousand": 55000.0, "fifty thousand": 50000.0,
            "forty five thousand": 45000.0, "forty thousand": 40000.0, "thirty five thousand": 35000.0,
        }
        word_budget_found = False
        for phrase, b_val in word_numbers.items():
            if phrase in msg:
                extracted["maximum_budget"] = b_val
                word_budget_found = True
                break

        if not word_budget_found:
            # Check for Lakh patterns (e.g. "1.5 lakh", "1 lakh", "2 lac")
            lakh_match = re.search(r"(\d+(\.\d+)?)\s*(lakh|lac|lacs)", msg)
            if lakh_match:
                extracted["maximum_budget"] = float(lakh_match.group(1)) * 100000
            else:
                # Check for thousand / k patterns (e.g. "80 thousand", "75k", "around ₹75k", "under 80 thousand")
                thousand_k_match = re.search(r"(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*(?:thousand|k)\b", msg)
                if thousand_k_match:
                    extracted["maximum_budget"] = float(thousand_k_match.group(1)) * 1000
                else:
                    # Check for standard numbers with optional commas (e.g. "₹75,000", "75000", "85,000")
                    num_matches = re.findall(r"(?:₹|rs\.?|inr)?\s*(\d{1,3}(?:,\d{3})+|\d{4,6})", msg)
                    if num_matches:
                        for num_str in num_matches:
                            val = float(num_str.replace(",", ""))
                            if 10000 <= val <= 500000:
                                extracted["maximum_budget"] = val
                                break

    # 3. RAM Extraction (e.g. "16GB", "16 GB RAM", "at least 16GB", "8gb", "32gb")
    ram_match = re.search(r"(\d+)\s*(?:gb|gigabytes?)\s*(?:ram)?", msg)
    if ram_match:
        ram_val = int(ram_match.group(1))
        if ram_val in [8, 16, 24, 32, 64]:
            extracted["minimum_ram"] = min(ram_val, 32)

    # 4. Storage Extraction
    storage_match = re.search(r"(512\s*gb|1\s*tb|2\s*tb|256\s*gb)", msg)
    if storage_match:
        extracted["minimum_storage"] = storage_match.group(1).upper().replace(" ", "")

    # 5. GPU Requirement & Preferred GPU
    # Specific GPU Model (e.g. RTX 4060, RTX 4050, RTX 3050, Apple M3, etc.)
    gpu_model_match = re.search(r"\b(rtx\s*\d{4}(?:\s*ti)?|gtx\s*\d{4}|radeon\s*\d+|apple\s*m\d(?:\s*pro|\s*max)?|m[123](?:\s*pro|\s*max)?)\b", msg)
    if gpu_model_match:
        extracted["preferred_gpu"] = gpu_model_match.group(1).upper()
        extracted["gpu_required"] = "Dedicated GPU"

    if "no gaming" in msg or "don't need gaming" in msg or "dont need gaming" in msg or "integrated" in msg:
        extracted["gpu_required"] = "Any"
        if state.get("primary_usage") == "Gaming":
            extracted["primary_usage"] = "Programming / Coding"
    elif any(phrase in msg for phrase in ["dedicated gpu", "dedicated graphics", "rtx", "graphics card", "nvidia", "dedicated"]):
        extracted["gpu_required"] = "Dedicated GPU"
    elif "gaming" in msg:
        extracted["gpu_required"] = "Dedicated GPU"

    # 6. Brand Preference
    brand_found = False
    for b in BRAND_LIST:
        if re.search(rf"\b{b.lower()}\b", msg):
            extracted["preferred_brand"] = b
            brand_found = True
            break
    if not brand_found and ("all brands" in msg or "any brand" in msg):
        extracted["preferred_brand"] = "All Brands"

    # 7. Portability & Battery Preferences
    if any(k in msg for k in ["lightweight", "portable", "thin and light", "easy to carry", "ultrabook", "compact"]):
        extracted["portability_preference"] = "High (Lightweight)"
    if any(k in msg for k in ["long battery", "all day battery", "good battery", "battery backup", "battery life"]):
        extracted["battery_preference"] = "High Battery Life"

    # 8. Primary & Secondary Usage
    detected_usages = []
    if any(k in msg for k in ["ai", "ml", "machine learning", "deep learning", "data science"]):
        detected_usages.append("AI / Data Science")
    if any(k in msg for k in ["coding", "programming", "developer", "cse", "software", "web dev"]):
        detected_usages.append("Programming / Coding")
    if any(k in msg for k in ["gaming", "game", "gamer", "gta", "fps"]) and "no gaming" not in msg and "don't need gaming" not in msg:
        detected_usages.append("Gaming")
    if any(k in msg for k in ["video editing", "render", "premier", "davinci", "animation"]):
        detected_usages.append("Video Editing / Content Creation")
    if any(k in msg for k in ["college", "student", "study", "general", "office", "everyday"]):
        detected_usages.append("General College Use / Everyday")

    if detected_usages:
        extracted["primary_usage"] = detected_usages[0]
        if len(detected_usages) > 1:
            extracted["secondary_usage"] = detected_usages[1]

    # 9. Check Missing Critical Information
    combined_budget = extracted.get("maximum_budget") or state.get("maximum_budget")
    missing = []
    if not combined_budget and extracted["intent"] == "recommendation":
        # Check if the query is too brief (e.g. "I want a laptop for college")
        if len(msg.split()) < 12 and not any(k in msg for k in ["budget", "price", "₹", "rs", "k", "thousand", "lakh"]):
            missing.append("maximum_budget")

    extracted["missing_critical_info"] = missing
    return validate_and_sanitize_requirements(extracted)
