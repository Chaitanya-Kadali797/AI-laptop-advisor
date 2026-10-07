# 💻 AI-Powered Intelligent Laptop Selection & Recommendation Chatbot

> **College Minor/Major Project — Complete Implementation (Phases 1, 2 & 3)**  
> Multi-Criteria Decision Analysis (MCDA), Explainable Recommendation AI, Google Gemini Conversational Chatbot, What-If Simulator & Plotly Analytics Dashboard.

---

## 📌 Project Overview
Choosing the right laptop is challenging because technical specifications (core architectures, thread counts, RAM speeds, TGP wattage, and color accuracy) do not directly tell a user whether a machine will support their daily workload (PyTorch/TensorFlow, Android Studio, Docker, competitive gaming, or video rendering).

This project solves this challenge with a 3-phase modular architecture:
- **Phase 1 (Deterministic Foundation):** An embedded SQLite database containing 126 verified laptops, a Multi-Criteria Decision Analysis (MCDA) recommendation engine, interactive Plotly charts, and a side-by-side spec comparison tool.
- **Phase 2 (Conversational AI Integration):** An intelligent conversational chatbot powered by **Google Gemini** that understands natural language queries, extracts structured constraints, executes queries against the SQLite database, and provides grounded technical explanations without hallucinations.
- **Phase 3 (Evaluation, Explainability & Analytics):** Recommendation sub-scores, transparent Recommendation Confidence index, What-If scenario simulation, alternative standout recommendations, and a comprehensive Evaluation Dashboard tested across 24 predefined scenarios.

---

## 🏗️ System Architecture & Communication Flow

```
                      [ User Input (Natural Language / Filters) ]
                                          │
                                          ▼
                   [ LLM / Rule-Based Requirement Parser ]
                        (services/requirement_parser.py)
                                          │
                                          ▼
                        [ Structured Validated JSON ]
          { budget: 75000, usage: "Coding", ram: 16, gpu: "Dedicated" }
                                          │
                                          ▼
                   [ Multi-Criteria Recommendation Engine ]
                          (services/recommender.py)
            ┌─────────────────────────────┼─────────────────────────────┐
            ▼                             ▼                             ▼
   [ Requirement Sub-Scores ]    [ Confidence Index ]     [ Plain-English Narrative ]
   - Budget Match %              - 0.40 * Completeness    - Form factor suitability
   - RAM Match %                 - 0.60 * Match Fit       - Battery endurance profile
   - GPU Match %
   - Workload Match %
                                          │
                                          ▼
                        [ SQLite Database Query ]
                       (Strict Source of Truth: 126 Laptops)
                                          │
                                          ▼
                      [ Top-Ranked Verified Laptops ]
                                          │
            ┌─────────────────────────────┴─────────────────────────────┐
            ▼                                                           ▼
 [ Grounded LLM Explainer ]                                  [ Decision Tools ]
   (services/ai_service.py)                                    - Alternative Options
            │                                                  - What-If Simulator
            ▼                                                  - Analytics Dashboard
 [ Streamlit Web Interface ]                                             │
   (Cards + Chat + Radar)  ◄─────────────────────────────────────────────┘
```

---

## 🧠 MCDA Algorithm & Scoring Formula (Phase 3)

### 1. Composite Match Score Formula
$$\text{Match Score} = (0.55 \times S_{\text{primary}}) + (0.20 \times S_{\text{overall}}) + (0.15 \times B_{\text{budget}}) + (0.10 \times R_{\text{ram}})$$

Where:
- $S_{\text{primary}}$: Domain performance score weighted by the user's workload (e.g. 50% AI/ML, 25% coding, 15% video, 10% gaming).
- $S_{\text{overall}}$: Arithmetic mean across all 4 domain benchmarks.
- $B_{\text{budget}}$: Budget utilization efficiency factor (rewards options that make optimal use of capital without exceeding the ceiling).
- $R_{\text{ram}}$: Memory headroom bonus rewarding 16GB/32GB configurations.

### 2. Requirement Match Sub-Scores (Explainability)
- **Budget Match %:** $100\%$ if $\text{Price} \le \text{Budget}$; scales down proportionally if budget is exceeded.
- **RAM Match %:** $100\%$ if $\text{RAM} \ge \text{Min RAM}$; otherwise $(\text{RAM} / \text{Min RAM}) \times 100\%$.
- **GPU Suitability %:** $100\%$ when the machine satisfies the required architecture (Dedicated vs Integrated).
- **Workload Match %:** The normalized domain performance rating out of 100.

### 3. Recommendation Confidence Index
$$\text{Confidence} = (0.40 \times C_{\text{completeness}}) + (0.60 \times \text{Match Score})$$
- **Not an ML probability:** Explicitly calculated from input completeness ($C_{\text{completeness}}$: budget, usage, RAM, GPU provided) and candidate fit.
- Categorized as **High Confidence ($\ge 85\%$)**, **Moderate Confidence ($70-84\%$)**, or **Preliminary Confidence ($< 70\%$)**.

---

## 📊 Quantitative Benchmark Results (24 Predefined Scenarios)

The system was evaluated using [`test_evaluation.py`](file:///C:/Users/chait/AI-laptop-advisor/test_evaluation.py) across **24 comprehensive scenarios**:

| Metric | Result | Benchmark Standard |
| :--- | :---: | :--- |
| **Total Scenarios Evaluated** | **24** | $\ge 20$ Scenarios across 10 Domains |
| **Budget Compliance Rate** | **100.0%** | Zero recommendations exceed user budget |
| **RAM Requirement Satisfaction Rate** | **100.0%** | 100% of recommendations meet or exceed required RAM |
| **GPU Requirement Compliance Rate** | **100.0%** | 100% adherence to Dedicated vs Integrated constraints |
| **Recommendation Availability Rate (Feasible)** | **100.0%** | All 20 feasible scenarios returned valid candidates |
| **Average Recommendation Score** | **77.5%** | High candidate match across all price bands |
| **Average Recommendation Confidence** | **84.3%** | High overall constraint confidence |
| **Edge Cases Handled Gracefully** | **4 / 4** | Zero crashes on impossible budgets or invalid inputs |
| **Final Verified Database Count** | **126 Laptops** | Real, commercial market configurations |

---

## 🔮 What-If Analysis Simulator & Alternatives Engine

- **What-If Analysis:** Allows users to modify a single constraint (e.g. $+₹15,000$ budget, upgrading to 32GB RAM, or removing dedicated GPU) and view side-by-side **Before vs After** recommendation cards, score deltas, and narrative trade-off summaries.
- **Alternative Standout Laptops:** Automatically identifies:
  - 🌟 **Best Overall:** Top ranked candidate.
  - 💰 **Best Budget Option:** Lowest priced viable option in the matching pool.
  - ⚡ **Best Performance Option:** Highest raw compute benchmark.
  - 🤖 **Best for AI/ML:** Highest AI compute score.
  - 🎮 **Best for Gaming:** Highest GPU/Gaming rating.
  - 💻 **Best for Programming:** Highest coding efficiency.

---

## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **Frontend / Framework:** Streamlit (`st.chat_message`, `st.chat_input`, `st.tabs`)
- **LLM API:** Google Gemini (`google-genai` / `google-generativeai`)
- **Schema Validation:** Pydantic
- **Data Manipulation:** Pandas
- **Database:** SQLite3 (standard library, local `.db` file)
- **Data Visualization:** Plotly Express & Plotly Graph Objects

---

## 📂 Project Directory Structure

```
AI-laptop-advisor/
├── app.py                      # Main Streamlit web application (7 interactive tabs)
├── requirements.txt            # All project dependencies
├── README.md                   # Full documentation & college viva guide
├── .env.example                # Template for Gemini API key configuration
├── test_evaluation.py          # Phase 3 evaluation test runner (24 scenarios)
├── test_chatbot.py             # Phase 2 conversational test suite (7 scenarios)
├── validate_dataset.py         # Database validation script (126 laptops)
├── data/
│   ├── seed_data.py            # 126 realistic laptop specifications across 10 major brands
│   └── laptops.db              # SQLite database (auto-initialized and seeded on launch)
├── database/
│   ├── __init__.py             # Database package initializer
│   └── db_manager.py           # SQLite connection, schema creation, seeding & queries
├── services/
│   ├── __init__.py             # Services package initializer
│   ├── recommender.py          # MCDA scoring engine, sub-scores, confidence, What-If
│   ├── evaluator.py            # Phase 3 evaluation battery (24 scenarios & metrics)
│   ├── requirement_parser.py   # Pydantic schema, state tracking & regex fallback parser
│   └── ai_service.py           # Google Gemini wrapper & grounded explanation generator
└── components/
    ├── __init__.py             # UI components package initializer
    ├── analytics_dashboard.py  # Phase 3 Recommendation Evaluation Dashboard
    ├── what_if_ui.py           # Phase 3 Interactive What-If Scenario Simulator
    ├── chat_ui.py              # Interactive Streamlit chatbot UI & message history
    ├── charts.py               # Plotly charts (Radar, Price vs Performance, Bar chart)
    └── ui_cards.py             # Reusable laptop spec cards, sub-scores & confidence badges
```

---

## 🔑 Secure Server-Side Gemini API Key Configuration (Public Deployment)

For security in public web deployments, the Gemini API key is managed **strictly on the server side** and is **never exposed to visitors** or rendered in the browser DOM.

### Option 1: Streamlit Community Cloud (Recommended for Public Deployment)
When deploying on [Streamlit Community Cloud](https://share.streamlit.io):
1. Go to your App Dashboard -> **Settings** -> **Secrets**.
2. Add your key in TOML format:
   ```toml
   GEMINI_API_KEY = "your_actual_gemini_api_key_here"
   ```
3. Save. Streamlit automatically makes it available via `st.secrets["GEMINI_API_KEY"]` on the server.

### Option 2: Environment Variable (Docker / Linux / Cloud VPS / PowerShell)
Set the environment variable on the server before starting the app:
- **Windows PowerShell:**
  ```powershell
  $env:GEMINI_API_KEY="your_actual_gemini_api_key_here"
  python -m streamlit run app.py
  ```
- **Linux / macOS / Docker:**
  ```bash
  export GEMINI_API_KEY="your_actual_gemini_api_key_here"
  python -m streamlit run app.py
  ```

### Option 3: Local Development `.env` File
Create a `.env` file in the project root (automatically ignored by `.gitignore`):
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

> **🛡️ Offline Fallback Safe Mode:** If no API key is configured or API quotas are exceeded, the application automatically operates using its built-in deterministic MCDA rule engine and response generator with **zero hallucinations** and zero crashes.

---

## 🚀 How to Run the Application & Tests

### Run the Web Application:
```powershell
cd C:\Users\chait\AI-laptop-advisor
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Run Phase 3 Evaluation Battery (24 Scenarios):
```powershell
python test_evaluation.py
```

### Run Phase 2 Chatbot Tests (7 Conversational Scenarios):
```powershell
python test_chatbot.py
```

---

## 🎓 College Viva Preparation Q&A

| Viva Question | Model Answer |
| :--- | :--- |
| **"How do the AI and the recommendation engine communicate?"** | *"The system uses a decoupled pipeline. The LLM converts natural language text into a validated JSON schema (`budget`, `ram`, `gpu`, `usage`). This JSON is passed to our Python/SQLite MCDA recommendation engine. The engine computes deterministic match scores, queries the database, and passes the retrieved records back to the LLM to format a grounded technical explanation."* |
| **"Why not let the LLM recommend laptops directly?"** | *"LLMs suffer from hallucinations—they can invent non-existent models, misquote specs, or cite outdated prices. By treating SQLite as the strict source of truth, our recommendations are 100% factual and mathematically verifiable."* |
| **"Is your confidence score an ML prediction probability?"** | *"No. It is a transparent **Recommendation Confidence Index** calculated as $(0.40 \times \text{Completeness}) + (0.60 \times \text{Match Fit})$. It reflects how thoroughly the user specified constraints and how well the best candidate satisfies them, avoiding deceptive probabilistic claims."* |
| **"How does the What-If analysis work?"** | *"It runs parallel executions of the MCDA scoring engine—one on the active baseline parameters and one on hypothetical modified inputs—then calculates the delta in price, capability, and score to show the user the exact trade-offs."* |

---

## ⚠️ Limitations & Future Improvements
1. **Real-time Price Fluctuations:** Currently uses verified retail seed data; future scope includes connecting live e-commerce APIs (Amazon, Flipkart) via scraping or webhooks.
2. **User Benchmark Submission:** Currently uses curated benchmark scores; future work could allow community benchmark submissions.
3. **Multi-Currency Support:** Currently defaults to INR (₹); future iterations will support USD/EUR currency toggles with real-time exchange rates.
