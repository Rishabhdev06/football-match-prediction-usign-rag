# AI-Based Football Match Outcome Prediction Using Machine Learning & RAG

An enterprise-grade, student-friendly, and professional sports analytics platform that pairs **chronological machine learning prediction models** with a **Retrieval-Augmented Generation (RAG) Football Analyst**.

Built with real historical English Premier League match records from [Football-Data.co.uk](https://www.football-data.co.uk/data.php), **ChromaDB** vector storage, **Sentence Transformers** (`all-MiniLM-L6-v2`), and hosted LLM reasoning (via **Google Gemini** or **OpenAI**) with a zero-configuration **Evidence-Only Offline Fallback**.

---

## 📌 Architecture & Key Highlights

```
                          ┌───────────────────────────┐
                          │   Historical Match Data   │
                          │   (760 Real EPL Matches)  │
                          └─────────────┬─────────────┘
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
   ┌───────────────────────────┐                 ┌───────────────────────────┐
   │    ML Feature Pipeline    │                 │    RAG Document Ingestion │
   │  - Rolling Goal & Form    │                 │  - 760 Match Recaps       │
   │  - Market Consensus Odds  │                 │  - 40 Team Summaries      │
   │  - Zero In-Game Leakage   │                 │  - 19 Rules/Tactical Docs │
   └─────────────┬─────────────┘                 └─────────────┬─────────────┘
                 │                                             │
                 ▼                                             ▼
   ┌───────────────────────────┐                 ┌───────────────────────────┐
   │  Chronological ML Models  │                 │    ChromaDB Vector Store  │
   │  - Logistic Regression    │                 │  - all-MiniLM-L6-v2       │
   │  - Random Forest (~60.5%) │                 │  - Date-Filtered Retr.    │
   └─────────────┬─────────────┘                 └─────────────┬─────────────┘
                 │                                             │
                 │   [Home / Draw / Away Probabilities]       │   [Grounded Evidence & Passages]
                 └──────────────────────┬──────────────────────┘
                                        ▼
                         ┌─────────────────────────────┐
                         │   Provider-Agnostic LLM     │
                         │   - Google Gemini API       │
                         │   - OpenAI API              │
                         │   - Offline Evidence Fallback│
                         └──────────────┬──────────────┘
                                        ▼
                         ┌─────────────────────────────┐
                         │   Streamlit Web Dashboard   │
                         │     (7 Interactive Pages)   │
                         └─────────────────────────────┘
```

### 1. Dual-Engine Intelligence: ML + Grounded RAG
- **Mathematical ML Probabilities**: Logistic Regression and Random Forest models trained strictly on pre-match features calculate objective outcome probabilities for Home Win, Draw, and Away Win.
- **Evidence-Grounded RAG Narrative**: ChromaDB semantic search retrieves verified historical head-to-head records, recent form passages, and tactical guides to generate objective post-prediction analysis without hallucinating.
- **Strict Separation of Concerns**: Model probabilities are never overridden or modified by the LLM text.

### 2. Zero Data Leakage & Chronological Integrity
- **Pre-Match Features Only**: Rest days, 5-match rolling goals scored/conceded, form points, and pre-kickoff market consensus odds (`AvgH`, `AvgD`, `AvgA`). All in-game statistics (shots, corners, fouls, halftime score) are strictly excluded from prediction inputs.
- **Temporal Train/Test Split**: Matches are sorted chronologically. The earliest 75% (570 matches: Aug 2022 – Dec 2023) are used for training; the latest 25% (190 matches: Dec 2023 – May 2024) serve as the unseen test set.
- **Date-Filtered Retrieval**: When simulating past matches, RAG queries enforce a date cutoff (`match_date < target_date`) so the LLM cannot access future results.

### 3. Production-Ready Vector Store & Offline Resilience
- **819 Deterministic Chunks**: Ingests 760 match reports, 40 home/away team summaries, and 19 reference documents covering Premier League rules, tactical profiles, and analytics methodologies.
- **Local Embeddings**: Runs locally via `all-MiniLM-L6-v2` ONNX embeddings stored in persistent ChromaDB.
- **Evidence-Only Mode**: Works out-of-the-box even without an LLM API key, displaying clean, numbered, verifiable evidence citations.

---

## 🖥️ Streamlit Multi-Page Dashboard (7 Pages)

The application features 7 specialized navigation views:

1. **📌 Project Overview & Architecture**: Executive summary, dual-engine ML + RAG architecture diagram, feature taxonomy, anti-leakage rules, and technical specifications.
2. **⚽ Match Outcome Predictor**: Select any Premier League matchup, customize odds, view real-time win/draw probabilities, feature breakdown, and auto-generated RAG tactical analysis with evidence citations.
3. **📈 Team Analytics & Form Trends**: Deep dive into individual team performance, 5-match rolling goals scored vs. conceded, home vs. away win percentages, and recent match histories.
4. **💬 RAG Football Assistant Chat**: Grounded conversational assistant. Ask questions about team tactics, historical rivalries, or form curves. Displays retrieved evidence chunks, similarity distances, and citation tags.
5. **🔍 Retrieved Evidence Explorer**: Direct vector search inspector. Enter any query, adjust similarity threshold, filter by category (`match_record`, `team_summary`, `reference_doc`), and view similarity distances and passage metadata.
6. **🏆 Model Evaluation & Comparison**: Side-by-side performance comparison (Accuracy, Macro Precision, Macro Recall, Macro F1), interactive confusion matrices, and college viva / presentation defense talking points.
7. **📚 Dataset & Reference Docs Explorer**: Search and filter the 760-match historical dataset, inspect raw pre-match vs. post-match columns, and read integrated markdown guides on tactical styles and rules.

---

## 🗂️ Project Directory Structure

```text
football-match-prediction/
├── app.py                     # 7-page interactive Streamlit dashboard
├── train_model.py             # Feature engineering, model training & evaluation pipeline
├── test_project.py            # Automated ML pipeline test suite (7 tests)
├── test_rag.py                # Automated RAG vector store & retrieval test suite (8 tests)
├── requirements.txt           # Project dependencies
├── .env.example               # Environment template for Gemini/OpenAI API keys
├── README.md                  # Comprehensive documentation and setup guide
├── .streamlit/
│   └── config.toml            # Headless server configuration (port 8501)
├── data/
│   ├── README.md              # Dataset documentation, column descriptions & leakage rules
│   ├── matches.csv            # 760 real EPL matches (2022–2024 seasons)
│   └── reference_docs/        # Markdown reference documentation
│       ├── match_analytics_guide.md
│       ├── premier_league_rules.md
│       └── tactical_profiles.md
├── models/                    # Trained models and artifacts
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── scaler.joblib
│   ├── feature_names.json
│   ├── team_history.json
│   └── metrics.json
├── rag/                       # RAG subsystem
│   ├── __init__.py
│   ├── document_loader.py     # Deterministic chunking (matches, summaries, guides)
│   ├── vector_store.py        # Persistent ChromaDB vector store
│   ├── retriever.py           # Semantic search & date-filtered evidence retrieval
│   └── llm_client.py          # Provider-agnostic LLM interface (Gemini/OpenAI/Offline)
└── chroma_db/                 # Persistent vector storage database (819 chunks)
    └── chroma.sqlite3
```

---

## ⚙️ Installation & Setup Guide

### Prerequisites
- Python 3.9, 3.10, or 3.11 installed on your system.

---

### 🍏 On macOS and Linux

1. **Open Terminal and navigate to the project directory**:
   ```bash
   cd football-match-prediction
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Configure API Keys for LLM Reasoning**:
   Create a `.env` file from the template:
   ```bash
   cp .env.example .env
   ```
   Add your Google Gemini API key:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
   *(If omitted, the app will run seamlessly in Evidence-Only Mode).*

---

### 🪟 On Windows (Command Prompt or PowerShell)

1. **Open Command Prompt (CMD) or PowerShell and navigate to the project**:
   ```cmd
   cd path\to\football-match-prediction
   ```

2. **Create and activate a virtual environment**:
   - In Command Prompt:
     ```cmd
     python -m venv venv
     venv\Scripts\activate.bat
     ```
   - In PowerShell:
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```

3. **Install dependencies**:
   ```cmd
   pip install -r requirements.txt
   ```

---

## 🚀 Execution Instructions

### 1. Run Automated Unit Tests (15 Tests)
Run both test suites to verify data processing, pre-match feature engineering, chronological splits, ChromaDB embeddings, and date-filtered retrieval:
```bash
python -m unittest test_project.py test_rag.py
```
*(All 15 tests should pass with `OK` in under a second.)*

### 2. Retrain ML Models (Optional)
The project comes with pre-trained models in `models/`. To retrain them from scratch:
```bash
python train_model.py
```

### 3. Re-index Vector Database (Optional)
The ChromaDB vector database is bundled and pre-indexed in `chroma_db/`. To re-index:
```bash
python -c "from rag.vector_store import get_vector_store; vs = get_vector_store(); vs.populate_collection(force_recreate=True)"
```

### 4. Launch the Streamlit Web Dashboard
Start the interactive application:
```bash
streamlit run app.py
```
Once launched, open your web browser at:
👉 **`http://localhost:8501`**

---

## 📊 Experimental Results (Chronological Test Set)

The models were evaluated strictly on the **190 most recent, unseen Premier League matches** (December 2023 to May 2024), after training on the preceding 570 matches:

| Metric | Logistic Regression | Random Forest | Baseline (Random Guess) |
|:---|:---:|:---:|:---:|
| **Accuracy** | **59.47%** | **60.53%** | 33.33% |
| **Macro Precision** | **0.5063** | **0.5662** | 0.3333 |
| **Macro Recall** | **0.5049** | **0.5227** | 0.3333 |
| **Macro F1-Score** | **0.4509** | **0.4634** | 0.3333 |

> **Context**: In 3-class professional sports forecasting, real-world models without data leakage typically score between **55% and 62%**. Unlike synthetic toy datasets claiming 95%+ accuracy, these scores reflect genuine predictive capability under real market conditions.

---

## 🧠 Viva & College Presentation Defense Points

### 1. Why combine Machine Learning with RAG?
Machine learning models excel at calculating calibrated numerical probabilities from quantitative features (rolling goal difference, odds, rest days), but they cannot explain *why* in natural language or synthesize tactical contexts. Conversely, large language models generate fluent prose but hallucinate numbers and cannot compute mathematical classification probabilities reliably. Combining them gives the best of both worlds: **ML predicts the numbers; RAG grounds the explanation in verified historical facts**.

### 2. What is data leakage and how is it prevented?
Data leakage occurs when information from during or after an event is included as a model input. In football forecasting, including in-game statistics like *shots on target*, *corners*, or *halftime scores* creates artificial 95% accuracy that completely fails in real life before kickoff. We strictly prevent leakage by using only pre-match rolling form and market consensus odds.

### 3. Why is chronological splitting mandatory in sports time-series?
Randomly shuffling matches (e.g. `train_test_split(shuffle=True)`) allows matches from the end of the season to leak information into earlier predictions. A team's tactical form, managerial changes, and confidence are continuous time-series. We preserve temporal causality by training on the first 75% of the timeline and testing on the subsequent 25%.

### 4. How does RAG prevent temporal knowledge leakage?
When generating an analysis for a match played on October 15, 2023, the retriever applies a metadata date filter `match_date < "2023-10-15"`. This guarantees that subsequent results, title wins, or later form slumps cannot leak into the historical context provided to the LLM.

---

## 📜 License & Acknowledgments
- Historical match data provided by [Football-Data.co.uk](https://www.football-data.co.uk/data.php), created by Joseph Buchdahl.
- Embeddings powered by Sentence Transformers (`all-MiniLM-L6-v2`).
- Vector storage managed with ChromaDB.
- LLM reasoning powered by Google Gemini API / OpenAI API.
