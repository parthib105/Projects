# Job Hunting Agent — Executive AI Career Assistant

An enterprise-grade, high-performance **AI Job Search & Application Orchestration Engine** built with Python, **LangGraph**, **Google Gemini LLM**, **DuckDuckGo Search**, **SQLite**, **Semantic Vector Matching**, and **Rich Terminal UI**.

---

## 🌟 Key Architecture & Feature Highlights

### ⚡ Phase 1: Code Quality & Core Architecture
- **Model Tiering Strategy**: Employs `gemini-2.0-flash` for high-speed query generation and `gemini-1.5-pro` for deep reasoning/ranking.
- **Structured Pydantic Schemas**: Standardized domain models (`UserPreferences`, `JobListing`, `JobMatchAnalysis`, `RankedJobList`, `ApplicationMaterials`) with modern Python 3.9+ built-in type hints (`list[T]`, `T | None`).
- **Unicode-Preserving Resume Parsing**: Preserves international names, foreign currencies (€, £, ₹), accents, and symbols across PDF, DOCX, and TXT files.

### 🔍 Phase 2: High-Performance Async Search & Deduplication
- **Plug-and-Play Strategy Engine (`BaseSearchProvider`)**: Abstracts search providers under a unified interface supporting **DuckDuckGo Search** (100% free open-source, no API key needed) and **Tavily API**.
- **Async Parallel Search**: Executes web searches concurrently via `asyncio.gather()`, reducing search runtime from ~25s to ~2s.
- **Canonical Deduplication Engine (`DeduplicationEngine`)**: Normalizes URLs by stripping tracking queries (`utm_source`, `gclid`, `ref`, etc.) and filters duplicate job listings across overlapping queries using composite `(company_name + job_title)` keys.

### 🧠 Phase 3: Map-Reduce Batch Ranking & Resume Customizer
- **Map-Reduce Parallel Batching**: Chunks raw job listings into parallel batches of 5, evaluating them concurrently using XML tags (`<job_listing id="...">`) to eliminate LLM context overload.
- **Multi-Criteria Scoring Rubric**: Evaluates jobs using a 4-part weighted scoring formula:
  - 🛠️ **Technical Skill Fit (40%)**
  - 📈 **Experience & Seniority Alignment (30%)**
  - 🏢 **Domain & Industry Relevance (20%)**
  - 🌍 **Work Conditions & Preferences (10%)**
- **Automated Application Tailoring (`tailor_application`)**: Generates 3–5 **Google XYZ-style resume bullet points** ("Accomplished [X] by doing [Y] as measured by [Z]") and a personalized 3-paragraph **cover letter** for top-ranked job matches.

### 💾 Phase 4: Local Database Storage & State Memory
- **Dedicated Database Package (`database/manager.py`)**: Manages SQLite storage for cached job listings, user search preferences, match evaluation history, and tailored application materials.
- **LangGraph Memory Checkpointing**: Integrated `MemorySaver` into `workflow.compile()` for thread-based state persistence, execution tracking, and workflow resumption.

### 🎨 Phase 5: Semantic Vector Search, Production Rich CLI & Export Engine
- **Semantic Vector Search Engine (`VectorSearchEngine`)**: Uses cosine similarity math between candidate resume embeddings and job vectors to uncover domain matches even when exact query keywords differ (with pure Python text similarity fallback).
- **Interactive Rich CLI (`cli.py`)**: Styled terminal interface with live status spinners, colored panels, and formatted `rich.table.Table` output.
- **Human-in-the-Loop Review & Export**: Interactively prompts candidates to review top matches and exports tailored application materials to Markdown files under `exports/`.

---

## 📁 Directory Structure

```
Job-Hunting-Agent/
├── cli.py                  # Production Interactive Rich CLI Entry Point
├── Job_hunting_agent.py    # Backwards-compatible CLI script
├── sample_resume.txt       # Sample candidate resume for testing
├── IMPROVEMENT_PLAN.md     # 5-Phase Progressive Evolution Roadmap & Matrix
├── README.md               # Project documentation
├── requirements.txt        # Python package dependencies
├── .env                    # Environment configuration
├── config/
│   ├── settings.py         # Config loader & ConfigurationError constructor
│   └── __init__.py
├── core/
│   ├── dependencies.py     # Gemini LLMs & Search Providers initialization
│   ├── deduplicator.py     # Canonical URL stripper & deduplication engine
│   ├── job_ranker.py       # Map-Reduce parallel batch ranking node
│   ├── job_searcher.py     # Async parallel job searching node
│   ├── query_generator.py  # Structured query generation node
│   ├── resume_parser.py   # Unicode-preserving resume parser
│   ├── search_providers.py # BaseSearchProvider strategy engine (DDGS & Tavily)
│   ├── state.py            # Pydantic schemas (AgentState, JobListing, etc.)
│   ├── tailor.py           # Resume bullet & cover letter generator node
│   ├── vector_search.py    # Semantic cosine similarity engine
│   └── workflow.py         # LangGraph workflow with MemorySaver checkpointer
├── database/
│   ├── manager.py          # SQLite database manager & caching engine
│   ├── job_hunting_agent.db# Local SQLite database file
│   └── __init__.py
├── exports/                # Generated Markdown application materials
├── tests/                  # 23 Automated pytest unit tests (100% pass rate)
└── utils/
    └── logging_config.py   # Centralized logger setup
```

---

## 🛠️ Installation & Prerequisites

### Prerequisites
- Python 3.11+
- Google API Key (for Gemini LLM)

### Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd Job-Hunting-Agent
   ```

2. **Create Virtual Environment & Install Dependencies**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Create or edit `.env` in the project root:
   ```env
   GOOGLE_API_KEY="your_google_api_key_here"
   SEARCH_PROVIDER=duckduckgo
   LLM_MODEL=gemini-2.0-flash
   REASONING_LLM_MODEL=gemini-1.5-pro
   LLM_TEMPERATURE=0.6
   MAX_SEARCH_RESULTS=5
   LOG_LEVEL=INFO
   ```

---

## 🚀 Quick Start & Usage

### 1. Interactive CLI Mode (Recommended)
Run the interactive terminal prompter:
```bash
python cli.py
```
You will be prompted for your resume path, target roles, preferred locations, remote choices, and search provider preference.

### 2. Command-Line Flag Mode (Non-Interactive)
Run with CLI flags and automatic Markdown export:
```bash
python cli.py --non-interactive --resume sample_resume.txt --provider duckduckgo --export
```

### 3. Backwards-Compatible Execution
Run the simple CLI script:
```bash
python Job_hunting_agent.py
```

---

## 🧪 Automated Testing Suite

Run the full automated `pytest` suite covering configuration settings, resume parsing, deduplication, search providers, vector similarity, SQLite storage, and CLI exports:

```bash
pytest -v
```

**Test Status**: **23 out of 23 unit tests passing (100% Pass Rate)** ✅

---

## 📄 License

This project is open-source and available under the MIT License.