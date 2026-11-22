# Enterprise RAG Quality Engineering Platform

> A production-grade, end-to-end Retrieval-Augmented Generation (RAG) system built as a portfolio project for an AI Quality Engineering company.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Running the Application](#running-the-application)
8. [Running Tests](#running-tests)
9. [Development Stages](#development-stages)
10. [Contributing](#contributing)

---

## Overview

This platform demonstrates enterprise-quality engineering practices applied to a RAG system. It covers the full lifecycle:

- **Document ingestion** – load PDFs, text files, and other formats
- **Chunking** – configurable splitting strategies
- **Embedding** – dense vector generation via Sentence Transformers
- **Vector storage** – ChromaDB with cosine similarity search
- **Retrieval** – semantic top-K search with optional reranking
- **Generation** – OpenAI-powered answer synthesis with source citations
- **Evaluation** – Precision@K, Recall@K, MRR, Ragas faithfulness & correctness
- **Reporting** – automated evaluation reports and visualisations

---

## Architecture

```
User / Streamlit UI
        │
        ▼
  FastAPI REST API
        │
        ▼
   RAG Pipeline
   ┌────────────────────────────────────────┐
   │  Retrieval Module ◄─── ChromaDB        │
   │        │                               │
   │  Generation Module ◄─── OpenAI API     │
   └────────────────────────────────────────┘
        │
        ▼
  Evaluation Framework ──► Reports & Charts
```

---

## Project Structure

```
Policy RAG/
├── config/
│   └── settings.py          # Centralised Pydantic settings (all tuneable knobs)
├── src/
│   ├── ingestion/           # File loading, format detection, pre-processing
│   ├── parsing/             # PDF/text extraction (PyMuPDF)
│   ├── chunking/            # Chunking strategies: fixed, sentence, recursive
│   ├── embeddings/          # SentenceTransformer wrapper + batch encoding
│   ├── vectordb/            # ChromaDB client, collection lifecycle
│   ├── retrieval/           # Semantic search, score filtering, reranking
│   ├── generation/          # LLM abstraction + OpenAI adapter
│   ├── pipeline/            # Orchestrates retrieval → generation (RAG chain)
│   ├── evaluation/          # Metrics: Precision@K, Recall@K, MRR, Ragas
│   └── reporting/           # Charts, HTML/PDF reports
├── api/
│   └── main.py              # FastAPI application entry point
├── frontend/
│   └── app.py               # Streamlit chatbot interface
├── tests/
│   ├── unit/                # Isolated unit tests per module
│   ├── integration/         # End-to-end pipeline tests
│   └── evaluation/          # Golden Q&A benchmark test suite (100+ cases)
├── data/
│   ├── raw/                 # Source documents (PDFs, text files, etc.)
│   ├── processed/           # Cleaned, chunked artefacts
│   ├── benchmarks/          # Golden Q&A JSONL datasets
│   └── vectorstore/         # ChromaDB local persistence (auto-created)
├── reports/                 # Generated evaluation reports and charts
├── notebooks/               # Exploratory analysis and prototyping
├── .env.example             # Environment variable template
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Tool configuration (pytest, ruff, mypy)
└── README.md                # This file
```

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python      | 3.11+   |
| pip         | 23+     |
| Git         | any     |

An **OpenAI API key** is required for LLM-based answer generation and evaluation.

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd "Policy RAG"
```

### 2. Create and activate a virtual environment

```bash
# Create the virtual environment
python3.11 -m venv .venv

# Activate it
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Open .env and fill in at minimum:
#   OPENAI_API_KEY=sk-...
```

---

## Configuration

All application settings live in `config/settings.py` and can be overridden via environment variables in `.env`.

| Setting group | Key env variables | Description |
|---|---|---|
| LLM | `OPENAI_API_KEY`, `LLM_MODEL`, `LLM_TEMPERATURE` | Language model parameters |
| Embedding | `EMBEDDING_MODEL` | HuggingFace model name |
| Chunking | `CHUNK_SIZE`, `CHUNK_OVERLAP` | Document splitting parameters |
| Vector DB | `CHROMA_MODE`, `CHROMA_PERSIST_DIR` | ChromaDB connection |
| Retrieval | `RETRIEVAL_TOP_K` | Number of retrieved chunks |
| Evaluation | `BENCHMARK_FILE`, `EVAL_OUTPUT_DIR` | Evaluation inputs/outputs |
| API | `API_HOST`, `API_PORT` | FastAPI server binding |
| Logging | `LOG_LEVEL` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

---

## Running the Application

> ⚠️ **Not yet implemented** — this section will be updated as development stages progress.

```bash
# FastAPI backend
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Streamlit frontend
streamlit run frontend/app.py
```

---

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run only unit tests
pytest tests/unit/

# Run evaluation benchmark
pytest tests/evaluation/ -v
```

---

## Development Stages

| Stage | Status | Description |
|-------|--------|-------------|
| 1 — Project scaffold | ✅ Complete | Folder structure, deps, config, README |
| 2 — Document ingestion & parsing | ⬜ Planned | PDF loading, text extraction |
| 3 — Chunking | ⬜ Planned | Fixed, sentence, recursive strategies |
| 4 — Embeddings & vector store | ⬜ Planned | SentenceTransformers + ChromaDB |
| 5 — Retrieval | ⬜ Planned | Semantic search, top-K |
| 6 — LLM generation | ⬜ Planned | OpenAI adapter, prompt templates |
| 7 — RAG pipeline | ⬜ Planned | End-to-end orchestration |
| 8 — FastAPI backend | ⬜ Planned | REST endpoints |
| 9 — Streamlit frontend | ⬜ Planned | Chatbot UI |
| 10 — Evaluation framework | ⬜ Planned | Metrics, golden Q&A, Ragas |
| 11 — Reporting | ⬜ Planned | Charts, HTML reports |

---

## Contributing

This is a portfolio project. Code quality standards enforced:

- **Formatting / linting:** `ruff check . && ruff format .`
- **Type checking:** `mypy src/`
- **Tests must pass:** `pytest`

All new modules must include type hints, docstrings on public functions, and corresponding unit tests.
