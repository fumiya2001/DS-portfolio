# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a data science portfolio containing multiple independent projects demonstrating ML, NLP, SQL, and statistical computing skills.

## Environment Setup

**Jupyter Lab (root-level projects):**
```bash
docker-compose up                  # Starts Jupyter on http://localhost:8888
# Or install locally:
pip install -r requirements.txt
jupyter lab
```

**Simple RAG system (separate environment):**
```bash
cd simple_RAG
docker-compose up -d               # Start PostgreSQL + pgvector on port 5432
pip install -r requirements.txt
python app/ingest.py               # Ingest PDFs into vector DB
uvicorn app.main:app --reload      # Start FastAPI on http://localhost:8000
```

## Project Architecture

Each subdirectory is a self-contained project:

### LLM_Classification_kaggle/
Disaster tweet classification pipeline with 4 sequential notebooks:
1. `01_Embeddings_Generation.ipynb` — extract embeddings from pretrained Transformers
2. `02_PCA_Logistic.ipynb` — PCA dimensionality reduction + baseline models
3. `03_Deep-learning_transformers.ipynb` — fine-tune DeBERTa-v3-base
4. `04_Model_Ensemble.ipynb` — 5-fold CV ensemble (requires GPU/Colab)

### titanic/
Kaggle Titanic prediction with feature engineering and hyperparameter tuning:
- `titanic-1.ipynb` — EDA and modeling
- `modeling.py` — model training logic
- `optuna_tuning.py` — Optuna-based hyperparameter search

### simple_RAG/
RAG system with a clear pipeline across separate modules:
- `app/ingest.py` — PDF → text chunks → embeddings → pgvector storage
- `app/search.py` — query embedding + cosine similarity + Cross-Encoder reranking
- `app/generate.py` — local LLM answer generation
- `app/main.py` — FastAPI endpoint (`POST /ask`)

Vector DB: PostgreSQL with pgvector extension. Embeddings via Sentence Transformers.

### LASSO_from_scratch/
R implementation comparing Coordinate Descent vs. ADMM for LASSO regression (`lasso.R`).

### SQL/
Bike store sales analysis (`queries.sql`) using CTEs, window functions, and JOINs on 9 CSV tables.

### transformer_architecture/
Single notebook (`transformer.ipynb`) exploring transformer architecture internals.

## Tech Stack

- **ML/DL:** scikit-learn, XGBoost, LightGBM, PyTorch, Hugging Face Transformers
- **Hyperparameter tuning:** Optuna
- **NLP:** Sentence Transformers, DeBERTa-v3, Japanese NLP (fugashi, ipadic)
- **RAG:** FastAPI, PostgreSQL+pgvector, LangChain text-splitters
- **Other:** R (statistical algorithms), SQL (MySQL/PostgreSQL)
