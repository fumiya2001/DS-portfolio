# Simple RAG System

This project implements a simple Retrieval-Augmented Generation (RAG) system using:

- pgvector (PostgreSQL) for vector search
- Sentence Transformers for embedding
- Cross-Encoder for reranking
- FastAPI for serving the API
- Local LLM for answer generation

---

## Architecture

1. PDF is processed and split into chunks
2. Each chunk is embedded and stored in PostgreSQL (pgvector)
3. User query is embedded and used to retrieve similar chunks
4. Retrieved chunks are reranked using a Cross-Encoder
5. Top chunks are passed to an LLM to generate the final answer

---

## Project Structure
``` text
simple_RAG/
|--app/
|   |-- ingest.py # PDF → chunk → embedding → DB
|   |-- search.py # retrieval + reranking
|   |-- generate.py # LLM answer generation
|   |-- main.py # FastAPI API
|
|--data/
|   |-- attention_is_all_you_need_paper.pdf
|
|--docker-compose.yml
|--requirements.txt
|--README.md
```

## Set Up
1) Run Database
```bash
docker-compose up -d
```

2) Install dependencies
``` bash
pip install -r requirements.txt
```

3) Ingest Data
``` bash
python app/ingest.py
```

4) Run API
``` bash
uvicorn app.main:app --reload
```

## API usage

Example Request
``` bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the self-attention mechanism?"}'
```

Example Response
```
{"query": "What is the self-attention mechanism?",
"answer": "..."}
```

## Notes
- Developed and tested on Python 3.11
- Currently supports a single PDF, but can be extented to multiple documents


## Future Work
- Support user-uploaded documents
- Improve prompt design for better answer quality
- Add a simple User Interface (UI)
- Containerize the Python application using Docker for full reproducibility