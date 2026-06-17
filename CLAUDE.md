# CineMind — Movie QA Agent (RAG)

CineMind is a Retrieval-Augmented Generation (RAG) question-answering agent for movies.
Users ask natural-language questions about a film's plot, cast, reviews, or subjective
topics (e.g. "is this good for kids?", "what do reviewers dislike about it?"), and the
agent answers from **retrieved, grounded context** rather than the model's memory alone.

This file is the single source of truth for the project. Read it fully before making changes.

## Goal & scope
- Build a reliable, well-structured **prototype first**; optimize later.
- **Local-first**: it must run on localhost before any deployment.
- **Free-tier friendly**: prefer tools with no or low cost.
- **Phase the work** (see Roadmap). Do NOT build everything at once.

## Tech stack (decided — do not swap without asking me first)
- **Language:** Python 3.11+
- **RAG framework:** LlamaIndex
- **Vector store:** ChromaDB (local, persistent)
- **Embeddings:** sentence-transformers `all-MiniLM-L6-v2` (free, runs locally)
- **LLM (generation)**: Google Gemini API via the `llama-index-llms-google-genai` package — model `gemini-3.5-flash` (free tier; Flash family). Keep the call isolated in `src/agent.py` so it can be swapped with one line.
- **UI:** Streamlit (chat interface)
- **Data sources:** TMDb API (structured metadata + reviews) and Wikipedia (plot/background).
  IMDb datasets are an optional later addition.
- **Deployment target:** Hugging Face Spaces (Docker SDK)

## Architecture
Ingestion → Chunking → Embedding → Vector store (Chroma) → Retrieval → Claude (generation) → Streamlit UI.

1. Ingest movie data (TMDb + Wikipedia) into normalized documents with metadata
   (title, year, content_type: plot | cast | review).
2. Chunk and embed; persist to a local Chroma collection.
3. At query time: embed the question, retrieve top-k relevant chunks (use metadata
   filtering where it helps, e.g. restrict to reviews for opinion questions).
4. Pass retrieved context + question to Claude with a **grounding prompt** that instructs
   it to answer only from context and to say clearly when it doesn't know.
5. Stream the answer in the Streamlit chat UI and show the sources used.

## Project structure (target)
```
CineMind/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── config.py            # central config; loads env vars
├── src/
│   ├── __init__.py
│   ├── ingest.py        # fetch TMDb + Wikipedia, build normalized documents
│   ├── index.py         # chunk, embed, persist to Chroma
│   ├── retriever.py     # query-time retrieval
│   └── agent.py         # RAG: retrieve + call Claude with grounding prompt
├── app.py               # Streamlit chat UI
├── data/                # raw/cached data (gitignored)
└── chroma_store/        # persisted vector DB (gitignored)
```

## Data sources
- **TMDb API:** movie details, cast, and user reviews. Needs a free key (`TMDB_API_KEY`).
- **Wikipedia:** plot summaries and background (via the `wikipedia` package or REST API).
- Start small — about **20–50 well-known movies** — so iteration is fast.
- **Cache** raw API/Wikipedia responses in `data/` so we never re-fetch unnecessarily.

## Development roadmap (build in this order — one phase per session)
- **Phase 0 — Scaffold:** structure, `requirements.txt`, `.gitignore`, `.env.example`,
  `config.py`. Stubs only, no logic.
- **Phase 1 — Ingestion:** fetch a small movie set from TMDb + Wikipedia into normalized
  documents; cache to `data/`.
- **Phase 2 — Indexing:** chunk, embed with MiniLM, persist to Chroma; a CLI entry point
  to (re)build the index.
- **Phase 3 — Retrieval + agent:** top-k retrieval and the Claude grounding call; testable
  from a small script before any UI.
- **Phase 4 — UI:** Streamlit chat with message history and source display.
- **Phase 5 — Polish + deploy:** error handling, `README.md`, and a `Dockerfile` for HF Spaces.

After each phase: **stop, let me test it, then I will commit via GitHub Desktop** before moving on.

## Coding conventions
- Readable code over cleverness. Type hints on function signatures.
- All secrets via environment variables (`python-dotenv`); never hardcode keys.
- Each module should be runnable/testable on its own where reasonable.
- One short docstring per module explaining its role.
- Pin major dependencies in `requirements.txt`.

## Environment variables (see `.env.example`)
- `GOOGLE_API_KEY` — Google Gemini API (from Google AI Studio)
- `TMDB_API_KEY` — TMDb API

## Commands
- **Setup:** `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- **Build index:** `python -m src.index`
- **Run app:** `streamlit run app.py`

## Guardrails (important)
- **NEVER** commit `.env` or real API keys. `.gitignore` must exclude
  `.env`, `data/`, `chroma_store/`, `.venv/`, and `__pycache__/`.
- Always provide a `.env.example` with placeholder values.
- The grounding prompt MUST instruct Claude to answer **only** from retrieved context and to
  state clearly when the answer isn't present — do not let it invent movie facts.
- Keep the free-tier footprint small; do not add paid services without asking.

## Deployment (Phase 5)
- Hugging Face Spaces, **Docker SDK** (the built-in Streamlit SDK is deprecated).
- Store `GOOGLE_API_KEY` and `TMDB_API_KEY` as **Space secrets**, never in the repo.
- The Chroma store can be rebuilt at startup or shipped pre-built — decide based on its size.