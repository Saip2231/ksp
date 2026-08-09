# CrimeAI

A bilingual (Kannada–English) intelligence platform that lets law enforcement query FIR (First Information Report) records in natural language and get back cited, hallucination-resistant answers — built for the Karnataka State Police (KSP) context.

At its core, CrimeAI is a **Retrieval-Augmented Generation (RAG)** system with three things that make it different from a typical RAG demo:
1. It understands **Kannada and English queries equally well, without translating first** (cross-lingual zero-shot retrieval).
2. It **routes** queries intelligently — a lookup question goes to the RAG pipeline, a pattern/prediction question goes elsewhere.
3. Every answer is **grounded in a literal snippet** from the original FIR, so the system can't quietly make facts up.

---

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [How It Works, Concept by Concept](#how-it-works-concept-by-concept)
  - [1. The RAG Pipeline](#1-the-rag-pipeline)
  - [2. Cross-Lingual Retrieval with LaBSE](#2-cross-lingual-retrieval-with-labse)
  - [3. Language Detection](#3-language-detection)
  - [4. Translation Engine](#4-translation-engine)
  - [5. Intent Routing](#5-intent-routing)
  - [6. Chunking Strategy](#6-chunking-strategy)
  - [7. Fallback Architecture](#7-fallback-architecture)
  - [8. Explainability & Auditing](#8-explainability--auditing)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Application](#running-the-application)
- [Environment Variables](#environment-variables)
- [Known Limitations](#known-limitations)

---

## Architecture Overview

```
┌─────────────────────┐        ┌──────────────────────────┐        ┌─────────────────────┐
│      Frontend        │  HTTP  │     Backend Gateway       │        │   PostgreSQL +       │
│  Next.js + React 19   │ ─────▶ │  FastAPI (Intelligence     │ ─────▶ │   pgvector            │
│  Leaflet map, brutalist│       │  Gateway API)              │        │   (FIR vector store)  │
│  UI                   │ ◀───── │  /query /firs /suspects    │ ◀───── │                       │
└─────────────────────┘        │  /telemetry                │        └─────────────────────┘
                                 │                            │
                                 │  ┌──────────────────────┐  │
                                 │  │   Intent Router       │  │
                                 │  │  LOOKUP → RAG          │  │
                                 │  │  PATTERN/PREDICT/      │  │
                                 │  │  NETWORK → stub model  │  │
                                 │  └──────────────────────┘  │
                                 └──────────────────────────┘
```

The frontend never talks to the database or any AI model directly — everything passes through the FastAPI gateway, which decides what kind of query it's looking at and which pipeline should handle it.

---

## How It Works, Concept by Concept

### 1. The RAG Pipeline

RAG (Retrieval-Augmented Generation) means: instead of asking an LLM to answer purely from what it memorized during training, you first **retrieve** relevant documents from your own database, then **feed those documents to the LLM** and ask it to answer using only that context.

CrimeAI's RAG pipeline is built with **LlamaIndex**, using `VectorStoreIndex` for indexing and `PGVectorStore` to persist vectors inside PostgreSQL rather than a separate vector database. This means one database serves both structured FIR fields and unstructured semantic vectors.

**Flow:**
1. Officer submits a query ("show me FIRs involving a stolen two-wheeler near Malleshwaram").
2. Query text is embedded into a vector.
3. That vector is compared against the vectors of stored FIRs.
4. The most similar FIR chunks are retrieved.
5. Those chunks are handed to the LLM, which writes a natural-language answer *and* keeps track of exactly which chunk each sentence came from.

### 2. Cross-Lingual Retrieval with LaBSE

Most multilingual RAG systems handle a non-English query like this: **translate the query to English → then search English documents.** That's an extra network call and an extra point of failure before you've even started retrieving anything.

CrimeAI skips the translation step entirely for retrieval, using **LaBSE** (`sentence-transformers/LaBSE`) — a 768-dimension, Google-built sentence embedding model that maps *many languages into the same shared vector space*. Practically, this means a Kannada sentence and its English translation land in almost the same spot in vector space. So a Kannada query can be embedded directly and matched against English FIR text — no intermediate translation, no added latency, and no risk of a bad translation corrupting the search.

This only applies to **retrieval**. Once the RAG pipeline has picked the relevant chunks, translation (if needed for the final answer) happens separately — see below.

### 3. Language Detection

Before anything else happens, the system needs to know what language the query is in. This is done with **Meta's FastText**, using the lightweight compressed model `lid.176.ftz` (176 languages, small enough to load instantly).

FastText returns a confidence score along with its guess. Short queries and acronyms (e.g., "FIR") are genuinely ambiguous across languages, so CrimeAI has a **fallback rule**: if confidence is below `0.55`, default to English rather than trust a low-confidence guess.

### 4. Translation Engine

While retrieval doesn't need translation, the **final answer delivered to the officer** may need to be in Kannada. Translation here is handled by:
- **`ai4bharat/indictrans2-indic-en-dist-200M`** — Kannada → English
- **`ai4bharat/indictrans2-en-indic-dist-200M`** — English → Kannada

Both are distilled (200M parameter) IndicTrans2 models from AI4Bharat, chosen to be light enough to run locally rather than requiring a hosted API for every request.

One subtlety: IndicTrans2 was trained on **Devanagari-unified script**. Kannada script and Devanagari script represent the same underlying phonemes differently, so before translation, Kannada text is **transliterated into Devanagari** using the `indic-transliteration` library. This isn't a language change — it's a script conversion so the model can read the text the way it was trained to.

If local models fail to load (common under memory constraints, since these still require meaningful RAM/VRAM), the system falls back automatically to the **Google Cloud Translation API**.

### 5. Intent Routing

Not every query is a "look something up" question. Some are "find a pattern" or "predict something" questions, which need a fundamentally different kind of model (e.g., graph analysis, time-series prediction) rather than retrieval.

The **Intent Router** (`intent_router.py`) classifies each query into one of:
- `LOOKUP` → routed to the RAG pipeline described above
- `PATTERN` / `PREDICT` / `NETWORK` → routed to a predictive/graph model stub

This separation matters architecturally: it means the RAG pipeline never has to "pretend" to answer a prediction question it has no business answering, and the gateway can grow a real predictive model behind that stub later without touching the RAG code at all.

### 6. Chunking Strategy

A common RAG mistake is chunking documents with a sliding window (e.g., 512 tokens with 50-token overlap), which can slice a document mid-thought — separating, say, a suspect's description from the crime scene description in the same FIR.

CrimeAI uses **Atomic Document Mapping**: one FIR = one chunk, via LlamaIndex's `SentenceSplitter(chunk_size=1024, chunk_overlap=0)`. Since most FIRs fit comfortably under 1024 tokens, this guarantees that **retrieving a chunk means retrieving the entire case file's context**, with nothing fragmented across chunk boundaries. The tradeoff is that this strategy assumes FIRs are short enough to fit in one chunk — it wouldn't scale as-is to much longer documents.

### 7. Fallback Architecture

Every external dependency in CrimeAI has a fallback, so a single API outage doesn't take down the whole platform:

| Component | Primary | Fallback | Trigger |
|---|---|---|---|
| LLM synthesis | Zoho Catalyst QuickML | Groq (`llama-3.1-8b-instant`) | Zoho Catalyst call fails |
| Translation | Local IndicTrans2 (CPU/GPU) | Google Cloud Translation API | Local model fails to load/process (e.g., memory limits) |
| Language detection | FastText confidence ≥ 0.55 | Defaults to English | Confidence < 0.55 |

### 8. Explainability & Auditing

Two safeguards ensure the platform is usable in a legal/investigative context, where hallucinated facts are unacceptable:

- **Deterministic Explainability Panel**: the UI shows the literal, lexically-matched snippet the answer was drawn from — not a paraphrase, not a summary generated separately. If the LLM says something the snippet doesn't literally support, that mismatch is visible to the officer.
- **Audit Logging**: every transaction is logged to `audit_log.jsonl` and `translated_audit_log.jsonl`, recording latency, confidence bands, the routing decision (`LOOKUP` vs `PATTERN`/etc.), and the exact retrieved chunk scores — useful both for debugging and for accountability in a policing context.

---

## Tech Stack

**Frontend**
- Next.js 16.2.10 (App Router)
- React 19.2.4
- Tailwind CSS v4
- Leaflet / `react-leaflet` for spatial/geographic mapping

**Backend**
- FastAPI (Python) on Uvicorn
- LlamaIndex (`VectorStoreIndex`, `PGVectorStore`)
- PostgreSQL + `pgvector`

**AI / NLP models**
- LaBSE (`sentence-transformers/LaBSE`) — cross-lingual embeddings
- AI4Bharat IndicTrans2 (`indic-en-dist-200M`, `en-indic-dist-200M`) — translation
- Meta FastText (`lid.176.ftz`) — language detection
- Zoho Catalyst QuickML → Groq (`llama-3.1-8b-instant`) — LLM synthesis

---

## Project Structure

```
ksp/
├── ingest/
│   └── ingest.py          # Loads and embeds FIRs into pgvector
├── query/
│   ├── app.py              # FastAPI entrypoint (Intelligence Gateway)
│   ├── intent_router.py    # Classifies LOOKUP vs PATTERN/PREDICT/NETWORK
│   └── .env                # DB + API credentials
├── frontend/
│   └── ...                 # Next.js dashboard
└── requirements.txt
```

---

## Setup & Installation

### Prerequisites
- Node.js v18+
- Python v3.9+
- PostgreSQL with the `pgvector` extension installed

### Backend
```bash
pip install -r ksp/requirements.txt
```
Key dependencies: `llama-index==0.10.50`, `llama-index-vector-stores-postgres==0.1.3`, `psycopg2-binary==2.9.9`, `sentence-transformers==3.0.1`, `python-dotenv==1.0.1`, `openai==1.35.10`, plus `fastapi`, `uvicorn`, `fasttext`, `torch`, `transformers`, `indic-transliteration`.

### Frontend
```bash
cd ksp/frontend
npm install
```

---

## Running the Application

**Step 1 — Ingest FIR data into the vector database**
```bash
cd ksp/ingest
python ingest.py
```

**Step 2 — Start the backend gateway**
```bash
cd ksp/query
python app.py
```
Server boots on `http://0.0.0.0:8001`.

**Step 3 — Start the frontend**
```bash
cd ksp/frontend
npm run dev
```
Dashboard available at `http://localhost:3000`.

> Note: local IndicTrans2 and LaBSE models are memory-hungry. If you're running this on a machine without a GPU or with limited RAM, expect longer cold-start times or automatic fallback to the Google Cloud Translation API.

---

## Environment Variables

Set these in `.env` (inside `ksp/query/` or `ksp/`):

```
DB_HOST=
DB_PORT=
DB_NAME=
DB_USER=
DB_PASSWORD=
GROQ_API_KEY=
CATALYST_API_KEY=
```

---

## Known Limitations

- **PATTERN/PREDICT/NETWORK routing** currently hands off to a *stub* model, not a fully trained predictive/graph model — the routing logic works end-to-end, but the downstream model is a placeholder.
- **Chunking strategy** (1 FIR = 1 chunk) assumes FIRs are short enough to fit within a 1024-token window; it isn't designed to scale to much longer documents without revision.
- **Local model inference** for LaBSE/IndicTrans2 requires meaningful memory; deployment on constrained/free-tier hosting may require running inference in a separate notebook/terminal environment rather than as part of a persistent server process.
