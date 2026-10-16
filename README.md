# RAG Chatbot — About Gobinath Rangaraj

A retrieval-augmented generation (RAG) chatbot that answers questions about
**Gobinath Rangaraj** (Senior Software Engineer & AI Product Engineer) using
only a curated 14-chunk knowledge base. Every answer is extractive — quoted
from the source chunks — with citations, similarity scores, and an abstention
fallback when nothing is relevant.

## Architecture

```
┌──────────────┐      HTTP /api      ┌─────────────────────────┐
│  React (Vite)│  ───────────────▶  │  FastAPI backend         │
│  chat UI     │  ◀───────────────  │  app.py                  │
└──────────────┘   JSON {answer,     └───────────┬─────────────┘
                   sources, confidence}          │
                                                 ▼
                                    ┌─────────────────────────┐
                                    │ 1. Embed question with  │
                                    │    all-MiniLM-L6-v2     │
                                    │ 2. Hybrid retrieval:    │
                                    │    0.75 cosine sim +    │
                                    │    0.25 keyword overlap │
                                    │    (top-3 chunks)       │
                                    │ 3. Extractive answer:   │
                                    │    best sentence quoted │
                                    │    + chunk title        │
                                    │ 4. Abstain if top score │
                                    │    < 0.35               │
                                    └─────────────────────────┘
```

## How the RAG pipeline works

1. **Chunking** — `backend/data/corpus.json` holds 14 hand-written chunks about
   Gobinath (summary, roles, skills, projects, certifications, contact, job
   preferences, brand). It is the only source of truth — the bot never invents
   facts.
2. **Embeddings** — `scripts/build_embeddings.py` encodes each chunk with
   `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim, L2-normalized) and
   writes `backend/data/embeddings.json`, which ships with the repo so the
   server starts fast.
3. **Hybrid retrieval** — at query time the question is embedded with the same
   model and scored against the stored embeddings: `0.75 * cosine similarity
   + 0.25 * normalized keyword overlap` (chunk title + text, stopwords removed,
   naive plural stemming). Pure cosine alone ranked the wrong chunk first for
   several natural questions (e.g. "What is Gobinath's experience?" surfaced
   "Job Preferences"), so the keyword term keeps lexical anchors like
   "experience", "contact", and "project" in play. The top-3 chunks are
   retrieved; scores stay in [0, 1] so the abstention calibration still holds.
4. **Cited extractive answer** — the most query-relevant sentence(s) from the
   top chunks are quoted verbatim with their chunk titles, and each source is
   returned with its title, excerpt, and similarity score.
5. **Abstention** — if the top similarity score is below **0.35**, the bot
   replies "I don't have information about that in my knowledge base." instead
   of guessing.

## Backend setup / run

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

- `GET /api/health` → `{"status": "ok"}`
- `POST /api/chat` with `{"question": "..."}` →
  `{"answer": "...", "sources": [{"title", "excerpt", "score"}], "confidence": 0.0}`

To regenerate embeddings from the corpus:

```bash
cd backend
source .venv/bin/activate
python scripts/build_embeddings.py
```

CORS is enabled for the Vite dev server (`http://localhost:5173`).

Note: on machines where the proxy env vars contain IPv6 bracket entries
(`no_proxy`/`NO_PROXY` with `[::1]`), `huggingface_hub`'s URL validator fails
and the model can't load — start the server with a sanitized proxy env, e.g.
`no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1 uvicorn app:app ...`.
The model is cached after the first successful load.

## Frontend setup / run

```bash
cd frontend
npm install
npm run dev        # opens on http://localhost:5173
```

`vite.config.js` proxies `/api/*` to `http://localhost:8000`, so the React app
calls the backend with plain `fetch('/api/chat', ...)`.

## Eval approach

The project follows eval-driven development (mirroring Gobinath's RAG Document
Q&A eval harness):

- A fixed question set over the 14 chunks measures **accuracy**: does the
  retrieved top-1 chunk match the question's expected chunk?
- **Abstention calibration**: sweep the similarity threshold and check that
  in-scope questions stay above it while off-topic questions (e.g. "What is
  the weather today?") score below it. The default `0.35` was chosen from this
  calibration.
- Retrieval-only scoring (TF-IDF vs SBERT style comparisons) can be re-run
  with the same corpus before changing the answerer, so improvements are
  measured, not guessed.

## Project structure

```
rag-chatbot/
├── backend/
│   ├── app.py                  # FastAPI app: retrieval + cited answers
│   ├── requirements.txt
│   ├── data/
│   │   ├── corpus.json         # 14-chunk knowledge base (source of truth)
│   │   └── embeddings.json     # pre-computed MiniLM embeddings
│   └── scripts/
│       └── build_embeddings.py # regenerate embeddings.json
├── frontend/
│   ├── package.json
│   ├── vite.config.js          # /api proxy → localhost:8000
│   ├── index.html
│   └── src/
│       ├── main.jsx
│       ├── App.jsx             # chat UI: bubbles, chips, citations
│       └── styles.css          # dark teal/indigo theme
├── README.md
└── .gitignore
```
