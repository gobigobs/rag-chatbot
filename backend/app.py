"""
RAG chatbot about Gobinath Rangaraj.

Pipeline: corpus chunks -> pre-computed sentence-transformer embeddings
-> cosine-similarity retrieval (top-3) -> extractive cited answer with
abstention when the top score falls below a calibrated threshold.
"""
import json
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3
ABSTAIN_THRESHOLD = 0.35
ABSTAIN_MESSAGE = "I don't have information about that in my knowledge base."

# Hybrid retrieval weights: dense cosine similarity plus a small keyword
# overlap term (chunk title + text, stemmed). Pure cosine ranked the wrong
# chunk top-1 for several natural questions (e.g. "What is Gobinath's
# experience?" -> "Job Preferences"), so the keyword term keeps lexical
# anchors like "experience", "contact", "project" in play. Scores stay in
# [0, 1], so the 0.35 abstention calibration still applies.
COSINE_WEIGHT = 0.75
KEYWORD_WEIGHT = 0.25

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "and", "or", "but", "with",
    "about", "his", "her", "he", "she", "it", "they", "what", "which",
    "who", "whom", "whose", "how", "when", "where", "why", "do", "does",
    "did", "can", "could", "would", "should", "has", "have", "had",
    "you", "your", "i", "me", "my", "we", "our", "as", "by", "from",
    "that", "this", "these", "those", "not", "no", "so", "if", "then",
    "than", "too", "very", "just", "also", "there", "their", "them",
}

app = FastAPI(title="Gobinath Rangaraj RAG Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------
# Lazy-loaded retrieval state
# --------------------------------------------------------------------------
_model = None
_chunks: List[dict] = []
_embeddings: Optional[np.ndarray] = None
_chunk_keywords: List[set] = []


def _load_corpus() -> None:
    """Load the corpus and pre-computed embeddings once, on first chat call."""
    global _model, _chunks, _embeddings, _chunk_keywords
    if _embeddings is not None:
        return

    corpus_path = DATA_DIR / "corpus.json"
    embeddings_path = DATA_DIR / "embeddings.json"
    if not corpus_path.exists():
        raise RuntimeError(f"Missing corpus file: {corpus_path}")
    if not embeddings_path.exists():
        raise RuntimeError(f"Missing embeddings file: {embeddings_path}")

    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)
    with open(embeddings_path, encoding="utf-8") as f:
        stored = json.load(f)

    if stored.get("model") != MODEL_NAME:
        raise RuntimeError(
            f"Embeddings were built with {stored.get('model')!r}, "
            f"expected {MODEL_NAME!r}. Re-run scripts/build_embeddings.py."
        )
    if stored.get("dim") != 384:
        raise RuntimeError(f"Unexpected embedding dim {stored.get('dim')}, expected 384.")

    by_id = {c["id"]: c for c in stored["chunks"]}
    _chunks = []
    for entry in corpus["chunks"]:
        stored_chunk = by_id.get(entry["id"])
        if stored_chunk is None:
            raise RuntimeError(f"No pre-computed embedding for chunk {entry['id']!r}.")
        _chunks.append(
            {
                "id": entry["id"],
                "title": stored_chunk["title"],
                "text": stored_chunk["text"],
                "embedding": stored_chunk["embedding"],
            }
        )
    _embeddings = np.array([c["embedding"] for c in _chunks], dtype=np.float32)
    _chunk_keywords = [
        _keywords(c["title"] + " " + c["text"]) for c in _chunks
    ]

    from sentence_transformers import SentenceTransformer

    _model = SentenceTransformer(MODEL_NAME)


def _embed_query(question: str) -> np.ndarray:
    vec = _model.encode(question, convert_to_numpy=True, normalize_embeddings=True)
    return vec.astype(np.float32)


def _retrieve(question: str, top_k: int = TOP_K) -> List[dict]:
    """Hybrid retrieval: dense cosine similarity blended with normalized
    keyword overlap (0 if the question has no keywords)."""
    query_vec = _embed_query(question)
    cosine = _embeddings @ query_vec  # cosine similarity on unit vectors
    q_kw = _keywords(question)
    if q_kw:
        overlap = np.array(
            [len(q_kw & ck_kw) / len(q_kw) for ck_kw in _chunk_keywords],
            dtype=np.float32,
        )
    else:
        overlap = np.zeros(len(_chunks), dtype=np.float32)
    scores = COSINE_WEIGHT * cosine + KEYWORD_WEIGHT * overlap
    idx = np.argsort(scores)[::-1][:top_k]
    return [
        {"chunk": _chunks[int(i)], "score": float(scores[int(i)])} for i in idx
    ]


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _stem(word: str) -> str:
    if word.endswith("s") and len(word) > 4 and not word.endswith("ss"):
        return word[:-1]
    return word


def _keywords(text: str) -> set:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {_stem(t) for t in tokens if t not in STOPWORDS and len(t) > 2}


def _extractive_answer(question: str, hits: List[dict]) -> str:
    """Pick the most query-relevant sentence(s) from the top chunks and quote them."""
    q_kw = _keywords(question)
    best: Optional[tuple] = None
    for hit in hits:
        chunk = hit["chunk"]
        for sent in _sentences(chunk["text"]):
            s_kw = _keywords(sent)
            overlap = len(q_kw & s_kw)
            # small bonus for sentences containing distinctive capitalized terms
            if not q_kw:
                overlap = 0
            score = overlap + 0.5 * hit["score"]
            if best is None or score > best[0]:
                best = (score, chunk, sent)
    assert best is not None
    _, chunk, sent = best
    quote = sent if len(sent) <= 280 else sent[:277].rstrip() + "..."
    return f'According to "{chunk["title"]}": "{quote}"'


def _excerpt(text: str, length: int = 200) -> str:
    text = " ".join(text.split())
    return text if len(text) <= length else text[:length].rstrip() + "..."


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str


class Source(BaseModel):
    title: str
    excerpt: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
    confidence: float


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    _load_corpus()
    hits = _retrieve(question)

    top_score = hits[0]["score"]
    if top_score < ABSTAIN_THRESHOLD:
        return ChatResponse(
            answer=ABSTAIN_MESSAGE,
            sources=[],
            confidence=round(top_score, 4),
        )

    answer = _extractive_answer(question, hits)
    sources = [
        Source(
            title=hit["chunk"]["title"],
            excerpt=_excerpt(hit["chunk"]["text"]),
            score=round(hit["score"], 4),
        )
        for hit in hits
    ]
    return ChatResponse(
        answer=answer,
        sources=sources,
        confidence=round(top_score, 4),
    )
