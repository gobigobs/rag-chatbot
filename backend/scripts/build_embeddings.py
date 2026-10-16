"""
Regenerate backend/data/embeddings.json from backend/data/corpus.json
using sentence-transformers (all-MiniLM-L6-v2, 384-dim, normalized).

Usage (from the backend/ directory):
    python scripts/build_embeddings.py
"""
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_NAME = "all-MiniLM-L6-v2"


def main() -> None:
    corpus_path = DATA_DIR / "corpus.json"
    out_path = DATA_DIR / "embeddings.json"

    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    texts = [chunk["text"] for chunk in corpus["chunks"]]
    vectors = model.encode(texts, normalize_embeddings=True)

    payload = {
        "model": MODEL_NAME,
        "dim": 384,
        "chunks": [
            {
                "id": chunk["id"],
                "title": chunk["title"],
                "text": chunk["text"],
                "embedding": [float(x) for x in vec],
            }
            for chunk, vec in zip(corpus["chunks"], vectors)
        ],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    print(f"Wrote {len(payload['chunks'])} embeddings -> {out_path}")


if __name__ == "__main__":
    main()
