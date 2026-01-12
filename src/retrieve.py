from pathlib import Path
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[1]
CHUNKS_PATH = BASE_DIR / "data" / "chunks.jsonl"
INDEX_PATH = BASE_DIR / "data" / "faiss.index"
META_PATH = BASE_DIR / "data" / "faiss_meta.jsonl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def main():
    if not INDEX_PATH.exists():
        raise SystemExit("Сначала запусти index_faiss.py")

    # грузим метаданные и тексты параллельно
    metas = list(read_jsonl(META_PATH))
    texts = [rec["text"] for rec in read_jsonl(CHUNKS_PATH)]

    index = faiss.read_index(str(INDEX_PATH))
    model = SentenceTransformer(MODEL_NAME)

    q = input("Вопрос: ").strip()
    qv = model.encode([q], normalize_embeddings=True).astype("float32")

    k = 5
    scores, ids = index.search(qv, k)

    print("\nTOP:", k)
    for rank, (idx, score) in enumerate(zip(ids[0], scores[0]), start=1):
        m = metas[idx]
        print(f"\n#{rank} score={float(score):.4f}  {m['pdf_name']}  page={m['page_index']}")
        print(texts[idx][:800], "..." if len(texts[idx]) > 800 else "")

if __name__ == "__main__":
    main()
