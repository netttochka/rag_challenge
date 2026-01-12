from pathlib import Path
import json
import numpy as np
import faiss
from tqdm import tqdm
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
    if not CHUNKS_PATH.exists():
        raise SystemExit("Сначала запусти build_index.py чтобы появился chunks.jsonl")

    model = SentenceTransformer(MODEL_NAME)

    texts = []
    metas = []
    for rec in tqdm(read_jsonl(CHUNKS_PATH), desc="Loading chunks"):
        texts.append(rec["text"])
        metas.append({"pdf_name": rec["pdf_name"], "page_index": rec["page_index"]})

    # эмбеддинги
    emb = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine ~ inner product при normalize_embeddings=True
    index.add(emb)

    faiss.write_index(index, str(INDEX_PATH))

    with META_PATH.open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print("Готово:")
    print(" -", INDEX_PATH)
    print(" -", META_PATH)
    print("Векторов:", index.ntotal)

if __name__ == "__main__":
    main()
