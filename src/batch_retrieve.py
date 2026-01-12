from pathlib import Path
import json
import faiss
from sentence_transformers import SentenceTransformer
from rerank import rerank


BASE_DIR = Path(__file__).resolve().parents[1]
INDEX_PATH = BASE_DIR / "data" / "faiss.index"
META_PATH = BASE_DIR / "data" / "faiss_meta.jsonl"
CHUNKS_PATH = BASE_DIR / "data" / "chunks.jsonl"

# TODO: поставь правильное имя файла с вопросами:
QUESTIONS_PATH = BASE_DIR / "data" / "questions.json"

OUT_PATH = BASE_DIR / "data" / "retrieved.jsonl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def load_questions(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    # Попробуем угадать формат
    if isinstance(data, list):
        # либо ["q", ...] либо [{"question":...}, ...]
        if data and isinstance(data[0], dict):
            return [x.get("question") or x.get("text") or x.get("query") for x in data]
        return data
    if isinstance(data, dict):
        # { "questions": [...] }
        if "questions" in data:
            q = data["questions"]
            if q and isinstance(q[0], dict):
                return [x.get("question") or x.get("text") or x.get("query") for x in q]
            return q
    raise ValueError("Не понял формат questions.json")

def main():
    if not INDEX_PATH.exists():
        raise SystemExit("Нет индекса. Запусти index_faiss.py")
    if not QUESTIONS_PATH.exists():
        raise SystemExit(f"Нет файла вопросов: {QUESTIONS_PATH}")

    metas = list(read_jsonl(META_PATH))
    texts = [rec["text"] for rec in read_jsonl(CHUNKS_PATH)]

    index = faiss.read_index(str(INDEX_PATH))
    model = SentenceTransformer(MODEL_NAME)

    questions = load_questions(QUESTIONS_PATH)

    k = 50
    with OUT_PATH.open("w", encoding="utf-8") as f_out:
        for qi, q in enumerate(questions):
            q = (q or "").strip()
            if not q:
                # сохраняем пустой вопрос как запись без hits, чтобы порядок не ломался
                rec = {"question_index": qi, "question": q, "hits": []}
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            qv = model.encode([q], normalize_embeddings=True).astype("float32")
            scores, ids = index.search(qv, k)

            hits = []
            for idx, score in zip(ids[0], scores[0]):
                m = metas[idx]
                hits.append({
                    "score": float(score),
                    "pdf_name": m["pdf_name"],
                    "page_index": m["page_index"],
                    "text": texts[idx]
                })



            rec = {
                "question_index": qi,
                "question": q,
                "hits": hits
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Готово:", OUT_PATH)

if __name__ == "__main__":
    main()
