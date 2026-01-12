from pathlib import Path
import json
import re
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[1]
EXTRACTED_DIR = BASE_DIR / "data" / "extracted"
OUT_PATH = BASE_DIR / "data" / "chunks.jsonl"

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def chunk_text(text: str, chunk_size: int = 900, chunk_overlap: int = 150):
    """
    Очень простой чанкер по словам.
    chunk_size/overlap — в символах (примерно), можно потом тюнить.
    """
    text = normalize_ws(text)
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap
        if start < 0:
            start = 0
        if end == len(text):
            break
    return chunks

def main():
    files = sorted(f for f in EXTRACTED_DIR.glob("*.json") if not f.name.startswith("_"))
    if not files:
        raise SystemExit(f"Не нашёл извлечённые json в {EXTRACTED_DIR}")

    total = 0
    with OUT_PATH.open("w", encoding="utf-8") as f_out:
        for fp in tqdm(files, desc="Chunking"):
            data = json.loads(fp.read_text(encoding="utf-8"))
            pdf_name = data["pdf_name"]
            data = json.loads(fp.read_text(encoding="utf-8"))

            # data может быть dict (нормальный формат) или list (иногда так сохраняют страницы)
            if isinstance(data, dict):
                pdf_name = data.get("pdf_name", fp.with_suffix(".pdf").name)
                pages = data.get("pages", [])
            elif isinstance(data, list):
                pdf_name = fp.with_suffix(".pdf").name
                pages = data
            else:
                raise TypeError(f"Неожиданный формат {type(data)} в файле {fp.name}")

            for i, p in enumerate(pages):
                page_index = p.get("page_index", i)
                text = p.get("text", "")
                for ch in chunk_text(text):
                    rec = {
                        "text": ch,
                        "pdf_name": pdf_name,
                        "page_index": page_index,
                    }
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total += 1

    print(f"Готово. Чанков: {total}")
    print(f"Файл: {OUT_PATH}")

if __name__ == "__main__":
    main()
