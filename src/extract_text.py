from pathlib import Path
import json
import fitz  # PyMuPDF
from tqdm import tqdm

PDF_DIR = Path("data/pdfs")
OUT_DIR = Path("data/extracted")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_pdf(pdf_path: Path):
    doc = fitz.open(pdf_path)
    pages = []
    total_chars = 0

    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text("text")  # обычное извлечение текста
        text = text.strip()

        pages.append({
            "page_index": i,
            "text": text
        })
        total_chars += len(text)

    return {
        "pdf_name": pdf_path.name,
        "num_pages": len(doc),
        "total_chars": total_chars,
        "pages": pages
    }

def main():
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"Не нашёл PDF в {PDF_DIR.resolve()}")

    stats = []
    for pdf in tqdm(pdfs, desc="Extracting"):
        data = extract_pdf(pdf)
        stats.append({
            "pdf_name": data["pdf_name"],
            "num_pages": data["num_pages"],
            "total_chars": data["total_chars"],
            "looks_like_scanned": data["total_chars"] < 2000,  # грубый признак
        })

        out_path = OUT_DIR / f"{pdf.stem}.json"
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    stats_path = OUT_DIR / "_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nГотово. Сводка:")
    for s in stats:
        flag = "⚠️ возможно скан" if s["looks_like_scanned"] else "ok"
        print(f"- {s['pdf_name']}: pages={s['num_pages']}, chars={s['total_chars']} -> {flag}")

if __name__ == "__main__":
    main()
