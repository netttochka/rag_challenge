from __future__ import annotations

from pathlib import Path
import json
import hashlib
import re
from typing import Any, Dict, List, Tuple, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


BASE_DIR = Path(__file__).resolve().parents[1]

QUESTIONS_PATH = BASE_DIR / "data" / "questions.json"
RETRIEVED_PATH = BASE_DIR / "data" / "retrieved.jsonl"
PDF_DIR = BASE_DIR / "data" / "pdfs"

OUT_PATH = BASE_DIR / "motovilova_v9.json"

EMAIL = "test@rag-tat.com"
SUBMISSION_NAME = "motovilova_v9"

TOP_K_HITS = 20


llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0,
    timeout=120,
)


_SHA1_CACHE: Dict[str, str] = {}


def sha1_of_file(path: Path) -> str:
    key = str(path)
    if key in _SHA1_CACHE:
        return _SHA1_CACHE[key]
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    digest = h.hexdigest()
    _SHA1_CACHE[key] = digest
    return digest


def clean_text(s: str) -> str:
    return (s or "").replace("\u2009", " ").replace("\xa0", " ").strip()


def norm_boolean(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = clean_text(str(x)).lower()
    s2 = re.sub(r"[^\w]", "", s)
    if s2 in ("true", "yes", "y", "1"):
        return True
    if s2 in ("false", "no", "n", "0"):
        return False
    if re.search(r"\b(true|yes|found|mentioned|includes|contains)\b", s, re.I):
        return True
    return False


def norm_number(x: Any) -> Any:
    if x is None:
        return "N/A"
    if isinstance(x, (int, float)):
        return float(x)
    s = clean_text(str(x))
    if re.search(r"\b(n/?a|n\.a\.|no data|not available|unknown)\b", s, re.I):
        return "N/A"
    m = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", s)
    if not m:
        return "N/A"
    num = m.group(0).replace(",", "")
    if num.endswith("."):
        num = num[:-1]
    try:
        return float(num)
    except Exception:
        return "N/A"


def norm_name(x: Any) -> str:
    if x is None:
        return "N/A"
    s = clean_text(str(x))
    if not s:
        return "N/A"
    if re.search(r"\b(n/?a|no data|not available|not mentioned|none|null|unknown)\b", s, re.I):
        return "N/A"
    first = s.splitlines()[0].strip()
    first = re.sub(r"^(answer|company|result)\s*:\s*", "", first, flags=re.I).strip()
    first = first.rstrip(" .;,")
    return first if first else "N/A"


def norm_names(x: Any) -> str:
    if x is None:
        return "N/A"
    s = clean_text(str(x))
    if not s:
        return "N/A"
    if re.search(r"\b(n/?a|no data|not available|not mentioned|none|null|unknown)\b", s, re.I):
        return "N/A"
    parts = re.split(r"[,\n;]+", s)
    parts = [p.strip().rstrip(" .;,") for p in parts if p.strip()]
    return ", ".join(parts) if parts else "N/A"


def is_empty_value(value: Any, kind: str) -> bool:
    if kind == "boolean":
        return value is False
    if kind == "number":
        return value == "N/A" or value is None
    if kind in ("name", "names"):
        return value == "N/A" or value is None or str(value).strip() == ""
    return value is None or str(value).strip() == ""


def safe_json_loads(s: str) -> Optional[dict]:
    if not s:
        return None
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def build_numbered_context(hits: List[Dict[str, Any]], k: int) -> str:
    blocks = []
    for idx, h in enumerate(hits[:k], start=1):
        pdf_name = h.get("pdf_name", "unknown.pdf")
        page = h.get("page_index", "?")
        txt = clean_text(h.get("text", ""))
        if not txt:
            continue
        blocks.append(f"ID {idx} | {pdf_name} | page {page}\n{txt}")
    return "\n\n".join(blocks)


def build_prompt(question: str, kind: str, context: str) -> str:
    if kind == "boolean":
        rule = "Return True only if the statement is clearly supported by at least one context block. Otherwise return False."
        out = '{"value": true/false, "support_ids": [1,2,...]}'
    elif kind == "number":
        rule = "Return the exact number only if it directly answers the question. Otherwise return N/A."
        out = '{"value": 123.45 or "N/A", "support_ids": [1,2,...]}'
    elif kind == "name":
        rule = "Return the exact name as written in the context. Otherwise return N/A."
        out = '{"value": "Name" or "N/A", "support_ids": [1,2,...]}'
    elif kind == "names":
        rule = "Return a comma-separated list exactly as in the context. Otherwise return N/A."
        out = '{"value": "A, B, C" or "N/A", "support_ids": [1,2,...]}'
    else:
        rule = "Return the answer only if supported. Otherwise return N/A."
        out = '{"value": "..." or "N/A", "support_ids": [1,2,...]}'

    return f"""
You must answer using ONLY the context blocks below.

{rule}

Output MUST be valid JSON with exactly these keys:
{out}

support_ids must include ONLY IDs of blocks that directly contain the evidence.
If not found, support_ids must be [].

Question type: {kind}
Question: {question}

Context blocks:
{context}
""".strip()


def pick_refs_from_support(hits: List[Dict[str, Any]], support_ids: List[int]) -> List[Dict[str, Any]]:
    seen: set[Tuple[str, int]] = set()
    refs: List[Dict[str, Any]] = []
    for sid in support_ids:
        if not isinstance(sid, int):
            continue
        if sid < 1 or sid > len(hits):
            continue
        h = hits[sid - 1]
        pdf_name = h.get("pdf_name")
        page_index = h.get("page_index")
        if not pdf_name or page_index is None:
            continue
        key = (pdf_name, int(page_index))
        if key in seen:
            continue
        seen.add(key)
        refs.append({
            "pdf_sha1": sha1_of_file(PDF_DIR / pdf_name),
            "page_index": int(page_index),
        })
    return refs


def answer_question(q_text: str, kind: str, hits: List[Dict[str, Any]]) -> Tuple[Any, List[Dict[str, Any]]]:
    hits = (hits or [])[:TOP_K_HITS]
    context = build_numbered_context(hits, TOP_K_HITS)

    if not context.strip():
        if kind == "boolean":
            return False, []
        if kind == "number":
            return "N/A", []
        return "N/A", []

    prompt = build_prompt(q_text, kind, context)
    resp = llm.invoke([HumanMessage(content=prompt)])
    raw = clean_text(resp.content or "")

    data = safe_json_loads(raw) or {}
    value_raw = data.get("value", None)
    support_ids = data.get("support_ids", [])
    if not isinstance(support_ids, list):
        support_ids = []

    if kind == "boolean":
        value = norm_boolean(value_raw)
    elif kind == "number":
        value = norm_number(value_raw)
    elif kind == "name":
        value = norm_name(value_raw)
    elif kind == "names":
        value = norm_names(value_raw)
    else:
        value = clean_text(str(value_raw)) if value_raw is not None else "N/A"
        if not value:
            value = "N/A"

    if is_empty_value(value, kind):
        return value, []

    refs = pick_refs_from_support(hits, [int(x) for x in support_ids if isinstance(x, (int, float, str)) and str(x).isdigit()])

    if not refs:
        refs = pick_refs_from_support(hits, [1])

    return value, refs


def main():
    questions = json.load(open(QUESTIONS_PATH, encoding="utf-8"))

    retrieved: List[Dict[str, Any]] = []
    with open(RETRIEVED_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            retrieved.append(json.loads(line))

    n = min(len(questions), len(retrieved))

    answers: List[Dict[str, Any]] = []

    for i in range(n):
        q = questions[i]
        r = retrieved[i]

        q_text = q["text"]
        kind = q.get("kind", "text")
        hits = (r.get("hits") or [])

        value, refs = answer_question(q_text, kind, hits)

        answers.append({
            "question_text": q_text,
            "value": value,
            "references": refs
        })

        if (i + 1) % 20 == 0:
            print(f"Progress {i+1}/{n}", flush=True)

    submission = {
        "email": EMAIL,
        "submission_name": SUBMISSION_NAME,
        "answers": answers
    }

    OUT_PATH.write_text(
        json.dumps(submission, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("DONE:", OUT_PATH, flush=True)


if __name__ == "__main__":
    main()
