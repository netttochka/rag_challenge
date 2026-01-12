import re
from rank_bm25 import BM25Okapi

def _tok(s: str):
    return re.findall(r"[A-Za-z0-9]+", s.lower())

def bm25_rerank(question: str, hits: list[dict], top_n: int = 15) -> list[dict]:
    docs = [_tok(h["text"]) for h in hits]
    bm25 = BM25Okapi(docs)
    scores = bm25.get_scores(_tok(question))
    for h, s in zip(hits, scores):
        h["bm25_score"] = float(s)
    hits.sort(key=lambda x: x["bm25_score"], reverse=True)
    return hits[:top_n]
