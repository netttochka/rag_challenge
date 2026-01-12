from sentence_transformers import CrossEncoder
_R = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(question: str, hits: list[dict], top_n: int = 5) -> list[dict]:
    pairs = [(question, h["text"]) for h in hits]
    scores = _R.predict(pairs, batch_size=16, show_progress_bar=False)
    for h, s in zip(hits, scores):
        h["rerank_score"] = float(s)
    hits.sort(key=lambda x: x["rerank_score"], reverse=True)
    return hits[:top_n]
