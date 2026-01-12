import re

NUM_RE = re.compile(r"(?<!\w)(-?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)(?!\w)")

STOP = {
    "what","was","the","value","of","at","in","on","for","to","and","or","is","are","by","a","an",
    "according","annual","report","end","year","year-end","period","listed","if","data","not","available",
    "return","true","false","mention","there","no","any","did","company","corporation","inc","limited",
    "value?","number","total"
}

def _keywords(question: str):
    q = question.lower()
    # вытащим годы
    years = set(re.findall(r"\b(19\d{2}|20\d{2})\b", q))
    # ключевые слова
    words = [w for w in re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", q) if w not in STOP]
    return set(words), years

def extract_number(question: str, hits: list[dict]):
    """
    Возвращает (число, hit) или (None, None)
    Идея: выбрать число, рядом с которым есть слова вопроса.
    """
    kw, years = _keywords(question)

    best = None  # (score, value, hit)
    for h in hits:
        txt = h.get("text", "")
        low = txt.lower()

        for m in NUM_RE.finditer(txt):
            raw = m.group(0).replace(" ", "").replace(",", "")
            try:
                val = float(raw)
            except:
                continue

            # отсекаем числа-ГОДЫ, если вопрос не про годовую метрику
            if int(val) in {int(y) for y in years}:
                continue
            if 1900 <= val <= 2100 and not years:
                # если в вопросе нет года, год в тексте почти всегда колонка таблицы
                continue

            # окно вокруг числа
            s = max(0, m.start() - 80)
            e = min(len(low), m.end() + 80)
            window = low[s:e]

            # сколько ключевых слов из вопроса встретилось рядом с числом
            hit_kw = sum(1 for w in kw if w in window)

            # небольшой бонус за хороший rerank_score если он есть
            bonus = float(h.get("rerank_score", 0.0))

            score = hit_kw + 0.1 * bonus

            if best is None or score > best[0]:
                best = (score, val, h)

    if best is None:
        return None, None
    window = low[s:e]

    mult = 1.0
    if "million" in window or "mn" in window:
        mult = 1_000_000.0
    elif "billion" in window or "bn" in window:
        mult = 1_000_000_000.0
    elif "thousand" in window or "in thousands" in window:
        mult = 1_000.0

    val = val * mult
    if years:
        hit_year = any(y in window for y in years)
        score = score + (2.0 if hit_year else 0.0)

    return best[1], best[2]
