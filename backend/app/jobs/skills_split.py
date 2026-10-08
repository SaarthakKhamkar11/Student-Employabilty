"""Split job skill text into skill phrases.

The dataset's `skills` field is phrases run together with no delimiter
("Event planning Conference logistics Budget management"), each starting
with a capital letter. A new phrase starts at a capitalised word unless:
- the previous word is a joiner ("and", "of", "e.g." ...),
- the previous word is itself capitalised ("Adobe Creative Suite"), or
- we are inside parentheses.

A run made only of capitalised words ("Problem-solving Communication") is
then split again if every part is a known skill in the shared alias table.
Free-text JDs with explicit separators (commas, bullets, new lines) are
split on those instead.

Known limits (docs/adr/0008): under-splitting of runs of single capitalised
words that are not all known skills ("Reporting GIS (...)"), and
over-splitting when a phrase contains a proper noun after a lowercase word.
"""

import re

from app.profile.skills import load_alias_table, skill_key

_JOINERS = {"and", "or", "of", "in", "for", "with", "to", "the", "a", "an", "&", "e.g.", "e.g.,", "on", "as",
            "including", "like", "such"}
_EXPLICIT_SEPARATORS = re.compile(r"[,;\n•▪·]|\s[-*]\s|^\s*[-*]\s", re.M)


def _known_segmentation(words: list[str]) -> list[str] | None:
    """Split a run of words into known skills (alias table) if that covers every word; else None."""
    table = load_alias_table()
    n = len(words)
    best: list[list[str] | None] = [None] * (n + 1)
    best[0] = []
    for i in range(n):
        if best[i] is None:
            continue
        for j in range(n, i, -1):  # longest match first
            if skill_key(" ".join(words[i:j])) in table and best[j] is None:
                best[j] = best[i] + [" ".join(words[i:j])]
    return best[n] if best[n] and len(best[n]) > 1 else None


def _split_undelimited(text: str) -> list[str]:
    phrases, current, depth = [], [], 0
    for word in text.split():
        prev = current[-1] if current else None
        if (prev is not None and depth == 0 and word[0].isupper() and prev.lower() not in _JOINERS
                and (not prev[0].isupper() or prev.endswith(")"))):
            phrases.append(current)
            current = []
        current.append(word)
        depth = max(0, depth + word.count("(") - word.count(")"))
    if current:
        phrases.append(current)
    out = []
    for words in phrases:
        capitalised_run = len(words) > 1 and all(w[0].isupper() for w in words) and "(" not in " ".join(words)
        out += (_known_segmentation(words) if capitalised_run else None) or [" ".join(words)]
    return out


def split_skill_phrases(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    # Separators count only outside parentheses: "(e.g., Node.js, Python)" is part of one phrase.
    if _EXPLICIT_SEPARATORS.search(re.sub(r"\([^)]*\)", "", text)):
        parts = [p.strip(" .-*•\t") for p in _EXPLICIT_SEPARATORS.split(text)]
        # "Social media platforms (e.g., Facebook, Twitter)" must not be split inside parentheses
        merged, depth = [], 0
        for p in parts:
            if depth > 0 and merged:
                merged[-1] += ", " + p
            elif p:
                merged.append(p)
            depth = max(0, depth + p.count("(") - p.count(")"))
        return [m for m in merged if m][:40]
    return _split_undelimited(text)[:40]
