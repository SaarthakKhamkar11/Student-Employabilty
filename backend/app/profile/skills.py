"""Skill normalisation against the auditable alias table (data/skill_aliases.csv).

normalize_skills() maps raw skill strings to canonical names and never invents
one. A string the table doesn't know is kept as written and marked
method="unmapped", so a reviewer can see exactly what still needs an alias.
"""

import csv
import re
from functools import lru_cache
from pathlib import Path

from app.schemas.student import NormalizedSkill

ALIAS_TABLE_PATH = Path(__file__).resolve().parent / "data" / "skill_aliases.csv"

# Kaggle resumes list skills as "Python- Exprience - 24 months" (sic). The suffix is evidence, not part of the name.
_EXPERIENCE_SUFFIX = re.compile(r"\s*-\s*Exp[e]?rience\s*-\s*(.*)$", re.I)
_LEADING_ENUM = re.compile(r"^\s*(?:\d+\s*[.)]|[-*•·▪➢✓])\s*")
# "(e.g., Docker, Kubernetes)" lists examples, not the skill itself (Phase 5, seen in jobs.csv).
_EXAMPLES = re.compile(r"\(\s*(?:e\.?\s?g\.?|i\.?\s?e\.?|such as|like|including)\b[^)]*\)?", re.I)
_TRIM = " \t\r\n.,;:-*•\"'"
_TRIM_LEFT = _TRIM.replace(".", "")  # keep the leading dot of ".NET"


def skill_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.rstrip(_TRIM).lstrip(_TRIM_LEFT).lower())


def clean_skill_item(item: str) -> tuple[list[str], str | None]:
    """Split one raw item into skill pieces, plus any stated experience. 'Python (Pandas' -> ['Python', 'Pandas']."""
    evidence = None
    match = _EXPERIENCE_SUFFIX.search(item)
    if match:
        evidence = re.sub(r"\s*months?\s*$", "", match.group(1).strip(), flags=re.I).strip() or None
        if evidence and evidence.isdigit():
            evidence += " months"
        item = item[: match.start()]
    item = _EXAMPLES.sub("", _LEADING_ENUM.sub("", item))
    pieces = [p.rstrip(_TRIM + "()").lstrip(_TRIM_LEFT + "()") for p in re.split(r"[()]", item)]
    return [p for p in pieces if 0 < len(p) <= 60], evidence


@lru_cache
def load_alias_table(path: Path = ALIAS_TABLE_PATH) -> dict[str, tuple[str, str, bool]]:
    """skill_key(alias) -> (canonical, category, is_canonical_itself)."""
    table = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            table[skill_key(row["alias"])] = (row["canonical"], row["category"], row["alias"] == row["canonical"])
    return table


def alias_table_version(path: Path = ALIAS_TABLE_PATH) -> str:
    import hashlib

    return "aliases-" + hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def normalize_skills(raw_skills: list[str]) -> list[NormalizedSkill]:
    table = load_alias_table()
    out: dict[str, NormalizedSkill] = {}
    for raw in raw_skills:
        pieces, evidence = clean_skill_item(raw)
        expanded = []
        for piece in pieces:
            parts = [p.strip() for p in piece.split("/")]
            # "Git/GitHub", "HTML/CSS": split only when the whole item is unknown and every part is known
            if skill_key(piece) not in table and len(parts) > 1 and all(skill_key(p) in table for p in parts):
                expanded += parts
            else:
                expanded.append(piece)
        for piece in expanded:
            hit = table.get(skill_key(piece))
            if hit:
                canonical, category, exact = hit
                skill = NormalizedSkill(raw=raw, canonical=canonical, category=category,
                                        method="exact" if exact else "alias", evidence=evidence)
            else:
                skill = NormalizedSkill(raw=raw, canonical=piece, category="unclassified", method="unmapped",
                                        evidence=evidence)
            out.setdefault(skill.canonical.lower(), skill)  # first occurrence wins
    return list(out.values())
