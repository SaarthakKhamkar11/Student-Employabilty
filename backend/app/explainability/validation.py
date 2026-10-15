"""validate_explanation: every generated line must trace to the evidence it cites (docs/adr/0019).

A line passes only if:
  1. it cites evidence ids that exist, of a kind its section may use (positive -> P, gaps / improve -> G,
     model evidence -> M);
  2. it names every anchor of every item it cites (the skill, or the feature's label);
  3. every skill name or feature label it names belongs to an item it cites (no unbacked claim);
  4. every number it states is one of its cited items' numbers (to the precision written);
  5. its wording matches the evidence: a model-evidence line states the SHAP direction of its items (all of one
     sign); a positive line doesn't describe a gap; a gap line says "not detected", never that the student lacks it.
And the explanation as a whole must cite every positive, gap and model-evidence item, and "How to Improve" must
cite the most important gap. Anything else fails: the agent retries once with stricter instructions, then falls
back to the structured evidence (spec §16: no unvalidated sentence is returned).
"""

import re

from app.schemas.agent_io import EvidenceBundle, EvidenceItem, ExplanationValidation, ProseDraft

SECTIONS = {"positive_evidence": "P", "potential_gaps": "G", "model_evidence": "M", "how_to_improve": "G"}
_POS = re.compile(r"\b(rais|increas|positiv|boost|help|support|strengthen|lift|improv)\w*")
_NEG = re.compile(r"\b(reduc|lower|decreas|negativ|hurt|weaken|penali[sz]|subtract|drag|pull\w* down)\w*")
_GAP_IN_POSITIVE = re.compile(r"\b(not detected|not found|missing|lack\w*|absent|no evidence|gap)\b")
_LACKS = re.compile(r"\b(lack\w*|does not have|doesn't have|has no|without any|incapable|unable)\b")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
MASK = "§"


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[-_‐-―]", " ", text.lower())).strip()


def _pattern(term: str) -> re.Pattern:
    return re.compile(r"(?<![a-z0-9])" + re.escape(normalise(term)) + r"(?![a-z0-9])")


def _contains(norm_text: str, term: str) -> bool:
    return bool(_pattern(term).search(norm_text))


def _number_ok(token: str, allowed: list[float]) -> bool:
    decimals = len(token.split(".")[1]) if "." in token else 0
    value, tolerance = float(token), 0.5 * 10 ** -decimals + 1e-9
    return any(abs(value - abs(a)) <= tolerance for a in allowed)


def check_line(section: str, text: str, ids: list[str], items: dict[str, EvidenceItem],
               known_terms: list[str]) -> list[str]:
    where = f"{section} line {text[:60]!r}"
    errors = []
    if "\n" in text:
        errors.append(f"{where}: contains a line break")
    unknown = [i for i in ids if i not in items]
    if unknown:
        return errors + [f"{where}: cites evidence that does not exist: {unknown}"]
    wrong = [i for i in ids if not i.startswith(SECTIONS[section])]
    if wrong:
        errors.append(f"{where}: cites {wrong}, which {section} may not use")
    cited = [items[i] for i in ids]
    norm = normalise(text)
    permitted = set()
    for item in cited:
        for anchor in item.anchors:
            if not _contains(norm, anchor):
                errors.append(f"{where}: cites {item.id} but does not name {anchor!r}")
        permitted |= {normalise(t) for t in item.anchors + item.mentions}
    masked = norm
    for term in known_terms:  # longest first: "data analysis tools" before "data analysis"
        pattern = _pattern(term)
        if pattern.search(masked):
            if normalise(term) not in permitted:
                errors.append(f"{where}: names {term!r}, which none of its cited evidence involves")
            masked = pattern.sub(lambda m: MASK * len(m.group(0)), masked)
    allowed = [n for item in cited for n in item.numbers]
    stray = [t for t in _NUMBER.findall(masked) if not _number_ok(t, allowed)]
    if stray:
        errors.append(f"{where}: states number(s) {stray} that its cited evidence does not contain")
    if section == "model_evidence":
        signs = {item.sign for item in cited}
        if len(signs) > 1:
            errors.append(f"{where}: mixes contributions that raised and lowered the score")
        elif signs == {1} and (not _POS.search(masked) or _NEG.search(masked)):
            errors.append(f"{where}: its SHAP evidence raised the score, the line must say so (and only that)")
        elif signs == {-1} and (not _NEG.search(masked) or _POS.search(masked)):
            errors.append(f"{where}: its SHAP evidence lowered the score, the line must say so (and only that)")
    elif section == "positive_evidence" and _GAP_IN_POSITIVE.search(masked):
        errors.append(f"{where}: positive evidence must not describe a gap")
    elif section in ("potential_gaps", "how_to_improve") and _LACKS.search(masked):
        errors.append(f"{where}: a missing skill is 'not detected', never something the student lacks")
    return errors


def validate(bundle: EvidenceBundle, draft: ProseDraft, attempt: int) -> ExplanationValidation:
    items = {i.id: i for i in bundle.positive + bundle.gaps + bundle.model}
    errors: list[str] = []
    lines = 0
    cited_in: dict[str, set[str]] = {s: set() for s in SECTIONS}
    for section in SECTIONS:
        for line in getattr(draft, section):
            lines += 1
            errors += check_line(section, line.text, line.evidence_ids, items, bundle.known_terms)
            cited_in[section] |= set(line.evidence_ids)
    for section, required in (("positive_evidence", bundle.positive), ("potential_gaps", bundle.gaps),
                              ("model_evidence", bundle.model)):
        left_out = [i.id for i in required if i.id not in cited_in[section]]
        if left_out:
            errors.append(f"{section}: leaves out evidence {left_out}")
    if bundle.improve:
        top = items[bundle.improve[0]].source.get("importance_score")
        best = {i for i in bundle.improve if items[i].source.get("importance_score") == top} or {bundle.improve[0]}
        if not cited_in["how_to_improve"] & best:
            errors.append(f"how_to_improve: must address the most important gap ({sorted(best)})")
        outside = cited_in["how_to_improve"] - set(bundle.improve)
        if outside:
            errors.append(f"how_to_improve: cites {sorted(outside)}, which are not improvement targets")
    elif draft.how_to_improve:
        errors.append("how_to_improve: there is no skill gap to improve on")
    return ExplanationValidation(attempt=attempt, valid=not errors, errors=errors[:25], lines_checked=lines)
