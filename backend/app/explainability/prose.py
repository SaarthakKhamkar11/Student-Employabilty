"""The LLM's only job here: restate validated evidence items as short lines (docs/adr/0019, spec §16).

The LLM receives the evidence items (facts, the terms each line must name, SHAP directions) and returns a
ProseDraft whose every line cites the items it states. It computes nothing: the score, the contributions and the
skill lists come from Phases 7-9's tools, and the "+"/"-" of each model-evidence line comes from the SHAP sign,
not from the LLM. The structured fallback prints each item's own fact instead.
"""

import json

from app.core.sanitize import INJECTION_GUARD, wrap_untrusted
from app.schemas.agent_io import EvidenceBundle, EvidenceItem, ProseDraft, ProseLine

_SYSTEM = (
    "You write the explanation of one student-job match for the student. You receive evidence items, each with an "
    "id, a fact and the terms a line about it must name. Rewrite the facts as short, plain lines in the requested "
    "sections. You never add, estimate or compute anything: every line restates one or more evidence items, lists "
    "their ids in evidence_ids, and contains every must_name term of every item it cites, verbatim. "
    "Name no skill or feature other than the cited items' must_name and may_also_name terms. State no number "
    "that is not in a cited fact. A skill that was not found is 'not detected': never say the student lacks it. "
    "Do not start lines with '+' or '-' (the system adds them). "
)


def system_prompt() -> str:
    from app.agents.base import GUARDRAILS  # lazy: app.agents imports the tools that import this module

    return _SYSTEM + GUARDRAILS + " " + INJECTION_GUARD

SECTION_RULES = (
    "Sections:\n"
    "- positive_evidence: cite every P item (several in one line is fine). Style: 'Python requirement satisfied', "
    "'Project relevance at the 91st percentile of student-job pairs'. Never describe a gap here.\n"
    "- potential_gaps: cite every G item. Style: 'Power BI not detected (required)'.\n"
    "- model_evidence: cite every M item. A line about items whose direction is 'raised' says the feature "
    "'contributed positively' or 'raised the score'; for 'lowered', say it 'reduced the score'. Never put a "
    "raised and a lowered item in one line.\n"
    "- how_to_improve: one or two sentences citing only improve_targets, starting with the first one, suggesting "
    "the student develops those skills. No course or resource names. Empty when improve_targets is empty.\n"
)


def _item(i: EvidenceItem) -> dict:
    out = {"id": i.id, "fact": i.fact, "must_name": i.anchors}
    if i.mentions:
        out["may_also_name"] = i.mentions
    if i.sign is not None:
        out["direction"] = "raised" if i.sign > 0 else "lowered"
    return out


def build_prompt(bundle: EvidenceBundle, strict: bool = False, previous_errors: list[str] | None = None) -> str:
    payload = {"positive_items": [_item(i) for i in bundle.positive], "gap_items": [_item(i) for i in bundle.gaps],
               "model_evidence_items": [_item(i) for i in bundle.model], "improve_targets": bundle.improve}
    # Skill names come from résumés and job descriptions: data, never instructions.
    prompt = ("Evidence for one student-job match (JSON):\n"
              + wrap_untrusted(json.dumps(payload, ensure_ascii=False), label="explanation_evidence",
                               max_chars=30_000)
              + "\n\n" + SECTION_RULES)
    if strict:
        prompt += (
            "\nSTRICT MODE. Your previous answer failed validation:\n"
            + "\n".join(f"* {e}" for e in (previous_errors or [])[:15])
            + "\nWrite exactly one line per evidence item, close to its fact; copy each must_name term exactly; "
              "copy numbers exactly as the fact writes them, or leave them out; add no other words that name a "
              "skill, feature or number.\n")
    return prompt


def structured_draft(bundle: EvidenceBundle) -> ProseDraft:
    """The fallback: each item's own fact, one line each. Built from the evidence, so it always validates."""
    line = lambda i: ProseLine(text=i.fact, evidence_ids=[i.id])  # noqa: E731
    improve = []
    if bundle.improve:
        gaps = {g.id: g for g in bundle.gaps}
        required = [gaps[i] for i in bundle.improve if gaps[i].kind == "missing_required"]
        targets = (required or [gaps[i] for i in bundle.improve])[:3]
        kind = "required" if targets[0].kind == "missing_required" else "preferred"
        names = [t.source["skill"] for t in targets]
        listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        tied = len({t.source.get("importance_score") for t in targets}) == 1 and len(targets) > 1
        improve.append(ProseLine(
            text=f"Develop {listed}: {kind} skills not detected, "
                 + ("of equal model importance (their order is not a ranking)." if tied
                    else "most important to the model first."),
            evidence_ids=[t.id for t in targets]))
    return ProseDraft(positive_evidence=[line(i) for i in bundle.positive],
                      potential_gaps=[line(i) for i in bundle.gaps],
                      model_evidence=[line(i) for i in bundle.model], how_to_improve=improve)


def rendered_lines(bundle: EvidenceBundle, draft: ProseDraft) -> dict[str, list[str]]:
    """Prefix each line: '+' positive, '-' gap, and for model evidence the SHAP sign of the items it cites."""
    signs = {i.id: i.sign for i in bundle.model}
    model = []
    for line in draft.model_evidence:
        sign = signs[line.evidence_ids[0]]
        model.append(f"{'+' if sign > 0 else '-'} {line.text}")
    return {"positive_evidence": [f"+ {line.text}" for line in draft.positive_evidence],
            "potential_gaps": [f"- {line.text}" for line in draft.potential_gaps],
            "model_evidence": model,
            "how_to_improve": [line.text for line in draft.how_to_improve]}


def output_line(contributions: dict) -> str:
    """The last Model Evidence line, computed here (never by the LLM): where the score starts and how all the
    contributions add up to it. Without it, a pair whose listed contributions all raise the score could still
    show a low score with no visible reason (the base value of a model trained on rare positives is low)."""
    values, space = contributions["values"], contributions["output_space"]
    total, n = sum(values.values()), sum(1 for v in values.values() if v)
    if contributions["base_value"] == 0 and space == "suitability":
        return f"= All {n} contributions sum to the suitability {contributions['model_output']:.3f}"
    return (f"= Base value {contributions['base_value']:+.3f} {space} (the model's starting point before this "
            f"pair's features) + all {n} contributions {total:+.3f} = {contributions['model_output']:+.3f} {space}")


def render(suitability: float, lines: dict[str, list[str]]) -> str:
    """The spec's format (MASTER_SPEC §16). Empty sections say so instead of inventing a line."""
    def block(title, rows, empty):
        return f"{title}:\n" + "\n".join(rows or [empty])

    return "\n\n".join([
        f"Suitability Score: {round(suitability * 100)}%",
        block("Positive Evidence", lines["positive_evidence"], "Insufficient evidence"),
        block("Potential Gaps", lines["potential_gaps"], "No skill gaps identified in the skill-gap report"),
        block("Model Evidence", lines["model_evidence"], "Insufficient evidence"),
        block("How to Improve", lines["how_to_improve"], "No skill gap to address in the skill-gap report."),
    ])
