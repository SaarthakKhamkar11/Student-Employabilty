"""Job visibility by the employer's stated gender preference (docs/adr/0009).

The project owner's decision: `gender_preference` is a VISIBILITY FILTER ONLY.
The Matching Agent calls is_visible_to() before scoring. The field is never a
scoring feature, and it is not among JobPosting.SCORING_FIELDS.

Safeguards:
- A student whose gender is unknown is never hidden from a job (166 Kaggle
  rows and every resume upload have no gender).
- A missing or unrecognised preference counts as "both".
- The whole filter can be switched off (JOB_GENDER_VISIBILITY_FILTER=false).

Concern recorded in ADR 0009: hiding postings by gender reproduces the
employer's gender restriction, which is unlawful or restricted in many
jurisdictions.
"""

_KNOWN = {"male", "female"}


def is_visible_to(gender_preference: str | None, student_gender: str | None, *, enabled: bool = True) -> bool:
    if not enabled:
        return True
    preference = (gender_preference or "").strip().lower()
    student = (student_gender or "").strip().lower()
    if preference not in _KNOWN or student not in _KNOWN:
        return True  # "both", missing or unknown on either side: never hide
    return preference == student


def posting_visible_to(visibility: dict | str | None, student_gender: str | None, *, enabled: bool = True) -> bool:
    """The same rule, given a JobPosting's `visibility` block (or its "Not yet evaluated" sentinel).

    The Matching Agent calls this, so the employer's preference stays confined to this module and the
    posting schema: matching code never reads or names the field (tests/test_fairness_guards.py).
    """
    preference = visibility.get("gender_preference") if isinstance(visibility, dict) else None
    return is_visible_to(preference, student_gender, enabled=enabled)
