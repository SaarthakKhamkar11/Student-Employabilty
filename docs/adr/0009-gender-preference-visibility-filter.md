# ADR 0009: An employer's gender preference is a visibility filter only, never a scoring feature

**Status:** Accepted, 2026-10-08, as the **project owner's decision**, with a recorded concern (below).

**Context.**
- The job data has `gender_preference` ∈ {both, male, female}. The Phase 5 prompt states the decision: it filters which postings a student sees, and is never an input to matching or scoring.
- The prompt refers to an earlier record of this ("ADR-002 in docs/ADR.md"), but that file never existed in this repository, so this ADR is its first written record.

**Decision.**
- `app/jobs/visibility.py` `is_visible_to(preference, student_gender)`, exposed as the `check_job_visibility` tool for the Matching Agent to call **before** scoring:
  - "both", or an unknown/missing preference → visible
  - an unknown/missing **student** gender → visible. Never hide on missing data: 166 Kaggle students and every resume upload have no gender.
  - otherwise visible only if the two match
- The field lives only in `JobPosting.visibility`. It is **not** in `JobPosting.SCORING_FIELDS`, and it is not in the matching feature contract (`app/matching/features.py`), which rejects protected attributes in code.
- `backend/tests/test_fairness_guards.py` asserts that no feature list contains it, that the guard rejects it, that a posting's scoring view never carries it, and that no module outside the visibility path even references it.
- The student's own gender is never part of `StudentProfile` (ADR 0007, spec §11).
- The filter can be switched off: `JOB_GENDER_VISIBILITY_FILTER=false`.

**Concern recorded.**
- Hiding job postings from students because of their gender reproduces the employer's gender restriction. Gender-restricted job advertising is unlawful or restricted in many jurisdictions.
- In a real placement setting this filter would itself be a discriminatory treatment of students, independent of the scoring model.
- **Recommendation:** default the filter to *off*, and treat a gender-restricted posting as a data-quality and compliance flag for the placement cell, rather than silently hiding it.
- The fairness audit (Phase 18) should report how many postings the filter hides per group.
