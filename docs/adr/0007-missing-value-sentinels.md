# ADR 0007: Missing values are sentinels, never defaults ("leave blank, never fabricate")

**Status:** Accepted, 2026-10-08. Formalises the convention `preprocess.py` already followed; the prompts call it "ADR-006 in docs/ADR.md", a file that never existed here.

**Decision.** A value that is not evidenced is one of these literal strings, never a default or guess:

| Sentinel | Meaning | Example |
|---|---|---|
| `Skill not detected` | the skills were looked for and none were found | S00256's resume has an empty "Skill Details" section |
| `Not detected` | a field or section was looked for and not found | a placement-cell student left certifications blank |
| `Not yet evaluated` | no extractor exists yet, the stage failed, or the model has not been selected | a resume's branch or CGPA; the Kaggle rows' projects (never extracted by `src/pipeline`); matching results before Phase 7 |
| `Insufficient evidence` | a result could not be produced from the evidence available | an agent skipped because its upstream agent failed |

**Rules.**
- Absence of evidence is not evidence of absence: "Skill not detected" never means "the student lacks the skill" (spec §5.2).
- A failed extraction is "Not yet evaluated", never "Skill not detected". The difference matters for skill-gap analysis.
- `missing_fields` lists every sentinel field in a profile, so gaps are explicit.
- In the database, a missing value is NULL (with the sentinel in the JSON profile). It is never 0, an empty string or a placeholder.
