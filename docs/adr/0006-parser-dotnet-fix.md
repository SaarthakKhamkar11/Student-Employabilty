# ADR 0006: Narrow bug fix in the existing resume parser (`.NET` lost its dot)

**Status:** Accepted, 2026-10-08

**Context.** The Phase 4 prompt says the existing parser must be wrapped, not rewritten, and that a genuine bug found during testing should be fixed narrowly and documented. In `src/parsing/resume_text_parser.py`, `extract_skills()` cleaned each item with `.strip(" .-*")`, which also removed a *leading* dot. So `.NET` became `NET`: a real skill was corrupted. It affected exactly 4 Kaggle resumes (S00340, S00347, S00351, S00353).

**Decision.** Only that line changed: a leading dot is kept when a letter follows it (`.NET`, `.js`), and trailing dots, bullets and dashes are still removed. A regression test is in `backend/tests/test_profile_components.py`.

**Consequences.**
- `python preprocess.py` regenerated `data/processed/students_unified.csv`: exactly those 4 rows changed. New SHA-256: `636d394f399c09e0c578a280b36e6294b333593418b80174161619d507c6b606`. `jobs.csv` and the data-quality report are unchanged.
- The database was re-seeded. The TF-IDF version is unchanged, because the tokenizer already ignored the dot.

**Known parser limitations found while testing, deliberately not changed** (they are heuristic behaviour, not narrow bugs; fixing them would change many profiles):
1. `extract_skills()` prefers a "Skill Details" block over a "Skills" block. Resume S00200 has a rich "Skills *" list at the top and a short "Skill Details" block at the end, so only 3 skills are extracted.
2. Section slicing can bleed: S00256's education text includes a job title, and S00200 has no education header, so its "education" is a job title.
3. Projects and certifications are split by line or bullet, never by comma, because titles contain commas. Comma-separated lists therefore stay one item.

These limitations belong to the existing parser. An LLM-assisted or rule-improved extractor must be measured against the same samples before it replaces anything.

**Addendum (Phase 5): second narrow fix, `guess_degree()` read the word "be" as a B.E.** Its `DEGREE_RE` is case-insensitive, so `\bB\.?\s?E\b` also matched the ordinary word "be": `guess_degree("candidates should be proficient")` returned `"BE"`. That mattered once job descriptions were parsed. The fix skips a bare `be`/`Be` that isn't all-caps; `BE`, `B.E` and `b.e` still match, and "Must be an MBA" now gives `MBA`. A check over all 166 Kaggle resumes found **0** degrees produced by the bug, so the processed data hashes are unchanged.

**Addendum (Phase 5): an extension, not a fix.** `_slice_section()` gained two optional, backward-compatible parameters, `markers` and `line_anchored`, and `parse_job_sections()` was added on top of them. With the defaults, resume behaviour is identical (verified by the unchanged data hashes).
