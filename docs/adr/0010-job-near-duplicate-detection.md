# ADR 0010: Near-duplicate job detection at ingest time

**Status:** Accepted, 2026-10-08

**Context.** Spec §5.3 says "remove duplicate jobs", and §26 requires near-duplicate detection. The job data is heavily templated: **4,990 postings share only 376 distinct descriptions, exactly one per role**, and all 376 roles reuse the same text across different companies.

**Decision** (`app/jobs/dedup.py`, `job-dedup-v1`). Two postings are near-duplicates when they have:
- the same company,
- the same job title (both case- and space-normalised), and
- a Jaccard similarity of at least 0.9 between word 3-shingles of description + responsibilities + skills.

Text similarity alone would flag almost every same-role posting.
- The canonical posting is the smallest `job_id`; the others get `jobs.near_duplicate_of`.
- **Batch:** `python -m app.db.seed_jobs` flags them across the set and logs the count.
- **Single JD:** the `check_duplicates` tool compares against stored jobs before saving. A free-text JD that duplicates a stored job is **not stored again**; its posting points at the existing `job_id`. If the check itself fails, the posting is not stored (fail closed).

**Measured on the real 4,990-row sample.** **51 near-duplicates** in 50 groups (49 pairs, 1 triple), all with similarity 1.0. Within each group the company, title, role and text are identical, but country, location, salary, work type, experience range, qualification and sometimes gender preference differ: the same posting text republished under different conditions. They are **flagged, not deleted**. They are separate rows with distinct IDs, and Phase 14's leakage report must keep each group within one data split.

**Limitations.** A changed company name or title evades detection, and "different company, identical template" is deliberately not a duplicate. The threshold of 0.9 is a choice, not a tuned value; on this data every flagged pair is exactly 1.0, so its exact value has no effect.

**Addendum, 2026-10-09 (Phase 7): `job-dedup-v2`.**
- **The gap.** v1 skipped the check for any posting without a company name. Uploaded JDs never carry one, so
  the same JD uploaded twice was stored twice. Matching then recommended both copies at ranks 1 and 2 (found
  while testing the Matching Agent with the demo's repeated upload).
- **The fix.** A posting with no company is compared only with stored postings that also have no company and
  the same title, using the same 0.9 Jaccard rule. A dataset posting (which always has a company) never
  merges with a company-less upload.
- **Verified.**
  - Re-running v2 over the 4,990 dataset rows flags the same 51 near-duplicates as v1.
  - The existing duplicate upload in the database (`JUc0bbd66eaa`) was flagged as `near_duplicate_of`
    `JU9119e88bb4` (similarity 1.0). Nothing was deleted.
