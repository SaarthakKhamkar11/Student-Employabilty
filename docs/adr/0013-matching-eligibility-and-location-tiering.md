# ADR 0013: Matching Stage 1, hard eligibility rules and same-country-first tiering

**Status:** Accepted, 2026-10-09 (Phase 7).

**Context.**
- Spec §14 Stage 1: remove only jobs or candidates that clearly fail a hard requirement (degree, minimum
  experience, and skills configured as mandatory), and "do not make all requirements hard filters".
- The Phase 7 prompt also asks for:
  - the Phase 5 visibility filter ([ADR 0009](0009-gender-preference-visibility-filter.md));
  - a same-country-first location rule, described as "already decided in docs/ADR.md", reusing
    `match_student()` in `src/embeddings/tfidf_baseline.py`.
- Neither exists in this repository:
  - There is no `docs/ADR.md` (ADRs live in `docs/adr/`) and no ADR about location.
  - `tfidf_baseline.py` has no `match_student()`. It was re-implemented in Phase 4, because the original file
    was never in the repository (PROJECT_AUDIT.md §0, [ADR 0004](0004-interim-tfidf-embeddings.md)).
  - This ADR is therefore the first record of the location rule, and its only implementation is
    `app/matching/eligibility.py`.

**Decision** (`app/matching/eligibility.py`, settings in `config/matching_baseline_v1.yaml`):

1. **Rules, checked in order.** An item is counted under the first rule it fails, and the counts are returned
   with every result.

   | Rule | Excludes when |
   |---|---|
   | `near_duplicate` | the posting is a repost ([ADR 0010](0010-job-near-duplicate-detection.md)); its canonical posting stays |
   | `not_visible` | jobs for a student only: the employer's preference hides the posting (ADR 0009) |
   | `degree` | the job's minimum degree level (1 diploma … 4 doctorate) is above the student's |
   | `experience` | the job's minimum years exceed the student's stated experience + `experience_tolerance_years` (0) |
   | `mandatory_skill` | a skill listed in `mandatory_skills` (by `job_id` or role) is missing |

2. **Unknown values never exclude.**
   - A degree that can't be parsed, or a job with no stated minimum, is kept ([ADR 0007](0007-missing-value-sentinels.md)).
   - Stated experience is the longest duration the student states anywhere (internship lines, per-skill
     "Exprience - 18 months"). It is a lower bound: date ranges aren't parsed, and no stated duration means
     0 years.
3. **Only configured skills are hard.** `mandatory_skills` is empty, so by default no skill excludes anything.
   Every missing required skill is a score penalty instead (`required_skill_match_score`, skill gaps).
4. **Location tiering** (jobs for a student):
   - Jobs in the student's country form tier 1.
   - If fewer than `min_same_country` (50 = `retrieval.top_n`) remain after the other rules, jobs in other
     countries are added as tier 2.
   - Retrieval takes every tier-1 job before any tier-2 job, then orders by similarity within each tier.
   - Each result records its tier and the rule that was applied.
   - A student with no recorded country (all 166 Kaggle resumes) gets no tiering, never an exclusion.
5. **The visibility filter and tiering apply only to what a student sees.**
   - Ranking candidates for an employer never hides a candidate by gender (ADR 0009's concern), and never
     by location (the employer's call).
   - Pair mode (a student asking about one named job) reports the rules the pair fails and does not hide
     the job.

**Measured effect on the real data** (student S00019, placement cell, India; 4,992 stored postings):
- Excluded: 51 near-duplicates, 1,646 not visible, 1,633 by degree, 1,371 by experience.
- 291 postings remain eligible. Only 2 of them are in India, so the rule widened to all countries.
- The size of this is a data effect. `qualification_required` and `experience_required` are assigned at
  random per posting (Phase 6: one role carries a median of 7 degrees and 10 experience ranges). For a
  fresher, the degree and experience rules therefore drop most postings of every role at random. The
  roles themselves mostly survive, through whichever postings drew a bachelor's degree and 0 years.

**Across all 286 seeded students** (each against the 4,990 dataset postings):

| | Per student |
|---|---|
| Eligible postings | median 434 (mean 967, min 0, max 4,939) |
| Excluded by experience | mean 1,690 |
| Excluded by degree | mean 1,598 |
| Not visible | mean 684 |
| Near-duplicates | 51 |

- Location: the rule widened for all 120 placement-cell students (India), and wasn't applied for the 166 with no
  recorded country.
- **All 18 diploma holders get no eligible job at all.** Every dataset posting requires a bachelor's degree or
  higher (levels 2 to 4 only), so the degree rule removes everything. On this data the hard degree rule
  silently gives these students an empty recommendation list.

**Consequences.**
- On real job data these rules mean what they say. On this synthetic data they mainly decide which copy of a
  role survives.
- **Recommendation for the owner.** Keep the rules (the spec requires them), but consider
  `degree_filter: false` / `experience_filter: false` for demonstrations on this dataset. The features
  `education_match`, `experience_match` and `experience_gap` still score the gap either way. At the least, decide
  what diploma students should see: today the answer is nothing.
- The job corpus can be restricted per request (`job_source`: all / dataset / uploaded), so test uploads can
  be kept out of a demonstration on the dataset.
