# ADR 0018: Learning Agent: a curated course catalogue plus live YouTube results, never an invented resource

**Status:** Accepted, 2026-10-14 (Phase 10).

**Context.**
- **Spec §17 and the Phase 10 prompt:** recommend learning resources for a student's skill gaps, and never invent a
  course name, URL or detail. Every resource must trace to a real catalogue row or API call. A skill with nothing
  relevant must say "No resources found for this skill in the current catalog".
- **Data sources, chosen by the owner on 2026-10-14:**
  - option (c): 20 Coursera courses with their URLs, supplied by hand (`data/learning/curated_courses.csv`);
  - option (b): the YouTube Data API v3, with the owner's key, used **for every skill** alongside the catalogue.
- **Input:** Phase 9's `SkillGapReport` (ADR 0016), consumed as is.
- **Embedding model:** the one in use: E5 `e5-base-v2`, or TF-IDF when the fallback is active (ADR 0012).

**Decision.**

1. **The catalogue is built from the course pages, deterministically**
   (`python -m app.learning.catalog build` → `data/learning/course_catalog.json` → `load` → `learning_resources`).
   - Each listed URL is fetched once, and each field comes from a fixed pattern on that page:

     | Field | Where on the page |
     |---|---|
     | title | `og:title` |
     | provider | "Offered by" |
     | difficulty | "… level" |
     | prerequisites | an explicit statement after the level, such as "No prior experience required" |
     | effort | "N weeks at H hours a week", or "… to complete" |
     | skills | "Skills you'll gain" |
     | description | the "What you'll learn" bullets |

   - **No model summarises a page.** A field the page doesn't state is NULL and is shown as "Not available in
     source data". A bare "Recommended experience" heading is not treated as a statement: its detail isn't in the
     page source.
   - The page's SHA-256 and fetch time are in `source_reference`.
2. **YouTube.**
   - **Calls:**
     - `search.list` with q = "<skill> tutorial", English relevance hint, safeSearch strict, 5 results;
     - `videos.list` for the full description and duration.
   - **What is stored:** each video is a `learning_resources` row (`youtube:<videoId>`), with the API calls in
     `source_reference`.
   - **Durations:** `contentDetails.duration` is reformatted ("2 h 24 min 25 s"), never estimated.
   - **No difficulty or prerequisites** are given for videos: the API has none.
   - **Quota:**
     - a search costs 100 of the default 10,000 daily units;
     - searches are cached per query (table `youtube_searches`, migration 0010, 7 days);
     - at most 10 searches per request.
   - **When YouTube is unavailable** (no key, quota, network), the catalogue is used alone and the reason is
     reported per skill.
3. **The pipeline.** Each stage is a validated tool, run through the BaseAgent loop (`app/tools/learning_tools.py`,
   logic in `app/learning/`):
   - `prioritize_gaps`: by Phase 9's `importance_score`; equal scores share a priority and are listed as tied;
   - `retrieve_courses`;
   - `compute_course_similarity`;
   - `rank_courses`;
   - `explain_recommendations`;
   - `save_learning_recommendations`: a database failure is reported, never fatal (ADR 0003).
4. **Ranking factors** (the output's `ranking.factor_availability` reports, per run, how many candidates have each
   one):

   | Factor | How it is used |
   |---|---|
   | Relevance | cosine of the skill name ("query: ") and the resource text ("passage: " title, the source's skill list, description) under the configured model (E5 adds the "query: "/"passage: " prefixes; TF-IDF ignores them). **Ranks resources** (ties: catalogue course first, then title) |
   | Skill importance | from Phase 9. **Orders the skills** |
   | Similarity score | the relevance itself, shown per resource |
   | Difficulty, prerequisites, effort | shown as the source states them, **not scored**: the sources state them for some resources only, and no evidence says which level or length suits a student better |

5. **Threshold.** A resource is recommended only at relevance ≥ `min_relevance`, tuned per model on real output
   (`config/learning.yaml`): **E5 0.80** (primary), TF-IDF 0.15 (fallback).
   - TF-IDF is lexical: a skill sharing no word with a resource scores 0, and a title that repeats the skill's
     words ranks first, whatever its language or length. Below it, the skill gets "No resources found for this skill in the current catalog",
   with the best candidate's relevance in the explanation.
   - Top 3 per skill.
   - A model with no tuned threshold refuses to rank, rather than borrowing one.
6. **The contract.**
   - **`LearningRecommendation`** (one per missing skill) holds `resources`, each a `LearningResource` with
     `resource_name`, `resource_url`, `source`, `relevance_score`, `difficulty`, `prerequisites`,
     `estimated_effort`, `kind` and `provider`. The `source` is `<resource_id> | <source_reference>`.
   - **`LearningAgentOutput`** adds `sources` (catalogue size, YouTube status per skill), `ranking`, `saved` and the
     not-detected `note`.

**Consequences.**
- **The catalogue is small** (20 courses): many skills get only YouTube results, or none. That is the honest result,
  and it grows with the catalogue: add rows to `curated_courses.csv`, then `build` and `load`.
- **YouTube results are only as good as search.** Relevance filters off-topic videos, but language is only a hint
  (`relevanceLanguage=en`).
- **Re-tune `min_relevance`** if the embedding model changes (done once: E5 → TF-IDF, 2026-10-14).
