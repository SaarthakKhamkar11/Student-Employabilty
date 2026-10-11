# ADR 0016: Skill-gap analysis: one shared normaliser, counterfactual importance, tuned relation tiers

**Status:** Accepted, 2026-10-11 (Phase 9).

**Context.**
- **What the Phase 9 prompt asks for:**
  - a standalone skill-gap module that the Matching Agent's Stage 6 wraps;
  - missing skills split into required and preferred;
  - importance derived from the actual matching model by counterfactual re-scoring;
  - transferable and partially related skills, by skill-name similarity under the configured embedding model;
  - a `SkillGapReport` contract that the Learning Agent (Phase 10) consumes.
- **Two premises of the prompt don't hold yet:**
  - **No matching model has been selected.** Phase 8 leaves Baseline 3 active until the human labels exist (ADR
    0015).
  - **No embedding model has been selected.** Phase 6 leaves the interim TF-IDF in place (ADR 0012).

  So "the actual model" means the **active** one: whatever `config/matching_model.yaml` names, the same predictor
  `predict_suitability` uses. "The Phase 6 model" means the **configured** one (`EMBEDDING_MODEL`).
- **Name clash.** Phase 7 already had a per-pair `SkillGap` schema. The Phase 9 contract needs `SkillGap` to be one
  missing skill.

**Decision.**

1. **The module.** `src/scoring/skill_gap_analysis.py`, with the method written in its docstring.
   - Stage 6 (`pipeline.skill_gaps` → the `identify_skill_gaps` tool) calls `analyse()` for each ranked candidate.
   - Phase 7's `skill_gap()` builder was removed.
   - Phase 7's `compare_skills` stays: the features (Stage 3) use it, and the counterfactuals re-run it.
2. **One normaliser.**
   - Student and job skills are canonical names from `app/profile/skills.py` `normalize_skills` (Phases 4 and 5),
     compared by its `skill_key`.
   - Raw lists go through the same function (`from_raw_skills`).
3. **Importance: counterfactual re-scoring.**
   - For each missing skill s: `suitability(profile + s) − suitability(profile)`.
   - Both runs go through `build_features` and the active predictor, batched per pair.
   - **Text-similarity features are held fixed.** The counterfactual is "the skill is detected in the skill list",
     not "the profile was rewritten".
   - The report names the model (`importance_model`). Missing lists are ordered by importance, then alphabetically.
4. **Relation tiers** (`config/skill_gap.yaml`). Per missing skill, the student's skill with the most similar
   name:
   - similarity ≥ `transferable` → `transferable_candidate`;
   - `partial` ≤ similarity < `transferable` → `partially_related`;
   - below `partial`: no relation is reported.

   Thresholds are per embedding model family. A model without an entry gets no tiers (the method says "Not yet
   evaluated"); thresholds are never borrowed from another model.
5. **Thresholds for TF-IDF: partial 0.40, transferable 0.60**, tuned on real output (evidence below).
   - These Phase 9 tiers are for the report only.
   - The features keep Phase 7's calibrated relatedness threshold (0.3), so `features-v1` and the Phase 8 artifacts
     are unchanged.
6. **The contract** (`app/schemas/agent_io.py`):
   - **`SkillGap`**: one not-detected skill. Fields: `skill`, `importance_score`, `transferable_candidate`,
     `transferable_similarity`.
   - **`PartiallyRelatedSkill`**: `skill`, `requirement`, `related_student_skill`, `similarity`.
   - **`SkillGapReport`**: `matched_skills`, `missing_required`, `missing_preferred`, `partially_related`, plus
     provenance: `importance_model`, `similarity_model`, `thresholds`, `method`, and a `note` that keeps the
     not-detected language.
   - **Uses:** `MatchResult.skill_gap` is a `SkillGapReport`. `LearningAgentInput` carries `skill_gap_report`, plus
     the missing skill names ordered by importance. Changing the shape means updating Phase 10.

**Evidence** (2026-10-11, `tfidf-953a75d166d8`).
- **Data for the thresholds.** Every job skill not detected in the profile, over the 360 pool pairs plus each
  student's 10 most similar jobs: 2,849 pairs, 12,022 not-detected skills.
  - Quantiles of the best-match similarity: median 0.00, p75 0.46, p90 0.70, p98 0.84.
  - Author's spot check of every distinct pair ≥ 0.40 (162) plus samples below:

    | Similarity band | Credibly related |
    |---|---|
    | 0.20–0.40 | ~50% |
    | 0.40–0.50 | 93% |
    | 0.50–0.60 | 91% |
    | 0.60–0.70 | 90% |
    | 0.70–0.80 | 80% |
    | 0.80–1.00 | 85% |

  - **The cliff is at 0.40.** Above it, precision doesn't rise: TF-IDF is lexical, so "Model Evaluation" ←
    "MODEL-VIEW-CONTROLLER" scores 0.804.
  - **0.60** is where the same skill in other words starts to dominate: "Python programming" ← "Python" 0.647.
  - **Known misses:** "Proficiency in HTML" ← "HTML" scores 0.542, so it lands in the partially related tier.
- **Real reports.** Three pairs (near-perfect, moderate, poor) were analysed with Baseline 3, and one of them also
  with LightGBM (forced by the override). They are reproduced in the Phase 9 entry of `PROGRESS.md`.
  - For S00001 × J0974289, the required gaps score +0.095 under Baseline 3 and +0.259 under LightGBM: **the number
    is the model's own.**

**Consequences and limits.**
- **Importance ties within a requirement class.** `features-v1` sees skills only through counts and ratios, so every
  model scores "add Data Modeling" exactly like "add Data Visualization". Importance does separate:
  - required from preferred;
  - unrelated from already-related;
  - one model from another.

  Ranking skills of the same class needs per-skill features (`features-v2`, then retraining). Until then, ties are
  broken alphabetically and labelled as ties.
- **Exact skill matches are rare** (most job skill phrases are wordier than student skills; Phase 5: 92% are not
  in the alias table). The transferable tier carries much of the signal: "Data analysis techniques" ← "Data
  Analysis" 0.78.
- **A semantic embedding model (ADR 0012) needs its own threshold entry**, tuned the same way.
