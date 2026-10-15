# ADR 0019: Explainability Agent: deterministic SHAP evidence, LLM prose only from cited evidence, validated or not returned

**Status:** Accepted, 2026-10-15 (Phase 11).

**Context.**
- **Spec §16 and the Phase 11 prompt:** explain a match (positive evidence, gaps, model evidence, how to improve)
  in the spec's format, and explain the model as a whole. **Never invent an explanation.** The LLM must not compute SHAP
  values, contributions or the score (spec §4, rule 5).
- **Active models (ADR 0012, 0015):** LightGBM on E5 features; Logistic Regression for the TF-IDF fallback;
  Baseline 3 can be forced with `MATCHING_MODEL`. LLM: Gemini (ADR 0001).
- **Inputs:** the Matching Agent's `MatchResult`. It carries the prediction, the features-v1 vector (`trace.features`),
  the semantic similarity, the Phase 9 `SkillGapReport` and the seven version fields. The Phase 4 profile and Phase 5
  posting are read as `StudentView` / `JobView` from the matching repository.

**Decision.**

1. **The explainer is chosen per model, never assumed** (`src/scoring/shap_explainer.py`, reusing Phase 8's `Explainer`):

   | Model | Explainer | Output space |
   |---|---|---|
   | LightGBM, XGBoost, sklearn GB | `shap.TreeExplainer`, exact TreeSHAP | log-odds |
   | Decision tree, random forest | `shap.TreeExplainer` | probability |
   | Logistic Regression | `shap.LinearExplainer` (Phase 8's 200-row train background) | log-odds |
   | Baseline 3 | its own additive decomposition, labelled "not SHAP" | suitability |

   - The model is loaded exactly as Stage 4 loads it (`build_predictor`, with the manifest's SHA-256 checked first).
   - `get_shap_values` refuses a prediction made by another model version. It also refuses a feature vector that the
     model doesn't score to the predicted value (within 1e-6).
   - The contributions must add up to the model's output within 1e-6, or nothing is returned.
2. **Global and individual are separate code paths and outputs.**
   - `global_explanation` → `get_global_explanation` → `GlobalExplanation` holds:
     - importance = mean |SHAP| over **every** row of the model's validation split (equal values share a rank);
     - per-feature distributions (mean, std, percentiles, share ±, missing-value rows, a 20-bin histogram);
     - summary-plot points from a seeded 300-row sample, as data for Phase 13.
   - `explain_pair` → the match pipeline → `MatchExplanation`.
   - The agent's `mode` input picks the pipeline. No function serves both.
3. **Evidence first, prose second.** `generate_evidence` turns each input value into an `EvidenceItem`. Each item
   holds an id, a plain-language `fact`, the `anchors` a line must name, the `mentions` it may name, the only
   `numbers` it may state, the SHAP `sign`, and its `source` value.
   - **positive:** skills in `matched_skills`; transferable candidates; `education_match` / `branch_match` = 1;
     `experience_match` = 1 when the job states a minimum; similarity percentiles ≥ 0.75 (top quarter of all pairs).
   - **gaps:** `missing_required` and `missing_preferred`, with their importance.
   - **model:** SHAP values with a share of the pair's total |SHAP| ≥ 5%, at most 3 per direction
     (`config/explainability.yaml`).
4. **The LLM writes lines that cite evidence ids** (`ProseDraft`); it sees facts, never asked to compute.
   - **`validate_explanation` fails the draft if any line breaks one of these rules:**
     - its ids don't exist, or belong to another section;
     - it misses a cited item's anchor;
     - it names a skill or feature label that no cited item involves;
     - it states a number that isn't among its cited items' numbers;
     - a model line's direction words don't match the SHAP sign (or it mixes signs);
     - a positive line describes a gap;
     - a gap line says the student "lacks" a skill.
   - **The draft as a whole must also:**
     - cite every positive, gap and model item;
     - address the most important gap in "How to Improve".
   - **Failure:** the agent calls `generate_explanation` once more in strict mode, passing the errors. If that fails
     too, `get_match_explanation` returns the structured fallback: each item's own fact. The fallback is validated
     too, and the run ends `partial` with `explanation_fallback`.
   - `get_match_explanation` re-validates any draft it is given, so an unvalidated line can't be returned even by
     mistake.
5. **Signs and the score are never the LLM's.**
   - Each model line's "+"/"−" comes from its cited SHAP sign.
   - The score is the prediction's own value.
   - A last, computed Model Evidence line states **base value + all contributions = model output**. LightGBM's
     TreeSHAP base value is −9.64 log-odds, so a pair can have net positive contributions and still score 0%.
     Without this line that score would look unexplained.
6. **Gemini response schema:** nested models are inlined, and length/count limits are left to local Pydantic
   validation (`response_schema` in `app/llm/client.py`). The nested `$defs` schema got `400 INVALID_ARGUMENT`.

**Consequences.**
- An explanation is either fully traceable or replaced by its evidence. Real runs fall back whenever Gemini is
  unavailable: the free tier allows **20 requests per day per model**, and each request also spends one routing call.
- **The model evidence shows what LightGBM learned from the weak labels (ADR 0015).**
  - `branch_match` and `career_interest_alignment` dominate: global mean |SHAP| 1.98 and 1.13.
  - `required_skill_match_score` ranks 17th of 20 (0.010).
  - Skill counts can point the "wrong" way: on S00023 × J0071584, a missing-skill count of 6 *raised* the score
    (+0.72).
  - Explanations make this visible. They do not correct it: that needs features-v2 and retraining (ADR 0016).
- **Rejected:**
  - letting the LLM write free text and checking it afterwards with keyword matching (it can't tie a claim to a
    value);
  - a static SHAP image (Phase 13 renders from data);
  - KernelExplainer for every model (approximate and slow, where exact explainers exist).
