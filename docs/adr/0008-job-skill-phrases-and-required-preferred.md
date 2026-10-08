# ADR 0008: Job skill phrases and the INFERRED required/preferred split

**Status:** Accepted, 2026-10-08 (heuristic v2)

**Context.**
- Spec §5.3 requires required skills to be kept distinct from preferred skills.
- The job data (`jobs.csv`) has one flat `skills` field: phrases run together with no delimiter ("Event planning Conference logistics Budget management").
- Its 4,990 descriptions contain **zero** occurrences of "must have", "nice to have", "preferred" or "a plus". **The data never says which skills are required**, so any split is an inference.

**Decision. Splitting phrases** (`backend/app/jobs/skills_split.py`): a new phrase starts at a capitalised word, unless the previous word is a joiner ("and", "e.g." ...), the previous word is itself capitalised, or the word is inside parentheses. A run made only of capitalised words is split again when every part is a known skill in the shared alias table ("Problem-solving Communication"). Text with explicit separators (comma, bullet, new line) *outside* parentheses is split on those. On the data this gives about 5.6 skills per job.

**Decision. The required/preferred heuristic** (`backend/app/jobs/requirements.py`, `required-preferred-heuristic-v2`). The first rule that fires is recorded per skill as `requirement_basis`:

| Rule | Result | Count on 4,990 jobs |
|---|---|---|
| Skill came from a "Requirements" / "Must have" section of a free-text JD | required | 0 (none in the data) |
| Skill came from a "Nice to have" / "Preferred" section | preferred | 0 |
| A sentence naming the skill uses preference phrasing ("a plus", "nice to have", "preferred") / requirement phrasing ("must have", "is required", "mandatory") | preferred / required | 0 |
| At least 2/3 of the skill's content words appear in the description or responsibilities (related forms match: plan/planning, coordinate/coordination, manage/management) | **required** | 11,938 |
| Otherwise | **preferred** | 15,741 |
| No skill was required: the first *inferred* one is promoted (explicitly preferred skills are never promoted) | required | 101 jobs |

Overall, 43.3% of 27,780 job skills are inferred as required.

**Iterations driven by real output** (all measured on the full dataset):
- **v1's cue words** "essential", "required" and "advantage" produced only false positives ("provide *essential* healthcare services", "infrastructure *required* for web applications", "competitive *advantage*"). They were narrowed to phrasing that really expresses a requirement.
- **Commas inside "(e.g., Node.js, Python)"** switched the splitter into comma mode and merged whole skill lists into one phrase. Separators now count only outside parentheses.
- **v1's stemmer** turned "planning" into "plann", so an Event Coordinator's "Event planning" was labelled *preferred*. v2 handles doubled consonants and related word forms, which cut the fallback from 379 jobs to 101.
- **The fallback** could override an explicitly preferred skill. It now never does.

**Known limitations. This is not ground truth.**
- **Required ≈ "named in the role text".** That is a proxy for importance, not a statement by the employer. Some core skills are labelled preferred because the templated descriptions don't repeat them. Example: J0334731 Transportation Planner, "Traffic engineering", where the text says "traffic management".
- **The 2/3 word-overlap rule** misses synonyms and paraphrases. Prefix matching can rarely link unrelated words (e.g. "event"/"eventual").
- **Splitting** still under-splits runs of single capitalised words that are not all known skills: "Documentation Empathy Communication skills" (J0558153), "Reporting GIS (...)". That is roughly 0.1% of phrases.
- **The shared alias table** maps only about 8% of job skill phrases; the rest are descriptive phrases kept as written (`unmapped`). Matching jobs to students by exact skill name will therefore rarely work, and Phase 7 needs token- or embedding-level skill similarity.
- **Accuracy is unmeasured**: there are no labels. Every surface (API `skill_split.inferred=true` with a note, the `job_skills.requirement_basis` column, these docs) states that the split is inferred.
