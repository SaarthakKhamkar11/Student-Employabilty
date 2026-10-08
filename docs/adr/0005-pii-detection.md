# ADR 0005: PII detection is an auditable regex baseline with measured blind spots

**Status:** Accepted, 2026-10-08

**Context.**
- Spec §5.2 and §38: detect PII before section extraction, and never expose it to the LLM, the ML pipeline or the logs.
- The Phase 4 prompt asked for "a simple, auditable regex/spaCy NER approach" with a documented false-negative rate.
- spaCy NER would add a large model dependency, and its name recall on Indian resumes is unmeasured here.

**Decision.** `backend/app/profile/pii.py` (`pii-regex-v2` since Phase 5; see the addendum):
- **Detection:** regexes for email, phone (10+ digits, excluding year ranges), URL, date of birth (labelled) and address (labelled, or a ≤160-character line with a 6-digit PIN plus an address word). Names are found only on the first line, or after a `Name:` label.
- **Output:** the type and character offsets of each finding, never the value. Later stages, the TF-IDF embedding and any LLM receive the **masked** text (`[EMAIL]`, `[NAME]` ...).
- **Logs:** only `{"type", "count"}` flags.
- **Validation gate:** `validate_profile` refuses to save a structured profile that still contains an email, phone, URL or date of birth.
- **Detector failure:** the agent falls back to email/phone masking, and skips both embedding and storage.

**Measured with v2** (`backend/tests/test_pii.py`, a labelled *synthetic* set: 27 snippets, 29 PII items, 12 clean lines):

| Type | Recall | False-negative rate | Known misses |
|---|---|---|---|
| url | 3/3 | 0% | |
| phone | 6/7 | 14% | fewer than 10 digits |
| email | 3/4 | 25% | obfuscated (`[at]`, `[dot]`) |
| address | 3/4 | 25% | no label and no PIN code |
| dob | 2/3 | 33% | unlabelled ("Born on ...") |
| name | 5/8 | **38%** | lowercase, not on the first line, single word |
| false positives | 0/12 clean lines | | |

On the 166 real Kaggle resumes it flagged 28 URLs (including some company websites, which is over-masking on the safe side), 1 email, 1 phone, 3 addresses and 2 dates of birth. Manual review of these found no false positives. False negatives on real data are **not measured**, because there are no labels.

**Consequences.** This is a baseline, not a guarantee. "No PII detected" must never be read as "contains no PII". Name detection is the weakest part. A better detector (NER, or an LLM-assisted pass on already-masked text) can replace `pii.py` behind the same tool interface, and must be measured on the same set.

**Addendum (Phase 5, v2).** A Phase 5 test found that v1 missed every phone number that ends a sentence ("... +91 98765 43210."): the regex's lookahead rejected any trailing ".". v2 allows a sentence-final "." but not ".<digit>". The same fix was applied to the log redactor's phone pattern. A sentence-final case was added to the labelled set, so phone recall is now 6/7. A known false positive, predating v2: long decimals such as `3.14159265358979` are flagged as phone numbers. That is over-masking, which is the safe side.

**Addendum (Phase 5): tool-call logging.** The tool registry used to log tool arguments with key-based redaction. That leaked short document text under keys it didn't list (`posting`, `job`, `description`). Tool-call logs now record only the *shape* of arguments (`summarize_arguments`: string and list lengths, numbers, booleans; sensitive keys masked). A regression test covers JDs (`test_jd_logs_carry_pii_flags_but_never_contact_details`) and the registry itself (`test_tool_call_logs_never_contain_argument_text`).
