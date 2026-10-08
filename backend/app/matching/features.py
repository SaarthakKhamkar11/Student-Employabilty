"""The matching feature contract (spec §11) and the protected-attribute guard.

Phase 7 builds feature vectors ONLY through this module: every feature name
must be in FEATURE_NAMES, and validate_feature_names() rejects protected
attributes. Gender (the student's) and gender_preference (the job's) are not
features. gender_preference is used only by the visibility filter, BEFORE
scoring (app/jobs/visibility.py, docs/adr/0009).
"""

FEATURE_NAMES: tuple[str, ...] = (  # spec §11 candidate features; Phase 7 implements and may prune them
    "skill_match_score", "required_skill_match_score", "preferred_skill_match_score", "semantic_similarity",
    "education_match", "degree_match", "branch_match", "experience_match", "experience_gap",
    "project_relevance", "certification_relevance", "career_interest_alignment", "transferable_skill_score",
    "matched_skill_count", "missing_skill_count", "required_skill_count", "matched_skill_ratio",
    "preferred_skill_ratio", "role_similarity", "domain_similarity",
)

# Never features (spec §11 and §38). A fairness audit may read them separately, never through this contract.
PROTECTED_ATTRIBUTES = frozenset({
    "gender", "gender_preference", "name", "email", "phone", "address", "date_of_birth", "dob",
    "religion", "race", "caste", "age", "marital_status", "nationality",
})


class ProtectedFeatureError(ValueError):
    """A protected attribute was offered as a matching feature."""


def validate_feature_names(names) -> tuple[str, ...]:
    names = tuple(names)
    protected = [n for n in names if n.lower() in PROTECTED_ATTRIBUTES or "gender" in n.lower()]
    if protected:
        raise ProtectedFeatureError(f"protected attributes cannot be matching features: {protected}")
    unknown = [n for n in names if n not in FEATURE_NAMES]
    if unknown:
        raise ValueError(f"not in the feature contract (app/matching/features.py): {unknown}")
    return names


validate_feature_names(FEATURE_NAMES)  # the contract itself must pass its own guard
