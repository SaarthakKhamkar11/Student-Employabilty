"""The matching configuration (config/matching_baseline_v1.yaml) and the job role families it uses.

matching_model_version is a content hash of the configuration (everything except its `registration`
block), so a changed weight is a new version, and predictions are refused until that version is
registered by a recorded experiment (spec §24, docs/adr/0014).
"""

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from app.core.config import REPO_ROOT
from app.matching.features import FEATURE_NAMES, validate_feature_names

MATCHING_CONFIG_PATH = REPO_ROOT / "config" / "matching_baseline_v1.yaml"
ROLE_FAMILIES_PATH = REPO_ROOT / "config" / "weak_label_rules.yaml"
DIMENSIONS = ("skill_match", "education_match", "experience_match", "project_relevance", "certification_relevance",
              "semantic_similarity", "career_interest_alignment", "transferable_skill_alignment")  # spec §15


class MatchingConfigError(ValueError):
    pass


@dataclass(frozen=True)
class MatchingConfig:
    matching_model: str
    feature_version: str
    top_n: int
    ranking: dict[str, Any]
    eligibility: dict[str, Any]
    features: dict[str, Any]
    weights: dict[str, float]
    reasons: dict[str, str]
    dimensions: dict[str, tuple[str, ...]]
    branch_rules: tuple[tuple[re.Pattern, tuple[str, ...]], ...]
    role_families: dict[str, re.Pattern]
    registration: dict[str, Any]
    sha256: str
    path: str = field(default="")

    @property
    def matching_model_version(self) -> str:
        return f"{self.matching_model}+{self.sha256[:12]}"

    @property
    def registered(self) -> bool:
        return bool(self.registration.get("experiment_id")) and self.registration.get("config_sha256") == self.sha256

    def families_of_role(self, role: str | None, title: str | None = None) -> set[str]:
        text = " ".join(t for t in (role, title) if t)
        return {name for name, pattern in self.role_families.items() if pattern.search(text)} if text else set()

    def families_of_branch(self, branch: str | None) -> set[str] | None:
        for pattern, families in self.branch_rules:
            if branch and pattern.search(branch):
                return set(families)
        return None  # unknown branch: branch_match is "Insufficient evidence", never a guess


def config_hash(raw: dict[str, Any]) -> str:
    body = {k: v for k, v in raw.items() if k != "registration"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def load_matching_config(path: Path = MATCHING_CONFIG_PATH, families_path: Path = ROLE_FAMILIES_PATH) -> MatchingConfig:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    with open(families_path, encoding="utf-8") as fh:
        families = {name: re.compile(p, re.I) for name, p in yaml.safe_load(fh)["families"].items()}

    weights = {name: float(v["weight"]) for name, v in raw["weights"].items()}
    reasons = {name: str(v.get("reason", "")).strip() for name, v in raw["weights"].items()}
    validate_feature_names(weights)  # rejects protected attributes and anything outside the contract
    if set(weights) != set(FEATURE_NAMES):
        raise MatchingConfigError(f"weights must list every feature in the contract; missing "
                                  f"{sorted(set(FEATURE_NAMES) - set(weights))}")
    if any(not r for r in reasons.values()):
        raise MatchingConfigError("every weight needs a one-line reason")
    if any(w < 0 for w in weights.values()) or not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
        raise MatchingConfigError(f"weights must be non-negative and sum to 1 (sum {sum(weights.values())})")
    dimensions = {d: tuple(fs) for d, fs in raw["dimensions"].items()}
    if set(dimensions) != set(DIMENSIONS):
        raise MatchingConfigError(f"dimensions must be exactly {list(DIMENSIONS)}")
    for d, fs in dimensions.items():
        validate_feature_names(fs)
    rules = []
    for rule in raw["branch_rules"]:
        unknown = set(rule["families"]) - set(families)
        if unknown:
            raise MatchingConfigError(f"branch rule {rule['pattern']!r} names unknown families {sorted(unknown)}")
        rules.append((re.compile(rule["pattern"], re.I), tuple(rule["families"])))
    return MatchingConfig(
        matching_model=raw["matching_model"], feature_version=raw["feature_version"],
        top_n=int(raw["retrieval"]["top_n"]), ranking=raw.get("ranking") or {}, eligibility=raw["eligibility"],
        features=raw["features"],
        weights=weights, reasons=reasons, dimensions=dimensions, branch_rules=tuple(rules),
        role_families=families, registration=raw.get("registration") or {}, sha256=config_hash(raw),
        path=str(path))


@lru_cache(maxsize=1)
def get_matching_config() -> MatchingConfig:
    return load_matching_config()
