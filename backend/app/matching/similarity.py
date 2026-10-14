"""Stage 2, semantic retrieval, and the per-pair similarity signals (spec §5.4: embedding similarity tool).

Retrieval scores every eligible job (or student) with Phase 6's section-weighted protocol: skills vs skills,
experience vs responsibilities, projects vs description (SECTION_PAIRS / SECTION_WEIGHTS in
src/embeddings/embedding_model_benchmark.py), the weights renormalised over the sections both sides have
(audit I-3). Education and certifications are matched as structured features instead: the job side has
no meaningful text for them in this data (qualification_required is randomly assigned, Phase 6).
Same-country jobs come before any other (docs/adr/0013); then the top_n by score go to feature engineering.

Signals, each student text vs job text: project_relevance (projects vs description), certification_relevance
(certifications vs skills), career_interest_alignment (stated interest vs title), role_similarity
(experience vs title), domain_similarity (branch + domain vs the full job text).

Raw cosine is model-specific (Phase 6: one model scored almost every pair 0.68-0.75, another 0.03-0.38), so every
similarity is also turned into a calibrated percentile by app/matching/calibration.py. Raw values are kept.
"""

from dataclasses import dataclass, field

from app.embeddings.provider import PASSAGE, QUERY, EmbeddingProvider, Role, Vector
from app.matching.eligibility import SAME
from app.matching.vectors import VectorIndex, VectorIndexError, cosine
from src.embeddings.embedding_model_benchmark import SECTION_PAIRS, SECTION_WEIGHTS

RETRIEVAL_PAIRS: tuple[tuple[str, str, float], ...] = tuple(
    (s, j, SECTION_WEIGHTS[s]) for s, j in SECTION_PAIRS.items() if j is not None)
SIGNALS: dict[str, tuple[str, str]] = {  # feature -> (student text, job text)
    "project_relevance": ("projects", "description"),
    "certification_relevance": ("certifications", "skills"),
    "career_interest_alignment": ("interest", "title"),
    "role_similarity": ("experience", "title"),
    "domain_similarity": ("field", "full"),
}
# Texts the Profile / Job agents store as embeddings (the rest are embedded per request).
STORED_STUDENT = {"skills", "experience", "education", "projects", "certifications", "full"}
STORED_JOB = {"skills", "required_skills", "responsibilities", "description", "education", "full"}


@dataclass
class Retrieved:
    entity_id: str
    score: float  # raw section-weighted cosine of the configured model
    sections: dict[str, float]  # "skills->skills": cosine, for the sections both sides have
    tier: str


@dataclass
class RetrievalResult:
    top: list[Retrieved]
    scored: int
    unscored: int  # eligible, but no section in common to compare (e.g. an empty profile)
    query_sections: list[str] = field(default_factory=list)


def role_of(kind: str) -> Role:
    """Student texts are queries, job texts passages, whichever side is being matched (provider.py)."""
    return QUERY if kind == "student" else PASSAGE


def embed_texts(provider: EmbeddingProvider, texts: dict[str, str], role: Role) -> dict[str, Vector | None]:
    keys = [k for k, t in texts.items() if t and t.strip()]
    vectors = provider.embed([texts[k] for k in keys], role) if keys else []
    return dict(zip(keys, vectors))


def retrieve(query_kind: str, query_texts: dict[str, str], eligible: list[str], tiers: dict[str, str],
             index: VectorIndex, provider: EmbeddingProvider, top_n: int) -> RetrievalResult:
    """query_kind "student" ranks jobs for a student; "job" ranks students for a job."""
    version, dimension = provider.describe()
    target = "job" if query_kind == "student" else "student"
    if eligible and index.count(target, provider.model_name, version) == 0:
        raise VectorIndexError(f"no stored {target} embeddings for {provider.model_name} {version}: "
                               "re-embed (seed --process) after changing EMBEDDING_MODEL")
    wanted = {(s if query_kind == "student" else j) for s, j, _ in RETRIEVAL_PAIRS}
    query = embed_texts(provider, {k: v for k, v in query_texts.items() if k in wanted}, role_of(query_kind))
    num: dict[str, float] = {}
    den: dict[str, float] = {}
    sections: dict[str, dict[str, float]] = {}
    for s_sec, j_sec, weight in RETRIEVAL_PAIRS:
        q_sec, stored_sec = (s_sec, j_sec) if query_kind == "student" else (j_sec, s_sec)
        vec = query.get(q_sec)
        if vec is None or not eligible:
            continue
        for entity_id, sim in index.similarities(target, stored_sec, vec, provider.model_name, version, dimension,
                                                 ids=eligible).items():
            num[entity_id] = num.get(entity_id, 0.0) + weight * sim
            den[entity_id] = den.get(entity_id, 0.0) + weight
            sections.setdefault(entity_id, {})[f"{s_sec}->{j_sec}"] = round(sim, 6)
    scored = [Retrieved(e, num[e] / den[e], sections[e], tiers.get(e, "any")) for e in num if den[e] > 0]
    scored.sort(key=lambda r: (r.tier != SAME, -r.score, r.entity_id))  # same country first, then by score
    return RetrievalResult(top=scored[:top_n], scored=len(scored), unscored=len(eligible) - len(scored),
                           query_sections=sorted(k for k, v in query.items() if v is not None))


def compute_signals(query_kind: str, query_texts: dict[str, str], candidate_texts: dict[str, dict[str, str]],
                    index: VectorIndex, provider: EmbeddingProvider) -> dict[str, dict[str, float | None]]:
    """Raw cosine of each signal for the query vs every candidate: {candidate_id: {signal: value}}.

    Stored candidate embeddings are compared in the database; only texts the agents don't store (job
    titles, or a student's interest and field) are embedded here. None: either side has no text.
    """
    version, dimension = provider.describe()
    ids = sorted(candidate_texts)
    target, stored = ("job", STORED_JOB) if query_kind == "student" else ("student", STORED_STUDENT)
    query_keys = {(s if query_kind == "student" else j) for s, j in SIGNALS.values()}
    query = embed_texts(provider, {k: v for k, v in query_texts.items() if k in query_keys}, role_of(query_kind))
    out: dict[str, dict[str, float | None]] = {i: {} for i in ids}
    for name, (s_key, j_key) in SIGNALS.items():
        q_key, c_key = (s_key, j_key) if query_kind == "student" else (j_key, s_key)
        qvec = query.get(q_key)
        if qvec is None or not ids:
            for i in ids:
                out[i][name] = None
            continue
        if c_key in stored:
            sims = index.similarities(target, c_key, qvec, provider.model_name, version, dimension, ids=ids)
        else:
            texts = {i: candidate_texts[i].get(c_key, "") for i in ids}
            vecs = embed_texts(provider, texts, role_of(target))
            sims = {i: cosine(qvec, v, dimension) for i, v in vecs.items()}
        for i in ids:
            out[i][name] = sims.get(i)
    return out
