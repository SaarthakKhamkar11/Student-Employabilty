"""The configured embedding provider: E5 selected by Phase 6 (docs/adr/0012), roles, caching, the TF-IDF fallback."""

import numpy as np

from app.core.config import Settings
from app.embeddings.provider import PASSAGE, QUERY, E5Provider, TfidfBaselineProvider, build_embedding_provider


class FakeST:
    """Stands in for SentenceTransformer: records what was encoded, returns unit vectors."""

    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        assert kwargs["normalize_embeddings"] is True
        return np.tile(np.eye(1, 768, 0, dtype=np.float32), (len(texts), 1))


def e5_with_fake():
    p = E5Provider()
    fake = FakeST()
    p._loaded = lambda: fake  # noqa: SLF001 - no model download in unit tests
    return p, fake


def test_e5_is_the_default_and_tfidf_stays_available():
    # docs/adr/0012: E5 is the primary model; TF-IDF is the baseline and the fallback only
    assert isinstance(build_embedding_provider(Settings(_env_file=None)), E5Provider)
    assert isinstance(build_embedding_provider(Settings(_env_file=None, embedding_model="tfidf_baseline")),
                      TfidfBaselineProvider)


def test_e5_version_is_the_one_phase_6_recorded():
    assert E5Provider().describe() == ("intfloat/e5-base-v2@f52bf8ec8c71", 768)
    assert not E5Provider.interim


def test_roles_become_e5_prefixes():
    p, fake = e5_with_fake()
    p.embed(["Python developer"], QUERY)
    p.embed(["Python developer"], PASSAGE)
    assert fake.calls == [["query: Python developer"], ["passage: Python developer"]]


def test_repeated_texts_are_embedded_once_and_empty_texts_have_no_signal():
    p, fake = e5_with_fake()
    out = p.embed(["SQL", "SQL", "", "Excel"], PASSAGE)
    assert fake.calls == [["passage: SQL", "passage: Excel"]]
    assert out[2] is None and out[0] is not None and len(out[0].values) == 768
    p.embed(["SQL"], PASSAGE)
    assert len(fake.calls) == 1  # cached


def test_agents_embed_students_as_queries_and_jobs_as_passages():
    import inspect

    from app.agents import job_intelligence_agent, profile_agent

    assert '"role": "query"' in inspect.getsource(profile_agent)
    assert '"role": "passage"' in inspect.getsource(job_intelligence_agent)


# -- the TF-IDF fallback (docs/adr/0012) -------------------------------------------------------------------------
def test_an_unloadable_e5_falls_back_to_tfidf_and_says_why(monkeypatch):
    def missing(self):
        raise OSError("no cached snapshot")

    monkeypatch.setattr(E5Provider, "check", missing)
    p = build_embedding_provider(Settings(_env_file=None), verify=True)
    assert isinstance(p, TfidfBaselineProvider)
    assert p.fallback_reason == "e5_base_v2 unavailable (OSError)"


def test_a_loadable_e5_is_used_and_no_fallback_is_reported(monkeypatch):
    monkeypatch.setattr(E5Provider, "check", lambda self: None)
    p = build_embedding_provider(Settings(_env_file=None), verify=True)
    assert isinstance(p, E5Provider)


def test_tfidf_vectors_are_kept_beside_e5_but_not_beside_themselves():
    from app.embeddings.provider import fallback_provider

    s = Settings(_env_file=None)
    assert isinstance(fallback_provider(E5Provider(), s), TfidfBaselineProvider)
    assert fallback_provider(TfidfBaselineProvider(s.processed_data_dir), s) is None
