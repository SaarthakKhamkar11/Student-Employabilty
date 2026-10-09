"""TF-IDF + cosine baseline (spec §34, Baseline 2). Re-implemented in Phase 4.

The original tfidf_baseline.py that the phase prompts describe was never in this
repository (PROJECT_AUDIT.md §0); this is a new implementation, not a recovery.

Until Phase 6 selects an embedding model from evidence, this baseline is also
the interim embedding model used by the Profile Agent (docs/adr/0004). It is
fitted on the processed corpus (every student and job), so students and jobs
share one vector space. Its version is a hash of the fitted vocabulary, the
IDF weights and the parameters, so the same data always gives the same version.

Usage:  python src/embeddings/tfidf_baseline.py     (fits on data/processed, prints the version)
"""

import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_NAME = "tfidf_baseline"
SECTIONS = ("skills", "experience", "education", "projects", "certifications")
PARAMS = {
    "lowercase": True,
    # keeps tokens such as c++, c#, node.js and asp.net intact
    "token_pattern": r"(?u)\b[a-zA-Z][a-zA-Z0-9]*(?:[+#]+|(?:\.[a-zA-Z0-9]+)+)?",
    "ngram_range": (1, 2),
    "min_df": 2,  # a term must occur in >= 2 documents: drops one-off tokens such as names
    "max_features": 4096,
    "sublinear_tf": True,
    "norm": "l2",
}
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_STUDENTS = os.path.join(ROOT, "data", "processed", "students_unified.csv")
DEFAULT_JOBS = os.path.join(ROOT, "data", "processed", "jobs.csv")


# The fitted vocabulary, and so the model version, depends on exactly these corpus texts. They are
# frozen here rather than imported from the Phase 6 benchmark, so the benchmark's corrected text
# views (audit I-3/I-4) can't silently change the deployed interim model.
def _student_corpus_sections(row) -> dict:
    return {
        "skills": str(row.get("skills_raw", "") or ""),
        "experience": str(row.get("internship_raw", "") or ""),
        "education": f"{row.get('degree', '') or ''} {row.get('branch', '') or ''}".strip(),
        "projects": str(row.get("technical_projects", "") or ""),
        "certifications": str(row.get("certifications", "") or ""),
    }


def _job_corpus_sections(row) -> dict:
    return {
        "skills": str(row.get("skills", "") or ""),
        "experience": f"{row.get('experience_required', '') or ''} {row.get('responsibilities', '') or ''}".strip(),
        "education": str(row.get("qualification_required", "") or ""),
        "description": str(row.get("job_description", "") or ""),
    }


def _full_text(sections: dict) -> str:
    seen, parts = set(), []
    for text in sections.values():
        if text and text not in seen:
            seen.add(text)
            parts.append(text)
    return "\n".join(parts)


def corpus_from_processed(students_csv: str = DEFAULT_STUDENTS, jobs_csv: str = DEFAULT_JOBS) -> list[str]:
    students = pd.read_csv(students_csv).fillna("")
    jobs = pd.read_csv(jobs_csv).fillna("")
    texts = [_full_text(_student_corpus_sections(r)) for _, r in students.iterrows()]
    texts += [_full_text(_job_corpus_sections(r)) for _, r in jobs.iterrows()]
    return [t for t in texts if t.strip()]


class TfidfBaseline:
    model_name = MODEL_NAME

    def __init__(self, vectorizer: TfidfVectorizer):
        self.vectorizer = vectorizer
        vocab = sorted((term, int(index)) for term, index in vectorizer.vocabulary_.items())
        digest = hashlib.sha256(json.dumps(
            {"sklearn": sklearn.__version__, "params": repr(sorted(PARAMS.items())), "vocab": vocab,
             "idf": np.round(vectorizer.idf_, 6).tolist()}).encode()).hexdigest()
        self.version = f"tfidf-{digest[:12]}"
        self.dimension = len(vocab)

    @classmethod
    def fit(cls, texts: list[str]) -> "TfidfBaseline":
        return cls(TfidfVectorizer(dtype=np.float32, **PARAMS).fit(texts))

    @classmethod
    def fit_from_processed(cls, students_csv: str = DEFAULT_STUDENTS, jobs_csv: str = DEFAULT_JOBS) -> "TfidfBaseline":
        return cls.fit(corpus_from_processed(students_csv, jobs_csv))

    def embed(self, texts: list[str]) -> np.ndarray:
        """Dense L2-normalised rows. Text with no in-vocabulary terms gives a zero row (check before storing)."""
        return self.vectorizer.transform(texts).toarray().astype(np.float32)


if __name__ == "__main__":
    t0 = time.time()
    corpus = corpus_from_processed()
    model = TfidfBaseline.fit(corpus)
    print(f"{MODEL_NAME}: version={model.version} dimension={model.dimension} "
          f"documents={len(corpus)} fit_seconds={time.time() - t0:.2f}")
