"""Phase 6 embedding benchmark: every model gets the same students, jobs, texts and measurements.

Models (spec §6):
  A  SBERT   sentence-transformers/all-mpnet-base-v2
  B  E5      intfloat/e5-base-v2
  C  BGE     BAAI/bge-base-en-v1.5
  D  Small   sentence-transformers/all-MiniLM-L6-v2: 6 layers, 384 dimensions, about 23M parameters.
             The most widely used small sentence-embedding model, from the same library and
             training recipe as A, so A vs D isolates the effect of model size.
  E  Custom  "Not feasible — insufficient labeled data" (fine-tuning needs thousands of labelled pairs)
  F  Hybrid  not an embedding model: semantic similarity + structured features, built in Phase 7
             and evaluated in Phase 14 (spec §6 "A hybrid model is not necessarily an embedding model")
  Floor      tfidf_baseline, the deployed interim model (ADR 0004). Every embedding model must beat it.

This module produces (1) ranking scores for (student, job) pairs and (2) cost measurements.
Quality metrics come only from labels (src/evaluation), never from raw cosine values: cosine
magnitudes are not comparable across model families (each family has its own score range).

Text views (fixes for audit I-3 and I-4):
  student  skills, internships/experience, degree + branch, projects, certifications, achievements
  job      title + role, skills, description, responsibilities: each field used once (I-4).
           experience_required and qualification_required are left out. In this dataset they are
           assigned at random per posting (one role carries a median of 7 different degrees and
           10 experience ranges), so they say nothing about the role.
  Empty sections are never embedded (I-3). The section-weighted protocol scores only the
  sections present on both sides, with the weights renormalised over those sections.
  The student's declared domain is never in the text: the weak labels are built from it.

Usage:  python -m src.embeddings.embedding_model_benchmark            (cost of every model, isolated)
        python -m src.embeddings.embedding_model_benchmark perf --model e5 --out x.json   (one model)
"""

import argparse
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_STUDENTS = os.path.join(ROOT, "data", "processed", "students_unified.csv")
DEFAULT_JOBS = os.path.join(ROOT, "data", "processed", "jobs.csv")

# Each family's recommended input format, checked against the model cards (2026-10-09):
# E5 needs "query: " / "passage: " on both sides; BGE v1.5 adds an instruction to short
# retrieval queries only; the SBERT models take plain text.
MODEL_CONFIGS = {
    "tfidf": {
        "spec_slot": "Floor", "display": "TF-IDF baseline (floor)", "family": "lexical baseline",
        "kind": "tfidf", "hf_name": None, "query_prefix": "", "passage_prefix": "",
        "runtime_packages": ["scikit-learn"], "corpus_fitted": True,
    },
    "sbert": {
        "spec_slot": "A", "display": "SBERT all-mpnet-base-v2", "family": "SBERT",
        "kind": "sentence_transformers", "hf_name": "sentence-transformers/all-mpnet-base-v2",
        "query_prefix": "", "passage_prefix": "",
        "runtime_packages": ["sentence-transformers"], "corpus_fitted": False,
    },
    "e5": {
        "spec_slot": "B", "display": "E5 e5-base-v2", "family": "E5",
        "kind": "sentence_transformers", "hf_name": "intfloat/e5-base-v2",
        "query_prefix": "query: ", "passage_prefix": "passage: ",
        "runtime_packages": ["sentence-transformers"], "corpus_fitted": False,
    },
    "bge": {
        "spec_slot": "C", "display": "BGE bge-base-en-v1.5", "family": "BGE",
        "kind": "sentence_transformers", "hf_name": "BAAI/bge-base-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ", "passage_prefix": "",
        "runtime_packages": ["sentence-transformers"], "corpus_fitted": False,
    },
    "minilm": {
        "spec_slot": "D", "display": "Small: all-MiniLM-L6-v2", "family": "SBERT (small)",
        "kind": "sentence_transformers", "hf_name": "sentence-transformers/all-MiniLM-L6-v2",
        "query_prefix": "", "passage_prefix": "",
        "runtime_packages": ["sentence-transformers"], "corpus_fitted": False,
    },
}
FLOOR = "tfidf"
CANDIDATES = ("sbert", "e5", "bge", "minilm")
NOT_RUN_MODELS = {
    "custom": {
        "spec_slot": "E", "display": "Custom / domain-adapted",
        "status": "Not feasible — insufficient labeled data",
        "reason": "Fine-tuning a domain model needs thousands of labelled student-job pairs; the project has "
                  "286 students and no labelled training pairs yet (the human benchmark is for evaluation only).",
    },
    "hybrid": {
        "spec_slot": "F", "display": "Hybrid (semantic + structured features)",
        "status": "Separate architecture — built in Phase 7, evaluated in Phase 14",
        "reason": "Spec §6: a hybrid model is not an embedding model. It combines the selected embedding's "
                  "similarity with structured features, so it is compared with the matching models, not here.",
    },
}

SECTION_WEIGHTS = {"skills": 0.35, "experience": 0.25, "education": 0.15, "projects": 0.15, "certifications": 0.10}
# student section -> job section scored against it. None: the job has no matching field, so the
# section is skipped and the weights renormalised (I-3). The old benchmark paired certifications
# with the job description, counting the description twice (I-4).
SECTION_PAIRS = {"skills": "skills", "experience": "responsibilities", "education": None,
                 "projects": "description", "certifications": None}
PROTOCOLS = ("full_text", "section_weighted")
RANDOMLY_ASSIGNED_JOB_FIELDS = ("experience_required", "qualification_required")


def _s(row, key: str) -> str:
    value = row.get(key, "")
    return "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value).strip()


def student_section_texts(row) -> dict[str, str]:
    return {
        "skills": _s(row, "skills_raw"),
        "experience": _s(row, "internship_raw"),
        "education": f"{_s(row, 'degree')} {_s(row, 'branch')}".strip(),
        "projects": _s(row, "technical_projects"),
        "certifications": _s(row, "certifications"),
        "achievements": _s(row, "achievements"),
    }


def job_section_texts(row) -> dict[str, str]:
    title = " - ".join(dict.fromkeys(t for t in (_s(row, "job_title"), _s(row, "role")) if t))
    return {
        "title": title,
        "skills": _s(row, "skills"),
        "description": _s(row, "job_description"),
        "responsibilities": _s(row, "responsibilities"),
    }


def _join(sections: dict[str, str]) -> str:
    return "\n".join(t for t in sections.values() if t)


def student_text(row) -> str:
    return _join(student_section_texts(row))


def job_text(row) -> str:
    return _join(job_section_texts(row))


def load_benchmark_data(students_csv: str = DEFAULT_STUDENTS, jobs_csv: str = DEFAULT_JOBS):
    """All students, and one representative job per role (the lowest job_id).

    Description, skills and responsibilities are identical for every posting of a role (checked
    below), so after dropping the two randomly assigned fields every posting of a role has the
    same benchmark text. Ranking 4,990 postings would only rank copies; the 376 representatives
    are the distinct jobs.
    """
    students = pd.read_csv(students_csv, dtype=str, keep_default_na=False)
    jobs = pd.read_csv(jobs_csv, dtype=str, keep_default_na=False)
    per_role = jobs.groupby("role")[["job_description", "skills", "responsibilities", "job_title"]].nunique()
    if (per_role > 1).any().any():
        raise ValueError("a role has more than one distinct description/skills/responsibilities/title: "
                         "role representatives would lose information")
    reps = jobs.sort_values("job_id").drop_duplicates("role").sort_values("job_id").reset_index(drop=True)
    return students, reps, jobs


# ---------------------------------------------------------------- encoders

def snapshot_path(hf_name: str) -> str:
    """The locally cached snapshot (downloaded once if missing and the network is allowed)."""
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(hf_name, local_files_only=True)
    except Exception:
        if os.environ.get("HF_HUB_OFFLINE") == "1":
            raise
        return snapshot_download(hf_name)


class SentenceTransformerEncoder:
    def __init__(self, key: str, device: str = "cpu", batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self.key, self.cfg, self.batch_size = key, MODEL_CONFIGS[key], batch_size
        path = snapshot_path(self.cfg["hf_name"])
        self.model = SentenceTransformer(path, device=device)  # loading the snapshot pins the revision
        self.revision = os.path.basename(path)
        self.version = f"{self.cfg['hf_name']}@{self.revision[:12]}"
        self.dimension = int(self.model.get_sentence_embedding_dimension())
        self.max_seq_length = int(self.model.max_seq_length)

    def prefixed(self, texts: list[str], role: str) -> list[str]:
        prefix = self.cfg["query_prefix"] if role == "query" else self.cfg["passage_prefix"]
        return [prefix + t for t in texts]

    def encode(self, texts: list[str], role: str) -> np.ndarray:
        if any(not t.strip() for t in texts):
            raise ValueError("empty text passed to an encoder (audit I-3): filter empty sections first")
        return self.model.encode(self.prefixed(texts, role), batch_size=self.batch_size, show_progress_bar=False,
                                 normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    def size_mb(self) -> float:
        return sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024**2

    def token_lengths(self, texts: list[str], role: str) -> list[int]:
        tokenizer = self.model.tokenizer
        return [len(ids) for ids in tokenizer(self.prefixed(texts, role), add_special_tokens=True,
                                              truncation=False)["input_ids"]]


class TfidfEncoder:
    """The deployed interim model: fitted on the processed corpus exactly as the backend fits it."""

    def __init__(self, key: str = "tfidf", device: str = "cpu"):
        from src.embeddings.tfidf_baseline import TfidfBaseline

        self.key, self.cfg = key, MODEL_CONFIGS[key]
        self.model = TfidfBaseline.fit_from_processed()
        self.version, self.dimension = self.model.version, int(self.model.dimension)
        self.revision, self.max_seq_length = self.version, None

    def encode(self, texts: list[str], role: str) -> np.ndarray:
        if any(not t.strip() for t in texts):
            raise ValueError("empty text passed to an encoder (audit I-3): filter empty sections first")
        return self.model.embed(texts)

    def size_mb(self) -> float:
        import pickle

        v = self.model.vectorizer  # vocabulary + IDF only; stop_words_ is a fitting by-product
        return len(pickle.dumps({"vocabulary": v.vocabulary_, "idf": v.idf_})) / 1024**2

    def mean_nnz(self, texts: list[str]) -> float:
        return float(self.model.vectorizer.transform(texts).getnnz(axis=1).mean())


def load_encoder(key: str, device: str = "cpu"):
    if MODEL_CONFIGS[key]["kind"] == "tfidf":
        return TfidfEncoder(key, device)
    return SentenceTransformerEncoder(key, device)


# ---------------------------------------------------------------- scoring

def score_matrix(encoder, students: pd.DataFrame, jobs: pd.DataFrame, protocol: str = "full_text"):
    """Cosine-based score for every (student, job) pair. Returns (student_ids, job_ids, matrix);
    NaN where a pair can't be scored (an entity with no text)."""
    sids, jids = list(students["student_id"]), list(jobs["job_id"])
    if protocol == "full_text":
        return sids, jids, _pair_cosines(encoder, [student_text(r) for r in students.to_dict("records")],
                                         [job_text(r) for r in jobs.to_dict("records")])
    if protocol != "section_weighted":
        raise ValueError(f"unknown protocol {protocol!r}")
    s_sections = [student_section_texts(r) for r in students.to_dict("records")]
    j_sections = [job_section_texts(r) for r in jobs.to_dict("records")]
    num = np.zeros((len(sids), len(jids)), dtype=np.float64)
    den = np.zeros_like(num)
    for s_name, j_name in SECTION_PAIRS.items():
        if j_name is None:
            continue
        sims = _pair_cosines(encoder, [x[s_name] for x in s_sections], [x[j_name] for x in j_sections])
        present = ~np.isnan(sims)
        num += np.where(present, SECTION_WEIGHTS[s_name] * sims, 0.0)
        den += np.where(present, SECTION_WEIGHTS[s_name], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return sids, jids, np.where(den > 0, num / den, np.nan)


def _pair_cosines(encoder, s_texts: list[str], j_texts: list[str]) -> np.ndarray:
    out = np.full((len(s_texts), len(j_texts)), np.nan)
    si = [i for i, t in enumerate(s_texts) if t.strip()]
    ji = [i for i, t in enumerate(j_texts) if t.strip()]
    if si and ji:
        s_emb = encoder.encode([s_texts[i] for i in si], "query")
        j_emb = encoder.encode([j_texts[i] for i in ji], "passage")
        out[np.ix_(si, ji)] = s_emb @ j_emb.T  # rows are L2-normalised: dot product = cosine
    return out


def as_pair_dict(sids, jids, matrix) -> dict[tuple[str, str], float]:
    rows, cols = np.nonzero(~np.isnan(matrix))
    return {(sids[r], jids[c]): float(matrix[r, c]) for r, c in zip(rows, cols)}


# ---------------------------------------------------------------- cost measurement

N_TRIALS = 30
N_WARMUP = 3
BATCH_TEXTS = 64
BATCH_TRIALS = 7
SEED = 42


def _timing_inputs(students: pd.DataFrame, jobs: pd.DataFrame):
    """The same 30 real (student, job) pairs and the same 64 job texts for every model."""
    s_texts = [t for t in (student_text(r) for r in students.to_dict("records")) if t]
    j_texts = [t for t in (job_text(r) for r in jobs.to_dict("records")) if t]
    rng = np.random.default_rng(SEED)
    pairs = [(s_texts[i], j_texts[j]) for i, j in zip(rng.choice(len(s_texts), N_TRIALS, replace=False),
                                                     rng.choice(len(j_texts), N_TRIALS, replace=False))]
    batch = [j_texts[i] for i in rng.choice(len(j_texts), BATCH_TEXTS, replace=False)]
    return pairs, batch, s_texts, j_texts


def _time_device(encoder, pairs, batch) -> dict:
    for s, j in pairs[:N_WARMUP]:  # warm-up: first calls pay one-off allocation and kernel setup costs
        encoder.encode([s], "query"), encoder.encode([j], "passage")
    single = []
    for s, j in pairs:
        t0 = time.perf_counter()
        q, p = encoder.encode([s], "query"), encoder.encode([j], "passage")
        float(q[0] @ p[0])
        single.append((time.perf_counter() - t0) * 1000)
    encoder.encode(batch, "passage")
    batched = []
    for _ in range(BATCH_TRIALS):
        t0 = time.perf_counter()
        encoder.encode(batch, "passage")
        batched.append(time.perf_counter() - t0)
    median_batch = statistics.median(batched)
    return {
        "single_pair_latency_ms": {"median": round(statistics.median(single), 2),
                                   "p95": round(float(np.percentile(single, 95)), 2),
                                   "stdev": round(statistics.stdev(single), 2), "trials": len(single)},
        "batch_latency_ms": {"median": round(median_batch * 1000, 1), "batch_size": len(batch),
                             "stdev": round(statistics.stdev(batched) * 1000, 1), "trials": len(batched)},
        "throughput_texts_per_s": round(len(batch) / median_batch, 1),
    }


def _peak_rss_mb() -> float:
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1024**2 if sys.platform == "darwin" else peak / 1024  # bytes on macOS, KiB on Linux


def measure_performance(key: str) -> dict:
    """Run inside a fresh process (see measure_isolated) so memory figures belong to one model."""
    import psutil

    proc = psutil.Process()
    students, jobs, _ = load_benchmark_data()
    pairs, batch, s_texts, j_texts = _timing_inputs(students, jobs)
    if MODEL_CONFIGS[key]["kind"] == "sentence_transformers":
        import torch
        import transformers
        from sentence_transformers import SentenceTransformer  # noqa: F401  (import cost is not model memory)

        transformers.logging.set_verbosity_error()
        threads = torch.get_num_threads()
    else:
        threads = 1
    gc.collect()
    rss_before = proc.memory_info().rss / 1024**2

    t0 = time.perf_counter()
    encoder = load_encoder(key, "cpu")
    load_seconds = time.perf_counter() - t0
    rss_loaded = proc.memory_info().rss / 1024**2
    cpu = _time_device(encoder, pairs, batch)
    rss_after = proc.memory_info().rss / 1024**2
    peak = _peak_rss_mb()

    det_texts = batch[:16]
    first, second = encoder.encode(det_texts, "passage"), encoder.encode(det_texts, "passage")
    result = {
        "model": key, "version": encoder.version, "revision": encoder.revision, "dimension": encoder.dimension,
        "size_mb": round(encoder.size_mb(), 1), "load_seconds": round(load_seconds, 2),
        "memory_mb": {"after_load": round(rss_loaded - rss_before, 1),
                      "after_encoding": round(rss_after - rss_before, 1),
                      "peak": round(max(peak - rss_before, rss_after - rss_before), 1)},
        "cpu": cpu, "cpu_threads": threads,
        "deterministic_cpu_max_abs_diff": float(np.abs(first - second).max()),
        "offline": os.environ.get("HF_HUB_OFFLINE") == "1",
    }
    if MODEL_CONFIGS[key]["kind"] == "tfidf":
        result["bytes_per_vector"] = round(encoder.mean_nnz(j_texts) * 8, 1)  # sparse: 4-byte index + 4-byte value
        result["mean_nonzeros_per_job_vector"] = round(encoder.mean_nnz(j_texts), 1)
        result["truncation"] = {"max_seq_length": None, "students_truncated_pct": 0.0, "jobs_truncated_pct": 0.0}
        result["mps"] = "Not applicable (CPU-only model)"
        return result

    result["bytes_per_vector"] = encoder.dimension * 4  # dense float32
    s_len, j_len = encoder.token_lengths(s_texts, "query"), encoder.token_lengths(j_texts, "passage")
    limit = encoder.max_seq_length
    result["truncation"] = {
        "max_seq_length": limit,
        "students_truncated_pct": round(100 * float(np.mean([n > limit for n in s_len])), 1),
        "jobs_truncated_pct": round(100 * float(np.mean([n > limit for n in j_len])), 1),
        "student_tokens_median": int(np.median(s_len)), "job_tokens_median": int(np.median(j_len)),
    }
    import torch

    if torch.backends.mps.is_available():
        del encoder
        gc.collect()
        mps = load_encoder(key, "mps")
        result["mps"] = _time_device(mps, pairs, batch)
        result["mps"]["cpu_vs_mps_max_abs_diff"] = float(np.abs(mps.encode(det_texts, "passage") - first).max())
    else:
        result["mps"] = "Not available on this machine"
    return result


def measure_isolated(key: str, offline: bool = True, timeout: int = 900) -> dict:
    """measure_performance() in a fresh Python process, so one model's memory can't count toward another's.
    offline=True also proves the model runs with no network access (local-deployment feasibility)."""
    env = dict(os.environ, PYTHONPATH=ROOT, TOKENIZERS_PARALLELISM="false")
    if offline:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, f"{key}.json")
        proc = subprocess.run([sys.executable, "-m", "src.embeddings.embedding_model_benchmark", "perf",
                               "--model", key, "--out", out], cwd=ROOT, env=env, capture_output=True,
                              text=True, timeout=timeout)
        if proc.returncode != 0:
            return {"model": key, "error": proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "failed"}
        with open(out, encoding="utf-8") as fh:
            return json.load(fh)


# ---------------------------------------------------------------- engineering facts

def dependency_footprint(packages: list[str]) -> dict:
    """Installed distributions needed at runtime (transitive, extras excluded) and their size on disk."""
    from importlib.metadata import PackageNotFoundError, distribution

    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    seen, stack = {}, [canonicalize_name(p) for p in packages]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        try:
            dist = distribution(name)
        except PackageNotFoundError:
            continue
        seen[name] = dist
        for raw in dist.requires or []:
            req = Requirement(raw)
            if req.marker is None or req.marker.evaluate({"extra": ""}):
                stack.append(canonicalize_name(req.name))
    size = sum(f.size or 0 for d in seen.values() for f in (d.files or []))
    return {"distributions": len(seen), "installed_mb": round(size / 1024**2, 1),
            "largest": sorted(seen, key=lambda n: -sum(f.size or 0 for f in (seen[n].files or [])))[:3]}


def complexity(key: str) -> dict:
    """Preprocessing + runtime complexity as points (lower is simpler); every point has a stated reason."""
    cfg = MODEL_CONFIGS[key]
    reasons = []
    if cfg["query_prefix"]:
        reasons.append("query (student) text needs a model-specific prefix")
    if cfg["passage_prefix"]:
        reasons.append("passage (job) text needs a model-specific prefix")
    if cfg["corpus_fitted"]:
        reasons.append("vocabulary is fitted on the corpus: refit and re-embed everything when the data changes")
    if cfg["kind"] == "sentence_transformers":
        reasons.append("needs the PyTorch + transformers runtime")
    return {"points": len(reasons), "reasons": reasons}


def hardware() -> dict:
    import psutil

    info = {"machine": platform.machine(), "processor": platform.processor() or platform.machine(),
            "system": f"{platform.system()} {platform.release()}", "python": platform.python_version(),
            "cpu_cores": psutil.cpu_count(logical=True), "ram_gb": round(psutil.virtual_memory().total / 1024**3, 1)}
    try:
        import torch

        info["torch"] = torch.__version__
        info["mps_available"] = bool(torch.backends.mps.is_available())
    except ImportError:
        pass
    return info


def run_all_performance(keys=tuple(MODEL_CONFIGS)) -> dict:
    return {key: measure_isolated(key) for key in keys}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase 6 embedding benchmark (cost side)")
    sub = ap.add_subparsers(dest="command")
    perf = sub.add_parser("perf", help="measure one model in this process (used by measure_isolated)")
    perf.add_argument("--model", required=True, choices=list(MODEL_CONFIGS))
    perf.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.command == "perf":
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(measure_performance(args.model), fh, indent=2)
    else:
        for key, r in run_all_performance().items():
            if "error" in r:
                print(f"{key:7s} FAILED: {r['error']}")
                continue
            c = r["cpu"]
            print(f"{key:7s} {r['dimension']:5d}d {r['size_mb']:7.1f} MB  single {c['single_pair_latency_ms']['median']:7.2f} ms "
                  f"(sd {c['single_pair_latency_ms']['stdev']:.2f})  {c['throughput_texts_per_s']:7.1f} texts/s  "
                  f"mem peak {r['memory_mb']['peak']:7.1f} MB")
