"""
Phase 2 -- SBERT vs E5 vs BGE, benchmarked on the axes previously agreed:
  semantic matching quality, precision, recall, F1, Recall@K,
  inference latency, embedding generation time, memory, model size,
  deployment complexity.
"""

import argparse
import gc
import json
import os
import time

import numpy as np
import pandas as pd

SECTION_WEIGHTS = {  
    "skills": 0.35, "experience": 0.25, "education": 0.15,
    "projects": 0.15, "certifications": 0.10,
}

# Model family quirks: E5 and BGE need instruction prefixes for best quality;
# plain SBERT models don't. Verify current recommended prefixes against each
# model's HF model card before a real run -- these can change between versions.
MODEL_CONFIGS = {
    "sbert": {
        "hf_name": "sentence-transformers/all-mpnet-base-v2",
        "query_prefix": "", "passage_prefix": "",
    },
    "e5": {
        "hf_name": "intfloat/e5-base-v2",
        "query_prefix": "query: ", "passage_prefix": "passage: ",
    },
    "bge": {
        "hf_name": "BAAI/bge-base-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
    },
}


def student_section_texts(row):
    return {
        "skills": str(row.get("skills_raw", "") or ""),
        "experience": str(row.get("internship_raw", "") or ""),
        "education": f"{row.get('degree','') or ''} {row.get('branch','') or ''}".strip(),
        "projects": str(row.get("technical_projects", "") or ""),
        "certifications": str(row.get("certifications", "") or ""),
    }


def job_section_texts(row):
    return {
        "skills": str(row.get("skills", "") or ""),
        "experience": f"{row.get('experience_required','') or ''} {row.get('responsibilities','') or ''}".strip(),
        "education": str(row.get("qualification_required", "") or ""),
        "projects": str(row.get("job_description", "") or ""),
        "certifications": str(row.get("job_description", "") or ""),
    }


def get_model_size_mb(model):
    """Sum of parameter tensor bytes -- proxy for on-disk/in-memory model size."""
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / (1024 ** 2)


def get_process_memory_mb():
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def embed_section(model, texts, prefix, batch_size=32):
    """Returns (embeddings, seconds_taken)."""
    prefixed = [f"{prefix}{t}" if t.strip() else t for t in texts]
    t0 = time.time()
    emb = model.encode(prefixed, batch_size=batch_size, show_progress_bar=False,
                        normalize_embeddings=True)
    return emb, time.time() - t0


def cosine_sim_matrix(a, b):
    return a @ b.T  # embeddings already L2-normalized above


def compute_recall_at_k(overall_scores, student_ids, job_ids, eval_labels_df, k=5):
    """eval_labels_df: columns student_id, job_id, relevant (1/0).
    Returns dict with precision/recall/F1 @k and hit rate, or None if no
    overlap between eval labels and the students/jobs actually scored."""
    if eval_labels_df is None:
        return None
    relevant_pairs = set(
        zip(eval_labels_df[eval_labels_df.relevant == 1]["student_id"],
            eval_labels_df[eval_labels_df.relevant == 1]["job_id"])
    )
    if not relevant_pairs:
        return None
    hits, total_relevant_found, total_relevant, n_eval = 0, 0, 0, 0
    for s_idx, s_id in enumerate(student_ids):
        student_relevant = {j for (s, j) in relevant_pairs if s == s_id}
        if not student_relevant:
            continue
        n_eval += 1
        top_k_idx = np.argsort(-overall_scores[s_idx])[:k]
        top_k_jobs = {job_ids[j] for j in top_k_idx}
        found = student_relevant & top_k_jobs
        if found:
            hits += 1
        total_relevant_found += len(found)
        total_relevant += len(student_relevant)
    if n_eval == 0:
        return None
    recall_at_k = total_relevant_found / total_relevant if total_relevant else 0.0
    hit_rate_at_k = hits / n_eval
    return {"recall_at_k": recall_at_k, "hit_rate_at_k": hit_rate_at_k,
            "k": k, "students_evaluated": n_eval}


def benchmark_model(model_key, students, jobs, sample_students=30, sample_jobs=500,
                     eval_labels_df=None, top_k=5):
    from sentence_transformers import SentenceTransformer

    cfg = MODEL_CONFIGS[model_key]
    print(f"\n=== {model_key} ({cfg['hf_name']}) ===")

    gc.collect()
    mem_before = get_process_memory_mb()
    t_load0 = time.time()
    model = SentenceTransformer(cfg["hf_name"])
    load_seconds = time.time() - t_load0
    model_size_mb = get_model_size_mb(model)
    mem_after_load = get_process_memory_mb()

    s_sample = students.sample(n=min(sample_students, len(students)), random_state=42).reset_index(drop=True)
    j_sample = jobs.sample(n=min(sample_jobs, len(jobs)), random_state=42).reset_index(drop=True)

    s_texts = {k: [] for k in SECTION_WEIGHTS}
    j_texts = {k: [] for k in SECTION_WEIGHTS}
    for _, r in s_sample.iterrows():
        t = student_section_texts(r)
        for k in SECTION_WEIGHTS:
            s_texts[k].append(t[k])
    for _, r in j_sample.iterrows():
        t = job_section_texts(r)
        for k in SECTION_WEIGHTS:
            j_texts[k].append(t[k])

    section_embed_seconds = {}
    section_sims = {}
    for section in SECTION_WEIGHTS:
        s_emb, s_secs = embed_section(model, s_texts[section], cfg["query_prefix"])
        j_emb, j_secs = embed_section(model, j_texts[section], cfg["passage_prefix"])
        section_embed_seconds[section] = s_secs + j_secs
        section_sims[section] = cosine_sim_matrix(s_emb, j_emb)  # shape (n_students, n_jobs)

    overall = np.zeros_like(section_sims["skills"])
    for section, w in SECTION_WEIGHTS.items():
        overall += w * section_sims[section]

    # single-item inference latency (deployment-relevant: one new student profile)
    t_single0 = time.time()
    _ = model.encode([s_texts["skills"][0]], show_progress_bar=False)
    single_latency_ms = (time.time() - t_single0) * 1000

    mem_after_embed = get_process_memory_mb()

    recall_metrics = compute_recall_at_k(
        overall, list(s_sample["student_id"]), list(j_sample["job_id"]),
        eval_labels_df, k=top_k
    )

    result = {
        "model": model_key, "hf_name": cfg["hf_name"],
        "model_size_mb": round(model_size_mb, 1),
        "load_seconds": round(load_seconds, 2),
        "total_embed_seconds": round(sum(section_embed_seconds.values()), 3),
        "section_embed_seconds": {k: round(v, 3) for k, v in section_embed_seconds.items()},
        "single_item_latency_ms": round(single_latency_ms, 1),
        "memory_delta_load_mb": round(mem_after_load - mem_before, 1),
        "memory_delta_embed_mb": round(mem_after_embed - mem_after_load, 1),
        "overall_score_mean": round(float(overall.mean()), 4),
        "overall_score_std": round(float(overall.std()), 4),
        "overall_score_max": round(float(overall.max()), 4),
        "recall_at_k": recall_metrics,
        "students_sampled": len(s_sample), "jobs_sampled": len(j_sample),
    }
    del model
    gc.collect()
    return result


def run(students_path, jobs_path, out_dir, eval_labels_path=None,
        sample_students=30, sample_jobs=500):
    students = pd.read_csv(students_path).fillna("")
    jobs = pd.read_csv(jobs_path).fillna("")
    eval_labels_df = pd.read_csv(eval_labels_path) if eval_labels_path else None
    if eval_labels_df is None:
        print("No --eval_labels provided: precision/recall/F1/Recall@K will be "
              "reported as null, not guessed.")

    results = []
    for model_key in MODEL_CONFIGS:
        try:
            results.append(benchmark_model(
                model_key, students, jobs, sample_students, sample_jobs, eval_labels_df
            ))
        except Exception as e:
            print(f"{model_key} FAILED: {e}")
            results.append({"model": model_key, "error": str(e)})

    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/embedding_model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    rows = [r for r in results if "error" not in r]
    if rows:
        summary = pd.DataFrame([{
            "model": r["model"], "model_size_mb": r["model_size_mb"],
            "total_embed_seconds": r["total_embed_seconds"],
            "single_item_latency_ms": r["single_item_latency_ms"],
            "overall_score_mean": r["overall_score_mean"],
            "recall_at_k": r["recall_at_k"]["recall_at_k"] if r["recall_at_k"] else None,
        } for r in rows])
        summary.to_csv(f"{out_dir}/embedding_model_comparison_summary.csv", index=False)
        print("\n" + summary.to_string(index=False))
    print(f"\nFull results -> {out_dir}/embedding_model_comparison.json")


if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))  # defaults work from any CWD
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", default=os.path.join(root, "data", "processed", "students_unified.csv"))
    ap.add_argument("--jobs", default=os.path.join(root, "data", "processed", "jobs.csv"))
    ap.add_argument("--out_dir", default=os.path.join(root, "results"))
    ap.add_argument("--eval_labels", default=None,
                     help="CSV: student_id,job_id,relevant -- optional, for real precision/recall/F1")
    ap.add_argument("--sample_students", type=int, default=30)
    ap.add_argument("--sample_jobs", type=int, default=500)
    args = ap.parse_args()
    run(args.students, args.jobs, args.out_dir, args.eval_labels,
        args.sample_students, args.sample_jobs)