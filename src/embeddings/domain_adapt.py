"""Domain-adapt E5 e5-base-v2 to student-job matching (Phase 6 slot E, docs/adr/0012).

Training signal: the rule-based weak labels (src/evaluation/weak_labels.py) of the TRAIN split only. The human
labels (validation and test students) are never seen, so they stay an independent judge, and the test split stays
held back for Phase 14.

Recipe, fixed before any evaluation (one run, no search over settings):
  pairs        every (train student, job role) with weak grade 3 (the student's primary role family) as a
               positive, plus one grade-0 job of the same student as a hard negative (seeded)
  texts        exactly the Phase 6 benchmark views: "query: " + student_text, "passage: " + job_text
  loss         MultipleNegativesRankingLoss (in-batch negatives + the hard negative), scale 20
  batches      16 triplets, no student twice in a batch (another of their positives would be a false negative)
  optimiser    AdamW, lr 2e-5, weight decay 0.01, linear warmup over 10% of steps, 1 epoch, seed 42, CPU

    python -m src.embeddings.domain_adapt      -> models/embeddings/e5-base-v2-domain-<hash>/ (+ manifest.json)
"""

import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.embeddings.embedding_model_benchmark import (MODEL_CONFIGS, ROOT, job_text, load_benchmark_data,
                                                      snapshot_path, student_text)

WEAK_LABELS = os.path.join(ROOT, "data", "benchmark", "weak_labels.csv")
OUT_DIR = os.path.join(ROOT, "models", "embeddings")
RECIPE = {"base": "intfloat/e5-base-v2", "positive_grade": 3, "negative_grade": 0, "hard_negatives_per_positive": 1,
          "loss": "MultipleNegativesRankingLoss", "scale": 20.0, "batch_size": 16, "lr": 2e-5, "weight_decay": 0.01,
          "warmup_ratio": 0.1, "epochs": 1, "seed": 42, "max_seq_length": 512, "split": "train"}


def triplets(seed: int = RECIPE["seed"]) -> list[tuple[str, str, str, str]]:
    """[(student_id, query text, positive passage, hard-negative passage)] from the train split's weak labels."""
    students, jobs, _ = load_benchmark_data()
    s_text = {r["student_id"]: student_text(r) for r in students.to_dict("records")}
    j_text = {r["job_id"]: job_text(r) for r in jobs.to_dict("records")}
    weak = pd.read_csv(WEAK_LABELS, dtype={"student_id": str, "job_id": str})
    weak = weak[(weak["split"] == RECIPE["split"]) & weak["job_id"].isin(j_text)]
    rng = random.Random(seed)
    out = []
    for sid, g in weak.groupby("student_id", sort=True):
        if not s_text.get(sid, "").strip():
            continue
        pos = sorted(g.loc[g["relevance_label"] == RECIPE["positive_grade"], "job_id"])
        neg = sorted(g.loc[g["relevance_label"] == RECIPE["negative_grade"], "job_id"])
        for p in pos:
            if neg:
                out.append((sid, "query: " + s_text[sid], "passage: " + j_text[p], "passage: " + j_text[rng.choice(neg)]))
    return out


def batches(rows, size: int, seed: int):
    """Shuffled batches in which no student appears twice."""
    rng = random.Random(seed)
    pool = rows[:]
    rng.shuffle(pool)
    out = []
    while pool:
        batch, seen, rest = [], set(), []
        for r in pool:
            if len(batch) < size and r[0] not in seen:
                batch.append(r)
                seen.add(r[0])
            else:
                rest.append(r)
        out.append(batch)
        pool = rest
    return out


def train() -> str:
    import torch
    from sentence_transformers import SentenceTransformer, losses
    from transformers import get_linear_schedule_with_warmup

    random.seed(RECIPE["seed"]), np.random.seed(RECIPE["seed"]), torch.manual_seed(RECIPE["seed"])
    base_path = snapshot_path(RECIPE["base"])
    model = SentenceTransformer(base_path, device="cpu")
    model.max_seq_length = RECIPE["max_seq_length"]
    # Recompute activations in the backward pass instead of keeping them: the same gradients, a fraction of the
    # memory (48 sequences per step otherwise exceed a 7.6 GB Docker VM).
    model[0].auto_model.gradient_checkpointing_enable()
    rows = triplets()
    plan = batches(rows, RECIPE["batch_size"], RECIPE["seed"])
    loss_fn = losses.MultipleNegativesRankingLoss(model, scale=RECIPE["scale"])
    opt = torch.optim.AdamW(model.parameters(), lr=RECIPE["lr"], weight_decay=RECIPE["weight_decay"])
    steps = len(plan) * RECIPE["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(RECIPE["warmup_ratio"] * steps), steps)
    model.train()
    t0, losses_seen = time.time(), []
    for step, batch in enumerate(plan, 1):
        feats = [model.tokenize([r[i] for r in batch]) for i in (1, 2, 3)]
        loss = loss_fn(feats, None)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(), sched.step(), opt.zero_grad()
        losses_seen.append(float(loss))
        if step % 10 == 0 or step == steps:
            print(f"step {step}/{steps} loss {np.mean(losses_seen[-10:]):.4f} ({time.time() - t0:.0f} s)", flush=True)
    model.eval()
    data_hash = hashlib.sha256(json.dumps([r[0] + r[2] + r[3] for r in rows]).encode()).hexdigest()
    tag = hashlib.sha256((json.dumps(RECIPE, sort_keys=True) + data_hash + os.path.basename(base_path)).encode()).hexdigest()[:12]
    out = os.path.join(OUT_DIR, f"e5-base-v2-domain-{tag}")
    model.save(out)
    manifest = {"model": f"e5-base-v2-domain-{tag}", "base": RECIPE["base"], "base_revision": os.path.basename(base_path),
                "recipe": RECIPE, "triplets": len(rows), "students": len({r[0] for r in rows}), "steps": steps,
                "training_data_sha256": data_hash, "loss_first_10": round(float(np.mean(losses_seen[:10])), 4),
                "loss_last_10": round(float(np.mean(losses_seen[-10:])), 4), "seconds": round(time.time() - t0),
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "weak_labels_sha256": hashlib.sha256(open(WEAK_LABELS, "rb").read()).hexdigest()}
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    print(json.dumps(manifest, indent=1))
    return out


if __name__ == "__main__":
    assert MODEL_CONFIGS["e5"]["hf_name"] == RECIPE["base"]
    train()
