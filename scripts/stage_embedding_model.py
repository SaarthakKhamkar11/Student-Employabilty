"""Stage the selected embedding model's weights for the backend image (docs/adr/0012).

    python scripts/stage_embedding_model.py      -> .model_cache/hf (gitignored), copied into the image at /srv/hf

Copies the pinned snapshot from the local Hugging Face cache (downloading it first if it isn't there), laid out as a
Hugging Face cache, so the container loads it fully offline with snapshot_download(local_files_only=True).
"""

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "backend"), str(REPO)]

from huggingface_hub import snapshot_download  # noqa: E402

from app.embeddings.provider import E5Provider  # noqa: E402

PATTERNS = ["*.json", "*.txt", "model.safetensors", "1_Pooling/*"]


def main() -> None:
    name, rev = E5Provider.HF_NAME, E5Provider.REVISION
    src = Path(snapshot_download(name, revision=rev, allow_patterns=PATTERNS))
    repo_dir = REPO / ".model_cache" / "hf" / "hub" / ("models--" + name.replace("/", "--"))
    dst = repo_dir / "snapshots" / rev
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.bin", "*.onnx", "*.h5", "*.ot", "*.msgpack"))
    (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
    (repo_dir / "refs" / "main").write_text(rev, encoding="utf-8")
    size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
    print(f"staged {name}@{rev[:12]} ({size / 1e6:.0f} MB) -> {dst}")


if __name__ == "__main__":
    main()
