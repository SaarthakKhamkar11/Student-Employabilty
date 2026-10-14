# Build context is the repo root (see docker-compose.yml), so the image can also
# include config/ and src/ (the existing data pipeline and resume parser).
FROM python:3.11.17-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/srv/backend:/srv

WORKDIR /srv/backend

# libgomp1: OpenMP runtime LightGBM needs (Phase 8 matching-model candidates)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# The selected embedding model (ADR 0012): CPU torch + sentence-transformers, then the E5 weights at the pinned
# revision (the one Phase 6 measured; app/embeddings/provider.py E5_REVISION). Separate layers, so a change to
# requirements.txt doesn't download them again. The container then runs fully offline (HF_HUB_OFFLINE=1).
COPY backend/requirements-embeddings.txt ./requirements-embeddings.txt
RUN pip install -r requirements-embeddings.txt
# The weights are staged into the build context first: python scripts/stage_embedding_model.py (-> .model_cache/hf,
# gitignored). It copies the pinned snapshot from the local Hugging Face cache, so the image build doesn't download it.
ENV HF_HOME=/srv/hf HF_HUB_OFFLINE=1
COPY .model_cache/hf /srv/hf

# pip downloads persist in a BuildKit cache mount, so a failed or repeated build doesn't fetch them again.
COPY backend/requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip env -u PIP_NO_CACHE_DIR pip install -r requirements.txt

# The existing data pipeline and parsers (config/, src/) live beside the backend: /srv/config, /srv/src
COPY config /srv/config
COPY src /srv/src
COPY backend/app ./app
COPY backend/migrations ./migrations
# Phase 8 trained matching models (hash-checked against their manifest before loading)
COPY models /srv/models

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
# Apply pending SQL migrations, then serve.
CMD ["sh", "-c", "python -m app.db.migrate && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
