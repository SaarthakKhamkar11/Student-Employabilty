# Build context is the repo root (see docker-compose.yml), so the image can also
# include config/ and src/ (the existing data pipeline and resume parser).
FROM python:3.11.17-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/srv/backend:/srv

WORKDIR /srv/backend

COPY backend/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# The existing data pipeline and parsers (config/, src/) live beside the backend: /srv/config, /srv/src
COPY config /srv/config
COPY src /srv/src
COPY backend/app ./app
COPY backend/migrations ./migrations

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
# Apply pending SQL migrations, then serve.
CMD ["sh", "-c", "python -m app.db.migrate && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
