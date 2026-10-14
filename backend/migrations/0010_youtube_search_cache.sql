-- Phase 10 (docs/adr/0018): cache of YouTube Data API searches, so a repeated skill doesn't spend quota again
-- (search.list costs 100 of the default 10,000 daily units). The videos themselves are learning_resources rows
-- (resource_id 'youtube:<videoId>'), with their API provenance in source_reference.
CREATE TABLE IF NOT EXISTS youtube_searches (
    query           TEXT PRIMARY KEY CHECK (btrim(query) <> ''),
    video_ids       TEXT[] NOT NULL DEFAULT '{}',   -- in the API's result order; empty = the API found nothing
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    request         JSONB NOT NULL DEFAULT '{}'::jsonb   -- the search parameters, never the API key
);
