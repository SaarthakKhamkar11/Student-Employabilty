-- Phase 4: the Profile Agent saves structured profiles.
-- An uploaded resume does not state branch, career interests or domain, so these
-- stay NULL (never guessed) and the NOT NULL constraints from 0002 are relaxed.
ALTER TABLE student_profiles ALTER COLUMN branch DROP NOT NULL;
ALTER TABLE student_profiles ALTER COLUMN career_interests DROP NOT NULL;
ALTER TABLE student_profiles ALTER COLUMN domain DROP NOT NULL;

-- The structured profile (PII-free by construction; see StudentProfile) and its provenance.
ALTER TABLE student_profiles ADD COLUMN IF NOT EXISTS profile JSONB;
ALTER TABLE student_profiles ADD COLUMN IF NOT EXISTS profile_extractor_version TEXT;
ALTER TABLE student_profiles ADD COLUMN IF NOT EXISTS profile_updated_at TIMESTAMPTZ;
-- What the PII detector flagged: types and counts only, never the values.
ALTER TABLE student_profiles ADD COLUMN IF NOT EXISTS pii_flags JSONB;
