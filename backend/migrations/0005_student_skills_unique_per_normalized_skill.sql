-- Phase 4 fix: one raw item can yield several skills ("Git/GitHub" -> Git, GitHub). The 0002 unique key
-- (student_id, skill_raw, source, extractor_version) made the second one collide, and ON CONFLICT DO
-- NOTHING silently dropped it. The normalised name is now part of the key.
ALTER TABLE student_skills DROP CONSTRAINT IF EXISTS student_skills_student_id_skill_raw_source_extractor_version_key;
ALTER TABLE student_skills ADD CONSTRAINT student_skills_unique_skill
    UNIQUE (student_id, skill_raw, skill_normalized, source, extractor_version);
