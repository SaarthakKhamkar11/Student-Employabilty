-- Fix for 0005: its DROP CONSTRAINT IF EXISTS named the old constraint by its full auto-generated
-- name, but Postgres truncates identifiers to 63 characters
-- (student_skills_student_id_skill_raw_source_extractor_versio_key), so the drop was a silent no-op.
-- Find the old constraint by its definition instead of by name.
DO $$
DECLARE old_name text;
BEGIN
    FOR old_name IN
        SELECT conname FROM pg_constraint
        WHERE conrelid = 'student_skills'::regclass AND contype = 'u'
          AND pg_get_constraintdef(oid) = 'UNIQUE (student_id, skill_raw, source, extractor_version)'
    LOOP
        EXECUTE format('ALTER TABLE student_skills DROP CONSTRAINT %I', old_name);
    END LOOP;
END $$;
