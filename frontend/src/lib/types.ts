// Shapes of the backend's Phase 3-12 contracts (backend/app/schemas/agent_io.py), as the portal returns them.
// A field can always hold a sentinel string instead of its value ("Not yet evaluated", "Insufficient evidence").

export type Sentinel = string;
export type Maybe<T> = T | Sentinel;

export interface AgentResult<T = Record<string, unknown>> {
  status: "success" | "partial" | "failed" | "skipped";
  data: T | Sentinel;
  skip_reason: string | null;
  error_codes: string[];
}

export interface Run {
  request_id: string;
  status: string;
  task_type: string;
  stored: boolean;
  finished_at: string;
  duration_ms: number;
  summary: string | null;
  model_versions: Record<string, string>;
  errors: { code?: string; message?: string; agent_name?: string }[];
  agents: Record<string, AgentResult>;
  routing: { source?: string; model_used?: string } | string;
}

export interface SkillGapItem {
  skill: string;
  importance_score: number | null;
  transferable_candidate: string | null;
  transferable_similarity: number | null;
}

export interface SkillGapReport {
  matched_skills: string[];
  missing_required: SkillGapItem[];
  missing_preferred: SkillGapItem[];
  partially_related: { skill: string; requirement: string; related_student_skill: string; similarity: number }[];
  importance_model: string;
  similarity_model: string;
  thresholds: Record<string, number>;
  method: string;
  note: string;
}

export interface Employability {
  overall: Maybe<number>;
  dimensions: Record<string, Maybe<number>>;
  method: string;
}

export interface MatchResult {
  student_id: string;
  job_id: string;
  rank: number;
  suitability_score: Maybe<number>;
  semantic_similarity: Maybe<number>;
  employability: Maybe<Employability>;
  skill_gap: Maybe<SkillGapReport>;
  summary: Record<string, string | null>;
  model_versions: Record<string, string>;
}

export interface MatchingOutput {
  mode: string;
  query_id: string;
  matches: Maybe<MatchResult[]>;
  funnel: Maybe<Record<string, unknown>>;
  pair_check: { eligible: boolean; failed_rules: string[] } | null;
  model_versions: Maybe<Record<string, string>>;
  notes: string[];
  cohort_benchmark: Maybe<CohortBenchmark>;
}

export interface LearningResource {
  resource_name: string;
  resource_url: string;
  source: string;
  relevance_score: number;
  kind: "course" | "video";
  provider: string;
  difficulty: string | null;
  prerequisites: string | null;
  estimated_effort: string | null;
}

export interface LearningRecommendation {
  skill: string;
  requirement: "required" | "preferred";
  priority: number;
  importance_score: number | null;
  tied_with: string[];
  transferable_candidate: string | null;
  resources: LearningResource[];
  message: string | null;
  explanation: string;
}

export interface LearningOutput {
  student_id: string;
  job_id: string;
  recommendations: Maybe<LearningRecommendation[]>;
  sources: Record<string, unknown>;
  ranking: Record<string, unknown>;
  note: string;
}

export interface MatchExplanation {
  student_id: string;
  job_id: string;
  job_title: string | null;
  suitability_score: number;
  positive_evidence: string[];
  potential_gaps: string[];
  model_evidence: string[];
  how_to_improve: string[];
  text: string;
  source: "llm_validated" | "structured_fallback";
  generation: Record<string, unknown>;
  model_versions: Record<string, string>;
}

export interface CohortBenchmark {
  student_id: string;
  job_id: string | null;
  method_version: string;
  target_role: Record<string, unknown>;
  filters: { name: string; value: string | null; applied: boolean; note: string }[];
  relaxed: string[];
  cohort_size: number;
  comparable_size: number;
  similarity: Record<string, unknown>;
  suitability: Maybe<{
    job_id: string; job_title: string | null; student_score: number; cohort_average: number; cohort_median: number;
    percentile: number; peers_scored: number; peers_unscored: number;
  }>;
  skill_coverage: Maybe<{ common_skills: { skill: string; peers_with_share: number }[]; coverage: Maybe<number>; covered: string[]; missing: string[] }>;
  relative_skill_gap: Maybe<{
    job_id: string; student_missing_count: number; cohort_average_missing_count: number; fewer_missing_than_percentile: number;
    distinctive_gaps: { skill: string; peers_with_share: number }[]; shared_gaps: { skill: string; peers_with_share: number }[];
    advantages: { skill: string; peers_with_share: number }[];
  }>;
  model_versions: Maybe<Record<string, string>>;
  privacy: Record<string, unknown>;
  notes: string[];
}

export interface AiSummary {
  status?: "ok" | "unavailable";
  explanation?: MatchExplanation;
  llm_model?: string | null;
  request_id?: string;
  created_at?: string;
  cached?: boolean;
  message?: string;
  reason?: string;
}

export function data<T>(run: Run | null | undefined, agent: string): T | null {
  const a = run?.agents?.[agent];
  if (!a || typeof a.data !== "object" || a.data === null) return null;
  return a.data as T;
}
