export interface RunInfo {
  key: string;
  label: string;
  slurm: string;
  has_loss: boolean;
  has_summary: boolean;
}

export interface DashboardConfig {
  host: string;
  user: string;
  remote_workdir: string;
  has_password: boolean;
  runs: RunInfo[];
}

export interface SshTestResult {
  ok: boolean;
  hostname: string;
  user: string;
  raw: string;
}

export interface QueueRow {
  [column: string]: string;
}

export interface TrainStats {
  device?: string;
  params?: number;
  params_millions?: number;
  examples?: { train?: number; val?: number; test?: number };
  train_ratio?: number;
  val_ratio?: number;
  vocab_size?: number;
  epochs?: number;
  batch_size?: number;
  max_seq_len?: number;
  d_model?: number;
  num_layers?: number;
  num_heads?: number;
  family_dropout?: number;
  precision?: string;
  ddp?: boolean;
  world_size?: number;
  best_epoch?: number;
  best_val_loss?: number | null;
  total_train_sec?: number;
  gpu_name?: string;
  gpu_total_gb?: number;
  gpu_peak_alloc_gb?: number;
  gpu_peak_reserved_gb?: number;
  gpu_peak_pct?: number;
}

export interface GpuTimelineSummary {
  samples?: number;
  avg_util?: number | null;
  max_util?: number | null;
  avg_mem_gb?: number | null;
  max_mem_gb?: number | null;
  avg_power_w?: number | null;
  max_power_w?: number | null;
}

export interface GpuTimelinePoint {
  t: number;
  util?: number | null;
  util_mem?: number | null;
  mem_gb?: number | null;
  mem_total_gb?: number | null;
  power?: number | null;
}

export interface JobResources {
  state?: string;
  elapsed?: string;
  total_cpu?: string;
  req_mem?: string;
  exit_code?: string;
  max_rss_mb?: number | null;
  max_vmsize_mb?: number | null;
  alloc_tres?: string;
  cpus?: number;
  alloc_mem_mb?: number | null;
  gpus?: number;
  train_stats?: TrainStats;
  gpu_timeline?: GpuTimelineSummary;
}

export interface DatasetSnapshot {
  count_param?: number | null;
  seed?: number | null;
  total_sequences?: number | null;
  total_step_rows?: number | null;
  generated_at?: number | null;
  generated_on?: string;
  families?: Record<string, number | null>;
}

export interface JobError {
  state?: string;
  exit_code?: string | null;
  message?: string;
  source?: "err" | "out" | null;
}

export interface JobRecord {
  run_key: string;
  label: string;
  job_id: string | null;
  slurm_script: string;
  status: string;
  submitted_at: number;
  updated_at: number;
  note: string;
  params?: Record<string, number | string | string[] | null>;
  archived?: boolean;
  submission_fetched?: boolean;
  resources?: JobResources;
  dataset?: DatasetSnapshot;
  error?: JobError;
}

export interface DatasetFamily {
  family: string;
  file: string;
  sequences: number | null;
  step_rows: number | null;
  remote_bytes: number | null;
  uploaded_at: number | null;
}

export interface DatasetInfo {
  present: boolean;
  last_upload: number | null;
  seed?: number | null;
  count_param?: number | null;
  generated_at?: number | null;
  total_sequences?: number | null;
  total_step_rows?: number | null;
  families: DatasetFamily[];
  stale: boolean;
  local_total_sequences?: number | null;
}

export interface DatasetPreview {
  family: string;
  file: string;
  lines: number;
  mtime: number | null;
  exists: boolean;
  rows: { seq_id: string; step: string }[];
  sequences_shown: number;
}

export interface DatasetFamilyDetail {
  file?: string;
  sequences?: number | null;
  step_rows?: number | null;
  bytes?: number | null;
}

export interface DatasetSummary {
  id: string;
  // The legacy/default training_data/ folder (not deletable, always usable).
  legacy: boolean;
  ready: boolean;
  families: string[];
  family_detail: Record<string, DatasetFamilyDetail>;
  count_param: number | null;
  seed: number | null;
  total_sequences: number | null;
  total_step_rows: number | null;
  generated_at: number | null;
  bytes: number | null;
  // Phase 2: a packed/ memmap blob exists, so train/eval run on the full set
  // with near-zero RAM.
  packed?: boolean;
}

export interface SubmissionFile {
  name: string;
  path: string;
  exists: boolean;
  rows: number | null;
  bytes: number | null;
  mtime: number | null;
}

export interface DirEntry {
  name: string;
  is_dir: boolean;
  bytes: number | null;
  mtime: number | null;
  rows: number | null;
}

export interface DirListing {
  exists: boolean;
  entries: DirEntry[];
}

export interface RemoteListing {
  dir: string;
  exists: boolean;
  entries: DirEntry[];
}

export interface SubmissionStatus {
  participant_dir: string;
  output_dir: string;
  script: SubmissionFile;
  checkpoint: SubmissionFile;
  vocab: SubmissionFile;
  inputs: Record<string, SubmissionFile>;
  outputs: Record<string, SubmissionFile>;
  output_listing: DirListing;
  anomaly_invalid: number | null;
  ready: {
    anomaly: boolean;
    next_step: boolean;
    completion: boolean;
  };
}

export interface CheckpointInfo {
  source: string;
  vocab: string;
  label: string;
  is_current: boolean;
  bytes: number | null;
  mtime: number | null;
}

export interface SubmissionRunResponse {
  ok: boolean;
  job_id: string | null;
  run_key: string;
  checkpoint: CheckpointInfo | null;
  raw: string;
  status: SubmissionStatus;
}

export interface CheckpointRemoveResponse {
  ok: boolean;
  removed: string;
  checkpoints: CheckpointInfo[];
}

export interface LossPoint {
  epoch: number;
  train_loss?: number;
  val_loss?: number;
  train_acc?: number;
  val_acc?: number;
  lr?: number;
  sec?: number;
  [key: string]: number | undefined;
}

export interface RunParams {
  epochs?: number;
  learning_rate?: number;
  batch_size?: number;
  max_seq_len?: number;
  d_model?: number;
  num_layers?: number;
  num_heads?: number;
  dropout?: number;
  family_dropout?: number;
  weight_decay?: number;
  label_smoothing?: number;
  lr_schedule?: string;
  warmup_ratio?: number;
  num_workers?: number;
  train_ratio?: number;
  val_ratio?: number;
  // Cap sequences read per family (0 = all); bounds RAM on huge datasets.
  max_sequences?: number;
  gpus?: number;
  ddp?: boolean;
  keep_checkpoint?: boolean;
  families?: string[];
  source_job_id?: string;
  seed?: number;
  count?: number;
  time_limit?: string;
  // Dataset folder id (datasets/<id>) to train/evaluate against.
  dataset?: string;
}

export interface ExperimentAnalysisCard {
  verdict: "bad" | "promising" | "good";
  englishSummary: string;
  diagnosis: string[];
  confidence: number;
  currentBottleneck: string;
  riskNotes: string[];
  fullSuggestion: {
    title: string;
    runKey: string;
    params: RunParams;
    reasonBySetting: Record<string, string>;
  };
  singleChangeSuggestion: {
    field: string;
    from: string | number | boolean | null;
    to: string | number | boolean;
    reason: string;
    expectedObservableDifference: string;
  };
  ablationPlan: { label: string; params: RunParams; reason: string }[];
  actionProposal: {
    nextAction:
      | "generate_data"
      | "generate_remote"
      | "upload"
      | "train"
      | "evaluate"
      | "cancel_job"
      | "wait";
    toolName: string;
    params: Record<string, unknown>;
    approvalText: string;
    expectedCostRisk: string;
    requiresApproval: boolean;
  };
}

export interface ExperimentAnalysisResponse {
  card: ExperimentAnalysisCard;
  snapshot?: unknown;
  model: string;
  warning?: string;
  cached?: boolean;
  cachedAt?: number;
}

export interface ResultsPayload {
  run: string;
  summary: Record<string, string>[];
  rule_counts: Record<string, string>[];
  split: Record<string, unknown> | null;
  job_id?: string | null;
  archived?: boolean;
}

export interface RuleViolation {
  rule: string;
  step_index: number;
  step_name: string;
  description: string;
}

export interface ValidateResult {
  steps: number;
  valid: boolean;
  violations: RuleViolation[];
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail =
        typeof body.detail === "string"
          ? body.detail
          : body.detail
            ? JSON.stringify(body.detail)
            : detail;
    } catch {
      // keep statusText
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  getConfig: () => request<DashboardConfig>("/config"),
  sshTest: () => request<SshTestResult>("/ssh/test", { method: "POST" }),
  upload: () =>
    request<{ uploaded: { file: string; ok: boolean }[]; count: number }>("/upload", {
      method: "POST",
    }),
  setup: () =>
    request<{ ok: boolean; torch_version: string; cuda_build: string; stdout: string; stderr: string }>(
      "/setup",
      { method: "POST" }
    ),
  runJob: (runKey: string, params?: RunParams) =>
    request<{ job_id: string | null; run_key: string; raw: string; dataset_id?: string | null }>(
      `/run/${runKey}`,
      {
        method: "POST",
        body: JSON.stringify(params ?? {}),
      }
    ),
  getQueue: () => request<{ rows: QueueRow[]; raw: string }>("/queue"),
  getDataset: () => request<DatasetInfo>("/dataset"),
  listDatasets: () => request<{ datasets: DatasetSummary[] }>("/datasets"),
  deleteDataset: (id: string) =>
    request<{ ok: boolean; deleted: string }>(`/datasets/${encodeURIComponent(id)}`, {
      method: "DELETE",
    }),
  getDatasetPreview: (family: string, lines = 200) =>
    request<DatasetPreview>(`/dataset/preview?family=${family}&lines=${lines}`),
  getSubmission: () => request<SubmissionStatus>("/submission"),
  getSubmissionRemote: () => request<RemoteListing>("/submission/remote"),
  runSubmission: (body: { source?: string; tasks?: string[] } = {}) =>
    request<SubmissionRunResponse>("/submission/run", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  getCheckpoints: () =>
    request<{ checkpoints: CheckpointInfo[] }>("/submission/checkpoints"),
  removeCheckpoint: (source: string) =>
    request<CheckpointRemoveResponse>("/submission/remove-checkpoint", {
      method: "POST",
      body: JSON.stringify({ source }),
    }),
  getJobs: () => request<{ jobs: JobRecord[] }>("/jobs"),
  clearJobs: (runs?: string[]) =>
    request<{ removed: number }>(
      `/jobs${runs && runs.length ? `?runs=${encodeURIComponent(runs.join(","))}` : ""}`,
      { method: "DELETE" }
    ),
  cancelJob: (jobId: string) =>
    request<{ job_id: string; cancelled: boolean; state: string; stderr?: string; detail?: string }>(
      `/jobs/${jobId}/cancel`,
      { method: "POST" }
    ),
  getJobStatus: (jobId: string) =>
    request<{ job_id: string; state: string | null; info: Record<string, string> | null; raw: string }>(
      `/jobs/${jobId}/status`
    ),
  getResults: (run: string, jobId?: string) =>
    request<ResultsPayload>(
      `/results?run=${run}${jobId ? `&job_id=${jobId}` : ""}`
    ),
  getLossSnapshot: (run: string, jobId?: string) =>
    request<{ run: string; rows: LossPoint[]; archived?: boolean; job_id?: string | null }>(
      `/loss/snapshot?run=${run}${jobId ? `&job_id=${jobId}` : ""}`
    ),
  getGpuTimeline: (run: string, jobId?: string) =>
    request<{
      run: string;
      rows: GpuTimelinePoint[];
      summary: GpuTimelineSummary;
      job_id?: string | null;
    }>(`/gpu/timeline?run=${run}${jobId ? `&job_id=${jobId}` : ""}`),
  validateSequence: (text: string) =>
    request<ValidateResult>("/validate", {
      method: "POST",
      body: JSON.stringify({ text }),
    }),
  analyzeExperiment: (body: {
    prompt?: string;
    currentControls?: RunParams & {
      train_run?: string;
      eval_run?: string;
      generation_time_limit?: string;
    };
    selectedJobId?: string | null;
    trainRun?: string;
    evalRun?: string;
    forceRefresh?: boolean;
  }) =>
    request<ExperimentAnalysisResponse>("/ai/analyze", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  aiChat: (messages: { role: "user" | "assistant"; content: string }[]) =>
    fetch("/api/ai/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages }),
    }),
};
